# Vincent - AI Artist

# Import dependencies
import numpy as np
import time
import os
import argparse
import h5py

from scipy.misc import imread, imresize, imsave
from scipy.optimize import fmin_l_bfgs_b

from sklearn.preprocessing import normalize

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, AveragePooling2D
from keras import backend as Kr

Kr.set_image_dim_ordering('th')

# Command line arguments
parser = argparse.ArgumentParser(description='AI Artist')

parser.add_argument('--base_img_path',  metavar='base', type=str, help='Path to base image')
parser.add_argument('--style_img_path', metavar='ref',  type=str, help='Path to artistic style reference image')
parser.add_argument('--result_prefix',  metavar='res',  type=str, help='Prefix for saved results')

parser.add_argument('--rescale',        dest='rescale',        default='True',    type=str,   help='Rescale image after execution')
parser.add_argument('--keep_aspect',    dest='keep_aspect',    default='True',    type=str,   help='Maintain aspect ratio of image')
parser.add_argument('--tot_var_weight', dest='tv_weight',      default=1e-3,      type=float, help='Total variation in weights')
parser.add_argument('--content_weight', dest='content_weight', default=0.025,     type=float, help='Weight of content')
parser.add_argument('--style_weight',   dest='style_weight',   default=1,         type=float, help='Weight of style')
parser.add_argument('--img_size',       dest='img_size',       default=512,       type=int,   help='Output image size')
parser.add_argument('--content_layer',  dest='content_layer',  default='conv5_2', type=str,   help="Optional: 'conv4_2'")
parser.add_argument('--init_image',     dest='init_image',     default='content', type=str,   help="Initial image used to generate the final image. Options are: 'content' or 'noise'")
parser.add_argument('--num_iter',       dest='num_iter',       default=10,        type=int,   help='Number of iterations')

# Helper methods

## Convert string to boolean
def strToBool(str):
    return str.lower() in ('true', 'yes', 't', 1)

## Open, resize and format pictures into tensors
def preprocess(img_path, load_dims=False):
    global img_WIDTH, img_HEIGHT, aspect_ratio

    img = imread(img_path, mode="RGB")

    if load_dims:
        img_WIDTH    = img.shape[0]
        img_HEIGHT   = img.shape[1]
        aspect_ratio = img_HEIGHT / img_WIDTH

    img = imresize(img, (img_width, img_height))
    img = img.transpose((2, 0, 1)).astype('float64')
    img = np.expand_dims(img, axis=0)
    return img

## Convert a tensor into a valid image
def deprocess(x):
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')

    return x

## Load weights
def load_weights(weight_path, model):
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'

    f = h5py.File(weights_path)

    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break

        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)

    f.close()
    print('Model loaded.')

## Gram matrix of an image tensor
def gram_matrix(x):
    assert Kr.ndim(x) == 3

    features = Kr.batch_flatten(x)
    gram = Kr.dot(features, Kr.transpose(features))

    return gram

## Evaluate loss and gradients
def eval_loss_and_grads(x):
    x = x.reshape((1, 3, img_width, img_height))
    outs = f_outputs([x])
    loss_value = outs[0]

    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')

    return loss_value, grad_values

## Style loss based on gram matrices
def style_loss(style, combination):
    assert Kr.ndim(style) == 3
    assert Kr.ndim(combination) == 3

    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_width * img_height

    return Kr.sum(Kr.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

## Content loss
def content_loss(base, combination):
    return Kr.sum(Kr.square(combination - base))

## Total variation loss
def total_variation_loss(x):
    assert Kr.ndim(x) == 4

    a = Kr.square(x[:, :, :img_width-1, :img_height-1] - x[:, :, 1:, :img_height-1])
    b = Kr.square(x[:, :, :img_width-1, :img_height-1] - x[:, :, :img_width-1, 1:])

    return Kr.sum(Kr.pow(a + b, 1.25))

## Combined loss function - combines all three losses into one single scalar
def get_total_loss(outputs_dict):
    loss = Kr.variable(0.)
    layer_features = outputs_dict[args.content_layer] # 'conv5_2' or 'conv4_2'
    base_image_features  = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(base_image_features, combination_features)
    feature_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    for layer_name in feature_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features     = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(feature_layers)) * sl

    loss += tv_weight * total_variation_loss(comb_img)

    return loss

## Combine loss and gradient
def combine_loss_and_gradient(loss, gradient):
    outputs = [loss]

    if type(grads) in {list, tuple}:
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = Kr.function([comb_img], outputs)

    return f_outputs

## Prepare image
def prepare_image():
    assert args.init_image in ['content', 'noise'] , "init_image must be one of ['content', 'noise']"

    if 'content' in args.init_image:
        x = preprocess(base_img_path, True)
    else:
        x = np.random.uniform(0, 255, (1, 3, img_width, img_height))

    num_iter = args.num_iter

    return x, num_iter

## The Evaluator class makes it possible to compute loss and gradients in one pass
class Evaluator(object):
    def __init__(self):
        self.loss_value   = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None

        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value  = loss_value
        self.grad_values = grad_values

        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None

        grad_values = np.copy(self.grad_values)
        self.loss_value  = None
        self.grad_values = None

        return grad_values

evaluator = Evaluator()

# Base image, style image, and result image paths
args = parser.parse_args()
base_img_path  = args.base_img_path
style_img_path = args.style_img_path
result_prefix  = args.result_prefix

# The weights file
weights_path = r"vgg16_weights.h5"

# Init bools to decide whether or not to resize
rescale     = strToBool(args.rescale)
keep_aspect = strToBool(args.keep_aspect)

# Init variables for style and content weights
tv_weight      = args.tv_weight
content_weight = args.content_weight
style_weight   = args.style_weight

# Init dimensions of the generated picture
img_width = img_height = args.img_size
img_WIDTH = img_HEIGHT = 0
aspect_ratio = 0

# Tensor representations of images
base_img  = Kr.variable(preprocess(base_img_path, True))
style_img = Kr.variable(preprocess(style_img_path))

# This will hold the output image
comb_img = Kr.placeholder((1, 3, img_width, img_height))

# Combining three images into one single tensor
inp_tensor = Kr.concatenate([base_img, style_img, comb_img], axis=0)

# Building the VGG16 network (31 layers) with our three images as input
layer0 = ZeroPadding2D((1, 1))
layer0.set_input(inp_tensor, shape=(3, 3, img_width, img_height))

model = Sequential()
model.add(layer0)
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

# Load weights for the VGG16 networks
load_weights(weights_path, model)

# Get symbolic output of each key layer (named layers)
out_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Combined loss (style, content, and total variation loss combined into one single scalar)
tot_loss = get_total_loss(out_dict)

# Gradients of the generated image with respect to the loss
grads = Kr.gradients(tot_loss, comb_img)

# Combine loss and gradient
f_outputs = combine_loss_and_gradient(tot_loss, grads)

# L-BFGS over pixels of the generated image to minimize neural style loss
x, num_iter = prepare_image()

for i in range(num_iter):
    # Step 1 : record iterations
    print('Starting iteration', (i+1))
    start_time = time.time()

    # Step 2 : L-BFGS optimization function using loss and gradient
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
    print('Current loss value: ', min_val)

    # Step 3 : get generated image
    img = deprocess(x.reshape((3, img_width, img_height)))

    # Step 4 : keep aspect ratio
    if (keep_aspect) & (not rescale):
        img_ht = int(img_width * aspect_ratio)
        img = imresize(img, (img_width, img_ht), interp='bilinear')

    if rescale:
        img = imresize(img, (img_WIDTH, img_HEIGHT), interp='bilinear')

    # Step 5 : save generated image
    fname = result_prefix + '_at_iteration_%d.jpg' % (i+1)
    imsave(fname, img)

    end_time = time.time()

    print('Image saved as: ', fname)
    print('Iteration %d completed in %ds' % (i+1, end_time - start_time))
