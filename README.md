# Vincent : AI Artist

A neural algorithm for artistic style transfer. Feel free to criticize, copy/share. Basically, do whatever you want to do with the code.

### Requirements

* Numpy (http://www.numpy.org/)
* Keras (http://keras.io/)
* Scipy  (https://www.scipy.org/)
* Pillow (https://python-pillow.org/)
* Theano (http://deeplearning.net/software/theano/)
* h5py (http://h5py.org/)
* sklearn (http://scikit-learn.org/)
* VGG16 file (https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing)

### Basic Usage

```
python main.py --base_img_path /path/to/base/image --style_img_path /path/to/artistic/image --result_prefix output
```

### Results

|![result_00](img/donelli/output.png)|
|-------------------------------|

### References

* Inceptionism: Going Deeper into Neural Networks (https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
* A Neural Algorithm of Artistic Style (http://arxiv.org/pdf/1508.06576v2.pdf)
