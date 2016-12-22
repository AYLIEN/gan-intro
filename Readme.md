# An introduction to Generative Adversarial Networks

This is the code that we used to generate our GAN 1D Gaussian approximation.
For more information see our blog post: [http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow).

## Installing dependencies

Written for Python 2.7.x.

For the Python dependencies, first install the requirements file:

    $ pip install -r requirements.txt

You should then install TensorFlow `0.12`, see: [https://www.tensorflow.org/get_started/os_setup#pip_installation](https://www.tensorflow.org/get_started/os_setup#pip_installation).

If you want to also generate the animations, you need to also make sure that `ffmpeg` is installed and on your path.

## Training

For a full list of parameters, run:

    $ python gan.py --help

To run without minibatch discrimination (and plot the resulting distributions):

    $ python gan.py

To run with minibatch discrimination (and plot the resulting distributions):

    $ python gan.py --minibatch True
