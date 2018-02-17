# StackGAN++
TensorFlow implementation of StackGAN++, 
described in the paper by Zhang, Xu et al.

## Run project

**Dependencies.**
Python 3, TensorFlow 1.4 (, TensorBoard)

**Data.**
Currently uses CIFAR-10 images to train on, which is downloaded when running the `train.py` script.

**Training.**
1. Clone the repo, including the TensorFlow models submodule:
   ```shell
   git clone --recurse-submodules https://github.com/jppgks/stackgan-pp.git
   ```

2. Run the training script
   ```shell
   python ./train.py
   ```
   optionally with arguments. 
   All possible arguments, with their doc strings, are listed when running:
   ```shell
   python ./train.py --help
   ```

3. Follow progress in TensorBoard:
   ```shell
   tensorboard --logdir=<TRAIN_LOG_DIR location>
   ```

## Project structure
The project aims to reproduce StackGAN++ paper results by introducing 
as little modifications as possible to the existing [TFGAN framework](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/gan).

**TFGAN &rarr; TFSTACKGAN.**
[`tfstackgan`](tfstackgan/) mimics the folder structure of TFGAN.
[`tfstackgan/python/train.py`](tfstackgan/python/train.py) contains 
modified TFGAN `train.py` functions. 
The color loss for the generator is defined in [`tfstackgan/python/losses/python/losses_impl.py`](tfstackgan/python/losses/python/losses_impl.py).
The [`./train.py`](train.py) and [`./networks.py`](networks.py) scripts are modeled after the TFGAN [CIFAR example](https://github.com/tensorflow/models/tree/master/research/gan/cifar).

## Comparison with paper
**Conditioning.**
Currently, only the unconditional setting is implemented.

**Upsampling.**
This implementation does not make use of GLUs and/or residual blocks at the moment.
Upsampling in all generator stages happens through [fractionally strided convolutions](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d_transpose).

**Loss.**
This implementation uses the Wasserstein loss, with optional gradient penalty, 
whereas the paper uses the non-saturating minimax loss.

## References
- [TFGAN](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/gan)
- [StackGAN++ PyTorch implementation](https://github.com/hanzhanggit/StackGAN-v2)
   
