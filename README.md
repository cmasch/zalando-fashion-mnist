# Image classification on fashion-MNIST
I would like to share my results (93.43% accuracy on average) on the [fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. You can find further informations about the dataset on [Zalando Research](https://github.com/zalandoresearch/fashion-mnist) and [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist)
This dataset is a great option instead of using traditional handwritten MNIST.<br>
Thanks to [Han](https://github.com/hanxiao) and [Kashif](https://github.com/kashif)!

## Evaluation procedure
I splitted the training data randomly in train (80%) and validation (20%). The testset of 10k images are used for final evaluation. I created 5 models with the same architecture but with random train/validation data. I only saved the weights of every model with best loss. Finally I used the models to evaluate them on the testset. The average loss/accuracy of the 5 models is the final result.<br>
Furthermore I evaluate the generalization of the network for classifying traditional MNIST at the end.

## Results
Actually I'm focussed on a very simple architecture with less than 500,000 parameters which could run on CPU with 4GB memory. If you are interested in further results, you can find them on [Zalando benchmark](https://github.com/zalandoresearch/fashion-mnist#benchmark). If it fits in time, I will evaluate architectures e.g. DenseNet.

I'm just using two convolutional layers, batchnorm, dropout and three fully connected layers. For a detailed implementation check the model definition in the [jupyter notebook](https://github.com/cmasch/zalando-fashion-mnist/blob/master/Simple_Convolutional_Neural_Network_Fashion-MNIST.ipynb).<br>
This is the summary of the current model:<br>
<kbd><img src="./images/scnn-model-summary.png"></kbd>

It's easy to reach an accuracy over 90% but around 92% it stucks very fast. If we take a look on the confusion matrix, we will find out that distinguishing between Shirt and T-Shirt/top is very difficult:<br>
<img src="./images/scnn-fashion_mnist-confusion_matrix.png">

Here's an example where my model fails. I'm not a fashion expert. Maybe thats the reason why I don't see any difference. If you could explain why this is a shirt and not a t-shirt I would appreciate it:
<img src="./images/fashion_mnist-samples.png">

Another two plots which illustrate the accuracy / loss of training and validation over the time:<br>
<kbd><img src="./images/scnn-fashion_mnist-training.png"><br>
<img src="./images/scnn-fashion_mnist-validation.png"></kbd>

And a detailed plot of iteration 0:<br>
<kbd><img src="./images/scnn-fashion_mnist-train_validation-model_0.png"></kbd>

Scores for training: 0.1066 loss / 95.99% accuracy.<br>
Scores for validation: 0.1245 loss / 95.64% accuracy.<br>
Scores for test: 0.2149 loss / 93.43% accuracy.

If you like, you can [download](https://github.com/cmasch/zalando-fashion-mnist/tree/master/models/simple_cnn/fashion_mnist) the saved models/weights and history for fashion-MNIST.

#### MNIST
I used the same architecture of neural network to train on traditional MNIST.<br>
Scores for training: 0.078 loss / 99.74% accuracy.<br>
Scores for validation: 0.161 loss / 99.61% accuracy.<br>
Scores for test: 0.0248 loss / 99.43% accuracy.

In the plot below you can track training / validation for MNIST:
<kbd><img src="./images/scnn-mnist-train-validation.png"></kbd>

If you like, you can [download](https://github.com/cmasch/zalando-fashion-mnist/tree/master/models/simple_cnn/mnist) the saved models/weights and history for MNIST.

### Overview

| Fashion-MNIST<br>test accuracy | Fashion-MNIST<br>train accuracy | Fashion-MNIST<br>validation accuracy | MNIST<br> test accuracy | Add. Settings | Trainable<br>Params |
| :---: | :---: | :---: | :---: | --- | :---: |
| 93.43% | 95.99% | 95.64% | 99.43% | BatchSize : 250<br> Epochs : 80<br> Data augmentation (2x) | 493,772

## Requirements
- [Anaconda](https://www.continuum.io/downloads)
- [Keras 2.x](https://keras.io/)
- [OpenCV 3.x](http://opencv.org/)
- [TensorFlow 1.x](https://www.tensorflow.org/)

## Usage
Feel free to use this code and models for improving on your own. I would appreciate it if you give any feedback.

The easiest way to use the code examples is to download / clone the whole repository. The only thing that’s missing is the folder `data` with the given subfolder structure for the input images:
```
data
  ├── test
  |   ├── fashion_mnist
  |   |   ├── 0
  |   |   ├── 1
  |   |   ├── ...
  |   |
  |   ├── mnist
  |   |   ├── 0
  |   |   ├── 1
  |   |   ├── ...
  |
  ├── train
  |   ├── fashion_mnist
  |   |   ├── 0
  |   |   ├── 1
  |   |   ├── ...
  |   |
  |   ├── mnist
  |   |   ├── 0
  |   |   ├── 1
  |   |   ├── ...
```
For getting this structure automatically I wrote a extractor finding in this [jupyter notebook](https://github.com/cmasch/zalando-fashion-mnist/blob/master/Extracting_MNIST.ipynb). This extracts the images of a MNIST IDX file.

If you have any questions or hints contact me through an [issue](https://github.com/cmasch/zalando-fashion-mnist/issues). Thanks!

## References
[1] [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)<br>
[2] [Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747)

## Author
Christopher Masch