A summary describing the purpose of each script:

[**mnist.py :**](mnist.py) Script to train a pre-trained model on the mnist dataset. The model has been slightly modified from the original [script](https://github.com/pytorch/examples/blob/master/mnist/main.py). All dropout layers have been removed. An additional fully connected layer and a conv have been added. The number of filters in the convolutions and the number of neurons in the FC layers have also been changed. 

**mnist_cnn.pt:** Pre-trained model weights trained with the [mnist.py](mnist.py) script

**mnist_cnn_nobias.pt:** Pre-trained model weights also trained with the mnist.py script except that the bias in the fully connected layer has been set to false.

[**fc_pruning_nobias.ipynb:**](fc_pruning_nobias.ipynb) Notebook demonstrating pruning of a Fully Connected layer with no bias. The output of the first FC layer is reduced from 128 to 64. Hence the input of the second FC layer is modified accordingly.See also [this figure](highlevel.png) which gives a rudimentary description of the high level overiew of the process.

[**fc_pruning_withbias.ipynb:**](fc_pruning_withbias.ipynb) Same as [fc_pruning_nobias.ipynb](fc_pruning_nobias.ipynb) except that the FC layers also have a bias. Therefore, not only the weights of the FC1 and FC2 need to be modified but also the biases.

[**conv_pruning.ipynb:**](conv_pruning.ipynb) Notebook demonstrating pruning of a conv layer together with a subsequent fully connected layer.
The Fully connected layer has no bias.

