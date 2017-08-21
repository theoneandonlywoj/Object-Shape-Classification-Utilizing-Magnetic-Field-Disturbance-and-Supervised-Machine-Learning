from __future__ import division, print_function, absolute_import

import tflearn
import time
from import_data import *
from network import *

# Recalling the network defined in network.py
available_networks = ['MLP-3', 'MLP-4', 'MLP-5', 'SimpleCNN','VGG-16', 'ResNet-18', 'PrimeInception-ES','PrimeInception-C', 
					  'SaraNet-P-ES', 'SaraNet-P-C', 'SaraNet-3x3-ES', 'SaraNet-3x3-C']

name = available_networks[8]

#model = ANN(NAME = name, WIDTH = 28, HEIGHT = 28, CHANNELS = 1, LABELS = 10)
#model = ANN(NAME = name, WIDTH = 32, HEIGHT = 32, CHANNELS = 3, LABELS = 10)
model = ANN(NAME = name, WIDTH = 50, HEIGHT = 50, CHANNELS = 1, LABELS = 4)

# Loading the best network
print('Evaluation for:', str(name))
model_name = input('Input name of the best model: ')
model_source = './checkpoints/best/' + name + '/' + str(model_name)
model.load(model_source)
print('*' * 70)
print('Model is successfully loaded for the best performance!')

#x_test, y_test =  get_data_MNIST_native(dataset = 'Test')
#x_test, y_test = get_data_CIFAR10(dataset = 'Test')
x_test, y_test =  get_data_magnetic(dataset = 'Test')

print('Data successfully loaded!')
# Evaluation
print('Evaluation in progress...')

acc = model.evaluate(x_test, y_test, batch_size = 10)
print('Size:', x_test.shape)
print('Test data accuracy:', acc)

#tensorboard --logdir=logs/	




