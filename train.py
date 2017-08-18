from __future__ import division, print_function, absolute_import

import tflearn
import time
from import_data import *
from network import *

# Getting the data
x_train, x_val, y_train, y_val = get_data_MNIST_native(dataset = 'Train + Val')
#x_train, x_val, y_train, y_val = get_data_CIFAR10(dataset = 'Train + Val')

# Recalling the network defined in network.py
available_networks = ['MLP-3', 'MLP-4', 'MLP-5', 'SimpleCNN','VGG-16', 'ResNet-18', 'PrimeInception-ES','PrimeInception-C', 
					  'SaraNet-P-ES', 'SaraNet-P-C', 'SaraNet-3x3-ES', 'SaraNet-3x3-C']

name = available_networks[11]
#'MLP-5'

model = ANN(NAME = name, WIDTH = 28, HEIGHT = 28, CHANNELS = 1, LABELS = 10)
#model = ANN(NAME = name, WIDTH = 32, HEIGHT = 32, CHANNELS = 3, LABELS = 10)

# Training
epochs = 100

start = time.time()
model.fit(x_train, y_train, n_epoch = epochs, validation_set = 0.2 , show_metric = True, 
		  batch_size = 100, shuffle = True, snapshot_epoch = True, run_id = name)
print("Training time for", name, ':' , (time.time() - start))
print("Per epoch:", ((time.time() - start) / 100))

# Loading the best network
model_name = input('Input name of the best model: ')
model_source = './checkpoints/best/' + name + '/' + str(model_name)
model.load(model_source)
print('*' * 70)
print('Model is successfully loaded for the best performance!')

# Evaluation
print('Evaluation in progress...')

x_test, y_test =  get_data_MNIST_native(dataset = 'Test')
#x_test, y_test = get_data_CIFAR10(dataset = 'Test')

acc = model.evaluate(x_test, y_test, batch_size = 10)
print('Size:', x_test.shape)
print('Test data accuracy:', acc)

#tensorboard --logdir=logs/	




