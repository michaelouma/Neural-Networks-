inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2,3,0.5]

output_layer = []

for neuron_weights, neuron_biases in zip(weights,biases):
    output = 0 
    for n_input, weight in zip(inputs, neuron_weights):
        output += n_input*weight
    output += neuron_biases
    output_layer.append(output)
        
print(output_layer)


## The .Dot product for the multiplication of matrices and vectors 
import numpy as np

output1 = np.dot(weights, inputs) + biases
print(output1)

## What about multiplying matrix and matrix

inputs = [[1,2,3,2.5],
          [2.0,5.0,-1.0, 2.0],
          [-1.5,2.7,3.3,-0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2,3,0.5]

output2 =np.dot(inputs, np.array(weights).T) + biases
print(output2)


## Putting the input layer to be X and then define the two hidden layers 
### It is hidden because we programmers we are not in charge of how the layers changes 
## We create an object 

X = [[1,2,3,2.5],
     [2.0,5.0,-1.0, 2.0],
     [-1.5,2.7,3.3,-0.8]]

## Initializing the layers 
 
# 1.we initialize wights in random values in range -1 and +1 (the smaller the range the better)
# 2. 

np.random.seed(0)
class Layer_dense:
    def __init__ (self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def foward(self, inputs):
        self.output = np.dot(inputs, self.weights + self.biases)

layer1 = Layer_dense(4,5)
layer2 = Layer_dense(5,2)

layer1.foward(X)
print(layer1.output)

layer2.foward(layer1.output)
print(layer2.output)


## Activation function
import nnfs
from nnfs.datasets import spiral_data

X,y = spiral_data(samples=100, classes=3)


class Activation_ReLu:
    def foward(self, inputs):
        self.output = np.maximum(0, inputs)    

layer3 = Layer_dense(2,5)
activation1 = Activation_ReLu()

print(layer3.output) # Before the activation function the output has negatives

layer3.foward(X)
activation1.foward(layer3.output)
print(activation1.output)  # After Rectified Linear activation function ReLU the negatives are reduced to zero

## Another activation function

## Softmax is the combination of the exponatiation and normalization

# Exponential Function will make good use of any negatives output ruther than just making them to be zero

import math

layer_output = [[4.5,1.21,2.34],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]]


# Euler's number
# E = math.e
exp_values =[]

for output in layer_output:
    exp_values.append(E**output)

print(exp_values)


## Now we normalize the exp_values to get probabilities

norm_base = sum(exp_values)
norm_values = []

for values in exp_values:
    norm_values.append(values/norm_base)

print(norm_values)
print(sum(norm_values))


## Anothe way using the numpy 

exp_values1 = np.exp(layer_output)

norm_values1 = exp_values1/np.sum(exp_values1, axis=1, keepdims= True)

print(norm_values1)
print(sum(norm_values1))


# Overflowing Error due to exponential 
## One way to tackle the overflow is to take all the value and subtract the largest value that is (v = u - max (u))


class Activation_softmax:
    def foward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims= True))
        probabilities = exp_values/ np.sum(exp_values,axis =1, keepdims = True)
        self.output = probabilities


# Now testing how wrong our model is ,,,,by calculationg Loss with categorical cross-entropy
class Loss:
    def calculate(self,output,y):
        sample_losses = self.foward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricaClassEntropy(Loss):  # This class inherites from the class Loss
    def foward(self,y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)


        if len(y_true.shape) == 1:
            correct_confidencs = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) ==2 :
            correct_confidencs = np.sum( y_pred_clipped*y_true, axis =1)

        negative_log_likelihood = -np.log(correct_confidencs)
        return negative_log_likelihood

        

layer4 = Layer_dense(2,3)
activation2 = Activation_ReLu()

layer5 = Layer_dense(3,3)
activation3 = Activation_softmax()

layer4.foward(X)
activation2.foward(layer4.output)

layer5.foward(activation2.output)
activation3.foward(layer5.output)

print(activation3.output[:5])


loss_function = Loss_CategoricaClassEntropy()
loss = loss_function.calculate(activation3.output, y)

print("Loss:", loss)


