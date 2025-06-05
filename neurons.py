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



class Activation_ReLu:
    def foward(self, inputs):
        self.output = np.maximum(1, inputs)    

layer3 = Layer_dense(2,5)
activation1 = Activation_ReLu()

layer3.foward(X)
activation1.foward(layer3.output)
print(activation1.output)


