import math 

# Say the output from softmax activation function is ,,,now we want to calculate the loss using categorical cross entropy

softmax_output = [0.7, 0.1, 0.2]

# lets say our target class is 0 ,this means that the one hot-encoder vector at index 0 is hot therefore [1, 0, 0], therefore the target output is [1,0,0]

target = [1, 0, 0]   # due to class length of 3

# Loss calculation
loss = -(math.log(softmax_output[0])*target[0]+
         math.log(softmax_output[1])*target[1]+
         math.log(softmax_output[2])*target[2])

print(loss)

# It is just buy saying

loss= -math.log(softmax_output[0])
print(loss)

## Now implementing the loss function

import numpy as np
softmax_output = np.array([[0.7,0.1,0.2],
                          [0.1,0.5,0.4],
                          [0.02,0.9,0.08]])

class_target = [0,1,1]

print(softmax_output[[0,1,2], class_target])
# Print the loss
print(-np.log(softmax_output[[0,1,2], class_target]))

# But since the log of 0 is infinit we have to rectify our code to fit all sort of samples by clipping ,,,check neurons.py