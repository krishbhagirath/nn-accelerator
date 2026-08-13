# simple implementation of a SINGLE neuron
# each neuron gets a set of inputs, weights, and a bias (# inputs must match # weights)
# different neurons may have different weights and biases, but the same inputs

inputs = [2,5,1,7]
weights = [1,2,3,4]
bias = 5
total = 0

for i in range(len(inputs)):
    total += inputs[i] * weights[i]

total += bias

output = max(0, total)

print("output: ", output)