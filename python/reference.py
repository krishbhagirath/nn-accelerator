# simple implementation of a SINGLE neuron
# each neuron gets a set of inputs, weights, and a bias (# inputs must match # weights)
# different neurons may have different weights and biases, but the same inputs

# inputs = [2,5,1,7]
# weights = [1,2,3,4]
# bias = 5
# total = 0

# for i in range(len(inputs)):
#     total += inputs[i] * weights[i]

# total += bias

# output = max(0, total)

# print("output: ", output)


inputs = [2, -1, 3, 4, 0, 5, -2, 1]

weights = [
    [1,  2,  0, -1,  3,  1,  2,  0],
    [0,  1,  2,  3, -1,  0,  1,  2],
    [2, -1,  1,  0,  2, -2,  1,  3],
    [1,  0, -2,  2,  1,  3,  0, -1],

    [3,  1,  0, -1,  2,  1, -2,  1],
    [1, -2,  3,  1,  0,  2,  1, -1],
    [0,  2,  1, -2,  3,  0,  1,  2],
    [2,  1, -1,  3,  0, -2,  2,  1],

    [1,  3,  2,  0, -1,  1, -2,  2],
    [2,  0,  1, -1,  3,  2,  1, -2],
    [-1, 2,  3,  1,  0, -2,  2,  1],
    [3, -1,  0,  2,  1,  1, -2,  2],

    [1,  2, -1,  3,  2,  0,  1, -2],
    [2, -2,  1,  0,  3,  1,  2, -1],
    [0,  1,  3, -2,  1,  2, -1,  3],
    [3,  0, -1,  2,  1, -2,  3,  1]
]

biases = [
     3, -2,  1,  4,
    -1,  2,  0,  3,
     1, -3,  2,  0,
     4,  1, -2,  3
]

outputs = []

for neuron in range(16):

    total = 0

    # MAC operation
    for i in range(8):
        total += inputs[i] * weights[neuron][i]

    # Add bias
    total += biases[neuron]

    # ReLU
    if total < 0:
        total = 0

    # Saturate to INT8 positive range
    if total > 127:
        total = 127

    outputs.append(total)


print("Reference outputs:")

for i, output in enumerate(outputs):
    print(f"output[{i}] = {output}")