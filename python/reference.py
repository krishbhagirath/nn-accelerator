inputs = [2,5,1,7]
weights = [1,2,3,4]
bias = 5
total = 0

for i in range(len(inputs)):
    total += inputs[i] * weights[i]

total += bias

output = max(0, total)

print("output: ", output)