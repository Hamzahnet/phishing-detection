import numpy as np
def relu(x):
    return np.maximum(0, x)
def layer(input, weight, bias):
    return relu(np.dot(input, weight)+ bias)
input = np.array([0.5, 0.3, 0.8])
weight = np.array([
    [0.4, 0.7, 0.2, 0.1],
    [0.3, 0.5, 0.8, 0.4],
    [0.2, 0.1, 0.6, 0.9],              
])
bias = np.array([0.1, 0.2, 0.1, 0.3])
output = layer(input, weight, bias)
print(f"Layer output: {output}")
print(f"Shape: {output.shape}")





import numpy as np 
arr2 = [1, 2, 3]
arr3 = [4, 5, 6]
mult = np.multiply(arr2,arr3)
print(mult)






import numpy as np
arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
matrix = arr
print(f"{matrix}\n")
print(f"{matrix.shape}\n")
print(matrix[1])








import numpy as np
arr2 = np.array([2,3])
arr3 = np.array([4,5])
mult = np.dot(arr2, arr3)
print(mult)


