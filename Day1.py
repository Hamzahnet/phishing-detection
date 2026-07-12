import numpy as np 
arr2 = [1, 2, 3]
arr3 = [4, 5, 6]                          #Numpy Fundamentals
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










# single neuron
import numpy as np 
def relu(x):
    return np.maximum(0, x)
def neuron(inputs, weights, biases):
    return relu(np.dot(inputs, weights)+biases)
inputs = np.array([0.5, 0.3, 0.8])
weights = np.array([0.4, 0.7, 0.2])
biases = 0.1
output = neuron(inputs, weights, biases)
print(f"Neuron output: {output}")






#leetcode House robber
from typing import List
class Solution:
    def rob(self, nums: List[int]) -> int:
        prev1 = 0
        prev2 = 0

        for num in nums:
            current = max(prev1, prev2 + num)

            prev2 = prev1
            prev1 = current

        return prev1
    
def main():
    sol = Solution()
    print(sol.rob([2, 7, 9, 3, 1]))  
    print(sol.rob([1, 2, 3, 1]))       

main()