import numpy as np
from Perceptron import Perceptron
import matplotlib.pyplot as plt

training_inputs = []

training_inputs.append(np.array([0,5]))
training_inputs.append(np.array([0,6]))
training_inputs.append(np.array([6,0]))
training_inputs.append(np.array([5,0]))
training_inputs.append(np.array([0,7]))
training_inputs.append(np.array([0,8]))
training_inputs.append(np.array([7,0]))
training_inputs.append(np.array([8,0]))
training_inputs.append(np.array([0,5]))
training_inputs.append(np.array([5,0]))


labels = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0])

perc = Perceptron(2)

perc.train(training_inputs, labels)

plt.plot(range(1, len(perc.errors_a) + 1), perc.errors_a, marker='o')
plt.xlabel =('Number of updates')
plt.ylabel=('Number of epochs')
plt.show()

inputs = []

# Should return a 1
# inputs.append(np.array([0,9]))
# Should return a 0
inputs.append(np.array([9,0]))

print(perc.predict(inputs))
