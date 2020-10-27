import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
digit = digits.data

plt.gray()
plt.matshow(digits.images[100])
plt.show()
print(digits.target[100])

model = KMeans(n_clusters=10)
model.fit(digit)
fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):
  ax = fig.add_subplot(2, 5, 1 + i)
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

new_samples = np.array([
[0.00,0.68,3.80,4.57,4.26,0.60,0.00,0.00,0.00,6.08,7.53,6.31,7.38,6.16,0.00,0.00,0.00,6.39,2.73,0.00,3.65,7.62,0.00,0.00,0.00,0.00,0.00,0.00,3.20,7.62,0.00,0.00,0.00,0.00,0.00,0.45,6.47,6.77,0.00,0.00,0.00,0.00,3.18,6.76,7.61,5.32,3.81,0.68,0.00,0.00,5.69,6.85,6.84,6.84,6.86,1.60,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.23,4.94,7.61,7.46,6.62,1.13,0.00,0.00,3.34,7.62,4.25,4.25,7.38,4.02,0.00,0.00,5.25,6.47,0.00,0.00,6.54,4.57,0.00,0.00,6.09,4.95,0.00,0.45,7.31,3.95,0.00,0.00,6.09,4.87,0.00,2.20,7.61,1.89,0.00,0.00,5.01,7.31,0.91,3.04,7.62,0.76,0.00,0.00,1.20,7.15,7.23,7.46,6.37,0.15,0.00,0.00,0.00,1.51,4.11,4.48,0.76,0.00,0.00],
[0.00,0.08,3.64,4.25,0.68,0.00,0.00,0.00,0.00,2.20,7.60,7.61,6.31,0.07,0.00,0.00,0.00,0.68,3.71,3.48,7.62,0.76,0.00,0.00,0.00,0.00,0.00,2.81,7.62,0.76,0.00,0.00,0.00,0.00,0.08,5.41,7.01,0.15,0.00,0.00,0.00,0.08,4.71,7.62,3.19,0.00,0.00,0.00,0.00,4.10,7.62,7.62,7.62,7.62,4.11,0.00,0.00,1.36,3.04,3.05,3.05,3.05,1.37,0.00],
[0.45,5.55,7.62,7.62,7.62,4.71,0.00,0.00,6.01,7.39,4.71,5.46,6.84,7.62,2.42,0.00,7.62,2.66,0.00,0.00,0.22,7.15,4.41,0.00,7.62,2.20,0.00,0.00,0.22,7.53,3.57,0.00,7.53,4.70,0.00,0.00,3.27,7.62,2.04,0.00,4.39,7.62,4.85,4.02,7.54,5.47,0.07,0.00,0.07,4.31,7.37,7.61,5.53,0.23,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00]
])

new_labels = model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
