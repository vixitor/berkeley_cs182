import numpy as np
b = np.array([[1,2,3],[4,5,6],[6,7,8],[10,11,12]])
print(b)
mean = np.mean(b, axis=0)
print(mean)
var = np.mean(np.square(b - mean),axis=0)
var1 = 0
# for i in range(b.shape[0]):
#     print(b[i, 0], mean[0, 0])
#     var1 += ((b[i, 0] - mean[0, 0] ) ** 2) / 4
print(var1)
print(var)
c = np.array([[1,2],[3,4],[5,6]])
d = np.array([1,2])
e = [3,5]
print(d.shape)
print(np.dot(c, np.diag(d)))
print(c + e)
