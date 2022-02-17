import math

import cvxpy
from decimal import Decimal
import numpy
import numpy as np
import matplotlib.pyplot as plt

f = open('../signal.txt', 'r')

y = []
while True:
    content = f.readline()
    if content == '':
        break
    y.append(float(content[:-1]))
y = numpy.array(y)
# y for 5001 * 1
# A for [sin(2pif_kt), cos(2pif_kt) f_k for range(1,100) 5001*200
# x for [a_2k, a_2k+1]
# min||x||_alpha + ||Ax-y||^2

#object c
# c = numpy.array()
A = []
cnt = 0
for t in range(0, 5001, 1):
    A.append([])
    t = t/1000
    for i in range(1, 101):
        A[cnt].append(math.sin(2 * math.pi * i * t))
        A[cnt].append(math.cos(2 * math.pi * i * t))
    cnt += 1
A = numpy.array(A)

n = 200
x = cvxpy.Variable(n)

# out = cvxpy.multiply(A, x)
# print(out.shape)
objective = cvxpy.Minimize(cvxpy.sum_squares(x) + 0.0001*cvxpy.sum_squares(A*x - y))
# objective = cvxpy.Minimize(cvxpy.sum(x) + 0.0015*cvxpy.sum_squares(A*x - y))
# objective = cvxpy.Minimize(cvxpy.sum(x) + 0.0015*cvxpy.sum_squares(A*(cvxpy.multiply(x, x)) - y))
constraints = []
prob = cvxpy.Problem(objective, constraints)
print(prob)
result = prob.solve(solver=cvxpy.SCS, verbose=True)
# print(result)

d = np.dot(A, x.value) - y
print(x.value)
# xAxis = [float(t)/1000 for t in range(0, 5001)]
xAxis = [t for t in range(200)]
plt.plot(xAxis, x.value)
# plt.title('title name')
plt.xlabel('a')
plt.ylabel('Weight')
plt.show()