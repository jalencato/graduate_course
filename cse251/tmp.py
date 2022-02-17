import numpy as np
from scipy.optimize import minimize


# 目标函数
def objective(x):
    return x[0] - 2*x[1] + x[2] - x[3]


# 约束条件
def constraint1(x):
    return -x[0] - x[3]


def constraint2(x):
    return -x[1]


def constraint3(x):
    return -x[2]

def constraint4(x):
    return -2*x[0] - 3*x[1] + x[2] - 2*x[3]


# 初始猜想
n = 4
x0 = np.zeros(n)
x0[0] = 0.12354
x0[1] = 0.23421
x0[2] = 0.3213
x0[3] = 0.12534583789

# show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))

# 边界约束
b1 = (1.0, None)
b2 = (-2.0, None)
b3 = (-3.0, None)
b4 = (4.0, None)
bnds = (b1, b2, b3, b4)  # 注意是两个变量都要有边界约束

con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
con4 = {'type': 'ineq', 'fun': constraint4}
cons = ([con1, con2, con3, con4])

# 优化计算
solution = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
print(solution)
x = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(x)))

# print solution
print('Solution')
print('x1 = ' + str(x[0]))
print('x2 = ' + str(x[1]))
print('x3 = ' + str(x[2]))
print('x4 = ' + str(x[3]))