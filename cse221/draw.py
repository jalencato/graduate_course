import matplotlib.pyplot as plt

#折线图
x = [1, 2, 3, 4, 5, 6]
k1 = [55.64, 83.24, 112.67, 135.23, 176.31, 186.89]
k2 = [148.34, 159.45, 166.56, 175.67, 181.34, 188.68]
plt.plot(x,k1,'s-',color='r',label="Sequential Acess")#s-:方形
plt.plot(x,k2,'o-',color='g',label="Random Access")#o-:圆形
plt.xlabel("Processes Numbers")
plt.ylabel("Average Read Time Per Block(us)")
plt.xticks(x, [1, 2, 3, 4, 5, 6],color='black')
plt.legend(loc="best")
plt.show()