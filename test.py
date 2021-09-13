import matplotlib.pyplot as plt
y = []
x = []

line = plt.plot(x, y)

for i in range(10):
    x.append(i)
    y.append(i*2)
    print(x, y)
    plt.plot(x, y)
    plt.pause(1)

#plt.plot([1,2,3,4])
plt.show()
