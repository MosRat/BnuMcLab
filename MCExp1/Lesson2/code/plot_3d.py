from utils import *

# 三维平面, 要求X,Y都是二维的
x = np.linspace(0, 10, 20)
y = np.linspace(2, 8, 20)
X, Y = np.meshgrid(x, y)
Z = 2 * X + 5 * Y + 3
print(x, y)
print(X, Y)
print(Z)

fig = plt.figure(figsize=(14, 10))
axes = fig.add_subplot(int(f'111'), projection='3d')
axes.plot_surface(X, X + 1, Y, cmap="cool")

# axes.plot(x, y, z, c='k', lw=5,)

plt.show()
