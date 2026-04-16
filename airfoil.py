import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,1,100)
y = x**(1/2)*(1-x)

plt.plot(x,y)
plt.plot(x,-y)
plt.axis('equal')
plt.show()