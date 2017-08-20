import numpy as np
import matplotlib.pyplot as plt
from opacity import *
a = opac('../Opacity Tables/Opal/GS98.txt', '../Opacity Tables/Ferguson/f05.gs98/', 0.7, 0.28)
tRan = [10 ** (i / 20.) for i in range(60, 180)]
rRan = [10 ** (i / 20.) for i in range(-200, 120)]
t,r = np.meshgrid(tRan,rRan)
z = a.opacity(t,r)
bigR = r/((t/1e6)**3)
z[bigR>10] = np.nan
print t.shape
print r.shape
print z
print z.shape
plt.imshow(z, extent=[3, 9, -10, 6], origin='lower',aspect=0.3)
cb = plt.colorbar()
cb.set_label('log $\kappa$')
plt.ylabel('log $\\rho$')
plt.xlabel('log T')
plt.show()
cs = plt.contourf(np.log10(tRan), np.log10(rRan), z)

cb.set_label('log $\kappa$')
plt.ylabel('log $\\rho$')
plt.xlabel('log T')
plt.show()
