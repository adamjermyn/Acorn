import numpy as np
import matplotlib.pyplot as plt
from constants import *
import thermoCache as tcache
import viscosity as visc
from opacity import *
aa = opac('../Opacity Tables/Opal/GS98.txt', '../Opacity Tables/Ferguson/f05.gs98/', 0.7, 0.28)
tc = tcache.thermCache(0.7,0.27)

tRan = [10 ** (i / 40.) for i in range(120, 320)]
rRan = [10 ** (i / 40.) for i in range(-400, 240)]
t,r = np.meshgrid(tRan,rRan)
z = aa.opacity(t,r)
p = tc.termo(r,t,name='p')
mu = kB*t*r/(p-a*(t**4)/3)
mu /= mP
b = 1e3
tShape = t.shape
t = np.reshape(t,(-1,))
p = np.reshape(p,(-1,))
r = np.reshape(r,(-1,))
z = np.reshape(z,(-1,))
mu = np.reshape(mu,(-1,))
mu[mu<0.5] = 0.5
mu[mu>1] = 1.
anis = 2*mu-1+2*(1-mu)*mu**2*16e-21*(p/b)**2
anis[anis>1] = 1.
print anis[::100]
va = visc.overall(t,p,r,z,anis=anis)
vi = visc.overall(t,p,r,z)
va = np.reshape(va,tShape)
vi = np.reshape(vi,tShape)
va = np.log10(va)
vi = np.log10(vi)
mu = np.reshape(mu,tShape)
plt.imshow(va-vi, extent=[3, 8, -10, 6], origin='lower',aspect=0.3)
cb = plt.colorbar()
cb.set_label('log $A$')
plt.ylabel('log $\\rho$')
plt.xlabel('log T')
plt.show()