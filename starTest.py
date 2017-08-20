import star
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from constants import *
from thermoCache import *
from eos import *

x = 0.7
y = 0.27
#thermcache = eos('opalEOS',x=x,z=1-x-y)
thermcache = thermCache(x,y)
rhocache = rhoCache(thermcache)
convcache = convGradCache()

color=plt.cm.rainbow(np.linspace(0,1,200))

st = star.star(x,y,mSun,rSun,1*lSun,1.5,thermcache,rhocache,convcache,lext=1*lSun)
eps = st.eps0
#eps = 0.2*np.exp((-st.state[:,16])/(4*pi*rSun**2*kappaG))*lSun/(4*pi*rSun**2*kappaG)
t0 = st.state[:,0]
t = 0
for i in range(20):
	st.stepController(1e8,eps)
	t0 = np.copy(st.state[:,0])
for i in range(400):
	tnext = 1e6
	if i%10==0:
		plt.subplot(211)
		plt.title('$L_{in}=100L_{sun}, \Delta t=1e8s$')
		plt.plot(np.log10(st.state[:,24]),(st.state[:,0]-t0)/t0,color = color[i])
		plt.ylabel('$\Delta T/T$')
		plt.subplot(212)
		plt.plot(np.log10(st.state[:,24]),st.l[:-1]/lSun,color = color[i])
		plt.ylabel('$L/L_{sun}$')
		plt.xlabel('Log $\Sigma$')
	t += tnext
	print 'Step',i
	if i<=100:
		st.stepController(tnext,(100-i)*eps/100)
	else:
		st.stepController(tnext,0*eps/100)		
	print 'Step',i
	if i%200==199:
		plt.show()
