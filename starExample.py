import star
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from constants import *
from thermoCache import *

x = 0.7
y = 0.27
thermcache = thermCache(x,y)
rhocache = rhoCache(thermcache)
convcache = convGradCache()

color=plt.cm.rainbow(np.linspace(0,1,200))
st = star.star(x,y,mSun,rSun,100*lSun,1.5,thermcache,rhocache,convcache,lext=0)
eps = 0.5*np.exp((-st.state[:,16])/(4*pi*rSun**2*1000))*st.l0/(4*pi*rSun**2*1000)
t0 = st.state[:,0]
t = 0
tnext = 1e6
for i in range(20):
	tnext = st.stepController(tnext,0)
	t0 = np.copy(st.state[:,0])
for i in range(200): # 501, 1001, 5001
	plt.subplot(211)
	plt.title('$L_{in}=100L_{sun}, \Delta t=2e8s$')
	plt.plot(np.log10(st.state[:,24]),(st.state[:,0]-t0)/t0,color = color[i])
	plt.ylabel('$\Delta T/T$')
	plt.subplot(212)
	plt.plot(np.log10(st.state[:,24]),st.l[:-1]/lSun,color = color[i])
	plt.ylabel('$L/L_{sun}$')
	plt.xlabel('Log $\Sigma$')
	if i%200==199:
		plt.show()
	t += tnext
	tnext = st.stepController(tnext,i*eps/100)
	if i>100:
		tnext = st.stepController(tnext,eps)