from starTracker import *
import numpy as np
import pickle
from numpy import pi
from multiprocessing import Pool

nM = 80
nR = 80

lRan = [1.,10.,25.,50.]
nL = len(lRan)
mRan = np.linspace(0.08,1.3,num=nM,endpoint=True)
minR = np.zeros(nM)*float('NaN')
maxR = np.zeros(nM)*float('NaN')
maxRL = np.zeros((nM,nL))*float('NaN')

li = np.zeros((nM,nL,nR))
dR = np.zeros((nM,nL,nR))
pbs = np.zeros((nM,nL,nR))
pb0 = np.zeros((nM,nL,nR))
ts = np.zeros((nM,nL,nR))

for i in range(nM):
	s = starT(mRan[i],sc=1e6)
	minR[i] = s.r0
	maxR[i] = s.rMax

	for j in range(len(lRan)):
		maxRL[i,j] = s.rmaxFromL(lRan[j])
		rRan = np.linspace(minR[i]+1e-2,maxRL[i,j],num=nR,endpoint=True)

		def f(k):
			rOrbit = (rRan[k]/0.46)*((2+mRan[i])/mRan[i])**(1./3)
			flux = lRan[j]/(4*pi*rOrbit**2)
			ret = None
			try:
				ret = s.lIn(rRan[k],flux,rm=maxRL[i,j])
			except ValueError as e:
				ret = (None,None,None,None,None)
			return ret
		p = Pool(16)
		ret = p.map(f,range(nR))
		for k in range(nR):
			li[i,j,k],ts[i,j,k],pb0[i,j,k],pbs[i,j,k],dR[i,j,k] = ret[k]
		p.close()
		print "You are at",i,j,". Percent done:",100.*(i*nL+j+1)/(nM*nL)

pickle.dump([minR,maxR,pbs,pb0,ts,maxRL,li,dR,mRan,nR,lRan],open('dataDump2','w+'))
