from starTracker import *
import numpy as np
import pickle
from numpy import pi
from multiprocessing import Pool

nM = 10
nC = 10
p = 22 # days

lRan = np.array([1.,10.,25.,50.])
nL = len(lRan)
mRan = np.linspace(1.,2.5,num=nM,endpoint=True)
cRan = np.linspace(0.2,0.3,num=nC,endpoint=True)

minR = np.zeros((nM,nC))
maxR = np.zeros((nM,nC))
tKel = np.zeros((nM,nC))
tExp = np.zeros((nM,nC,nL))
li = np.zeros((nM,nC))

for i in range(nM):
	for j in range(nC):
		s = starT(mRan[i],sc=1e6,mc=cRan[j])
		minR[i,j] = s.r0
		maxR[i,j] = s.rMax
		tKel[i,j] = 6*10**14*mRan[i]**2/s.r0/s.l
		li[i,j] = s.l
		r = (216./25)*p**(2./3)
		print i,j
		for k in range(nL):
			le = (1./4)*lRan[k]*(s.r0**2/r**2)
			tExp[i,j,k] = tKel[i,j]*s.l/min(le,s.l)

pickle.dump([nM,nC,nL,p,mRan,cRan,lRan,minR,maxR,li,tKel,tExp],open('dataDumpRed','w+'))