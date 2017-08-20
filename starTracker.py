import star
import numpy as np
from numpy import pi
from constants import *
from thermoCache import *
from scipy.interpolate import interp1d
import os.path
import pickle

q = 4.5 # Kramer's P,T opacity law
gA = 0.4 # Adiabatic gradient in ionized matter
x = 0.7
y = 0.27

if not os.path.exists('cachesLowRes'):
	thermcache = thermCache(x,y,resRho=100,resT=100)
	rhocache = rhoCache(thermcache)
	convcache = convGradCache()
	pickle.dump([thermcache,rhocache,convcache],open('cachesLowRes','w+'))
else:
	thermcache,rhocache,convcache = pickle.load(open('cachesLowRes','rb'))

def lum(m,mc=None):
	if m<0.43 and mc is None:
		return 0.23*m**2.3
	elif mc is None:
		return m**4
	else:
		return (10**5.3)*(mc**6)/(1+10**0.4*mc**4+10**0.5*mc**5)

def r(m,mc=None):
	if mc is None:
		return m**0.9
	else:
		return 3.7*10**3*mc**4/(1+mc**3+1.75*mc**4)

class starT:
	def __init__(self,m,mc=None,sc=1e6):
		# Compute unperturbed equilibrium properties
		self.m = m
		self.mc = mc
		self.sc = sc
		self.l = lum(m,mc=mc)
		self.r0 = r(m,mc=mc)
		self.delMmult = 32702.6*(1e6/sc)
		self.eq = star.star(x,y,mSun*m,rSun*self.r0,lSun*self.l,1.5,\
					thermcache,rhocache,convcache,\
					delM=self.r0**2/m/self.delMmult,lext=0,\
					minRes=350,caution=50,quiet=True)

		# Compute expansion potential
		sg = self.eq.retrieve('sigma','steady')
		gR = self.eq.retrieve('gradR','steady')
		p = self.eq.retrieve('p','steady')
		t = self.eq.retrieve('t','steady')
		bS = np.argmin(np.abs(sg-self.sc))
		self.pbs = p[bS]
		self.pb0 = p[bS]*(gR[bS]/gA)**(1./(gA*(4+q)-2))
		self.ts = t[bS]
		self.rMax = self.r0*max(1,(self.pb0/self.pbs)**(2./(3*(4+q))))

	def rmax(self,flux):
		print 'Computing self-consistent maximum radius...'
		r = self.r0
		dev = 1
		while abs(dev)>1e-3:
			st = star.star(x,y,mSun*self.m,rSun*r,lSun*self.l,1.5,\
					thermcache,rhocache,convcache,\
					delM=r**2/self.m/self.delMmult,lext=flux*lSun*pi*r**2,\
					minRes=350,caution=50,quiet=True)
			sg = st.retrieve('sigma','steady')
			gR = st.retrieve('gradR','steady')
			p = st.retrieve('p','steady')
			bS = np.argmin(np.abs(sg-self.sc))
			pb = (r**2/self.r0**2)*p[bS]*(gR[bS]/gA)**(1./(gA*(4+q)-2))
			rNew = self.r0*max(1,(self.pb0/max(pb,self.pbs))**(2./(3*(4+q))))
			r = (rNew + r)/2
			dev = rNew - r
		print 'Done!',r
		return r

	def rmaxFromL(self,lum):
		print 'Computing self-consistent maximum period...'
		r = self.r0
		dev = 1
		while abs(dev)>1e-3:
			rOrbit = (r/0.46)*((2+self.m)/self.m)**(1./3)
			flux = lum/(4*pi*rOrbit**2)
			st = star.star(x,y,mSun*self.m,rSun*r,lSun*self.l,1.5,\
					thermcache,rhocache,convcache,\
					delM=r**2/self.m/self.delMmult,lext=flux*lSun*pi*r**2,\
					minRes=350,caution=50,quiet=True)
			sg = st.retrieve('sigma','steady')
			gR = st.retrieve('gradR','steady')
			p = st.retrieve('p','steady')
			bS = np.argmin(np.abs(sg-self.sc))
			pb = (r**2/self.r0**2)*p[bS]*(gR[bS]/gA)**(1./(gA*(4+q)-2))
			rNew = self.r0*max(1,(self.pb0/max(pb,self.pbs))**(2./(3*(4+q))))
			r = (rNew + r)/2
			dev = rNew - r
			print r,dev
		print 'Done!',r
		return r

	def lIn(self,r,flux,rm=None):
		print 'Computing flux...'
		if rm is None:
			rm = self.rmax(flux)
		print rm
		if r>rm:
			print 'Error: Requested radius greater than possible.'
			return None,None,None,None,None
		elif r<self.r0:
			print 'Error: Requested radius less than possible.'
			return None,None,None,None,None
		pbExp = (self.r0/r)**2*self.pb0*(self.r0/r)**(3.*(4+q)/2)
		l = self.l/2
		lower = 0
		upper = self.l
		dev = 1
		counter = 0
		st = None
		while abs(dev)>3e-3:
			counter += 1
			st = star.star(x,y,mSun*self.m,rSun*r,lSun*l,1.5,\
					thermcache,rhocache,convcache,\
					delM=r**2/self.m/self.delMmult,lext=flux*lSun*pi*r**2,\
					minRes=350,caution=50,quiet=True)
			sg = st.retrieve('sigma','steady')
			gR = st.retrieve('gradR','steady')
			p = st.retrieve('p','steady')
			bS = np.argmin(np.abs(sg-self.sc))
			pb = (r/self.r0)**2*p[bS]*((self.l/l)*gR[bS]/gA)**(1./(gA*(4+q)-2))
			dev = (r-self.r0*max(1,(self.pb0/max(pb,self.pbs))**(2./(3*(4+q)))))/(r-self.r0)
			if dev<0:
				upper = l
			elif dev>0:
				lower = l
			l = (upper + lower)/2
#			print l,dev,self.l,r,self.r0,flux
			if l<1e-4*self.l:
				print 'Need more mass!',self.sc
				s = starT(self.m,sc=self.sc*2)
				return s.lIn(r,flux,rm=None)
			if counter > 100:
				print 'Error: No solution found!'
				return None,None,None,None,None
		print 'Done!',self.m,r,flux

		# Process outputs
		dm = st.retrieve('dm','steady')
		gR = st.retrieve('gradR','steady')
		sg = st.retrieve('sigma','steady')
		t = st.retrieve('t','steady')
		dr = dm/(4*pi*r**2*rSun**2*st.retrieve('rho','steady'))
		bs = np.argmin(np.abs(sg-self.sc))
		ts = t[bs]
		pbs = p[bs]
		pb0 = (r/self.r0)**2*p[bS]*((self.l/l)*gR[bS]/gA)**(1./(gA*(4+q)-2))
		tau = st.retrieve('tau','steady')
		bt = np.argmin(np.abs(tau-2./3))
		dRpre = np.sum(dr[bt:bs])

		# Compute sudden collapse star
		print 'Computing sudden collapse...'
		stpre = st
		dev = 1
		st = None
		lower = 0
		upper = l + flux*pi*r**2
		lSurf = (lower+upper)/2
		counter = 0		
		while abs(dev)>3e-3:
			counter += 1
			st = star.star(x,y,mSun*self.m,rSun*r,lSurf*lSun,1.5,\
					thermcache,rhocache,convcache,\
					delM=r**2/self.m/self.delMmult,lext=0,\
					minRes=350,caution=50,quiet=True)
			sg = st.retrieve('sigma','steady')
			t = st.retrieve('t','steady')
			bS = np.argmin(np.abs(sg-self.sc))
			dev = (t[bS] - ts)/ts
			if dev>0:
				upper = lSurf
			elif dev<0:
				lower = lSurf
			lSurf = (upper+lower)/2
#			print lSurf,dev,self.l,r,self.r0,flux
			if counter > 100:
				print 'Error: No solution found!'
				return None,None,None,None,None
		# Determine change in radius
		sg = st.retrieve('sigma','steady')
		dm = st.retrieve('dm','steady')
		dr = dm/(4*pi*r**2*rSun**2*st.retrieve('rho','steady'))
		bs = np.argmin(np.abs(sg-self.sc))
		tau = st.retrieve('tau','steady')
		bt = np.argmin(np.abs(tau-2./3))
		dR = np.sum(dr[bt:bs])
		print 'Done!!',self.m,r,flux
		return l,ts,pb0,pbs,dRpre-dR
