import numpy as np
import pyximport
pyximport.install()
import gob
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import newton
from constants import *

class thermCache: # Cache object allowing for precomputation of thermodynamic quantities
	def __init__(self, x, y,minLogRho=-13,maxLogRho=8,minLogT=2.5,maxLogT=8,resRho=500,resT=500):
		rran = 10**np.linspace(minLogRho,maxLogRho,num=resRho)
		tran = 10**np.linspace(minLogT,maxLogT,num=resT)
		data = np.zeros((len(rran),len(tran),13))
		for i in range(len(rran)):
			for j in range(len(tran)):
				data[i,j] = gob.termo(rran[i],tran[j],x,y)
		self.rran = rran
		self.tran = tran
		self.data = data
		self.interp = []
		self.needsLog = [3,4,6,8]
		data[:,:,self.needsLog] = np.log10(data[:,:,self.needsLog])
		self.interp = RegularGridInterpolator((rran,tran),data,bounds_error=False,fill_value=np.nan)
		self.indDict = {'q':0,'cp':1,'gradad':2,'p':3,'dpro':4,'dpt':5,'u':6,'dut':7,'vad':8,'err':9,\
						'xh1':10,'xhe1':11,'xhe2':12}

	def termo(self,rho,t,name=None):
		# Passing name causes this to just return the specified quantity,
		# otherwise all computed quantities are returned.
		ret = np.transpose(self.interp(np.transpose([rho,t])))
		ret[self.needsLog] = 10**ret[self.needsLog]
		if name==None:
			return ret
		return ret[self.indDict[name]]

	def rhoFromP(self,p,t):
		pg = p-(a*t**4)/3
		if pg<0: # Inconsistent with Eddington limit
			return np.nan
		rho0 = mP*pg/(kB*t)
		f = lambda rho: 1-self.termo(abs(rho),t,name='p')[0]/p
		fp = lambda rho: -(1/p)*self.termo(abs(rho),t,name='dpro')[0]
		ret = np.nan
		try:
			ret = np.abs(newton(f,rho0,fprime=fp,maxiter=50))
		except:
			print 'WARNING: Convergence Failure in Rho-solving! Inputs are log(p),',np.log10(p),'log(t),',np.log10(t)
		return ret


class rhoCache: # Cache object allowing for inversion of the equation of state
	def __init__(self,thermcache,minLogP=-6,maxLogP=16,minLogT=2.5,maxLogT=8,resP=150,resT=150):
		pran = 10**np.linspace(minLogP,maxLogP,num=resP)
		tran = 10**np.linspace(minLogT,maxLogT,num=resT)
		data = np.zeros((len(pran),len(tran)))
		for i in range(len(pran)):
			for j in range(len(tran)):
				data[i,j] = np.log10(thermcache.rhoFromP(pran[i],tran[j]))
		self.pran = pran
		self.tran = tran
		self.data = data
		self.interp = RegularGridInterpolator((pran,tran),data,bounds_error=False,fill_value=np.nan)

	def rho(self,p,t):
		pg = p-(a*t**4)/3
		if not isinstance(t, np.ndarray):
			if pg<0: # Inconsistent with Eddington limit
				return np.nan
		else:
			ret = np.zeros(len(p))
			ret[pg<0] = np.nan # Inconsistent with Eddington limit
		# The above code is necessary because the photosphere sometimes interpolates to NaN values
		# due to proximity to Eddington-violating parameters.
		ret[ret==0] = (10**self.interp(np.transpose([p,t])))[ret==0]
		return ret

	def drhodT(self,p,t,eps=1e-3): # Scipy's RectBivariateSpline differentiator is broken so I wrote my own
		return (self.rho(p,t*(1+eps))-self.rho(p,t*(1-eps)))/(2*eps*t)

class convGradCache: # Cache object for the root-finding problem of the convective gradient
	def __init__(self,minLogV=-20,maxLogV=20,minLogA=-20,maxLogA=20,resV=100,resA=100):
		vran = 10**np.linspace(minLogV,maxLogV,num=resV)
		aran = 10**np.linspace(minLogA,maxLogA,num=resA)
		data = np.zeros((len(vran),len(aran)))
		for i in range(len(vran)):
			for j in range(len(aran)):
				roots = np.roots([2*aran[j],vran[i],vran[i]**2,-vran[i]])
				data[i,j] = np.max(np.real(roots[np.where(np.isreal(roots))]))
		self.vran = vran
		self.aran = aran
		self.data = data
		self.interp = RegularGridInterpolator((vran,aran),data,bounds_error=False,fill_value=np.nan)

	def convGrad(self,v,a):
		return self.interp(np.transpose([v,a]))