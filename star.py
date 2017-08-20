import opacity
import numpy as np
from thermoCache import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from numpy import pi
from constants import *
import plotUtils as pu

def f(tau):
	d = np.array(1-1.5*tau)
	d[d<0] = 0
	return d

def s(t,rho,l,r,m):
	return (2./3)*a*t**3*r**0.5*(np.abs(l)/(8*pi*sigma))**0.25/(newtonG*m*rho)

def gradR(kappa,l,m,ff,ss,p,t):
	return 3*p*(kappa*l+ff*ss*4*pi*newtonG*m*c)/(16*pi*newtonG*m*c*a*t**4*(1+ff*ss))

def gradRho(p,dpdt,t,gradd,dpdrho,rho):
	return (p-dpdt*t*gradd)/(dpdrho*rho)

def gradConv(gradRr,gradAd,kappa,rho,hs,alpha,m,r,q,cp,t,convcache):
	lt = hs*alpha
	w = kappa*rho*lt
	g0 = cp*rho*(1+(w**2)/3)/(8*sigma*t**3*w)
	d = newtonG*m*lt**2*q/(8*hs*r**2)
	aa = 9*w**2/(8*(3+w**2))
	v = 1/(g0*d**0.5*(gradRr-gradAd)**0.5)
	y0 = convcache.convGrad(v,aa)
	return gradAd+(gradRr-gradAd)*y0*(y0+v),y0/v,y0/(v*g0)

def gradFull(m,r,tau,l,t,rho,opac,x,y,alpha,thermcache,convcache):
	ff = f(tau)
	ss = s(t,rho,l,r,m)
	q,cp,gradad,p,dpro,dpt,u,dut,vad,error,xh1,xhe1,xhe2 = thermcache.termo(rho,t)
	dlnp = -newtonG*m*(1+ff*ss)/(4*pi*p*r**4)
	kappa = 10**opac.opacity(t,rho)
	gradRr = gradR(kappa,l,m,ff,ss,p,t)
	hs = p*r**2/(rho*newtonG*m)
	gradd = gradR(kappa,l,m,ff,ss,p,t)
	if not isinstance(gradd, np.ndarray):
		if gradRr>gradad:
			gradd = gradConv(gradRr,gradad,kappa,rho,hs,alpha,m,r,q,cp,t,convcache)[0][0]
	else:
		gradd[gradRr>gradad] = gradConv(gradRr,gradad,kappa,rho,hs,alpha,m,r,q,cp,t,convcache)[0][gradRr>gradad]    	
		if len(np.where(np.isnan(gradd))[0])>0:
			print 'Error: Invalid numerics detected in gradient calculation.'
			print 'Inputs are:'
			print 't',t
			print 'rho',rho
			print 'p',p
			print 'kappa',kappa
			print 'l',l
			print 'Intermediate values are:'
			print 'Radiative Gradient',gradRr
			print 'Adiabatic Gradient',gradad
			exit()
	return gradd

def dgraddT(m,r,p,tau,l,t,opac,x,y,alpha,thermcache,convcache,rhocache,eps=1e-3):
	rho0 = rhocache.rho(p,t*(1-eps))
	rho1 = rhocache.rho(p,t*(1+eps))
	g0 = gradFull(m,r,tau,l,t*(1-eps),rho0,opac,x,y,alpha,thermcache,convcache)
	g1 = gradFull(m,r,tau,l,t*(1+eps),rho1,opac,x,y,alpha,thermcache,convcache)
	return (g1-g0)/(2*t*eps)

def dgraddL(m,r,p,tau,l,t,opac,x,y,alpha,thermcache,convcache,rhocache,l0,eps=1e-3):	
	rho = rhocache.rho(p,t)
	dlp = eps*l0
	g0 = gradFull(m,r,tau,l-dlp,t,rho,opac,x,y,alpha,thermcache,convcache)
	g1 = gradFull(m,r,tau,l+dlp,t,rho,opac,x,y,alpha,thermcache,convcache)
	return (g1-g0)/(2*dlp)

class star:
	def __init__(self,x,y,m0,r0,l0,alpha,thermcache,rhocache,convcache,fnameOpal='../Opacity Tables/Opal/GS98.txt',fnameFerg='../Opacity Tables/Ferguson/f05.gs98/',delM=3e-3,lext=0,minRes=500,caution=500,quiet=False):
		# Store inputs
		self.x = x
		self.y = y
		self.m0 = m0
		self.r0 = r0
		self.t0 = ((l0 + lext)/(8*pi*sigma*self.r0**2))**0.25 # T0 is the surface temp, not the photosphere temp: related by 2^(1/4)
		self.l0 = l0
		self.l = None
		self.lext = lext
		self.alpha = alpha
		self.delM = delM
		self.quiet = quiet

		# Prepare opacity interpolator
		self.opalName = fnameOpal
		self.fergName = fnameFerg
		self.opac = opacity.opac(fnameOpal,fnameFerg,x,y)

		# Caches
		self.thermcache = thermcache
		self.rhocache = rhocache
		self.convcache = convcache

		# Helper for reading out data
		self.indDict = {'t':0,'rho':1,'r':2,'tau':3,'p':4,'cp':5,'gradad':6,'dpro':7,'dpt':8,'u':9,'dut':10,\
						'vad':11,'grad':12,'gradRho':13,'gradR':14,'q':15,'mUp':16,'dm':17,'mDown':18,\
						'kappa':19,'hs':20,'gamma':21,'vc':22,'mu':23,'sigma':24}

		# Prepare initial star state
		self.steady = self.steadyIntegrate(minRes=minRes,caution=caution) 
		sg = np.transpose([self.steady[:,16]/(4*pi*self.r0**2)])
		self.steady = np.concatenate((self.steady,sg),axis=1)
		print self.steady[::10,-1]
		sel = np.where(self.steady[:,-1]>1e-2)
		self.steady = self.steady[sel] # For plotting convenience
		self.l = self.l[sel]

		# Prepare for time integration
		self.state = np.copy(self.steady)
		print self.state[:,3]
		sel = np.where(self.state[:,3]>2./3)
		self.state = self.state[sel] # Chop off top of photosphere for time integration
		self.l = self.l[sel]
		self.l = np.concatenate((self.l,[self.l0]))
		self.tb = self.steady[-1,0]
		# Prepare helper variables for time integration
		self.m = self.state[:,18]
		print self.m.shape
		self.dm = self.state[:-1,17]
		self.mUp = self.state[:,16]
		self.mL = np.concatenate(([self.m[0]],(self.m[1:]+self.m[:-1])/2,[self.m[-1]]))
		self.mLup = np.concatenate(([self.mUp[0]],(self.mUp[1:]+self.mUp[:-1])/2,[self.mUp[-1]]))
		self.dmL = np.concatenate(([self.dm[0]/2],(self.dm[1:]+self.dm[:-1])/2,[self.dm[-1]/2]))
		self.fact = self.l[0]/(4*pi*self.r0**2*sigma*self.state[0,0]**4) # Temp BC correction
		if not self.quiet:
			print self.fact,self.l[0]/(4*pi*self.r0**2*sigma*self.state[0,0]**4)

		# Prepare derivatives matrix
		# L oupies 0 through N, T oupies N+1 through 2N
		n = self.state.shape[0]
		if not self.quiet:
			print n
		ij = np.zeros((4*n-2,2))
		vs = np.zeros(4*n-2)

		# Luminosity derivatives
		ij[:n]   	= [[i+1,i] for i in range(n)]
		vs[:n] 		= -1/self.dmL
		ij[n:2*n]	= [[i+1,i+1] for i in range(n)]
		vs[n:2*n]	= 1/self.dmL

		# Temperature derivatives
		ij[2*n:3*n-1] 	= [[i+n+2,i+n+1] for i in range(n-1)]
		vs[2*n:3*n-1] 	= -self.mUp[:-1]/self.dm
		ij[3*n-1:4*n-2] = [[i+n+2,i+n+2] for i in range(n-1)]
		vs[3*n-1:4*n-2] = self.mUp[:-1]/self.dm

		# Put it all together
		ij = np.transpose(ij)
		self.diffMat = csr_matrix((vs,ij),shape=(2*n+1,2*n+1))
		self.eps0 = -(self.diffMat*np.concatenate((self.l,self.state[:,0])))[1:1+len(self.state)]

	def plot(self,kind,xVar,yVar,logX,logY,xlab,ylab,axis,endMarker=None,endMarkerSize=None):
		x = self.retrieve(xVar,kind)
		y = self.retrieve(yVar,kind)
		r = self.retrieve('r',kind)

		# Compute marker on the heating depth
		sig = self.retrieve('sigma',kind)
		heatLoc = np.argmin(np.abs(sig-1e3))
		xH = x[heatLoc]
		yH = y[heatLoc]

		# Compute marker on the photosphere
		tau = self.retrieve('tau',kind)
		pLoc = np.argmin(np.abs(tau-2./3))
		xP = x[pLoc]
		yP = y[pLoc]

		# Continue with plotting
		isConv = (self.retrieve('gradad',kind)>=self.retrieve('gradR',kind))
		if len(x)>500:
			red = len(x)/500
			x = np.copy(x[::red])
			y = np.copy(y[::red])
			r = np.copy(r[::red])
			isConv = isConv[::red]
		if logX:
			x = np.log10(x)
			xH = np.log10(xH)
			xP = np.log10(xP)
		if logY:
			y = np.log10(y)
			yH = np.log10(yH)
			yP = np.log10(yP)
		if yVar!='r':
		 	thinApproxFilter = np.abs(r-self.r0)<0.5*self.r0
		 	x = x[thinApproxFilter]
		 	y = y[thinApproxFilter]
		pu.colorline(axis,x,y,z=0.15+0.7*isConv)
		ranX = np.nanmax(x)-np.nanmin(x)
		ranY = np.nanmax(y)-np.nanmin(y)
		axis.set_xlim([np.nanmin(x)-ranX/10,np.nanmax(x)+ranX/10])
		axis.set_ylim([np.nanmin(y)-ranY/10,np.nanmax(y)+ranY/10])
		axis.set_xlabel(xlab)
		axis.set_ylabel(ylab)
		heatLoc = np.argmin(np.abs(x-1e3))

		# Place markers
		if xVar=='sigma':
			axis.axvspan(xP, xH, alpha=0.3, color='grey')
		if not endMarker is None:
			axis.scatter(x[-1],y[-1],marker=endMarker,s=endMarkerSize,c='k',zorder=100)


	def retrieve(self,name,kind):
		if kind=='steady':
			if name=='l':
				return self.l0
			return self.steady[:,self.indDict[name]]
		elif kind=='timedep':
			if name=='l':
				return self.l
			return self.state[:,self.indDict[name]]
		else:
			print 'Error: Invalid kind. Please specify either steady or timedep.'

	def sigma(self,kind):
		return self.retrieve('p',kind)*self.r0**2/(newtonG*self.m0)

	def mu(self,kind):
		return self.retrieve('rho',kind)*kB*self.retrieve('t',kind)/self.retrieve('p',kind)

	def steadyIntegrate(self,minRes=500,caution=500):
		z = np.array([np.log(self.t0),np.log(1e-12),self.r0,0]) # rho0 = 1e-12
		i = 0
		data = []
		mUp = 1e-30
		while mUp<self.delM*self.m0:
			# Prepare luminosity
			le = self.lext*np.exp(-mUp/(kappaG*4*pi*self.r0**2))
			l = self.l0 + le
			# Prepare thermodynamics
			t = np.exp(z[0])
			rho = np.exp(z[1])
			tau = z[3]
			r = self.r0
			ff = f(tau)
			ss = s(t,rho,l,r,self.m0)
			kappa = 10**self.opac.opacity(t,rho)
			q,cp,gradad,p,dpro,dpt,u,dut,vad,error,xh1,xhe1,xhe2 = self.thermcache.termo(rho,t)[:,0]
			gradRr = gradR(kappa,l,self.m0,ff,ss,p,t)
			
			# Compute derivatives
			dlnp = -newtonG*self.m0*(1+ff*ss)/(4*pi*p*r**4)
			dr = 1./(4*pi*r**2*rho)
			dtau = -kappa/(4*pi*r**2)
			dp = p*dlnp
			
			# Compute other quantities of interest
			hs = p*r**2/(rho*newtonG*self.m0)
			gradC,gam,vc = gradConv(gradRr,gradad,kappa,rho,hs,self.alpha,self.m0,r,q,cp,t,self.convcache)
			gam = gam[0]
			vc = vc[0]
			mu = kB*t*rho/(p-a*(t**4)/3)

			# More derivatives	
			gradd = gradRr
			if gradRr>gradad:
				gradd = gradC[0]
			gradRhoo = gradRho(p,dpt,t,gradd,dpro,rho)
			dlnrho = gradRhoo*dlnp
			dlnt = gradd*dlnp
			
			# Wrap derivatives
			derivs = np.array([dlnt,dlnrho,dr,dtau])
			
			# Set step size
			h = min(self.m0*self.delM/minRes,(1./caution)*(np.min([1,1,z[2]]/np.abs(derivs[:-1]))))

			# Compute current state
			nums = np.array([t,rho,r,tau,p,cp,gradad,dpro,dpt,u,dut,vad,gradd,gradRhoo,gradRr,q,0,0,self.m0,kappa,hs,gam,vc,mu])
			nums[self.indDict['r']] = z[2]
			nums[self.indDict['mDown']] = self.m0-mUp
			nums[self.indDict['dm']] = h
			nums[self.indDict['mUp']] = mUp

			# Step forward
			data.append(nums)
			z -= h*derivs
			mUp += h
			i+=1
			# Check for errors
			if not self.quiet:
				if i%1000==0:
					print 'Mass Step:',h
					print 'Net Mass:',mUp/self.m0
			if i>1000000:
				print 'Warning: Mass step too low!'
				raise ValueError('Mass step too low!')
		self.l = np.ones(len(data))*self.l0
		return np.array(data)

	def jac(self,vec,tstep):
		# Read in state
		r    = 	self.state[:,2]
		tau = self.state[:,3]
		p    =	self.state[:,4]
		cp   =	self.state[:,5]
		n    =	self.state.shape[0]

		# Produce updated quantities
		t = self.state[:,0] + vec[n+1:2*n+1]
		l = self.l + vec[:n+1]*self.l0
		rho = self.rhocache.rho(p,t)
		
		# Compute helper term
		g    =	newtonG*self.m0/self.r0**2

		# Compute grad
		grad =	gradFull(self.m0,self.r0,tau,l[:-1],t,rho,self.opac,self.x,self.y,self.alpha,self.thermcache,self.convcache)

		# Compute grad derivatives
		dgdt =	dgraddT(self.m0,self.r0,p,tau,l[:-1],t,self.opac,self.x,self.y,self.alpha,self.thermcache,self.convcache,self.rhocache)
		dgdl =	dgraddL(self.m0,self.r0,p,tau,l[:-1],t,self.opac,self.x,self.y,self.alpha,self.thermcache,self.convcache,self.rhocache,self.l0)

		# Prepare sparse matrix
		# L occupies 0 through N, T occupies N+1 through 2N, tau goes 2N+1 to 3N

		ij		=	np.zeros((2+3*n,2))
		vs 		=	np.zeros(2+3*n)

		# Boundary condition on T at base
		ij[0]	=	[n+1,2*n]
		vs[0]	=	1.

		# Boundary condition on L at top
		ij[1]	=	[0,0]
		vs[1]	=	1.
		ij[2]	=	[0,n+1]
		vs[2]	=	-self.fact*16*pi*self.r0**2*sigma*t[0]**3/self.l0

		# Derivatives

		# Output L
		ij[4:4+n]	=	[[i+1,i+n+1] for i in range(n)]
		vs[4:4+n] 	= 	-cp/tstep/self.l0

		# Output T
		ij[4+n:3+2*n]	=	[[i+n+2,i] for i in range(n-1)]
		vs[4+n:3+2*n] 	= 	-t[:-1]*dgdl[:-1]*self.l0
		ij[3+2*n:2+3*n]	=	[[i+n+2,i+n+1] for i in range(n-1)]
		vs[3+2*n:2+3*n] =	-((t*dgdt+grad)[:-1])

		# Put it all together and solve
		ij = np.transpose(ij)
		mat = csr_matrix((vs,ij),shape=(2*n+1,2*n+1))
		amat = mat + self.diffMat
		return amat

	def func(self,vec,tstep,eps):
		# Read in unchanged things
		r    = 	self.state[:,2]
		tau  =	self.state[:,3]
		p    =	self.state[:,4]
		cp   =	self.state[:,5]
		n    =	self.state.shape[0]

		# Produce updated quantities
		t1 = self.state[:,0] + vec[n+1:2*n+1]
		l1 = self.l + vec[:n+1]*self.l0
		rho1 = self.rhocache.rho(p,t1)
		grad1 =	gradFull(self.m0,self.r0,tau,l1[:-1],t1,rho1,self.opac,self.x,self.y,self.alpha,self.thermcache,self.convcache)
		k = self.opac.opacity(t1,rho1)
		# Evaluate derivative conditions
		ders = self.diffMat*np.concatenate((l1,t1))

		# Evaluate left side BC's
		lbcs = np.zeros(len(ders))
		lbcs[0] = l1[0]
		lbcs[n+1] = t1[n-1]

		# Put it together
		left = ders + lbcs

		# Evaluate right side
		rght = np.concatenate(([self.fact*4*pi*self.r0**2*sigma*t1[0]**4]\
								,(cp*(vec[n+1:2*n+1]/tstep))-eps\
								,[self.tb],(grad1*t1)[:-1]))
		bNew = left-rght
		bNew[:n+2]/=self.l0
		return bNew

	def stepController(self,tstep,eps):
		dt = tstep
		delta = 0
		while delta<tstep:
			backup = np.copy(self.state)
			backupL = np.copy(self.l)
			done = False
			while not done:
				if not self.quiet:
					print 'dt =',dt
				ret = self.newStep(dt,eps)
				if ret==-1 or np.sum(1.0*(self.state[:,0]<0))>0 or np.sum(1.0*(self.state[:,0]>1e11))>0:
					self.state = np.copy(backup)
					self.l = np.copy(backupL)
					dt /= 2
				else:
					done = True
			delta += dt

	def newStep(self,tstep,eps,rtol=1e-3,stepSize=0.3):
		n = self.state.shape[0]
		vec = np.zeros(2*self.state.shape[0]+1)
		err = 1.0
		i=0
		while err>rtol:
			j = self.jac(vec,tstep)
			b = self.func(vec,tstep,eps)
			dVec = spsolve(j,-b)
			vec += dVec*stepSize
			if np.sum(1.0*np.isnan(vec))>0:
				return -1
			b[n+2:2*n+1]/=self.state[:-1,0]
			b[1:n+2]*=self.delM*self.m0
			err = np.sum(b**2)**0.5/len(b)
			if not self.quiet:
				print err,stepSize,n,np.argmax(np.abs(b)),np.max(np.abs(vec/np.concatenate((self.l/self.l0,self.state[:,0]))))
			if i>100 and i%200==0:
				stepSize/=2
			if i>1000:
				return -1
			i+=1
		if not self.quiet:
			print 'Step done. Error:',err
			print vec[:n+1:100]
			print vec[n+1::100]
		self.l += self.l0*vec[:n+1]
		self.state[:,0] += vec[n+1:2*n+1]
		self.state[:,1] = self.rhocache.rho(self.state[:,4],self.state[:,0])
		kappa = 10**self.opac.opacity(self.state[:,0],self.state[:,1])
		q,cp,gradad,p,dpro,dpt,u,dut,vad,error,xh1,xhe1,xhe2 = self.thermcache.termo(self.state[:,1],self.state[:,0])[:,0]
		ff = f(self.state[:,3])
		ss = s(self.state[:,0],self.state[:,1],self.l[:-1],self.r0,self.m0)
		self.state[:,5] = cp
		self.state[:,6] = gradad
		self.state[:,7] = dpro
		self.state[:,8] = dpt
		self.state[:,9] = u
		self.state[:,10] = dut
		self.state[:,11] = vad
		self.state[:,12] = gradFull(self.m0,self.r0,self.state[:,3],self.l[:-1],self.state[:,0],self.state[:,1]\
							,self.opac,self.x,self.y,self.alpha,self.thermcache,self.convcache)
		self.state[:,13] = gradRho(self.state[:,4],self.state[:,8],self.state[:,0],self.state[:,12],\
							self.state[:,7],self.state[:,1])
		self.state[:,14] = gradR(kappa,self.l[:-1],self.m0,ff,ss,self.state[:,4],self.state[:,0])
		self.state[:,15] = q
		self.state[:,19] = kappa
		# TODO: Update r
