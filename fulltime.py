import numpy as np
from numpy import pi
from constants import *

def drdt(dl,r,p):
	return lSun*dl/(4*pi*(r*rSun)**2*p)

def drMin(dl,r,m,pbs,pb0,rho0):
	g = m*mSun*newtonG/(r*rSun)**2
	return g*rho0*(0.4+1)*dl*lSun/(12*pi*r*rSun*pbs**2*(pb0/pbs)**(0.4+1))

def dlnrhodt(li,rho,m,r,p):
	g = m*mSun*newtonG/(r*rSun)**2
	l = p/(g*rho)
	vc = (lSun*li/(4*pi*(r*rSun)**2*rho))**(1./3)
	return vc/(10*l)

def f(hs,init,r,m,dl,li,rho0,pb0,pbs,eps=1e-2):
	togo = hs - init
	t = 0
	rho = rho0
	p = pbs
	drdt0 = drdt(dl,r,rho)
	dt = 1.
	drM = drMin(dl,r,m,pbs,pb0,rho0)
	counter = 0
	while togo > 1.:
		dr = drdt(dl,r,p)
		if dr > drM:
			dln = dlnrhodt(li,rho,m,r,p)
			dt = eps*np.min([1./dln,togo/dr])
			dln *= dt
			rho *= 1+dln
			p *= 1+1.4*dln
			togo -= dr*dt
			t += dt
		else:
			dr = drM
			dt = eps*togo/dr
			togo -= dr*dt
		if counter%4000==3999:
			print counter,togo
		counter += 1
	return t