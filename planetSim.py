import numpy as np
import opacity
from constants import *
from numpy import pi

# Useful definitions
gradad = 0.4
rJupiter = 6.99e9
rJorbit = 8.16e13
mJupiter = 1.9e30

# System parameters
x = 0.7
y = 0.27
m = mJupiter
ms = mSun
rorb = 0.015*rJorbit
le = lSun
r0 = rJupiter*0.5

# Read in opacity tables
fnameOpal='../Opacity Tables/Opal/GS98.txt'
fnameFerg='../Opacity Tables/Ferguson/f05.gs98/'
opac = opacity.opac(fnameOpal,fnameFerg,x,y)

def periodFromRadius(ms,rorb):
	return 2*pi*np.sqrt(rorb**3/(newtonG*ms))

p = periodFromRadius(ms,rorb)
print rorb/rJorbit,p/(24*3600)

def temperature(le,rorb):
	return (le/(4*pi*rorb**2*sigma)/2)**0.25

def density(r,r0,temp):
	rr = r/r0
	psi = (np.sqrt(rr**2+4*rr-4)+rr-2)/4
	return (8e-6*(1.15**(2./3))*temp/psi)**2.5

def pressure(r,r0,temp):
	return (1e13/(1.15**(2./3)))*density(r,r0,temp)**(5./3)*(r/r0)

def ltides(r,temp):
	return 6e28*((temp/2000)**3*(r/rJupiter)**4*(p/(4*3600*24))**(-2))

def lrle(r,r0,temp):
#	op = 10**opac.opacity(temp,density(r,r0,temp))
	op = 1e-2+1e-4*pressure(r,r0,temp)
	return gradad*(4*newtonG*m/(3*rorb**2*pressure(r,r0,temp)*op))

def tcm(m):
	return 8.1e7*(m/mSun)**(4./3)

def drdt(r,r0,le,rorb):
	rr = r/r0
	temp = temperature(le,rorb)
	return r0*rr**2*(1/(1+1/rr))*(mP*(ltides(r,temp)-le*lrle(r,r0,temp))/(3.5*m*kB*tcm(m)))

r = rJupiter
rd = [r]
ti = [0]
for i in range(10000):
	dr = drdt(r,r0,le,rorb)
	dt = 1e-4*r/np.abs(dr)
	r += drdt(r,r0,le,rorb)*dt
	rd.append(r)
	ti.append(ti[-1]+dt)

rd = np.array(rd)
ti = np.array(ti)

import matplotlib.pyplot as plt
plt.plot(np.log10(1+ti/3e7),rd/rJupiter)
plt.show()