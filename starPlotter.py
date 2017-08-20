import star
import numpy as np
from numpy import pi
from constants import *
from thermoCache import *
#from eos import *
import matplotlib.pyplot as plt
from mpltools import style
from mpltools import layout
import os

style.use('ggplot')
LeS = ['0','1','10']

def makePlot(figname,stGrid,kind,xvar,yvar,logX,logY,xlab,ylab):
	plt.figure(figsize=(10,12))
	for i in range(3):
		for j in range(3):
			xl = ''
			yl = ''
			if i==0:
				yl = ylab+', $L_e='+LeS[j]+'(R/R_\odot)^2 L_\odot$'
			if j==2:
				xl = xlab
			stGrid[i][j].plot(kind,xvar,yvar,logX,logY,xl,yl,plt.subplot(3,3,3*j+i+1))
			if j==0:
				plt.title('M='+str(round(stGrid[i][j].m0/mSun,2)))
	plt.savefig('Plots/'+figname,dpi=200)

# Compute stellar structures

x = 0.7
y = 0.27
#thermcache = eos('opalEOS',x=x,z=1-x-y)
thermcache = thermCache(x,y)
rhocache = rhoCache(thermcache)
convcache = convGradCache()
masses = [1.0,0.3,0.02]
rs = [1.0,0.43,0.14]
ls = lSun*10**np.array([0.,-2.,-4.])
delM = [[0.1,0.1,0.1],[0.05,0.05,0.05],[1.4e-4,1.4e-4,1.4e-4]]
caution = 500
lext = 10**np.array([-10,0,1.])
stGrid = [[] for i in range(3)]
for i in range(3):
	for j in range(3):
		st = star.star(x,y,mSun*masses[i],rSun*rs[i],ls[i],1.5,thermcache,rhocache,convcache,\
			delM=delM[i][j],lext=lSun*rs[i]**2*lext[j],caution=caution)
		stGrid[i].append(st)

# Make opacity plot
from opacity import *
a = opac('../Opacity Tables/Opal/GS98.txt', '../Opacity Tables/Ferguson/f05.gs98/', 0.7, 0.27)
tRan = [10 ** (i / 20.) for i in range(60, 140)]
rRan = [10 ** (i / 20.) for i in range(-200, 20)]
t,r = np.meshgrid(tRan,rRan)
z = a.opacity(t,r)
bigR = r/((t/1e6)**3)
z[bigR>10] = np.nan
axis = plt.subplot(1,1,1)
cax = axis.imshow(z, extent=[3, 7, -10, 1], origin='lower',aspect=0.3)
cb = plt.colorbar(cax)
cb.set_label('log $\kappa$')
for i in range(3):
	for j in range(3):
		s = (3-i)*30
		if j==0:
			em='^'
		elif j==1:
			em='s'
		else:
			em='h'
		stGrid[i][j].plot('steady','t','rho',True,True,'log T','log $\\rho$',axis,em,s)
axis.set_xlim([3,7])
axis.set_ylim([-10,1])
		
plt.savefig('Plots/opacityOverlay.eps',dpi=200)


# Make the other plots
makePlot('heatGrid.eps',stGrid,'steady','sigma','t',True,True,'Log $\Sigma$','Log T')
makePlot('rhoGrid.eps',stGrid,'steady','sigma','rho',True,True,'Log $\Sigma$','Log $\\rho$')
makePlot('kappaGrid.eps',stGrid,'steady','sigma','kappa',True,True,'Log $\Sigma$','Log $\kappa$')
makePlot('tauGrid.eps',stGrid,'steady','sigma','tau',True,True,'Log $\Sigma$','Log $\\tau$')
makePlot('rGrid.eps',stGrid,'steady','sigma','r',True,False,'Log $\Sigma$','R')
makePlot('vsGrid.eps',stGrid,'steady','sigma','vad',True,True,'Log $\Sigma$','Log $v_{s}$')
makePlot('gammaGrid.eps',stGrid,'steady','sigma','gamma',True,True,'Log $\Sigma$','Log $\Gamma$')
makePlot('sigmaGrid.eps',stGrid,'steady','sigma','hs',True,True,'Log $\Sigma$','Log $h_s$')


# Write tables
for i in range(3):
	for j in range(3):
		f = file('Tables/'+str(i)+'.'+str(j)+'.0','w+')
		f.write('Log(Sigma),Log(M+),Log(M-),Log(Rho),Log(p),Log(T),Log(hs),Log(vs),Log(vc)\n')
		data = np.transpose(np.log10([\
			stGrid[i][j].retrieve('sigma','steady'),\
			stGrid[i][j].retrieve('mUp','steady'),\
			stGrid[i][j].retrieve('mDown','steady'),\
			stGrid[i][j].retrieve('rho','steady'),\
			stGrid[i][j].retrieve('p','steady'),\
			stGrid[i][j].retrieve('t','steady'),\
			stGrid[i][j].retrieve('hs','steady'),\
			stGrid[i][j].retrieve('vad','steady'),\
			stGrid[i][j].retrieve('vc','steady')\
			]))
		data = data[::200]
		np.savetxt(f,data,delimiter=',',fmt='%.3f')
		f = file('Tables/'+str(i)+'.'+str(j)+'.1','w+')
		f.write('Log(Sigma),Log(Gamma),Log(mu),Log(Grad),Log(GradA),Log(GradR),Log(Tau),Log(R)\n')
		data = np.transpose(np.log10([\
			stGrid[i][j].retrieve('sigma','steady'),\
			stGrid[i][j].retrieve('gamma','steady'),\
			stGrid[i][j].retrieve('mu','steady'),\
			stGrid[i][j].retrieve('grad','steady'),\
			stGrid[i][j].retrieve('gradad','steady'),\
			stGrid[i][j].retrieve('gradR','steady'),\
			stGrid[i][j].retrieve('tau','steady'),\
			stGrid[i][j].retrieve('r','steady')\
			]))
		data = data[::200]
		np.savetxt(f,data,delimiter=',',fmt='%.3f')		

