import pickle
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d,griddata
from constants import *
import fulltime

def pFromR(r,m):
	return (r/(0.46*0.0021538*m**(1./3)))**(3./2)

def lumm(m):
	return 0.23*m**2.3*(m<0.43)+(m>=0.43)*m**4

def starTrackerR(m,mc=None):
	if mc is None:
		return m**0.9
	else:
		return 3.7*10**3*mc**4/(1+mc**3+1.75*mc**4)

# Read in parameter sweep
minR,maxR,pbs,pb0,ts,maxRL,li,dR,mRan,nR,lRan = pickle.load(open('dataDump2'))

# Set sweep parameters
nM = len(mRan)
nL = len(lRan)

numFigs = 7
figs = []
imm = []
for i in range(numFigs):
	figs.append(plt.figure())
	imm.append([])

# Loop over luminosities
for lIndex in range(nL):

	# Set orbital parameters
	nP = 150
	pRan = 10**np.linspace(3.5,4.8,num=nP,endpoint=True)
	rRoche = 0.46*0.0021538*(np.outer(mRan,pRan**2))**(1./3) # 0.0021538 is (GM_sun/R_sun^3*s^2/(4pi^2))^(1/3)
	rOrbit = (rRoche/0.46)*((mRan[:,np.newaxis]+2)/mRan[:,np.newaxis])**(1./3)
	flux = np.array([lRan[i]/(4*pi*rOrbit**2) for i in range(nL)])

	# Find max/min period as a function of mass
	pMax = np.zeros((nM,nL))
	pMin = np.zeros((nM,nL))
	for i in range(nM):
		for j in range(nL):
			pMax[i,j] = pFromR(maxRL[i,j],mRan[i])
			pMin[i,j] = pFromR(minR[i],mRan[i])

	# Filter pb0
	for i in range(nM):
		rRan = np.linspace(minR[i]+1e-2,maxRL[i,lIndex],num=nR,endpoint=True)
		for j in range(nR):
			if pb0[i,lIndex,j] > newtonG*mSun**2*mRan[i]**2/(4*pi*rRan[j]**4*rSun**4):
				pb0[i,lIndex,j] = newtonG*mSun**2*mRan[i]**2/(4*pi*rRan[j]**4*rSun**4)

	# Interpolate various properties
	lum = np.zeros((nP,nM))
	tss = np.zeros((nP,nM))
	radP = np.zeros((nP,nM))
	netP = np.zeros((nP,nM))
	pb00 = np.zeros((nP,nM))
	pbss = np.zeros((nP,nM))
	dRR = np.zeros((nP,nM))
	for i in range(nM):
		rRan = np.linspace(minR[i]+1e-2,maxR[i],num=nR,endpoint=True)
		interp = interp1d(rRan,li[i,lIndex,:],bounds_error=False)
		lum[:,i] = interp(rRoche[i])
		interp = interp1d(rRan,ts[i,lIndex,:],bounds_error=False)
		tss[:,i] = interp(rRoche[i])
		interp = interp1d(rRan,pb0[i,lIndex,:],bounds_error=False)
		pb00[:,i] = interp(rRoche[i])
		interp = interp1d(rRan,pbs[i,lIndex,:],bounds_error=False)
		pbss[:,i] = interp(rRoche[i])
		interp = interp1d(rRan,dR[i,lIndex,:],bounds_error=False)
		dRR[:,i] = interp(rRoche[i])

	# Compute Roche scale height
	v0 = rSun*rOrbit*2*pi/pRan[np.newaxis,:]
	hs = rSun*np.transpose(rRoche)*(5*kB/(3*mP))*(((lumm(mRan)[np.newaxis,:]+lum)*lSun/(4*pi*rSun**2*np.transpose(rRoche)**2*sigma))**(1./4))/np.transpose(v0)**2
	vs = (5*kB*(((lumm(mRan)[np.newaxis,:]+lum)*lSun/(4*pi*rSun**2*np.transpose(rRoche)**2*sigma))**(1./4))/(3*mP))**0.5
	for i in range(nM):
		hs[pRan>pMax[i,lIndex],i] = np.nan

	# Compute g
	g = newtonG*mRan[:,np.newaxis]*mSun/(rSun**2*rRoche**2)

	# Compute \dot{R} for expansion
	rdot = 1.4*(mP/2)*lSun*(lumm(mRan)[np.newaxis,:]-lum)*np.transpose(g)/(12*pi*np.transpose(rRoche)*rSun*kB*tss*pbss*(pb00/pbss)**1.4)

	# Compute contraction timescale
	time = np.zeros((nP,nM))
#	fullTime = np.zeros((nP,nM))
	for i in range(nP):
		for j in range(nM):
			rho0 = pbss[i,j]*mP/(2*tss[i,j]*kB)
			time[i,j] = fulltime.f(hs[i,j]*np.log(10),dRR[i,j],rRoche[j,i],mRan[j],lumm(mRan[j])-lum[i,j],lumm(mRan[j]),rho0,pb00[i,j],pbss[i,j])
#			fullTime[i,j] = fulltime.f(rSun*(rRoche[j,i]-starTrackerR(mRan[j])),dRR[i,j],rRoche[j,i],mRan[j],lumm(mRan[j])-lum[i,j],lumm(mRan[j]),rho0,pb00[i,j],pbss[i,j])
			lSurf = lum[i,j] + pi*rRoche[j,i]**2*flux[lIndex][j,i]
			time[i,j] += 1e4*(lumm(mRan[j])/(lumm(mRan[j])-lum[i,j]))/(lSurf)**(3./4)
#			fullTime[i,j] += 1e4*(lumm(mRan[j])/(lumm(mRan[j])-lum[i,j]))/(lSurf)**(3./4)
	for i in range(nM):
		time[pRan>pMax[i,lIndex],i] = np.nan
		time[pRan<pMin[i,lIndex],i] = np.nan

	ax = figs[0].add_subplot(2,2,lIndex+1)
	# Plot maximum radius, minimum radius
	ax.plot(mRan,np.log10(pMax[:,lIndex]),c='k',linewidth=2)
	ax.plot(mRan,np.log10(pMin[:,lIndex]),c='k',linewidth=2)
	# Color by expansion timescale
	im = ax.imshow(np.log10(np.abs(hs/rdot/(365.254*3600*24))),origin='lower',extent = [0.08,1.3,3.5,4.8])
	imm[0].append(im)
	ax.set_xlabel('M')
	ax.set_ylabel('Log P')
	ax.set_title('$L_p='+str(lRan[lIndex])+'L_\odot$')

	ax = figs[1].add_subplot(2,2,lIndex+1)	
	# Plot maximum radius, minimum radius
	ax.plot(mRan,np.log10(pMax[:,lIndex]),c='k',linewidth=2)
	ax.plot(mRan,np.log10(pMin[:,lIndex]),c='k',linewidth=2)
	# Color by rapid contraction
	im = ax.imshow(np.log10(np.abs(dRR/hs)),origin='lower',extent = [0.08,1.3,3.5,4.8])
	imm[1].append(im)
	ax.set_xlabel('M')
	ax.set_ylabel('Log P')
	ax.set_title('$L_p='+str(lRan[lIndex])+'L_\odot$')

	ax = figs[2].add_subplot(2,2,lIndex+1)	
	# Plot maximum radius, minimum radius
	ax.plot(mRan,np.log10(pMax[:,lIndex]),c='k',linewidth=2)
	ax.plot(mRan,np.log10(pMin[:,lIndex]),c='k',linewidth=2)
	# Color by disk:expansion timescale ratio.
	logTimescaleRat = np.log10(np.abs((3e5*lRan[lIndex]**(-1./8)*np.transpose(rRoche**(5./8)))/(hs/rdot)))
	im = ax.imshow(logTimescaleRat,origin='lower',extent = [0.08,1.3,3.5,4.8])
	imm[2].append(im)
	ax.set_xlabel('M')
	ax.set_ylabel('Log P')
	ax.set_title('$L_p='+str(lRan[lIndex])+'L_\odot$')

	ax = figs[3].add_subplot(2,2,lIndex+1)	
	# Plot maximum radius, minimum radius
	ax.plot(mRan,np.log10(pMax[:,lIndex]),c='k',linewidth=2)
	ax.plot(mRan,np.log10(pMin[:,lIndex]),c='k',linewidth=2)
	# Color by contraction timescale.
	im = ax.imshow(np.log10(time),origin='lower',extent = [0.08,1.3,3.5,4.8])
	imm[3].append(im)
	ax.set_xlabel('M')
	ax.set_ylabel('Log P')
	ax.set_title('$L_p='+str(lRan[lIndex])+'L_\odot$')

	ax = figs[4].add_subplot(2,2,lIndex+1)	
	# Plot maximum radius, minimum radius
	ax.plot(mRan,np.log10(pMax[:,lIndex]),c='k',linewidth=2)
	ax.plot(mRan,np.log10(pMin[:,lIndex]),c='k',linewidth=2)
	# Color by contraction timescale.
	im = ax.imshow(np.log10((3e5*lRan[lIndex]**(-1./8)*np.transpose(rRoche**(5./8)))/time),origin='lower',extent = [0.08,1.3,3.5,4.8])
	imm[4].append(im)
	ax.set_xlabel('M')
	ax.set_ylabel('Log P')
	ax.set_title('$L_p='+str(lRan[lIndex])+'L_\odot$')

	ax = figs[5].add_subplot(2,2,lIndex+1)	
	# Plot maximum radius, minimum radius
	ax.plot(mRan,np.log10(pMax[:,lIndex]),c='k',linewidth=2)
	ax.plot(mRan,np.log10(pMin[:,lIndex]),c='k',linewidth=2)
	# Color by contraction timescale.
	tML = time
	tDisk = 3e5*lRan[lIndex]**(-1./8)*np.transpose(rRoche**(5./8))
	tSpread = (2./5)*((mRan/(mRan+2))**(1./3)*(vs/np.transpose(v0))**2)**(-3./8)*tDisk
	pdata = 1.0*(tML>tDisk)+1.0*(tML>tSpread)
#	pdata = np.log10(np.abs(hs/rdot))
	pdata[np.isnan(np.log10(np.abs(hs/rdot)))] = np.nan
	im = ax.imshow(pdata,origin='lower',extent = [0.08,1.3,3.5,4.8])
	imm[5].append(im)
	ax.set_xlabel('M')
	ax.set_ylabel('Log P')
	ax.set_title('$L_p='+str(lRan[lIndex])+'L_\odot$')

	ax = figs[6].add_subplot(2,2,lIndex+1)	
	# Plot maximum radius, minimum radius
	ax.plot(mRan,np.log10(pMax[:,lIndex]),c='k',linewidth=2)
	ax.plot(mRan,np.log10(pMin[:,lIndex]),c='k',linewidth=2)
	# Color by tDisk+tContraction
	# Note: This plot is broken!!!
	im = ax.imshow(np.log10((2*tML+tDisk+2*tSpread)/(365.25*3600*24)),origin='lower',extent = [0.08,1.3,3.5,4.8])
	imm[6].append(im)
	ax.set_xlabel('M')
	ax.set_ylabel('Log P')
	ax.set_title('$L_p='+str(lRan[lIndex])+'L_\odot$')



mins = [1e10 for i in range(numFigs)]
maxs = [-1e10 for i in range(numFigs)]
for i in range(numFigs):
	for j in range(nL):
		ran = imm[i][j].get_clim()
		if ran[0]<mins[i]:
			mins[i] = ran[0]
		if ran[1]>maxs[i]:
			maxs[i] = ran[1]
for i in range(numFigs):
	for j in range(nL):
		imm[i][j].set_clim(mins[i],maxs[i])

cax = figs[0].add_axes([0.85,0.1,0.03,0.8])
cbar = figs[0].colorbar(imm[0][0],cax=cax)
cbar.set_clim(mins[0],maxs[0])
cbar.set_label('Log $h_s/\dot{R}$')
cax = figs[1].add_axes([0.85,0.1,0.03,0.8])
cbar = figs[1].colorbar(imm[1][0],cax=cax)
cbar.set_clim(mins[1],maxs[1])
cbar.set_label('Log $\Delta R/h_s$')
cax = figs[2].add_axes([0.85,0.1,0.03,0.8])
cbar = figs[2].colorbar(imm[2][0],cax=cax)
cbar.set_clim(mins[2],maxs[2])
cbar.set_label('Log $\\tau_\mathrm{disk}/\\tau_\mathrm{exp}$')
cax = figs[3].add_axes([0.85,0.1,0.03,0.8])
cbar = figs[3].colorbar(imm[3][0],cax=cax)
cbar.set_clim(mins[3],maxs[3])
cbar.set_label('Log $\\tau_\mathrm{contraction}$')
cax = figs[4].add_axes([0.85,0.1,0.03,0.8])
cbar = figs[4].colorbar(imm[4][0],cax=cax)
cbar.set_clim(mins[4],maxs[4])
cbar.set_label('Log $\\tau_\mathrm{disk}/\\tau_\mathrm{contraction}$')
#cax = figs[5].add_axes([0.85,0.1,0.03,0.8])
#cbar = figs[5].colorbar(imm[5][0],cax=cax)
#cbar.set_clim(mins[5],maxs[5])
#cbar.set_label('Cycle Type')
cax = figs[6].add_axes([0.85,0.1,0.03,0.8])
cbar = figs[6].colorbar(imm[6][0],cax=cax)
cbar.set_clim(mins[6],maxs[6])
cbar.set_label('Log $\\tau_\mathrm{cycle} \mathrm{(years)}$')


for i in range(len(figs)):
	figs[i].tight_layout()
	if i!=5:
		figs[i].subplots_adjust(right=0.8)

figs[0].savefig('Plots/'+'L_expansionTime.pdf',dpi=200)
figs[1].savefig('Plots/'+'L_contraction.pdf',dpi=200)
figs[2].savefig('Plots/'+'L_TimeRatio.pdf',dpi=200)
figs[3].savefig('Plots/'+'L_contractionTime.pdf',dpi=200)
figs[4].savefig('Plots/'+'L_contractionTimeRatio.pdf',dpi=200)
figs[5].savefig('Plots/'+'L_cycleType.pdf',dpi=200)
# Note: This plot is broken!!! Do not uncomment below without fixing.
figs[6].savefig('Plots/'+'L_cycleTime.pdf',dpi=200)


