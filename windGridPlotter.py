import numpy as np
import matplotlib.pyplot as plt
from winds import *

lpr = [1,10,25,50]
figs = []
imm = []
strs = ['$\log F_{day}/F_{night}$','$\log F_{day}/(F_e+F_i)$','$\log F_{night}/F_i$','$\log \Delta F/F$','$\log W/(F_i+F_e)$']
numFigs = 5
for i in range(numFigs):
	figs.append(plt.figure())
	imm.append([])
for q in range(4):
	lp = lpr[q]
	mr = 10**np.linspace(np.log10(0.08),np.log10(20),num=200,endpoint=True)
	pr = 10**np.linspace(-0.5,3,num=200,endpoint=True)
	dfr = np.concatenate([[0],10**np.linspace(-10,-2,num=150,endpoint=True)\
						,10**np.linspace(-2,0,num=400,endpoint=True)])
	m,pp,df = np.meshgrid(mr,pr,dfr,indexing='ij')
	li,rr = lr(m)
	fi = li/rr**2
	fe = lp/(((2+m)*pp**2)**(2./3))/2
	df *= fe/fi
	
	w,ar,lra,df,rov = wind(m,pp,lp,df)
	# These locations get NaN'd because we zero-out fe in the
	# convective full-bottling zone, and then update df accordingly.
	# Everywhere that this occurs, df should properly be zero.
	df[np.isnan(df)] = 0
	res = (df - lra + w)
	fN = np.zeros((len(mr),len(pr)))
	fD = np.zeros((len(mr),len(pr)))
	lraa = np.zeros((len(mr),len(pr)))
	lrr = lr(mr)
	for i in range(len(mr)):
		for j in range(len(pr)):
			lraa[i,j] = lra[i,j,np.argmin(res[i,j]**2)]
			fN[i,j] = fi[i,j,0]*(lraa[i,j]+2-df[i,j,np.argmin(res[i,j]**2)])/2
			fD[i,j] = fi[i,j,0]*(lraa[i,j]+2+df[i,j,np.argmin(res[i,j]**2)])/2
			w[i,j,0] = w[i,j,np.argmin(res[i,j]**2)]
			if lrr[1][i]>0.49*((mr[i]+2)**0.5*pr[j]*(mr[i]/2))**(2./3)/(0.6*(mr[i]/2)**(2./3)+np.log(1+(mr[i]/2)**(1./3))):
				fN[i,j] = 0
				fD[i,j] = 0
				w[i,j,0] = 0
	w = w[:,:,0]
	fN = np.transpose(fN)
	fD = np.transpose(fD)
	ax = figs[4].add_subplot(2,2,q+1)
	ax.plot(mr,10**4*(0.49*((mr+2)**0.5*(mr/2))**(2./3)/(0.6*(mr/2)**(2./3)+np.log(1+(mr/2)**(1./3)))/lrr[1])**(-3./2),c='k',linewidth=2)
	im = ax.imshow(np.log10(np.transpose((w*fi[:,:,0])/(fi[:,:,0]+(lp/(((2+m)*pp**2)**(2./3))/2)[:,:,0]))),origin='lower',extent=[0.08,20,3*10**3,10**7],aspect=0.65)
	ax.set_xlim([0.08,20])
	ax.set_ylim([3*10**3,10**7])
	imm[4].append(im)
	ax.set_title('$L_p='+str(lpr[q])+'$')
	ax.set_xscale('log')
	ax.set_yscale('log')
	if q==2 or q==3:
		ax.set_xlabel('M ($M_\odot$)')
	if q==0 or q==2:
		ax.set_ylabel('P (s)')
	ax = figs[3].add_subplot(2,2,q+1)
	ax.plot(mr,10**4*(0.49*((mr+2)**0.5*(mr/2))**(2./3)/(0.6*(mr/2)**(2./3)+np.log(1+(mr/2)**(1./3)))/lrr[1])**(-3./2),c='k',linewidth=2)
	im = ax.imshow(np.log10(2*(fD-fN)/(fD+fN)),origin='lower',extent=[0.08,20,3*10**3,10**7],aspect=0.65)
	ax.set_xlim([0.08,20])
	ax.set_ylim([3*10**3,10**7])
	imm[3].append(im)
	ax.set_title('$L_p='+str(lpr[q])+'$')
	ax.set_xscale('log')
	ax.set_yscale('log')
	if q==2 or q==3:
		ax.set_xlabel('M ($M_\odot$)')
	if q==0 or q==2:
		ax.set_ylabel('P (s)')
	ax = figs[2].add_subplot(2,2,q+1)
	ax.plot(mr,10**4*(0.49*((mr+2)**0.5*(mr/2))**(2./3)/(0.6*(mr/2)**(2./3)+np.log(1+(mr/2)**(1./3)))/lrr[1])**(-3./2),c='k',linewidth=2)
	im = ax.imshow(np.log10(fN/np.transpose(fi[:,:,0])),origin='lower',extent=[0.08,20,3*10**3,10**7],aspect=0.65)
	ax.set_xlim([0.08,20])
	ax.set_ylim([3*10**3,10**7])
	imm[2].append(im)
	ax.set_title('$L_p='+str(lpr[q])+'$')
	ax.set_xscale('log')
	ax.set_yscale('log')
	if q==2 or q==3:
		ax.set_xlabel('M ($M_\odot$)')
	if q==0 or q==2:
		ax.set_ylabel('P (s)')
	ax = figs[1].add_subplot(2,2,q+1)
	ax.plot(mr,10**4*(0.49*((mr+2)**0.5*(mr/2))**(2./3)/(0.6*(mr/2)**(2./3)+np.log(1+(mr/2)**(1./3)))/lrr[1])**(-3./2),c='k',linewidth=2)
	im = ax.imshow(np.log10(fD/np.transpose(fi[:,:,0]+(lp/(((2+m)*pp**2)**(2./3))/2)[:,:,0])),origin='lower',extent=[0.08,20,3*10**3,10**7],aspect=0.65)
	ax.set_xlim([0.08,20])
	ax.set_ylim([3*10**3,10**7])
	imm[1].append(im)
	ax.set_title('$L_p='+str(lpr[q])+'$')
	ax.set_xscale('log')
	ax.set_yscale('log')
	if q==2 or q==3:
		ax.set_xlabel('M ($M_\odot$)')
	if q==0 or q==2:
		ax.set_ylabel('P (s)')
	ax = figs[0].add_subplot(2,2,q+1)
	ax.plot(mr,10**4*(0.49*((mr+2)**0.5*(mr/2))**(2./3)/(0.6*(mr/2)**(2./3)+np.log(1+(mr/2)**(1./3)))/lrr[1])**(-3./2),c='k',linewidth=2)
	im = ax.imshow(np.log10(fD/fN),origin='lower',extent=[0.08,20,3*10**3,10**7],aspect=0.65)
	ax.set_xlim([0.08,20])
	ax.set_ylim([3*10**3,10**7])
	imm[0].append(im)
	ax.set_title('$L_p='+str(lpr[q])+'$')
	ax.set_xscale('log')
	ax.set_yscale('log')
	if q==2 or q==3:
		ax.set_xlabel('M ($M_\odot$)')
	if q==0 or q==2:
		ax.set_ylabel('P (s)')

for j in range(numFigs):
	minn = 1e10
	maxx = -1e10
	for i in range(4):
		ran = imm[j][i].get_clim()
		if ran[0]<minn:
			minn=ran[0]
		if ran[1]>maxx:
			maxx=ran[1]
	for i in range(4):
		if j!=4:
			imm[j][i].set_clim(minn,maxx)
		else:
			imm[j][i].set_clim(minn,0)

	cax = figs[j].add_axes([0.85,0.1,0.03,0.8])
	cbar = figs[j].colorbar(imm[j][0],cax=cax)
	cbar.set_label(strs[j])
	figs[j].subplots_adjust(right=0.8)
	figs[j].savefig('../Thesis/anisotropy'+str(j+1)+'.pdf',dpi=200)

