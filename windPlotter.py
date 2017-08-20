import numpy as np
import matplotlib.pyplot as plt
from mpltools import style
from mpltools import layout
from winds import *

style.use('ggplot')

lpr = np.linspace(0.,10.,num=100)

data = []
data2 = []

for q in range(len(lpr)):
	lp = lpr[q]
	mr = np.array([0.071])
	pr = np.array([1.2816])
	dfr = np.concatenate([[0],10**np.linspace(-10,-2,num=10000,endpoint=True)\
						,10**np.linspace(-2,0,num=10000,endpoint=True)])

	m,pp,df = np.meshgrid(mr,pr,dfr,indexing='ij')
	li,rr = lr(m,minM=0.04) # We use minM to push the brown dwarf limit down below 0.06Msun
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
	lrr = lr(mr,minM=0.04)
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
	data.append(fD[0][0]/fN[0][0])
	data2.append(5778*(fN[0][0]**0.25))

data = np.array(data)
data2 = np.array(data2)
print data
plt.figure(figsize=(10,4))
plt.subplot(111)
plt.plot(lpr,data)
plt.plot(lpr,data2/4425)
print data2/4425
plt.show()