import pickle
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from constants import *

nM,nC,nL,p,mRan,cRan,lRan,minR,maxR,li,tKel,tExp = pickle.load(open('dataDumpRed'))

plt.imshow(maxR/30,origin='lower',extent=[0.2,0.3,1,2.5],aspect=0.07)
plt.xlabel('$M_c (M_\odot)$')
plt.ylabel('$M (M_\odot)$')
plt.colorbar()
plt.savefig('Plots/redgiants.pdf',dpi=200)