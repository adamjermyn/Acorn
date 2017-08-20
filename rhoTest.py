import numpy as np
from constants import *
import thermoCache as tc

t = tc.thermCache(0.7,0.27)
r = tc.rhoCache(t)

n = 100

rho = 10**(np.random.rand(n)*22-14)
temp = 10**(np.random.rand(n)*8+2)
p = t.termo(rho,temp)[3]
pg = p-(a*temp**4)/3


b = r.rho(p,temp)
data = np.vstack((temp,rho,b,(rho-b)/rho,p,pg))
print np.transpose(data)