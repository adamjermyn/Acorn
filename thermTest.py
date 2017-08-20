import numpy as np
import time
import thermoCache as tc
import pyximport
pyximport.install()
import gob
t = tc.thermCache(0.7,0.27)

r = 10**(np.random.rand(1000)*22-14)
te = 10**(np.random.rand(1000)*8+2)

for i in range(len(r)):
	print '---'
	print 1-t.termo(r[i],te[i])[:,0]/np.array(gob.termo(r[i],te[i],0.7,0.27))

a = time.clock()
c = t.termo(r,te)
b = time.clock()
print b-a
a = time.clock()
c = [np.array(gob.termo(r[i],te[i],0.7,0.27)) for i in range(1000)]
b = time.clock()
print b-a
