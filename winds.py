import numpy as np

def lr(m,minM=0.04,eps=1e-4): # Implements L,R main sequence relations
	s = m.shape
	m = np.reshape(m,(-1,))
	l = np.zeros(m.shape)
	l[(m>=2) & (m<20+eps)] = 2**(4-3.6)*m[(m>=2) & (m<20+eps)]**3.6
	l[(m>=0.43) & (m<2)] = m[(m>=0.43) & (m<2)]**4
	l[(m>=minM-eps) & (m<0.43)] = (0.43)**(4-2.3)*m[(m>=minM-eps) & (m<0.43)]**2.3
	r = np.zeros(m.shape)
	r[(m>=2) & (m<20+eps)] = 2**(0.72-0.57)*m[(m>=2) & (m<20+eps)]**0.57
	r[(m>=0.43) & (m<2)] = m[(m>=0.43) & (m<2)]**0.72
	r[(m>=minM-eps) & (m<0.43)] = m[(m>=minM-eps) & (m<0.43)]**0.72
	l = np.reshape(l,s)
	r = np.reshape(r,s)
	m = np.reshape(m,s)
	return l,r

def ut(m,p,lp,df,fe,fi): # Compute T and dT/T
	fn = (fe+fi*(2-df))/2
	fd = (fe+fi*(2+df))/2
	t = 0.6*(fd**(1./4)+fn**(1./4))/2
	u = 2*(fd**(1./4)-fn**(1./4))/(fd**(1./4)+fn**(1./4))
	return u,t

def wind(m,p,lp,df,minM=0.04): # Self-consistently compute wind flux/fi
	li,rr = lr(m,minM=minM)
	fi = li/rr**2
	fe = lp/(((2+m)*p**2)**(2./3))/2
	for i in range(5): # Self-consistency loop
		u,t = ut(m,p,lp,df,fe,fi)
		ro = 0.012*t**0.5*p/rr
		rov = ro*u**2/(16*np.pi)

		a = 0
		b = 3*np.tanh(rov)**2
		q = 5-3*np.tanh(rov)**2
		y = 1+9*np.tanh(rov)**2

		w = 5*y*t**(3./2)*u**q*ro**b*(2*t*rr/m)**a/(fi*rr)

		rho = 2*10**4*m*10**3/rr**2/(10**13*t)
		y = ((4*10**10*10**(-2)*rho/t**(7))*(m/rr**2))**(1./3)
		s = m.shape
		m = np.reshape(m,(-1,))
		p = np.reshape(p,(-1,))
		t = np.reshape(t,(-1,))
		y = np.reshape(y,(-1,))
		u = np.reshape(y,(-1,))
		fe = np.reshape(fe,(-1,))
		fi = np.reshape(fi,(-1,))
		rr = np.reshape(rr,(-1,))
		ro = np.reshape(ro,(-1,))
		w = np.reshape(w,(-1,))
		a = 1./3
		q = 3
		b = 0
		w[1>25*((fe+fi)/2)**(3./4)*u**2*p] = (5*y*t**(3./2)*u**q*ro**b*(2*t*rr/m)**a/(fi*rr))[1>25*((fe+fi)/2)**(3./4)*u**2*p]
		t = np.reshape(m,s)
		p = np.reshape(p,s)
		y = np.reshape(y,s)
		u = np.reshape(u,s)
		fe = np.reshape(fe,s)
		fi = np.reshape(fi,s)
		rr = np.reshape(rr,s)
		ro = np.reshape(ro,s)
		w = np.reshape(w,s)
		m = np.reshape(m,s)

		# Calculate bottled area fraction
		ar = (1./2)*(1+w/2)
		ar[ar>1]=1
		ar[np.isnan(ar)]=1./2 # In case a previous iteration messed up

		# Calculate revised fe
		lrat = lp/(((2+m)*p**2)**(2./3))/2/fi
		m = np.reshape(m,(-1,))
		ar = np.reshape(ar,(-1,))
		lrat = np.reshape(lrat,(-1,))
		w = np.reshape(w,(-1,))
		lrat[(m<2) & (lrat<2)] = 0
		lrat[(m<2) & (lrat>2)] -= 4*(ar[(lrat>2) & (m<2)])
		m = np.reshape(m,s)
		w = np.reshape(w,s)
		ar = np.reshape(ar,s)
		lrat = np.reshape(lrat,s)
		df *= lrat*fi/fe
		fe = lrat*fi
	return w,ar,lrat,df,rov
