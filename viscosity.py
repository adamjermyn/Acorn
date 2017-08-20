import numpy as np
import constants
from scipy.interpolate import RegularGridInterpolator

naan = float('nan')

values = np.array([
    34.1, 34.1, 34.1, 34.1, 34.1, 34.1, 34.1, 34.1, naan, naan, naan, naan,
    38.1, 38.1, 38.1, 38.1, 38.1, 38.1, 38.1, 38.1, naan, naan, naan, naan,
    43.5, 43.5, 43.5, 43.5, 43.5, 43.5, 43.5, 43.5, 43.5, naan, naan, naan,
    51.5, 51.5, 51.5, 51.5, 51.5, 51.5, 51.5, 51.5, 51.5, naan, naan, naan,
    52.7, 62.6, 62.0, 63.2, 63.2, 63.2, 63.2, 63.2, 63.2, naan, naan, naan,
    12.5, 16.2, 22.4, 31.3, 44.5, 56.4, 66.3, 82.2, 82.5, naan, naan, naan,
    3.98, 7.61, 12.2, 15.9, 18.6, 23.6, 32.1, 61.2, 85.8, naan, naan, naan,
    2.84, 3.05, 3.26, 3.56, 4.83, 9.67, 21.5, 27.8, 61.3, 97.1, 117, naan,
    4.74, 6.31, 6.80, 7.29, 7.87, 8.46, 9.24, 13.3, 29.1, 79.6, 109, 159,
    9.76, 10.3, 10.9, 11.6, 12.5, 14.1, 17.3, 26.7, 33.0, 46.1, 100, 163,
    22.0, 23.2, 24.4, 25.8, 27.4, 28.6, 30.7, 37.5, 52.6, 86.6, 119, 185,
    47.9, 50.8, 53.3, 56.1, 59.7, 63.3, 67.3, 77.4, 91.2, 112, 198, 284
])  # dynamic viscosity

theta = np.array(
    [1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05])
ts = 5040. / theta
tmax = max(ts)
logp = np.array([3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 11])
interp = RegularGridInterpolator((ts,logp),np.reshape(values,(12,12)),bounds_error=False,fill_value=np.nan)

def interpolate(t, p, rho):
    return 1e-5*interp(np.transpose([t,np.log10(p)]))/rho

def loglam(t, rho):
    d0 = -17.4 + 1.5 * np.log(t) - 0.5 * np.log(rho)
    d1 = -12.7 + np.log(t) - 0.5 * np.log(rho)
    boolarr = 1.0 * (t < 1.1e5 * np.ones(t.shape))
    return d0 * boolarr + d1 * (1 - boolarr)


def spitzer(t, rho):
    return 5.2e-15 * np.power(t, 5. / 2) / (rho * loglam(t, rho))

def overall(t, p, rho, kappa,anis=1):
    t = np.array(t)
    p = np.array(p)
    rho = np.array(rho)
    # First, produce nu from actual data
    d0 = interpolate(t, p, rho)
    # Next, compute Spitzer values
    d1 = spitzer(t, rho)
    # Replace Spitzer values with NaN if they aren't above the ionization zone
    d1[t<10**4.1] = np.nan
    # Replace data values with Spitzer values if they are NaN
    d0[np.isnan(d0)] = d1[np.isnan(d0)]
    d0*=anis
    return d0

q = 4.80320451e-10
c = 29979245800.
def isotropicB(t,p,rho,mu):
    nu = overall(t,p,rho)
    return 3*p*mu*c/(rho*q*nu)
