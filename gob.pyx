#cython: cdivision=True
#cython: infer_types=True
import numpy as np
from numpy import exp
from numpy import sqrt
from scipy.interpolate import griddata

# Useful constants
# ------- set the values of critical densities for pressure ionization:
#                         "rhc1", "rhc2", "rhc3"
# ------- the value of critical second helium ionization: "helim2"
# ------- and the average charge of "metals": "zav" .
cdef double rhcl1 = -1.0
cdef double rhcl2 = -0.5
cdef double rhcl3 = 0.0
cdef double he2lim = 0.99
cdef double zav = 10.0
cdef double rhc1 = 10.0 ** rhcl1
cdef double rhc2 = 10.0 ** rhcl2
cdef double rhc3 = 10.0 ** rhcl3

# p = pressure (cgs)
# ro = density (cgs)
# u = energy density per unit mass (cgs)
# x = hydrogen mass fraction
# y = helium mass fraction
# Returns p,u, as well as
#		xh1		hydrogen ionization fraction
#		xhe1	helium first ionization fraction
#		xhe2	helium second ionization fraction


cdef energ(double ro,double t,double x,double y):
    # rhc1, rhc2, rhc3  are critical densities for "pressure ionization"
    cdef double teta = 5040. / t
    cdef double dm = 2.302585
    cdef double tm = 1 / teta / dm
    cdef double logt = np.log10(t)
    cdef double mue = (1. + x) / 2.
    cdef double h = 1.6734e-24
    cdef double nh = x * ro / h
    cdef double nhe = 0.25 * y * ro / h
    cdef double nmet = 1. / zav / 2. * (1.0 - x - y) * ro / h
    # Assume metals fully ionized
    cdef double nmetel = nmet * zav
    cdef double ne = 0.0
    cdef double nhi = nh
    cdef double nhii = 0.0
    cdef double nhei = nhe
    cdef double nheii = 0.
    cdef double nheiii = 0.
    cdef double xh1 = 0.0
    cdef double xhe1 = 0.0
    cdef double xhe2 = 0.0
    cdef double fac1 = 0.0
    cdef double fac2 = 0.0
    cdef double fac3 = 0.0
    # Hydrogen ionization
    cdef double hi1 = 13.595
    cdef double hi = hi1 * (1 - ro / rhc1 * (1 + tm / hi1))
    cdef double fhl = np.log10(nh)
    cdef double b10 = 15.3828 + 1.5 * logt - hi * teta - fhl
    if b10 > 10.0:
        b10 = 10.0
    if b10 > -10:
        b = 10.0 ** b10
        c = b
        fac1 = c * nh
        bc = 0.5 * b / c
        xx = 1.0 / (sqrt(bc * bc + 1.0 / c) + bc)
        # xx is the positive root of equation: xx**2 + b*xx - c = 0
        xx1 = 1.0 - xx
        if xx1 < 1.0e-10:
            xx1 = 1.0e-10
        nhii = nh * xx
        ne = nhii
        nhi = nh * xx1
        xh1 = xx
        # Helium ionization
        hi2 = 24.580
        hi = hi2 * (1 - ro / rhc2 * (1 + tm / hi2))
        fhel = np.log10(nhe)
        b10 = 15.9849 + 1.5 * logt - hi * teta - fhel
        if b10 > 10.0:
            b10 = 10.0
        if b10 > -10:
            c = 10.0 ** b10
            b = c + ne / nhe
            fac2 = c * nhe
            bc = 0.5 * b / c
            xx = 1.0 / (sqrt(bc * bc + 1.0 / c) + bc)
            xx1 = 1.0 - xx
            if xx1 < 1e-10:
                xx1 = 1e-10
            nheii = nhe * xx
            ne = ne + nheii
            nhei = nhe * xx1
            xhe1 = xx
            # Second Helium ionization
            hi3 = 54.403
            hi = hi3 * (1 - ro / rhc2 * (1 + tm / hi3))
            fhel = np.log10(nheii)
            b10 = 15.3828 + 1.5 * logt - hi * teta - fhel
            if b10 > 10:
                b10 = 10
            if b10 > -10:
                c = 10.0 ** b10
                b = c + ne / nheii
                fac3 = c * nheii
                bc = 0.5 * b / c
                xx = 1.0 / (sqrt(bc * bc + 1.0 / c) + bc)
                xx1 = 1.0 - xx
                if xx1 < 1e-10:
                    xx1 = 1e-10
                nheiii = nheii * xx
                ne = ne + nheiii
                nheii = nheii * xx1
                xhe2 = xx
            f1 = fac1 / ne
            f2 = fac2 / ne
            f3 = fac3 / ne
            f4 = nh / ne
            f5 = y / 4 / x
            zz = 1.0
            zz = fzz(zz, f1, f2, f3, f4, f5)
            ne = ne * zz
            xh1 = f1 / (1 + f1)
            xhe1 = f2 / (1 + f2 * (1 + f3))
            xhe2 = xhe1 * f3
            nhi = nh * (1 - xh1)
            nhii = nh * xh1
            nhei = nhe * (1 - xhe1 - xhe2)
            nheii = nhe * xhe1
            nheiii = nhe * xhe2
    nh2 = 0.0
    if nhi > 0.001 * nh and t < 20000:
        fac = 28.0925 - teta * (4.92516 - teta * (0.056191 + teta * 0.0032688)) - logt
        if t < 12000:
            fac = fac + (t - 12000) / 1000.
        fac = exp(dm * fac)
        if fac > 1e-20 * nhi:
            b = fac / nhi
            bc = 0.5
            xx = 1.0 / (sqrt(bc * bc + 1.0 / b) + bc)
            nh2 = 0.5 * nhi * (1 - xx)
            nhi = nhi * xx
        else:
            nh2 = 0.5 * nhi
            nhi = 0.0
    # Correction for slight electron degeneracy
    nedgen = (nmetel + ne) * (1. + 2.19e-2 * (ro / mue) * (t / 1.e6) ** (-1.5))
    nt = nh - nh2 + nhe + nedgen + nmet
    pg = 1.3805e-16 * nt * t
    pr = 2.521922460548802e-15*t**4
    p = pg + pr
    uh2 = t * (2.1 + t * 2.5e-4)
    if t > 3000:
        uh2 = -1890. + t * (3.36 + t * 0.4e-4)
    u = (1.5 * pg + 3. * pr + 1.3805e-16 * nh2 * uh2 + 3.585e-12 * nhi + 25.36e-12 * nhii +
         39.37e-12 * nheii + 126.52e-12 * nheiii) / ro
    return p, u, xh1, xhe1, xhe2

# input:
#    zz = a guess of correcting factor to the electron density (=1.0)
#    f1,f2,f3 = ionization factors divided by electron density
#    f4 = number density of hydrogen ions and atoms / electron number density
#    f5 = ratio of helium to hydrogen nuclei
# output:
#    zz = the iterated value of the correcting factor
#    f1,f2,f3 = ionization factors divided by the corrected electron density
# Helper method for correcting electron density


cdef double fzz(double zz,double f1,double f2,double f3,double f4,double f5,double delta=0.001,double acc=0.00001, int itmax = 30):
    cdef int iterations = 1
    fz = funzz(zz, f1, f2, f3, f4, f5)
    while abs(fz) > acc and iterations <= itmax:
        zz1 = zz + delta
        fz1 = funzz(zz1, f1, f2, f3, f4, f5)
        dz = delta * fz / (fz - fz1)
        zz = zz + dz
        iterations += 1
        fz = funzz(zz, f1, f2, f3, f4, f5)
    if iterations == itmax:
        print 'Warning: fzz iterations do not converge.'
    return zz


cdef double funzz(double zz,double f1,double f2,double f3,double f4,double f5):  # Helper method
    return f1 / (f1 + zz) + f5 * f2 * (zz + 2 * f3) / (zz * zz + f2 * (zz + f3)) - zz / f4

#   input:
#       ro      density(c.g.s.)
#       t       temp(k)
#       x       hydrogen mass fraction
#       y       helium mass fraction
#       typ     control variable:
#                       > 0 include radiation in ionization region
#                       <=0 neglect radiation in ionization region
#   output:
#       q       -(d ln rho /d ln t)p                           |
#       cp      (du/dt)p   specific heat cap. at const p       |
#       gradad  (d ln t/d ln p)s   adiabatic gradient          |
#       p       pressure (c.g.s.)                              |
#       dpro    (dp/drho)t                                     |--> c.g.s.k.
#       dpt     (dp/dt)rho                                     |
#       u       specific internal energy                       |
#       dut     (du/dt)rho  specific heat cap. at const vol.   |
#       vad     adiabatic sound speed                          |
#       error   log(error)
#       xh1     hydrogen ionization fraction
#       xhe1    helium first ionization fraction
#       xhe2    helium second ionization fraction
#       an adjustment is made to take into account weak electron degen.,
#       here for full ionization, in energ for partial ionization


cpdef termo(double ro,double t,double x,double y):
    p, u, xh1, xhe1, xhe2 = energ(ro, t, x, y)
    z = 1. - x - y
    if xhe2 >= he2lim:
        xh1 = 1.0
        xhe1 = 0.0
        xhe2 = 1.0
        # full ionization
        nelect = (x + y / 2. + z / 2.)
        nnucl = (x + y / 4. + z / zav / 2.)
        mue = (1. + x) / 2.
        ndgen = nelect * (1. + 2.19e-2 * (ro / mue) * (t / 1.e6) ** (-1.5))
        # p1 is for particles, p2 is for photons
        p1 = 0.825075e8 * (ndgen + nnucl)
        p2 = 2.523e-15 * t ** 3 / ro
        p = (p1 + p2) * ro * t
        u = (1.5 * p1 + 3. * p2) * t + 1.516e13 * x + 1.890e13 * y
        dpro = p1 * t
        dpt = (p1 + 4. * p2) * ro
        duro = -3. * p2 * t / ro
        dut = 1.5 * p1 + 12. * p2
    else:
        # partial ionization of hydrogen and helium
        p1, u1, xh1, xhe1, xhe2 = energ(ro, 0.999 * t, x, y)
        p2, u2, xh1, xhe1, xhe2 = energ(ro, 1.001 * t, x, y)
        p3, u3, xh1, xhe1, xhe2 = energ(0.999 * ro, t, x, y)
        p4, u4, xh1, xhe1, xhe2 = energ(1.001 * ro, t, x, y)
        p = (p1 + p2 + p3 + p4) / 4
        u = (u1 + u2 + u3 + u4) / 4
        dpro = (p4 - p3) * 500 / ro
        dpt = (p2 - p1) * 500 / t
        duro = (u4 - u3) * 500 / ro
        dut = (u2 - u1) * 500 / t
    # evaluation of more complex thermodynamic functions and the error
    q = t / ro * dpt / dpro
    cp = dut + q / ro * dpt
    gradad = p * q / (cp * ro * t)
    vad = sqrt(dpro * cp / dut)
    er1 = np.abs(1 - t / p * dpt - duro * ro ** 2 / p) + 1.0e-10
    error = np.log10(er1)
    return q, cp, gradad, p, dpro, dpt, u, dut, vad, error, xh1, xhe1, xhe2