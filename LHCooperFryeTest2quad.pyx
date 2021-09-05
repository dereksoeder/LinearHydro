#cython: language_level=3

cimport libc.math
from scipy.integrate import _quadpack

cdef double integrand_phip(double phip, double pT, double uT_over_TFO, double sigmaT, double farg_terms, double dot_terms) nogil:
    cdef double pTsinphip = pT * libc.math.sin(phip)
    cdef double farg      = farg_terms - pTsinphip*uT_over_TFO
    cdef double f         = 0. if (farg > 20.) else 1. / (libc.math.exp(farg) - 1.)  # don't bother evaluating if exp will be huge
    return pT * (dot_terms + pTsinphip*sigmaT) * f

cdef double integrand_phip_pT(double pT, double TFO, double coshrap, double sinhrap, double sigma0, double sigmaT, double sigmaL, double u0, double uT, double uL):
    cdef double mT    = libc.math.sqrt(0.13957039*0.13957039 + pT*pT)  # charged pion mass in GeV
    cdef double ptau  = mT * coshrap
    cdef double petas = mT * sinhrap
    cdef double farg_terms = (ptau*u0 - petas*uL) / TFO
    cdef double dot_terms  = sigma0*ptau + sigmaL*petas
    return _quadpack._qagse(integrand_phip, 0., 2.*libc.math.pi, (pT, uT/TFO, sigmaT, farg_terms, dot_terms), 0, 1.49E-8, 1.49E-8, 50)[0]

cpdef double integrate(double TFO, double coshrap, double sinhrap, double sigma0, double sigmaT, double sigmaL, double u0, double uT, double uL):
    return _quadpack._qagie(integrand_phip_pT, 0., 1, (TFO, coshrap, sinhrap, sigma0, sigmaT, sigmaL, u0, uT, uL), 0, 1.49E-8, 1.49E-8, 50)[0]  # 0. .. inf
