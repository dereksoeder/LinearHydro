# Cooper-Frye computation for use with linear hydro, assuming a 2D Gaussian transverse profile
#
# Derek Soeder (@dereksoeder), last updated 2021-09-03

from sys import argv, stderr
import numpy as np
import scipy.integrate

from Hydro1p1 import Hydro1p1


if (len(argv[1:]) < 7):
    print(f"Usage:   LHCooperFryeTest.py   dEdetasfile  transverse_Gaussian_width_in_fm  tau0_in_fm/c  dtau_in_fm/c  epsilon_FO_in_GeV/fm^3  T_FO_in_GeV  {{rapidity | rapidity1,rapidity2}}  [...]", file=stderr)
    exit(1)

dEdetasfile = argv[1]           # one row per etas slice (2 or more slices); each row is:  etas  dEdetas_in_GeV
Gaussian_width, tau0, dtau, epsilon_FO, TFO = map(float, argv[2:7])

raplist = [ tuple(map(float, arg.split(","))) if "," in arg else float(arg) for arg in argv[7:] ]

for rapspec in raplist:
    if (type(rapspec) is tuple) and ((len(rapspec) != 2) or (rapspec[0] >= rapspec[1])):
        raise ValueError(f"invalid rapidity range {rapspec}")


hbarc  = 0.19733                # h-bar times speed of light, in GeV-fm
mpi    = 0.13957039             # pion mass in GeV
mpi_sq = mpi**2
gpi    = 1.                     # degeneracy factor

Gaussian_width_sq = Gaussian_width**2  # square of width parameter for Gaussian transverse energy density profile ansatz, in fm^2


#
# load profile and perform linear hydro evolution
#

etaslist, profile0 = np.loadtxt(dEdetasfile, ndmin=2).T
detas = etaslist[1] - etaslist[0]

hydro = Hydro1p1()
hydro.load(etaslist, profile0)

rf_array = []
u_array = []
taulist = []

print("Starting linear hydro...", file=stderr)

n = 0
while True:
    n += 1
    tau = tau0 + (n * dtau)
    taulist.append(tau)

    Eprofile, vLprofile, _, _ = hydro.evolve(tau0, tau, False, None)

    print(f"  tau = {tau:g} fm/c, max epsilon = {np.max(Eprofile / (2. * np.pi * Gaussian_width_sq))} GeV/fm^3", file=stderr)

    rflist = np.sqrt(
            (2. * Gaussian_width_sq) *
            np.maximum(0., np.log( Eprofile / (2. * np.pi * Gaussian_width_sq * epsilon_FO) ))
        )

    vTprofile = 0.150 * rflist * (tau - tau0) / Gaussian_width_sq  # TODO: figure out the correct proportionality
    ulist = np.array([ (1., vT, vL) for vT, vL in zip(vTprofile, vLprofile) ])

    ulist[:,1:] *= np.minimum(1., 0.9 / (np.linalg.norm(ulist[:,1:], axis=1) + 1.E-8))[:,np.newaxis]  # limit beta to <= 0.9
    ulist /= np.sqrt(1. - np.linalg.norm(ulist[:,1:], axis=1))[:,np.newaxis]  # apply gamma factor

    rf_array.append(rflist)
    u_array.append(ulist)

    if not np.any(rflist > 0.): break

print(f"All slices frozen out at {taulist[-1]} fm/c.", file=stderr)


#
# evaluate Cooper-Frye
#

def integrand2(phip, pT, coshrap, sinhrap, sigma0, sigmaT, sigma3, u0, uT, u3):
    sinphip = np.sin(phip)
    mT      = np.sqrt(mpi_sq + pT**2)
    ptau    = mT * coshrap
    petas   = mT * sinhrap
    farg    = (ptau*u0 - pT*uT*sinphip - petas*u3) / TFO
    f       = 0. if (farg > 20.) else 1. / (np.exp(farg) - 1.)                      # assumes that flow velocity is always radially outward, which appears to be true
    return pT * ((sigma0 * ptau) + (sigmaT * sinphip * pT) + (sigma3 * petas)) * f  # assumes that surface normal is always radially outward, which appears to be true

def integrand3(phip, pT, rap, etas, *sigma_and_u):
    return integrand2(phip, pT, np.cosh(rap - etas), np.sinh(rap - etas), *sigma_and_u)

def differential(arr, i, j=None):
    f = (lambda idx: arr[idx]) if j is None else (lambda idx: arr[idx][j])

    if (i == 0):
        return (f(i+1) - f(i))
    elif (i == len(arr) - 1):
        return (f(i) - f(i-1))
    elif (i == 1) or (i == len(arr) - 2):
        return (f(i+1) - f(i-1)) / 2.
    else:
        return (-f(i+2) + 8.*f(i+1) - 8.*f(i-1) + f(i-2)) / 12.  # five-point stencil


for rapspec in raplist:
    Npi = 0.

    for ietas, etas in enumerate(etaslist):
        # note: MUSIC covers only |y-etas| < 4.0 (see MUSIC/src/freeze_pseudo.cpp: Freeze::ComputeParticleSpectrum_pseudo_improved, y_minus_eta_cut), but the difference is minor
        if np.isscalar(rapspec):
            coshrap, sinhrap = np.cosh(rapspec - etas), np.sinh(rapspec - etas)

        for itau, tau in enumerate(taulist):
            rf = rf_array[itau][ietas]
            if (rf < 1.E-8): continue

            drf_in_tau  = differential(rf_array, itau, ietas)
            drf_in_etas = differential(rf_array[itau], ietas)

            sigma0 = -drf_in_tau * tau * detas
            sigmaT = dtau * tau * detas
            sigma3 = -drf_in_etas * dtau

            if np.isscalar(rapspec):
                extras = ( coshrap, sinhrap, sigma0, sigmaT, sigma3, *u_array[itau][ietas] )

                npi = scipy.integrate.dblquad(
                    integrand2,         # first argument is y, second is x; additional arguments are passed via `extras`
                    0., np.inf,         # pT (x) limits of integration
                    0., 2.*np.pi,       # phi_p (y) limits of integration
                    extras )[0]         # [0] to get result of integration ([1] is error estimate)
            else:
                raplo, raphi = rapspec

                extras = ( etas, sigma0, sigmaT, sigma3, *u_array[itau][ietas] )

                npi = scipy.integrate.tplquad(
                    integrand3,         # first argument is z, second is y, third is x; additional arguments are passed via `extras`
                    raplo, raphi,       # rap (x) limits of integration
                    0., np.inf,         # pT (y) limits of integration
                    0., 2.*np.pi,       # phi_p (z) limits of integration
                    extras )[0]         # [0] to get result of integration ([1] is error estimate)

            Npi += (2. * np.pi * rf) * gpi * npi / (2. * np.pi)**3 / (hbarc**3)

    print(rapspec, Npi)
