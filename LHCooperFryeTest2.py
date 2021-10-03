# Cooper-Frye computation for use with linear hydro, assuming a 2D Gaussian transverse profile
# v2: calls into scipy.integrate at a low level from Cython code for much faster numerical integration
#
# Derek Soeder (@dereksoeder), last updated 2021-10-02

from sys import argv, stderr
import numpy as np

from Hydro1p1 import Hydro1p1

import pyximport ; pyximport.install()
import LHCooperFryeTest2quad


def compute_sigmas(grid, dx, dy):
    sigmas = []

    xmesh, ymesh = np.meshgrid(dx * np.arange(grid.shape[2]), dy * np.arange(grid.shape[1]), indexing="xy")

    for slice in grid:
        slicesum = slice.sum()
        xcm, ycm = ( np.sum(mesh * slice) / slicesum for mesh in (xmesh, ymesh) )

        rs = np.sqrt( np.square(xmesh - xcm) + np.square(ymesh - ycm) )
        rmin, rmax = np.sqrt(dx**2 + dy**2) / 2., rs.max() + np.sqrt(dx**2 + dy**2)

        cutoff = slicesum * 0.39346934028736696  # \int_0^1 r dr exp(-r**2/2) / \int_0^\infty r dr exp(-r**2/2)

        for steps in range(30):
            r = (rmin + rmax) / 2.
            rmin, rmax = (r, rmax) if (slice[rs <= r].sum() <= cutoff) else (rmin, r)

        sigmas.append(rmax)

    return np.array(sigmas)


if (len(argv[1:]) < 7):
    print("Usage:   LHCooperFryeTest2.py   " +
                   "{dETdetasfile transverse_Gaussian_width_in_fm | epsilongridfile dxy,detas}  " +
                   "tau0_in_fm/c  dtau_in_fm/c  " +
                   "epsilon_FO_in_GeV/fm^3  T_FO_in_GeV  " +
                   "{rapidity | rapidity1,rapidity2}  [...]",
          file=stderr)
    exit(1)

if "," in argv[2]:
    grid = np.loadtxt(argv[1], ndmin=2)  # nxy columns, nxy*netas rows representing netas slices
    grid = grid.reshape(-1, grid.shape[1], grid.shape[1])
    dxy, detas = map(float, argv[2].split(","))

    profile0 = grid.sum(axis=(1,2)) * dxy**2
    etassize = detas * (len(profile0) - 1)
    etaslist = np.linspace(-etassize/2., etassize/2., len(profile0), endpoint=True)
    Gaussian_width = np.average( compute_sigmas(grid, dxy, dxy), weights=profile0 )
    del grid
else:
    etaslist, profile0 = np.loadtxt(argv[1], ndmin=2).T  # one row per etas slice (2 or more slices); each row is:  etas  dETdetas_in_GeV
    detas = etaslist[1] - etaslist[0]
    Gaussian_width = float(argv[2])

tau0, dtau, epsilon_FO, TFO = map(float, argv[3:7])

raplist = list(map(float, argv[7:]))


hbarc = 0.19733                 # h-bar times speed of light, in GeV-fm
gpi   = 1.                      # degeneracy factor for \pi^+
# NOTE: charged pion mass is hard-coded in accompanying .pyx

Gaussian_width_sq = Gaussian_width**2  # square of width parameter for Gaussian transverse energy density profile ansatz, in fm^2


#
# load profile and perform linear hydro evolution
#

hydro = Hydro1p1()
hydro.load(etaslist, profile0)

rf_array = []
u_array = []
taulist = []

print("Starting linear hydro...", file=stderr)

n = -1  # include surface contribution at time tau_0
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
    ulist /= np.sqrt(1. - np.square(ulist[:,1:]).sum(axis=1))[:,np.newaxis]  # apply gamma factor

    rf_array.append(rflist)
    u_array.append(ulist)

    if not np.any(rflist > 0.): break

print(f"All slices frozen out at {taulist[-1]} fm/c.", file=stderr)


#
# evaluate Cooper-Frye
#

def differential(arr, i, j=None):
    if (len(arr) == 1):
        return 0.

    f = (lambda idx: arr[idx]) if j is None else (lambda idx: arr[idx][j])

    if (i == 0):
        return (f(i+1) - f(i))
    elif (i == len(arr) - 1):
        return (f(i) - f(i-1))
    elif (i == 1) or (i == len(arr) - 2):
        return (f(i+1) - f(i-1)) / 2.
    else:
        return (-f(i+2) + 8.*f(i+1) - 8.*f(i-1) + f(i-2)) / 12.  # five-point stencil


for rap in raplist:
    Npi = 0.

    for ietas, etas in enumerate(etaslist):
        # note: MUSIC covers only |y-etas| < 4.0 (see MUSIC/src/freeze_pseudo.cpp: Freeze::ComputeParticleSpectrum_pseudo_improved, y_minus_eta_cut), but the difference is minor
        coshrap, sinhrap = np.cosh(rap - etas), np.sinh(rap - etas)

        for itau, tau in enumerate(taulist):
            rf = rf_array[itau][ietas]
            if (rf < 1.E-8): continue

            drf_in_tau  = differential(rf_array, itau, ietas)
            drf_in_etas = differential(rf_array[itau], ietas)

            sigma0 = -drf_in_tau * tau * detas
            sigmaT = dtau * tau * detas
            sigmaL = -drf_in_etas * dtau

            Npi += (2. * np.pi * rf) * gpi * \
                   LHCooperFryeTest2quad.integrate(TFO, coshrap, sinhrap, sigma0, sigmaT, sigmaL, *u_array[itau][ietas]) \
                   / (2. * np.pi)**3 / (hbarc**3)

    print(rap, Npi)
