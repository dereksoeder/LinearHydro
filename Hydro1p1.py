# 1+1D linearized hydro
#
# Credit: Weiyao Ke (@keweiyao)
# This version is maintained by Derek Soeder (@dereksoeder), last updated 2021-09-01

from sys import argv, stderr
import numpy as np
from scipy.interpolate import interp1d


class Hydro1p1:
    """Implementation of linear 1+1D hydro, based on code by Weiyao Ke."""

    def __init__(self):
        self.cs2 = 1./3.                # (speed of sound / c) squared; see e.g. arXiv:1806.09407
        self.cs  = np.sqrt(self.cs2)
        self.kc2 = (1.-self.cs2)**2 / 4. / self.cs2

        self.etasmax = 10.              # this might need to be adjusted depending on the profile or sqrts
        self.etasmin = -self.etasmax

        self.Npoint = 4095              # recommended to be odd
        self.x      = np.linspace(self.etasmin, self.etasmax, self.Npoint)
        self.dx     = self.x[1] - self.x[0]
        self.kmax   = np.pi / self.dx
        self.k      = np.linspace(-self.kmax, self.kmax, self.Npoint)
        self.dk     = self.k[1] - self.k[0]
        self.k      = np.fft.fftshift(self.k) - (self.dk / 2.)
        self.Y      = np.exp(-0.5 * np.square(self.k) / (4.*self.kmax)**2).astype(np.complex)

        self.Gee_cache, self.Gepz_cache = {}, {}
        self.Gee_highest_t = self.Gepz_highest_t = -np.inf

        self.etas = self.profile = None

    def __Gee(self, t):
        if (t <= self.Gee_highest_t):  # cheap way to avoid cache lookups in the regular use case (monotonically increasing t)
            if t in self.Gee_cache:  # using a float as a key isn't always a good idea, but it should be fine for this purpose
                return self.Gee_cache[t]
        else:
            self.Gee_highest_t = t

        Y = self.Y.copy()
        for i, ik in enumerate(self.k):
            Omega = self.cs * np.sqrt(np.abs(ik**2 - self.kc2))
            if ik**2 <= self.kc2:
                F = np.cosh(Omega*t) + 0.5*(1.-self.cs2)*np.sinh(Omega*t)/Omega
            else:
                F = np.cos(Omega*t) + 0.5*(1.-self.cs2)*np.sin(Omega*t)/Omega
            Y[i] *= F
        Z = np.fft.ifft(Y)
        Z = np.fft.fftshift(Z)
        ret = Z * np.exp(-(3.+self.cs2)/2.*t) * self.Npoint/np.pi/2. * self.dk

        self.Gee_cache[t] = ret
        return ret

    def __Gepz(self, t):
        if (t <= self.Gepz_highest_t):  # cheap way to avoid cache lookups in the regular use case (monotonically increasing t)
            if t in self.Gepz_cache:  # using a float as a key isn't always a good idea, but it should be fine for this purpose
                return self.Gepz_cache[t]
        else:
            self.Gepz_highest_t = t

        Y = self.Y.copy()
        for i, ik in enumerate(self.k):
            Omega = self.cs * np.sqrt(np.abs(ik**2 - self.kc2))
            if np.abs(ik)<1e-5:
                F = 0.
            elif ik**2 <= self.kc2:
                F = -1j * np.sinh(Omega*t) * (-Omega**2 + (self.cs2-1)**2 / 4.)/(ik*Omega)
            else:
                F = -1j * np.sin(Omega*t) * (Omega**2 + (self.cs2-1)**2 / 4.)/(ik*Omega)
            Y[i] *= F
        Z = np.fft.ifft(Y)
        Z = np.fft.fftshift(Z)
        ret = Z * np.exp(-(3.+self.cs2)/2.*t) * self.Npoint/np.pi/2. * self.dk

        self.Gepz_cache[t] = ret
        return ret

    @staticmethod
    def linterp(inxs, inys, outxs):
        return interp1d(inxs, inys, kind="linear", bounds_error=False, fill_value=0.)(outxs)

    def load(self, etas, profile, sqrts=None):
        """Prepare an energy density profile for hydrodynamic evolution."""

        profile = np.array(profile, ndmin=1)

        if (len(profile.shape) == 2):
            if (profile.shape[1] == 1):
                profile = profile[:,0]
            elif (profile.shape[1] == 2):
                if etas is not None:
                    raise ValueError("etas cannot be specified if profile includes an eta_s column")
                if sqrts is not None:
                    raise ValueError("sqrts must not be specified if profile includes an eta_s column")
                etas, profile = profile.T

        if (len(profile.shape) != 1) or (len(profile) < 2):
            raise ValueError(f"invalid profile shape {profile.shape}")

        if etas is None:
            if sqrts is None:
                raise ValueError("either etas or sqrts must be provided")
            if not np.all([fn(sqrts) for fn in (np.isscalar, np.isfinite, np.isreal, lambda _: _ > 0.)]):
                raise ValueError("sqrts must be a positive real number")

            etasmax = np.arccosh(0.5 * sqrts / 0.2)
            etas = np.linspace(-etasmax, etasmax, len(profile))
        else:
            if sqrts is not None:
                raise ValueError("sqrts must not be specified if etas is provided")

            etas = np.array(etas, ndmin=1)
            if not np.all(etas.shape == profile.shape):
                raise ValueError(f"etas shape {etas.shape} does not match profile shape {profile.shape}")

        self.etas = etas
        #self.profile = np.maximum(0., interp1d(self.etas, profile, kind="cubic", bounds_error=False, fill_value=0.)( self.x ))  # cubic spline can dip negative; use np.maximum to mitigate
        self.profile = self.linterp(self.etas, profile, self.x)  # linear seems to perform better when comparing to coarse hydro

    def __Freezeout(self, dy, v=0.):
        return 1. / (np.cosh(self.x-dy) - v**2)**2

    def __y2eta(self, dNdrap, m_over_pT):
        MoverMT = 1. / np.sqrt(1. + m_over_pT**-2)
        beta = 1. / np.sqrt( 1. + 1. / (np.cosh(self.x)**2 / MoverMT**2 - 1.) )
        eta = 0.5 * np.log( (beta + np.tanh(self.x)) / (beta - np.tanh(self.x)) )
        Jacobian = np.sqrt( 1. - (MoverMT / np.cosh(self.x))**2 )
        return self.linterp(eta, Jacobian*dNdrap, self.x)

    def evolve(self, tau0, tau, freezeout=True, m_over_pT=0.6039437166879817):  # arXiv:1207.6517 (Roehrscheid & Wolschin)
        """Compute the hydrodynamic evolution of a previously loaded energy density profile."""

        if not (tau >= tau0 > 0.):
            raise ValueError("evolution time must be tau >= tau0 > 0.")

        if self.profile is None:
            raise RuntimeError("a profile must be loaded before calling evolve")

        t = np.log(tau / tau0)

        Gee = self.__Gee(t)
        dEdetas = self.dx * np.convolve(self.profile, Gee)[self.Npoint//2: self.Npoint//2+self.Npoint]

        Gepz = self.__Gepz(t)
        dPzdetas = self.dx * (tau / tau0) * np.convolve(self.profile, Gepz)[self.Npoint//2: self.Npoint//2+self.Npoint]
        vz = dPzdetas / (1.+self.cs2) / dEdetas * tau0 / tau

        if freezeout:
            deltaEtas = 0.5 * np.log((1.+vz)/(1.-vz))
            GFreezeOut = self.__Freezeout(deltaEtas)
            dNdrap = self.dx/2. * np.convolve(dEdetas, GFreezeOut)[self.Npoint//2: self.Npoint//2+self.Npoint]
            dNdeta = None if m_over_pT is None else self.linterp(self.x, np.real(self.__y2eta(dNdrap, m_over_pT)), self.etas)
            dNdrap = self.linterp(self.x, np.real(dNdrap), self.etas)
        else:
            dNdrap = dNdeta = None

        return self.linterp(self.x, np.real(dEdetas), self.etas), self.linterp(self.x, np.real(vz), self.etas), dNdrap, dNdeta


def Hydro1p1_main(*args):
    if (len(args) != 4):
        print(f"Usage: Hydro1p1.py profilefile tau0_in_fm/c tau1_in_fm/c dtau_in_fm/c", file=stderr)
        exit(1)

    profilefile, tau0, tau1, dtau = args[0], *map(float, args[1:])

    hydro = Hydro1p1()
    hydro.load( *np.loadtxt(profilefile).T )

    n = 0
    while True:
        n += 1
        tau = tau0 + (n * dtau)
        if (tau > tau1): break

        dEdetas, _, _, _ = hydro.evolve(tau0, tau, freezeout=False, m_over_pT=None)
        print(" ".join(map(str, dEdetas)))

if __name__ == "__main__":
    Hydro1p1_main(*argv[1:])
