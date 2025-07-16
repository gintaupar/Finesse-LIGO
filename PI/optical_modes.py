import numpy as np
from abc import ABC, abstractmethod
from scipy.integrate import trapezoid
import scipy.special as sp
import matplotlib.pyplot as plt

from finesse.cymath.homs import HGModes as HGModes_calc


def wrap_rad(ang_rad):
    return np.mod(ang_rad + np.pi, 2 * np.pi) - np.pi


class OpticalModes(ABC):
    def __init__(self):
        pass

    @property
    def homs(self):
        return self._homs

    @property
    def Nhoms(self):
        return len(self.homs)

    @property
    def q(self):
        return self._q

    def mode_shape(self, hom_idx):
        if self._mode_shapes is None:
            raise RuntimeError("must compute mode shapes")
        return self._mode_shapes[hom_idx]

    @abstractmethod
    def calc_mode_shapes(self, x_m, y_m, dx_m=0, dy_m=0):
        pass

    def calc_mode_overlaps(self, idxn, idxm):
        psi_n = self.mode_shape(idxn)
        psi_m = self.mode_shape(idxm)
        overlap_integrand = psi_n.conj() * psi_m
        overlap = trapezoid(
            [
                trapezoid(ovrlp, self._x_m) for ovrlp in overlap_integrand
            ],
            self._y_m
        )
        return np.abs(overlap)**2

    def _plot_mode(self, mode_shape, ptype="abs", colorbar=True, figsize=(8, 8)):
        if ptype == "abs":
            label = r"Intensity [W/m$^2$]"
            func = lambda x: np.abs(x)**2
        elif ptype in ["real", "imag"]:
            label = r"Amplitude [$\sqrt{\mathrm{W}/\mathrm{m}}$]"
            func = lambda x: getattr(np, ptype)(x)
        else:
            raise ValueError("Unrecognized plot type")
        # mode = np.abs(mode_shape)**2
        mode = func(mode_shape)
        fig, ax = plt.subplots(figsize=figsize)
        msh = ax.pcolormesh(self._x_m, self._y_m, mode)
        ax.set_aspect("equal")
        ax.set_xlabel("X position [m]")
        ax.set_ylabel("Y position [m]")
        ax.set_rasterized(True)
        if colorbar:
            plt.colorbar(msh, ax=ax, label=label)
        return fig

    def plot_mode(
            self, hom_idx, ptype="abs", colorbar=True, figsize=(8, 8),
    ):
        return self._plot_mode(
            self.mode_shape(hom_idx), ptype=ptype, colorbar=colorbar, figsize=figsize,
        )


class CavityModes(OpticalModes):
    def __init__(self, eigen_sol, homs, maxtem, q):
        gouy_rad = np.angle(eigen_sol.eigvalues)
        loss_rt = 1 - np.abs(eigen_sol.eigvalues)**2
        # sort by increasing loss
        self._sidx = np.argsort(loss_rt)
        self._eigvalues = eigen_sol.eigvalues
        self._eigvectors = eigen_sol.eigvectors
        self._gouy = gouy_rad
        self._loss_rt = loss_rt
        self._homs = homs
        self._q = q
        self._mode_shapes = None

        # gouy_wrapped = wrap_rad(gouy_rad)[0]
        # int_mult_gouy = wrap_rad(np.arange(0, maxtem + 1) * gouy_wrapped)
        # print(int_mult_gouy)
        # ord_m, gouy_m = np.meshgrid(int_mult_gouy, gouy_wrapped)
        # print(ord_m)
        # print(gouy_m)
        # # # self._mode_order = np.argmin(np.abs(gouy_m - ord_m), axis=0)
        # # print("mins")
        # # self._mode_order = np.abs(gouy_m - ord_m)

    @property
    def loss_rt(self):
        return self._loss_rt[self._sidx]

    @property
    def gouy_rad(self):
        return self._gouy[self._sidx]

    @property
    def gouy_deg(self):
        return self.gouy_rad * 180 / np.pi

    @property
    def eigvalues(self):
        return self._eigvalues[self._sidx]

    @property
    def eigvectors(self):
        return self._eigvectors[self._sidx]

    @property
    def mode_order(self):
        return self._mode_order[self._sidx]

    def mode_shape(self, hom_idx):
        # overloading this to make sure everything is sorted
        if self._mode_shapes is None:
            raise RuntimeError("must compute mode shapes")
        return self._mode_shapes[self._sidx][hom_idx]

    def calc_mode_shapes(self, x_m, y_m, dx_m=0, dy_m=0):
        self._x_m = x_m - dx_m
        self._y_m = y_m - dy_m
        modes = HGModes_calc(self.q, np.array(self.homs).astype(np.int32))
        psi_inf = modes.compute_2d_modes(self._x_m, self._y_m)
        self._mode_shapes = np.zeros_like(psi_inf)
        for mi, psi in enumerate(psi_inf):
            self._mode_shapes[mi] = np.einsum(
                "i,ijk->jk", self._eigvectors[:, mi], psi_inf,
            )


class HGModes(OpticalModes):
    def __init__(self, homs, q):
        self._homs = homs
        self._q = q

    def _1d(self, x_m, n):
        # FIXME: put phase back
        w_m = self.q.w
        norm = (2 / np.pi)**(1 / 4) / np.sqrt(2**n * sp.factorial(n) * w_m)
        poly = sp.eval_hermite(n, np.sqrt(2) * x_m / w_m)
        exp = np.exp(-x_m**2 / w_m**2)
        return  norm * poly * exp

    def calc_mode_shapes(self, x_m, y_m, dx_m=0, dy_m=0):
        self._x_m = x_m - dx_m
        self._y_m = y_m - dy_m
        # FIXME: put phase back
        self._mode_shapes = np.zeros((self.Nhoms, len(y_m), len(x_m)))
        for (idx, [n, m]) in enumerate(self.homs):
            un = self._1d(self._x_m, n)
            um = self._1d(self._y_m, m)
            self._mode_shapes[idx] = np.outer(um, un)


class LGModes(OpticalModes):
    def __init__(self, homs, q):
        self._homs = homs
        self._q = q

    def _single_mode(self, x_m, y_m, p, m):
        x_msh, y_msh = np.meshgrid(x_m, y_m)
        r_m = np.sqrt(x_msh**2 + y_msh**2)
        theta_rad = np.angle(y_msh / x_msh)
        ma = np.abs(m)
        w_m = self.q.w
        norm = np.sqrt(2 * sp.factorial(p) / (np.pi * sp.factorial(p + ma)))
        r_norm = np.sqrt(2) * r_m / w_m
        poly = r_norm**ma * sp.eval_genlaguerre(p, ma, r_norm**2) / w_m
        exp = np.exp(-r_norm**2 / 2 + 1j * m * theta_rad)
        return norm * poly * exp

    def calc_mode_shapes(self, x_m, y_m, dx_m=0, dy_m=0):
        self._x_m = x_m - dx_m
        self._y_m = y_m - dy_m
        # FIXME: put phase back
        self._mode_shapes = np.zeros((self.Nhoms, len(y_m), len(x_m)), dtype=complex)
        for (idx, [p, m]) in enumerate(self.homs):
            self._mode_shapes[idx] = self._single_mode(self._x_m, self._y_m, p, m)
