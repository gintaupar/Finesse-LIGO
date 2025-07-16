import numpy as np
from scipy.interpolate import griddata
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt


class MechanicalMode:
    def __init__(self, disp_z_data, coords_m, F_Hz, x_m=None, y_m=None):
        self._F_Hz = F_Hz
        self._coords_data = coords_m
        self._disp_z_data = disp_z_data
        if x_m is not None and y_m is not None:
            self.update_mesh(x_m, y_m)
        else:
            self._disp_z = None
            self._x_m = None
            self._y_m = None

    @property
    def F_Hz(self):
        return self._F_Hz

    @property
    def disp_z(self):
        if self._disp_z is None:
            raise RuntimeError('Need to interpolate the data')
        return self._disp_z

    @property
    def x_m(self):
        return self._x_m

    @property
    def y_m(self):
        return self._y_m

    def update_mesh(self, x_m, y_m):
        self._x_m = x_m
        self._y_m = y_m
        x_msh, y_msh = np.meshgrid(x_m, y_m)
        self._disp_z = griddata(
            self._coords_data,
            self._disp_z_data,
            (x_msh, y_msh),
            fill_value=0,
        )

    @classmethod
    def from_slawek(cls, fname):
        data = np.loadtxt(fname)
        Fm_Hz = data[0, -1]
        assert np.all(data[:, -1] == Fm_Hz)
        mode = cls(
            disp_z_data=data[:, 3],
            coords_m=data[:, 1:3],
            F_Hz=Fm_Hz,
        )
        return mode

    def optical_mode_overlap(self, idxn, idxm, optical_modes):
        psi_n = optical_modes.mode_shape(idxn)
        psi_m = optical_modes.mode_shape(idxm)
        assert psi_n.shape == self.disp_z.shape
        assert psi_m.shape == self.disp_z.shape
        overlap_integrand = psi_n.conj() * psi_m * self.disp_z
        overlap = trapezoid(
            [
                trapezoid(ovrlp, self.x_m) for ovrlp in overlap_integrand
            ],
            self.y_m
        )
        return overlap

    def _surface_area(self):
        overlap_integrand = np.ones_like(self.disp_z)
        overlap = trapezoid(
            [
                trapezoid(ovrlp, self.x_m) for ovrlp in overlap_integrand
            ],
            self.y_m
        )
        return overlap

    def compute_all_mode_overlaps(self, optical_modes):
        self._overlaps = np.zeros(optical_modes.Nhoms, dtype=complex)
        for idx in range(optical_modes.Nhoms):
            self._overlaps[idx] = self.optical_mode_overlap(0, idx, optical_modes)
        return self._overlaps

    def plot_mode(
            self, colorbar=True, plot_nodes=False,
            label_freq=True, figsize=(8, 8),
    ):
        fig, ax = plt.subplots(figsize=figsize)
        msh = ax.pcolormesh(self.x_m, self.y_m, self.disp_z)
        if plot_nodes:
            ax.plot(
                self._coords_data[:, 0], self._coords_data[:, 1],
                '.', c='xkcd:white', markersize=2,
                )
        ax.set_aspect('equal')
        ax.set_xlabel('X position [m]')
        ax.set_ylabel('Y position [m]')
        ax.set_rasterized(True)
        if label_freq:
            ax.text(
                0.95, 0.95,
                '{:0.2f} kHz'.format(self.F_Hz * 1e-3),
                color='xkcd:maroon',
                fontsize='x-large',
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top',
            )
        if colorbar:
            plt.colorbar(msh, ax=ax, label='Z displacement')
        return fig
