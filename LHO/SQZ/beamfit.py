import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from finesse import BeamParam
from finesse.utilities.text import scale_si


class BeamParamFit:
    def __init__(
            self,
            w_meas,
            z_meas,
            w0,
            z0,
            method="Powell",
            wavelength=1064e-9,
            **fit_kw,
    ):
        self._w_meas = w_meas
        self._z_meas = z_meas
        self._wavelength = wavelength

        def cost(x):
            # x[0]: beam radius
            # x[1]: distance from the waist
            zr = np.pi * x[0]**2 / wavelength
            return np.sum((w_meas - x[0] * np.sqrt(1 + ((z_meas - x[1]) / zr)**2))**2)

        self._res = minimize(cost, x0=(w0, z0), method=method, **fit_kw)
        self._w0 = self._res.x[0]
        self._z0 = self._res.x[1]

    @property
    def w_meas(self):
        return self._w_meas

    @property
    def z_meas(self):
        return self._z_meas

    @property
    def optimization_result(self):
        return self._res

    @property
    def w0(self):
        return self._w0

    @property
    def z0(self):
        return self._z0

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def zr(self):
        return np.pi * self.w0**2 / self.wavelength

    def beamsize(self, dist):
        return self.w0 * np.sqrt(1 + ((dist - self.z0) / self.zr)**2)

    def curvature(self, dist):
        raise NotImplementedError()

    def q_at(self, dist):
        return BeamParam(w0=self.w0, z=dist - self.z0, wavelength=self.wavelength, nr=1)

    def __call__(self, dist):
        return self.q_at(dist)

    def plot_fit(
            self,
            dist,
            fig=None,
            ax=None,
            plot_curvature=False,
            plot_residuals=False,
            annotate=False,
            fit_kw={},
            data_kw={},
            xscale=None,
            yscale=None,
    ):
        if plot_curvature:
            raise NotImplementedError()
        if plot_residuals:
            raise NotImplementedError()
        if annotate:
            raise NotImplementedError()

        def set_mapping_default(mapping, default, key, *aliases):
            for k in [key] + list(aliases):
                if k in mapping:
                    return
            mapping[key] = default

        set_mapping_default(fit_kw, "Fit", "label")
        set_mapping_default(data_kw, "Data", "label")
        set_mapping_default(fit_kw, "-", "linestyle", "ls")
        set_mapping_default(data_kw, "", "linestyle", "ls")
        set_mapping_default(data_kw, "o", "marker")

        if fig is None and ax is None:
            fig, ax = plt.subplots()
        ax.plot(dist, self.beamsize(dist), **fit_kw)
        ax.plot(self.z_meas, self.w_meas, **data_kw)
        ax.legend()
        ax.grid(True, which="major", alpha=0.5)
        ax.grid(True, which="minor", alpha=0.5)
        ax.set_xlim(dist[0], dist[-1])
        if xscale is not None:
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda v, p: f"{v * xscale:0.1f}")
            )
        if yscale is not None:
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, p: f"{v * yscale:0.1f}")
            )
        return fig, ax

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"w0={scale_si(self.w0, units='m')}, "
            f"z0={scale_si(self.z0, units='m')}, "
            f"zr={scale_si(self.zr, units='m')})"
        )
