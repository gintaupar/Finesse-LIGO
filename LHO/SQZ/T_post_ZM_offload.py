import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pytest

from munch import Munch
from finesse import BeamParam, Model
from finesse.symbols import Variable
from finesse.tracing import abcd
import finesse.components as fc

from beamfit import BeamParamFit
from sqz_factory import LIGOwSQZFactory, SQZ_params


class SQZFactory(LIGOwSQZFactory):
    def set_default_options(self):
        super().set_default_options()
        self.options.SQZ.use_ZM_strain_gauge = False

    def post_make(self, model, params):
        if self.options.SQZ.use_ZM_strain_gauge:
            ZM5_V = model.add_parameter("ZM5_V", 8).ref
            ZM4_V = model.add_parameter("ZM4_V", -4).ref
            ZM5_0 = model.add_parameter("ZM5_0", 1.035).ref
            ZM5_1 = model.add_parameter("ZM5_1", -0.0311).ref
            ZM4_0 = model.add_parameter("ZM4_0", -0.236).ref
            ZM4_1 = model.add_parameter("ZM4_1", -0.0262).ref
            model.ZM4.Rc = 2 / (ZM4_0 + ZM4_1 * ZM4_V)
            model.ZM5.Rc = 2 / (ZM5_0 + ZM5_1 * ZM5_V)


lengths = Munch({
    k: v for k, v in SQZ_params.SQZ.items() if k.split("_")[0] == "length"
})
lengths.update(Munch(
    length_SRM_OFI=0.9046,
    length_OFI_OM1=2.5354, # SRM->OM1 - SRM->OFI = 3.440-0.9046
))


def length(m1, m2):
    return lengths.get(f"length_{m1}_{m2}")


in2m = 0.0254
OM1_ZM5_m = (
    length("OFI", "OM1") +
    2 * length("SRM", "OFI") +
    length("ZM6", "OFI") +
    length("ZM5", "ZM6")
)
VIP_ZM5_m = length("VIP", "ZM4") + length("ZM4", "ZM5")
VIP_OM1_m = VIP_ZM5_m + OM1_ZM5_m


class BeamMeasurement:
    def __init__(
            self, location, zm4_gauge, zm4_rqst, zm5_gauge, zm5_rqst,
            wx_um, sx_um, wy_um, sy_um,
    ):
        self.location = location
        self.HAM = int(location[0])
        assert self.HAM in [6, 7]
        to_m = 1e-6 / 2
        self.wx_m = wx_um * to_m
        self.wy_m = wy_um * to_m
        self.sx_m = sx_um * to_m
        self.sy_m = sy_um * to_m
        self.zm4_gauge = zm4_gauge
        self.zm5_gauge = zm5_gauge
        self.zm4_rqst = zm4_rqst
        self.zm5_rqst = zm5_rqst

    @property
    def ZM5_D(self):
        """Curvature of ZM5 [D]"""
        return (-31.1 * self.zm5_gauge + 1035) * 1e-3

    @property
    def ZM4_D(self):
        """Curvature of ZM4 [D]"""
        return (-26.2 * self.zm4_gauge - 236) * 1e-3

    @property
    def distance_from_ref_m(self):
        """Distance from the reference

        For HAM7 this is ZM5, positive towards HAM6
        For HAM6 this is OM1, positive towards OMC

        See figure in *LAST* comment of LHO:75770 for HAM6
        """
        if self.HAM == 7:
            zmap_in = {
                "7A": 62.25,
                "7B": 60.25,
                "7C": 56.5,
            }
            return zmap_in[self.location] * in2m

        elif self.HAM == 6:
            zmap_in = {
                "6A": 52,
                "6B": 51.75,
                "6C": -23.25,
                "6D": 13.5,
            }
            return -zmap_in[self.location] * in2m

    @property
    def z_m(self):
        """Distance from VIP [m]"""
        if self.HAM == 7:
            return self.distance_from_ref_m + VIP_ZM5_m
        elif self.HAM == 6:
            return self.distance_from_ref_m + VIP_OM1_m

    def subs(self, subs0=dict()):
        """Dictionary for symbolic substitutions"""
        subs = subs0.copy()
        subs.update({
            "ZM4.Rcx": 2 / self.ZM4_D,
            "ZM4.Rcy": 2 / self.ZM4_D,
            "ZM5.Rcx": 2 / self.ZM5_D,
            "ZM5.Rcy": 2 / self.ZM5_D,
        })
        return subs

    def subs_strain_gauge(self, subs0=dict()):
        subs = subs0.copy()
        subs.update({
            "ZM4_V": self.zm4_gauge,
            "ZM5_V": self.zm5_gauge,
        })
        return subs

    def update_ROCs(self, model):
        """Update the ROCs of a model with those from this measurement"""
        model.ZM4.Rc = 2 / self.ZM4_D
        model.ZM5.Rc = 2 / self.ZM5_D

    def update_ROCs_strain_gauge(self, model):
        model.ZM4_V = self.zm4_gauge
        model.ZM5_V = self.zm5_gauge

    @property
    def ABCD(self):
        """ABCD matrix to progate from reference"""
        return abcd.space(L=self.distance_from_ref_m)


# HAM6 measurements
bm6 = Munch({
    "B1": BeamMeasurement("6B", 0.82, 0, -4.14, 0, 1392, 8.7, 1573, 9.9),
    "B2": BeamMeasurement("6B", 4.38, 100, -4.14, 0, 1406, 7.3, 1584, 6.3),
    "A1": BeamMeasurement("6A", 5.7, 100, 0.625, 100, 2074, 9.9, 1611, 7.4),
    "C1": BeamMeasurement("6C", 5.7, 100, 0.612, 100, 1670, 10.2, 1618, 7.0),
    "D1": BeamMeasurement("6D", 5.68, 100, 0.52, 100, 1285, 6.7, 1426, 9.2),
})
# HAM7 measurements
bm7 = Munch({
    "B1": BeamMeasurement("7B", 4.39, 100, -4.14, 0, 2001, 6.4, 2017, 6.1),
    "B2": BeamMeasurement("7B", 1, 0, -4.14, 0, 1928, 5.6, 1924, 5.9),
    "B3": BeamMeasurement("7B", 0.98, 0, -0.52, 100, 2200, 8.2, 2212, 9.4),
    "B4": BeamMeasurement("7B", 4.31, 100, -0.52, 100, 2276, 9.5, 2303, 9.3),
    "B5": BeamMeasurement("7B", 8.49, 200, -0.52, 100, 2363, 11, 2412, 13.8),
    "B6": BeamMeasurement("7B", 8.53, 200, 3.57, 200, 2611, 9.0, 2691, 7.2),
    "B7": BeamMeasurement("7B", 5.74, 100, 3.59, 200, 2539, 13.7, 2603, 18.1),
    "A1": BeamMeasurement("7A", 5.72, 100, 0.64, 100, 2387, 13.5, 2446, 13.1),
    "C1": BeamMeasurement("7C", 5.715, 100, 0.63, 100, 2374, 11, 2415, 9.5),
})


@pytest.fixture
def simple_sqz_factory(ppath_join):
    factory = SQZFactory(ppath_join("LHO", "lho_O4.yaml"), SQZ_params)
    factory.update_parameters(ppath_join("LHO", "lho_mcmc_RC_lengths.yaml"))
    factory.options.add_ifo = False
    factory.options.SQZ.add_seed_laser = False
    factory.options.detailed_OFI = False
    factory.options.SQZ.use_ZM_strain_gauge = True
    return factory


def simple_HAM7_model(use_ZM_strain_gauge=True):
    par = SQZ_params.SQZ
    model = Model()
    model.add(fc.Laser("VIP"))
    for bs in ["ZM4", "ZM5", "ZM6"]:
        model.add(fc.Beamsplitter(bs, **par[bs]))
    model.connect(model.VIP.p1, model.ZM4.p1, par.length_VIP_ZM4)
    model.connect(model.ZM4.p2, model.ZM5.p1, par.length_ZM4_ZM5)
    model.connect(model.ZM5.p2, model.ZM6.p1, par.length_ZM5_ZM6)
    if use_ZM_strain_gauge:
        ZM5_V = model.add_parameter("ZM5_V", 8).ref
        ZM4_V = model.add_parameter("ZM4_V", -4).ref
        ZM5_0 = model.add_parameter("ZM5_0", 1.035).ref
        ZM5_1 = model.add_parameter("ZM5_1", -0.0311).ref
        ZM4_0 = model.add_parameter("ZM4_0", -0.236).ref
        ZM4_1 = model.add_parameter("ZM4_1", -0.0262).ref
        model.ZM4.Rc = 2 / (ZM4_0 + ZM4_1 * ZM4_V)
        model.ZM5.Rc = 2 / (ZM5_0 + ZM5_1 * ZM5_V)
    return model


# some fit parameters from simple HAM7
fpars = Munch(
    NM=Munch(w0x=564e-6, w0y=567e-6, zx=-289e-3, zy=-297e-3, Z4=0.76, Z5=0.29),
    Pow=Munch(w0x=516e-6, w0y=522e-6, zx=-278e-3, zy=-283e-3, Z4=0.97, Z5=0.29),
    CG=Munch(w0x=429e-6, w0y=464e-6, zx=236e-3, zy=290e-3, Z4=1.03, Z5=-0.14),
    BFGS1=Munch(w0x=673e-6, w0y=688e-6, zx=160e-3, zy=138e-3, Z4=0.69, Z5=0.09),
    BFGS2=Munch(w0x=659e-6, w0y=673e-6, zx=108e-3, zy=89e-3, Z4=0.70, Z5=0.11),
)


def T_simple_HAM7(tpath_join):
    model = simple_HAM7_model(use_ZM_strain_gauge=True)
    # qvip = BeamParam(w0=Variable("w0"), z=Variable("z"))
    par_key = "BFGS1"
    p = fpars[par_key]
    p_to = "ZM6.p1.i"
    p_fr = "VIP.p1.o"
    bm = bm7.B1
    qvip_x = BeamParam(w0=p.w0x, z=p.zx)
    qvip_y = BeamParam(w0=p.w0y, z=p.zy)
    model.ZM4_0 = p.Z4
    model.ZM5_0 = p.Z5

    bm.update_ROCs_strain_gauge(model)
    symbolic = False
    # symbolic = ("ZM5.Rcx", "ZM5.Rcy", "ZM4.Rcx", "ZM4.Rcy")
    psol = Munch(
        x=model.propagate_beam(p_fr, p_to, q_in=qvip_x, symbolic=symbolic, direction="x"),
        y=model.propagate_beam(p_fr, p_to, q_in=qvip_y, symbolic=symbolic, direction="y"),
    )
    ps_opts = dict(radius_scale=1e3, radius_pref="m")
    save_opts = dict(bbox_inches="tight", pad_inches=0.05)

    def save_profile(direction):
        w_m = getattr(bm, f"w{direction}_m")
        s_m = getattr(bm, f"s{direction}_m")
        fig, axs = psol[direction].plot_wield(**ps_opts)
        axs[0].errorbar(bm.z_m, w_m, 2 * s_m, marker="X", ls="", capsize=6)
        fig.savefig(tpath_join(f"{direction}_profile.pdf"), **save_opts)

    save_profile("x")
    save_profile("y")


def T_fit_simple_HAM7(tpath_join, makegrid):
    print("")
    model = simple_HAM7_model(use_ZM_strain_gauge=True)
    qvip = BeamParam(w0=Variable("w0"), z=Variable("z"))
    p_to = "ZM6.p1.i"
    p_fr = "VIP.p1.o"
    symbolic = ("ZM5.Rcx", "ZM5.Rcy", "ZM4.Rcx", "ZM4.Rcy")
    psol = Munch(
        x=model.propagate_beam(p_fr, p_to, q_in=qvip, symbolic=symbolic, direction="x"),
        y=model.propagate_beam(p_fr, p_to, q_in=qvip, symbolic=symbolic, direction="y"),
    )
    p_ref = "ZM5.p2.o"
    q_measB = Munch(
        x=Munch({k: psol.x.q(p_ref).transform(bm.ABCD) for k, bm in bm7.items()}),
        y=Munch({k: psol.y.q(p_ref).transform(bm.ABCD) for k, bm in bm7.items()}),
    )

    def cost(x):
        # w0x, zx, w0y, zy, ZM4, ZM5
        subs_x = {"w0": x[0], "z": x[1], "ZM4_0": x[4], "ZM5_0": x[5]}
        subs_y = {"w0": x[2], "z": x[3], "ZM4_0": x[4], "ZM5_0": x[5]}
        chi2 = 0
        for key, bm in bm7.items():
            wx_prop = q_measB.x[key].w.eval(subs=bm.subs_strain_gauge(subs_x))
            wy_prop = q_measB.y[key].w.eval(subs=bm.subs_strain_gauge(subs_y))
            chi2 += (wx_prop - bm.wx_m)**2 / bm.sx_m**2
            chi2 += (wy_prop - bm.wy_m)**2 / bm.sy_m**2
        return chi2

    x0 = (550e-6, 250e-3, 550e-6, 250e-3, 1, -0.2)
    res = minimize(cost, x0=x0, method="L-BFGS-B")
    print(res)
    subs_x = {"w0": res.x[0], "z": res.x[1], "ZM4_0": res.x[4], "ZM5_0": res.x[5]}
    subs_y = {"w0": res.x[2], "z": res.x[3], "ZM4_0": res.x[4], "ZM5_0": res.x[5]}
    print(f"w0x: {subs_x['w0'] * 1e6} um, zx: {subs_x['z'] * 1e3} mm")
    print(f"w0y: {subs_y['w0'] * 1e6} um, zy: {subs_y['z'] * 1e3} mm")
    print(f"ZM4 offset: {subs_x['ZM4_0']}, ZM5 offset {subs_x['ZM5_0']}")

    nlocs = len(bm7.keys())
    w_fit = np.zeros(2 * nlocs)
    w_meas = np.zeros_like(w_fit)
    w_err = np.zeros_like(w_fit)
    labels = []
    for ki, key in enumerate(bm7.keys()):
        wx = q_measB.x[key].w.eval(subs=bm7[key].subs_strain_gauge(subs_x))
        wy = q_measB.y[key].w.eval(subs=bm7[key].subs_strain_gauge(subs_y))
        w_fit[ki] = wx
        w_fit[ki + nlocs] = wy
        w_meas[ki] = bm7[key].wx_m
        w_meas[ki + nlocs] = bm7[key].wy_m
        w_err[ki] = 2 * bm7[key].sx_m
        w_err[ki + nlocs] = 2 * bm7[key].sy_m
        labels.append(key)

    xx = np.arange(2 * nlocs)
    fig, ax = plt.subplots()
    ax.errorbar(
        xx, w_meas * 1e3, w_err * 1e3, capsize=7,
        marker="X", ls="", c="xkcd:cerulean", label="data",
    )
    ax.plot(xx, w_fit * 1e3, marker="P", ls="", c="xkcd:blood red", label="fit")
    ax.set_xticks(xx)
    ax.set_xticklabels(2 * labels)
    ax.axvline(nlocs - 0.5, ls=":", c="xkcd:slate")
    ax.legend()
    makegrid(ax)
    ax.set_ylabel("Radius [mm]")
    ax.set_title("Beam size")
    fig.savefig(tpath_join("joint_fit.pdf"))


def T_fit_HAM7(simple_sqz_factory, tpath_join, makegrid):
    print("")
    subs0 = dict(w0=650e-6, z=240e-3)
    factory = simple_sqz_factory
    factory.options.SQZ.use_ZM_strain_gauge = True
    model = factory.make()
    qvip = BeamParam(w0=Variable("w0"), z=Variable("z"))
    p_to = "OMC_IC.p3.o"
    p_fr = "SFI2.p3.o"
    symbolic = ("ZM5.Rcx", "ZM5.Rcy", "ZM4.Rcx", "ZM4.Rcy")
    psol = Munch(
        x=model.propagate_beam(p_fr, p_to, q_in=qvip, symbolic=symbolic, direction="x"),
        y=model.propagate_beam(p_fr, p_to, q_in=qvip, symbolic=symbolic, direction="y"),
    )

    direction = "x"
    fit_zm = True
    ps = psol[direction]
    qref = ps.q("ZM5.p2.o")
    q_measB = Munch({k: qref.transform(bm.ABCD) for k, bm in bm7.items()})
    w_measB = Munch({k: getattr(bm, f"w{direction}_m") for k, bm in bm7.items()})
    s_measB = Munch({k: getattr(bm, f"s{direction}_m") for k, bm in bm7.items()})

    def cost(x):
        subs = {"w0": x[0], "z": x[1]}
        if fit_zm:
            subs.update({"ZM4_0": x[2], "ZM5_0": x[3]})
        chi2 = 0
        for k, w_meas in w_measB.items():
            w_prop = q_measB[k].w.eval(subs=bm7[k].subs_strain_gauge(subs))
            chi2 += (w_prop - w_meas)**2 / s_measB[k]**2
        return chi2

    x0 = (550e-6, 200e-3)
    if fit_zm:
        x0 = x0 + (1, -0.2)
    res = minimize(cost, x0=x0, method="Powell")
    print(res)
    subs_sol = {"w0": res.x[0], "z": res.x[1]}
    if fit_zm:
        subs_sol.update({"ZM4_0": res.x[2], "ZM5_0": res.x[3]})
    print(f"w0: {subs_sol['w0'] * 1e6} um, z: {subs_sol['z'] * 1e3} mm")
    if fit_zm:
        print(f"ZM4 offset: {subs_sol['ZM4_0']}, ZM5 offset {subs_sol['ZM5_0']}")
    w_fit = np.zeros(len(q_measB))
    w_meas = np.zeros_like(w_fit)
    w_err = np.zeros_like(w_fit)
    labels = []
    for ki, key in enumerate(q_measB.keys()):
        w = q_measB[key].w.eval(subs=bm7[key].subs_strain_gauge(subs_sol))
        w_fit[ki] = w
        w_meas[ki] = w_measB[key]
        w_err[ki] = s_measB[key]
        labels.append(key)

    fig, ax = plt.subplots()
    xx = np.arange(len(w_meas))
    # ax.plot(xx, w_meas * 1e3, marker="X", ls="", c="xkcd:cerulean", label="data")
    ax.errorbar(
        xx, w_meas * 1e3, 2 * w_err * 1e3, capsize=7,
        marker="X", ls="", c="xkcd:cerulean", label="data",
    )
    ax.plot(xx, w_fit * 1e3, marker="P", ls="", c="xkcd:blood red", label="fit")
    ax.set_xticks(list(range(9)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Radius [mm]")
    ax.legend()
    ax.set_title("Beam size")
    makegrid(ax)
    fig.savefig(tpath_join("fit_x.pdf"))


def T_fit_HAM6(simple_sqz_factory, tpath_join, makegrid):
    print("")
    factory = simple_sqz_factory
    factory.options.SQZ.use_ZM_strain_gauge = True
    model = factory.make()
    qvip = BeamParam(w0=Variable("w0"), z=Variable("z"))
    p_to = "OM2.p1.i"
    p_fr = "SFI2.p3.o"

    symbolic = ("ZM5.Rcx", "ZM5.Rcy", "ZM4.Rcx", "ZM4.Rcy")
    exclude = ["A1", "C1"]
    psol = Munch(
        x=model.propagate_beam(p_fr, p_to, q_in=qvip, symbolic=symbolic, direction="x"),
        y=model.propagate_beam(p_fr, p_to, q_in=qvip, symbolic=symbolic, direction="y"),
    )
    p_ref = "OM1.p1.i"
    q_measB = Munch(
        x=Munch({
            k: psol.x.q(p_ref).transform(bm.ABCD)
            for k, bm in bm6.items() if k not in exclude
        }),
        y=Munch({
            k: psol.y.q(p_ref).transform(bm.ABCD)
            for k, bm in bm6.items() if k not in exclude
        }),
    )

    def cost(x):
        # w0x, zx, w0y, zy, ZM4, ZM5
        subs_x = {"w0": x[0], "z": x[1], "ZM4_0": x[4], "ZM5_0": x[5]}
        subs_y = {"w0": x[2], "z": x[3], "ZM4_0": x[4], "ZM5_0": x[5]}
        chi2 = 0
        for key, bm in bm6.items():
            if key in exclude:
                continue
            wx_prop = q_measB.x[key].w.eval(subs=bm.subs_strain_gauge(subs_x))
            wy_prop = q_measB.y[key].w.eval(subs=bm.subs_strain_gauge(subs_y))
            chi2 += (wx_prop - bm.wx_m)**2 / bm.sx_m**2
            chi2 += (wy_prop - bm.wy_m)**2 / bm.sy_m**2
        return chi2

    x0 = (650e-6, 250e-3, 650e-6, 250e-3, 1, -0.2)
    bounds = (
        (1e-6, 0.1),
        (1e-6, 0.1),
        (1e-6, 0.1),
        (1e-6, 0.1),
        (None, None),
        (None, None),
    )
    res = minimize(cost, x0=x0, method="trust-constr", bounds=bounds)
    print(res)
    subs_x = {"w0": res.x[0], "z": res.x[1], "ZM4_0": res.x[4], "ZM5_0": res.x[5]}
    subs_y = {"w0": res.x[2], "z": res.x[3], "ZM4_0": res.x[4], "ZM5_0": res.x[5]}
    print(f"w0x: {subs_x['w0'] * 1e6} um, zx: {subs_x['z'] * 1e3} mm")
    print(f"w0y: {subs_y['w0'] * 1e6} um, zy: {subs_y['z'] * 1e3} mm")
    print(f"ZM4 offset: {subs_x['ZM4_0']}, ZM5 offset {subs_x['ZM5_0']}")

    nlocs = len(q_measB.x.keys())
    w_fit = np.zeros(2 * nlocs)
    w_meas = np.zeros_like(w_fit)
    w_err = np.zeros_like(w_fit)
    labels = []
    for ki, key in enumerate(q_measB.x.keys()):
        wx = q_measB.x[key].w.eval(subs=bm6[key].subs_strain_gauge(subs_x))
        wy = q_measB.y[key].w.eval(subs=bm6[key].subs_strain_gauge(subs_y))
        w_fit[ki] = wx
        w_fit[ki + nlocs] = wy
        w_meas[ki] = bm6[key].wx_m
        w_meas[ki + nlocs] = bm6[key].wy_m
        w_err[ki] = 2 * bm6[key].sx_m
        w_err[ki + nlocs] = 2 * bm6[key].sy_m
        labels.append(key)

    xx = np.arange(2 * nlocs)
    fig, ax = plt.subplots()
    ax.errorbar(
        xx, w_meas * 1e3, w_err * 1e3, capsize=7,
        marker="X", ls="", c="xkcd:cerulean", label="data",
    )
    ax.plot(xx, w_fit * 1e3, marker="P", ls="", c="xkcd:blood red", label="fit")
    ax.set_xticks(xx)
    ax.set_xticklabels(2 * labels)
    ax.axvline(nlocs - 0.5, ls=":", c="xkcd:slate")
    ax.legend()
    makegrid(ax)
    ax.set_ylabel("Radius [mm]")
    ax.set_title("Beam size")
    fig.savefig(tpath_join("joint_fit.pdf"))


def T_fit_all(simple_sqz_factory, tpath_join, makegrid):
    print("")
    factory = simple_sqz_factory
    model = factory.make()
    qvip = BeamParam(w0=Variable("w0"), z=Variable("z"))
    p_to = "OM2.p1.i"
    p_fr = "SFI2.p3.o"
    symbolic = ("ZM5.Rcx", "ZM5.Rcy", "ZM4.Rcx", "ZM4.Rcy")
    psol = Munch(
        x=model.propagate_beam(p_fr, p_to, q_in=qvip, symbolic=symbolic, direction="x"),
        y=model.propagate_beam(p_fr, p_to, q_in=qvip, symbolic=symbolic, direction="y"),
    )

    p_ref7 = "ZM5.p2.o"
    p_ref6 = "OM1.p1.i"
    q_measB = Munch(
        x=Munch({k: psol.x.q(p_ref7).transform(bm.ABCD) for k, bm in bm7.items()}),
        y=Munch({k: psol.y.q(p_ref7).transform(bm.ABCD) for k, bm in bm7.items()}),
    )
    # Don't fit 6C1 so that we don't need to worry about OM1
    q_measB.x.update(Munch({
        k: psol.x.q(p_ref6).transform(bm.ABCD) for k, bm in bm6.items() if k != "C1"
    }))
    q_measB.y.update(Munch({
        k: psol.y.q(p_ref6).transform(bm.ABCD) for k, bm in bm6.items() if k != "C1"
    }))


def T_propagate_single(simple_sqz_factory, tpath_join):
    print("")
    factory = simple_sqz_factory
    model = factory.make()
    model.beam_trace()
    par_key = "CG"
    p = fpars[par_key]
    # p_to = "ZM6.p1.i"
    # p_to = "OMC_IC.p3.o"
    p_to = "OM2.p1.i"
    p_fr = "SFI2.p3.o"
    bm = bm6.A1
    qvip_x = BeamParam(w0=p.w0x, z=p.zx)
    qvip_y = BeamParam(w0=p.w0y, z=p.zy)
    model.ZM4_0 = p.Z4
    model.ZM5_0 = p.Z5

    bm.update_ROCs_strain_gauge(model)
    symbolic = False
    psol = Munch(
        x=model.propagate_beam(p_fr, p_to, q_in=qvip_x, symbolic=symbolic, direction="x"),
        y=model.propagate_beam(p_fr, p_to, q_in=qvip_y, symbolic=symbolic, direction="y"),
    )

    save_opts = dict(bbox_inches="tight", pad_inches=0.05)

    def save_profile(direction):
        w_m = getattr(bm, f"w{direction}_m")
        s_m = getattr(bm, f"s{direction}_m")
        # q_target = psol[direction].q(p_to)
        q_target = model.get(f"{p_to}.q{direction}")
        ps_opts = dict(
            radius_scale=1e3,
            radius_pref="m",
            q_target=q_target,
            target_kw=dict(name="OMC Target", c="xkcd:blood red")
        )
        fig, axs = psol[direction].plot_wield(**ps_opts)
        axs[0].errorbar(bm.z_m, w_m, 2 * s_m, marker="X", ls="", capsize=6)
        fig.savefig(tpath_join(f"{direction}_profile.pdf"), **save_opts)

    save_profile("x")
    save_profile("y")
