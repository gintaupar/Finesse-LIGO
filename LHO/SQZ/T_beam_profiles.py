import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from munch import Munch
import finesse.components as fc
import finesse.analysis.actions as fa
from finesse import BeamParam, gaussian

from beamfit import BeamParamFit
from sqz_factory import LIGOwSQZFactory, SQZ_params


def length(m1, m2):
    return SQZ_params.SQZ.get(f"length_{m1}_{m2}")


length_OFI_VIP = (
    length("VIP", "ZM4") + length("ZM4", "ZM5")
    + length("ZM5", "ZM6") + length("ZM6", "OFI")
)

length_SFI1_SFI2 = 0.73
length_SFI2_M4 = 0.3
z_ZM6_m = length_OFI_VIP - length("ZM6", "OFI")
z_ZM5_BD_m = 1184e-3  # distance between ZM5 and beam diverter
z_ZM4_m = length("VIP", "ZM4")
z_ZM5_m = z_ZM4_m + length("ZM4", "ZM5")


def olap_contour(w_m, s_D, olap, *, w0_m=None, s0_D=None, plot_cbar=False):
    fig, ax = plt.subplots()
    contourf = ax.contourf(w_m, s_D, 1 - olap, 200, cmap="viridis_r")
    contour = ax.contour(w_m, s_D, 1 - olap, colors="xkcd:white")
    if plot_cbar:
        cbar = fig.colorbar(
            contourf,
            ax=ax,
            format=FuncFormatter(lambda v, p: f"{v*100:0.0f}%"),
        )
    clabel = ax.clabel(
        contour,
        fmt=FuncFormatter(lambda v, p: f"{v*100:0.0f}"),
    )
    if w0_m is not None and s0_D is not None:
        ax.plot(w0_m, s0_D, marker="X", markersize=10, zorder=100)
    for cc in contourf.collections:
        cc.set_edgecolor("face")
    return fig, ax


def fit_61698():
    d_rail_m = np.array([0, 4, 8]) * 1e-2
    wx_m = np.array([1518, 1497, 1470]) * 1e-6 / 2
    wy_m = np.array([1636, 1618, 1578]) * 1e-6 / 2
    z_SFI1_m = 5.96  # distance from SFI1 to d_rail origin
    # 0.978 m = 38 7/8"
    d_M4_m = length("VIP", "ZM4") + length("ZM4", "ZM5") + z_ZM5_BD_m + 0.987
    print(d_M4_m)
    dist_m = d_rail_m + d_M4_m
    # dist_m = d_rail_m + d_ZM4_m - length("VIP", "ZM4")
    qx_fit = BeamParamFit(w_meas=wx_m, z_meas=dist_m, w0=500e-6, z0=-0.5)
    qy_fit = BeamParamFit(w_meas=wy_m, z_meas=dist_m, w0=500e-6, z0=-0.5)
    return Munch(qx=qx_fit, qy=qy_fit)


def fit_LLO_T2200151():
    z_meas = np.array([-400, -297, -137, 0, 80, 160]) / 1000
    wx_m = np.array([3510, 3670, 3710, 3970, 4090, 4130]) * 1e-6 / 2
    wy_m = np.array([3240, 3370, 3520, 3725, 3810, 3900]) * 1e-6 / 2
    qx_fit = BeamParamFit(w_meas=wx_m, z_meas=z_meas, w0=100e-5, z0=-1)
    qy_fit = BeamParamFit(w_meas=wy_m, z_meas=z_meas, w0=100e-5, z0=-1)
    return Munch(qx=qx_fit, qy=qy_fit)


def find_dep(node, tf):
    return tf.find_dependency_from_node(node).name


def T_fit_61698(tpath_join):
    print("")
    fit = fit_61698()
    print(fit.qx.optimization_result.success, fit.qy.optimization_result.success)
    print(fit.qx)
    print(fit.qy)
    qx_ZM5 = fit.qx(z_ZM5_m)
    qy_ZM5 = fit.qy(z_ZM5_m)
    print("mismatch", qx_ZM5.mismatch(qx_ZM5, qy_ZM5))
    d_BD_m = np.linspace(z_ZM5_m - 0.2, 6.5, 100)
    fig, ax = fit.qx.plot_fit(
        d_BD_m,
        fit_kw=dict(label="X Fit", c="xkcd:cerulean"),
        data_kw=dict(label="X Data", c="xkcd:tangerine"),
        yscale=1e3,
    )
    fit.qy.plot_fit(
        d_BD_m,
        fig=fig,
        ax=ax,
        fit_kw=dict(label="Y Fit", c="xkcd:red"),
        data_kw=dict(label="Y Data", c="xkcd:kelly green", marker="P"),
    )
    z_ZM4 = length("VIP", "ZM4")
    z_ZM5 = z_ZM4 + length("ZM4", "ZM5")
    z_ZM6 = z_ZM5 + length("ZM5", "ZM6")
    print("z_ZM5_m", z_ZM5)
    print("ZM5 X", fit.qx(z_ZM5))
    print("ZM5 Y", fit.qy(z_ZM5))
    print("ZM6 X", fit.qx(z_ZM6))
    print("ZM6 Y", fit.qy(z_ZM6))
    ax.axvline(z_ZM5, ls=":", c="xkcd:slate")
    ax.axvline(z_ZM6, ls=":", c="xkcd:slate")
    ax.set_xlabel("Distance from VIP (M4) [m]")
    ax.set_ylabel("Beam radius [mm]")
    ax.set_title("LHO aLOG 61698 Fit")
    fig.savefig(tpath_join("fit.pdf"))


def T_fit_LLO_T2200151(tpath_join):
    print("")
    fit = fit_LLO_T2200151()
    print(fit.qx.optimization_result.success, fit.qy.optimization_result.success)
    print(fit.qx)
    print(fit.qy)
    dZM5_m = np.linspace(-2, 0.5, 100)
    fig, ax = fit.qx.plot_fit(
        dZM5_m,
        fit_kw=dict(label="X Fit", c="xkcd:cerulean"),
        data_kw=dict(label="X Data", c="xkcd:tangerine"),
        yscale=1e6,
    )
    fit.qy.plot_fit(
        dZM5_m,
        fig=fig,
        ax=ax,
        fit_kw=dict(label="Y Fit", c="xkcd:red"),
        data_kw=dict(label="Y Data", c="xkcd:kelly green", marker="P"),
    )
    ax.set_xlabel("Distance from ZM5 [cm]")
    ax.set_ylabel("Beam radius [um]")
    ax.set_title("LLO T2200151 ZM5 Fit")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{1e2 * v:0.0f}"))
    fig.savefig(tpath_join("fit.pdf"))


def T_propagate_61698(tpath_join, ppath_join):
    print("")
    use_keita = False
    fit = fit_61698()
    # fit = fit_LLO_T2200151()
    print("z0", fit.qx.z0)
    print("w0", fit.qx.w0)
    z0_ref = fit.qx.z0
    qx_fit = fit.qx.q_at(z_ZM5_m)
    qy_fit = fit.qy.q_at(z_ZM5_m)
    print(qx_fit)
    print(qy_fit)
    # qx_fit = fit.qx.q_at(z_ZM6_m)
    # qx_fit = fit.qx.q_at(-length("ZM5", "ZM6")).reverse()  # for LLO, z0 is referenced to ZM5
    factory = LIGOwSQZFactory(ppath_join("LHO", "lho_O4.yaml"), SQZ_params)
    factory.update_parameters(ppath_join("LHO", "lho_mcmc_RC_lengths.yaml"))
    factory.options.add_ifo = False
    factory.options.SQZ.add_seed_laser = False
    factory.options.detailed_OFI = False
    if use_keita:
        factory.params.OUTPUT.length_OM3_OMC = 0.268
        factory.params.OUTPUT.length_OM2_OM3 = 0.708
        factory.params.OUTPUT.length_OM1_OM2 = 1.395
    model = factory.make()
    set_ROCs(model, 100, "cold")
    model.beam_trace()
    p_to = model.OMC_IC.p3.o
    # p_to = model.OFI.p2.i
    # p_to = model.ZM6.p1.i
    p_fr = model.ZM5.p2.o
    # p_fr = model.SFI2.p3.o
    # p_fr = model.ZM6.p2.o
    # ps1 = model.propagate_beam(p_fr, p_to)
    ps2 = model.propagate_beam(p_fr, p_to, q_in=qx_fit, direction="x")
    qf = ps2.q(p_to)
    q_target = p_to.qx
    print("mismatch", qf.mismatch(qf, q_target))

    ps_opts = dict(
        # include="SRMAR",
        # ignore="*OFI*",
        # ignore=["ZM6", "*OFI*"],
        ignore=["*OFI*"],
        radius_scale=1e3,
        radius_pref="m",
        q_target=p_to.qx,
        target_kw=dict(name="OMC Target", c="xkcd:blood red"),
    )
    save_opts = dict(bbox_inches="tight", pad_inches=0.05)
    # fig, ax = ps1.plot_wield(**ps_opts)
    # fig.savefig(tpath_join("default.pdf"), **save_opts)
    fig, ax = ps2.plot_wield(**ps_opts)
    # dist_VIP_m = fit.qx.z_meas
    dist_VIP_m = fit.qx.z_meas - z_ZM6_m
    ax[0].plot(dist_VIP_m, fit.qx.w_meas, "o")
    fig.savefig(tpath_join("profile.pdf"), **save_opts)


def T_fit_path_61698(tpath_join):
    print("")
    fit = fit_61698()
    print(fit.qx.z0)
    z0_ref = fit.qx.z0
    # fit.qx._z0 = 6
    qx_fit = fit.qx.q_at(0).reverse()
    qy_fit = fit.qy.q_at(0).reverse()
    factory = LIGOwSQZFactory("lho_O4.yaml", SQZ_params)
    factory.options.add_ifo = False
    factory.options.SQZ.add_seed_laser = False
    model = factory.make()
    set_ROCs(model, 100, "cold")
    # p_to = model.OMC_IC.p1.i
    p_to = model.ZM6.p1.i
    p_fr = model.SFI2.p3.o
    symbolic = False
    # symbolic = (model.ZM4.Rcx, model.ZM5.Rcx, model.VIP_ZM4.L, model.ZM4_ZM5.L)
    # subs = {model.ZM4.Rcx: -7.7, model.ZM5.Rcx: 5}
    # subs = {model.ZM4_ZM5.L: 10}
    subs = {}
    # path = model.path(p_fr, p_to, symbolic=True)
    # print(path.physical_length)
    qx_fit_target = fit.qx.q_at(z_ZM6_m)
    qy_fit_target = fit.qy.q_at(z_ZM6_m)
    ps = model.propagate_beam(p_fr, p_to, q_in=qx_fit, symbolic=symbolic, direction="x")
    # ps = model.propagate_beam(p_fr, p_to, symbolic=symbolic, direction="x")
    # fig, ax = ps.plot("all", subs=subs, show=False)
    fig, ax = ps.plot_wield(
        q_target=qx_fit_target,
        radius_scale=1e3,
        radius_pref="m",
    )
    dist_VIP_m = fit.qx.z_meas + fit.qx.z0 - z0_ref

    ax[0].plot(dist_VIP_m, fit.qx.w_meas, "o")
    fig.savefig(tpath_join("fit.pdf"), bbox_inches="tight")


def T_weird_nondeterministic_trace_forest(tpath_join):
    print("")
    fit = fit_61698()
    qx_VIP = fit.qx.q_at(length_OFI_VIP)
    qy_VIP = fit.qy.q_at(length_OFI_VIP)
    factory = LIGOwSQZFactory("lho_O4.yaml", SQZ_params)
    factory.options.add_ifo = False
    factory.options.SQZ.add_seed_laser = False
    model = factory.make()
    # cycling through not setting q at all, setting it at at the node, and setting it
    # with a Gauss, randomly lead to different trace dependencies.
    # model.SFI2.p3.o.q = [qx_VIP, qy_VIP]
    model.add(fc.Gauss("qSFI2_p3_o", model.SFI2.p3.o, qx=qx_VIP, qy=qy_VIP, priority=1000))
    model.beam_trace()
    tf1 = model.trace_forest
    print_dep = lambda node, tf_: print(node, find_dep(model.get(node), tf_))
    disp_nodes = [
        "SFI2.p3.o",
        "ZM4.p2.o",
        "ZM5.p2.o",
        "ZM6.p2.o",
        "OFI.p2.i",
        "SRMAR.p2.i",
        "SRMAR.p2.o",
        "OFI.p1.i",
        "OFI.p3.i",
        "OFI.p3.o",
        "OM1.p2.o",
        "OM2.p2.o",
        "OM3.p2.o",
        "OMC_IC.p1.o",
        "OMC_IC.p1.i",
        "OMC_IC.p3.o",
    ]
    for node in disp_nodes:
        print_dep(node, tf1)

    # Likewise, even saving and not saving here (while still propagating) changes
    # the dependencies above. It can be wield plots or native finesse.
    # Is this some set() thing and the plotting just a red herring?
    ps = model.propagate_beam(model.SFI2.p3.o, model.OMC_IC.p1.i)
    fig, ax = ps.plot_wield(ignore=["ZM6", "*OFI*"], radius_scale=1e3, radius_pref="m")
    # fig, ax = ps.plot()
    fig.savefig(tpath_join("beam_profile.pdf"), bbox_inches="tight", pad_inches=0.05)
    print("again")
    for node in disp_nodes:
        print_dep(node, tf1)


ROC_m = Munch(
    ZM45_100_100={
        # trends from our OMC scans
        # "ZM4": np.array([-7.7, -7.7]),
        # "ZM5": np.array([+2.07, +2.07]),
        # numbers from 60411
        "ZM4": np.array([-11.8, -11.8]),
        "ZM5": np.array([+3.3, +3.3]),
    },
    ZM45_200_200={
        "ZM4": np.array([-9.9, -9.9]),
        "ZM5": np.array([+1.94, +1.94]),
    },
    OM2_cold={"OM2": np.array([+2.05, +2.05])},
    OM2_hot={"OM2": np.array([+1.706, +1.706])},
)


def make_ROCs(ZM45, OM2):
    # assert ZM45 in ["ZM45_100_100", "ZM45_200_200"]
    # assert OM2 in ["OM2_cold", "OM2_hot"]
    ROCs = dict()
    keys = [f"ZM45_{ZM45}_{ZM45}", f"OM2_{OM2}"]
    for key in keys:
        for opt, Rc in ROC_m[key].items():
            ROCs[f"{opt}.Rcx"] = Rc[0]
            ROCs[f"{opt}.Rcy"] = Rc[1]
    return ROCs


def set_ROCs(model, ZM45, OM2):
    ROCs = make_ROCs(ZM45, OM2)
    for k, v in ROCs.items():
        model.set(k, v)


def T_OMC_scan(tpath_join, makegrid):
    print("")
    fit = fit_61698()
    qx_VIP = fit.qx.q_at(length_OFI_VIP)
    qy_VIP = fit.qy.q_at(length_OFI_VIP)
    factory = LIGOwSQZFactory("lho_O4.yaml", SQZ_params)
    factory.options.add_ifo = False
    factory.options.SQZ.add_seed_laser = True
    model = factory.make()
    # model.SFI2.p3.o.q = [qx_VIP, qy_VIP]
    ROCs = make_ROCs(100, "hot")
    for k, v in ROCs.items():
        model.set(k, v)
    model.add(fc.Gauss(
        "qSFI2_p3_o", model.SFI2.p3.o, qx=qx_VIP, qy=qy_VIP, priority=1000,
    ))
    model.beam_trace()
    tf = model.trace_forest
    print_dep = lambda node: print(node, find_dep(model.get(node), tf))
    disp_nodes = [
        "SFI2.p3.o",
        "ZM4.p2.o",
        "ZM5.p2.o",
        "ZM6.p2.o",
        "OFI.p2.i",
        "SRMAR.p2.i",
        "SRMAR.p2.o",
        "OFI.p1.i",
        "OFI.p3.i",
        "OFI.p3.o",
        "OM1.p2.o",
        "OM2.p2.o",
        "OM3.p2.o",
        "OMC_IC.p1.o",
        "OMC_IC.p1.i",
        "OMC_IC.p3.o",
    ]
    for node in disp_nodes:
        print_dep(node)

    ps = model.propagate_beam(model.SFI2.p3.o, model.OMC_IC.p1.i)
    fig, ax = ps.plot_wield(ignore=["ZM6", "*OFI*"], radius_scale=1e3, radius_pref="m")
    fig.savefig(tpath_join("beam_profile.pdf"), bbox_inches="tight", pad_inches=0.05)

    def scan_actions(ZM45, OM2):
        return [
            fa.Change(make_ROCs(ZM45, OM2)),
            fa.Noxaxis(name=f"DC_{ZM45}_{OM2}"),
            fa.Xaxis(
                "OMC_CM1.phi",
                "lin",
                -180,
                180,
                200,
                relative=True,
                name=f"scan_{ZM45}_{OM2}",
            ),
        ]

    model.modes(maxtem=6)
    print(model.ZM4.Rc)
    sol = model.run(fa.Series(
        *scan_actions(100, "hot"),
        *scan_actions(200, "hot"),
        *scan_actions(100, "cold"),
        *scan_actions(200, "cold"),
    ))

    print("100, 100, hot", sol["DC_100_hot", "P_OMC"])
    print("200, 200, hot", sol["DC_200_hot", "P_OMC"])
    print("100, 100, cold", sol["DC_100_cold", "P_OMC"])
    print("200, 200, cold", sol["DC_200_cold", "P_OMC"])

    def normalize_scan(ZM45, OM2):
        scan = sol[f"scan_{ZM45}_{OM2}", "P_OMC"]
        # return scan / np.max(scan)
        return scan

    x_nm = sol["scan_100_hot"].x[0] * 1064 / 360
    fig, ax = plt.subplots()
    # ax.plot(x_nm, sol["scan", "P_OMC"] / Pmax)
    ax.plot(x_nm, normalize_scan(100, "hot"), label="100, hot")
    ax.plot(x_nm, normalize_scan(200, "hot"), label="200, hot")
    ax.plot(x_nm, normalize_scan(100, "cold"), label="100, cold", ls="--")
    ax.plot(x_nm, normalize_scan(200, "cold"), label="200, cold", ls="--")
    makegrid(ax, x_nm)
    ax.set_yscale("log")
    ax.set_xlabel("OMC detuning [nm]")
    ax.set_ylabel("Power relative to maximum")
    ax.legend()
    fig.savefig(tpath_join("scan.pdf"))


def T_OMC_scan_from_ZM6(tpath_join):
    print("")
    fit = fit_61698()
    length_ZM6_VIP = length_OFI_VIP - length("ZM6", "OFI")
    qx_ZM6 = fit.qx.q_at(length_ZM6_VIP)
    qy_ZM6 = fit.qy.q_at(length_ZM6_VIP)
    factory = LIGOwSQZFactory("lho_O4.yaml", SQZ_params)
    factory.options.add_ifo = False
    model = factory.make()
    ROCs = make_ROCs(100, "hot")


def T_OMC_olap(tpath_join):
    print("")
    fit = fit_61698()
    qx_VIP = fit.qx.q_at(length_OFI_VIP)
    qy_VIP = fit.qy.q_at(length_OFI_VIP)
    factory = LIGOwSQZFactory("lho_O4.yaml", SQZ_params)
    factory.options.add_ifo = False
    model = factory.make()
    model.add(fc.Gauss(
        "qSFI2_p3_o", model.SFI2.p3.o, qx=qx_VIP, qy=qy_VIP, priority=1000,
    ))
    model.beam_trace()
    omc_port = model.OMC_IC.p3.o
    vip_port = model.SFI2.p3.o
    tf = model.trace_forest
    q_omc = omc_port.qx
    print(find_dep(omc_port, tf))

    w_m, s_D, olap = gaussian.ws_overlap_grid(
        q_omc, woffset=(0.5e-3, 1e-2), soffset=(1, 4),
    )
    fig, ax = olap_contour(w_m, s_D, olap, w0_m=q_omc.w, s0_D=q_omc.S)

    def calc_overlap(ZM45, OM2):
        # ROCs = make_ROCs(100, "hot")
        ROCs = make_ROCs(ZM45, OM2)
        for k, v in ROCs.items():
            model.set(k, v)
        model.beam_trace()
        ps = model.propagate_beam(vip_port, omc_port, direction="x", q_in=qx_VIP)
        tf = model.trace_forest
        q_vip = ps.q(omc_port)
        print(ZM45, OM2)
        print("OMC dependency", find_dep(omc_port, tf))
        print("OMC input dependency", find_dep(model.OMC_IC.p1.i, tf))
        print("OMC cavity mismatch", q_omc.mismatch(q_omc, omc_port.qx))
        print("Mode mismatch", q_omc.mismatch(q_omc, q_vip))
        return q_vip

    q100_hot = calc_overlap(100, "hot")
    q200_hot = calc_overlap(200, "hot")
    q100_cold = calc_overlap(100, "cold")
    q200_cold = calc_overlap(200, "cold")
    ax.plot(
        q100_hot.w, q100_hot.S, marker="P", markersize=10, c="xkcd:tangerine",
        label="100/100, hot",
    )
    ax.plot(
        q200_hot.w, q200_hot.S, marker="s", markersize=10, c="xkcd:sky blue",
        label="200/200, hot",
    )
    ax.plot(
        q100_hot.w, q100_cold.S, marker="d", markersize=10, c="xkcd:kelly green",
        label="100/100, cold",
    )
    ax.plot(
        q200_hot.w, q200_cold.S, marker="H", markersize=10, c="xkcd:pink",
        label="200/200, cold",
    )
    ax.legend()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v * 1e3:0.0f}"))
    ax.set_xlabel("Beam radius [mm]")
    ax.set_ylabel("Defocus [D]")
    ax.set_title("Mode at OMC")
    fig.savefig(tpath_join("contour.pdf"))

def T_OMC2_olap(tpath_join):
    factory = LIGOwSQZFactory("lho_O4.yaml", SQZ_params)
    factory.options.add_ifo = True
    factory.options.SQZ.add_seed_laser = True
    model = factory.make()
    model.beam_trace()
    qx = BeamParam(w=625e-6, Rc=13.81)
    q_omc = model.OMC_IC.p3.i.qx
    p_fr = model.ZM4.p2.o
    # p_fr = model.SRM.p1.i
    ps = model.propagate_beam(
        p_fr, model.OMC_IC.p3.o, direction="x", reverse_propagate=False,
    )
    q_fr = ps.q(p_fr)
    print(q_fr)
    print(1 - q_fr.overlap(q_fr, q_omc))
    tf = model.trace_forest
    # print(tf)
    # for (n1, n2) in tf.find_potential_mismatch_couplings():
    #     print(n1.full_name, n2.full_name)

    find_dep = lambda x: tf.find_dependency_from_node(x).name
    print("ZM4", find_dep(model.ZM4.p2.o))
    print("OMC_IC", find_dep(model.OMC_IC.p3.o))
    print("SRM", find_dep(model.SRM.p1.i))
    print("ZM3", find_dep(model.ZM3.p3.o))
    # w_m, s_D, olap = gaussian.ws_overlap_grid(q_omc, woffset=0.5e-3, soffset=1)
    w_m, s_D, olap = gaussian.ws_overlap_grid(q_omc, woffset=(0.5e-3, 0.8e-3), soffset=(1, 1))
    # w_m, s_D, olap = gaussian.ws_overlap_grid(q_omc, woffset=(0.5e-3, 1e-3), soffset=(3.5, 1))
    fig, ax = olap_contour(w_m, s_D, olap, w0_m=q_omc.w, s0_D=q_omc.S)
    ax.plot(q_fr.w, q_fr.S, marker="P", markersize="10", c="xkcd:tangerine")
    fig.savefig(tpath_join("contour.pdf"))
