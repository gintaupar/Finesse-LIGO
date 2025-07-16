"""
Compare methods of calculating PIs since it's not straightforward in finesse yet.

Use finesse to sum optical CLG's to get the PI gain and compare with wield's
direct calculation using the mechanical CLG.

Check both methods with finesse using HG00 and the center of mass "PI" where
optomechanics works.

Where would we be without FrequencyResponse3? Nowhere.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as scc
from copy import deepcopy

from munch import Munch
from finesse import Model
import finesse.components as fc
import finesse.analysis.actions as fa
import finesse.detectors as fd
from wield.control.plotting import plotTF
from wield.control.SFLU import SFLU
import wield.control.SFLU.optics as wield_optics

import wield_edges
import wield_elements
from wield_lib import MatrixLib

FP_PARAMS = Munch(
    Ti=0.014,
    Te=0,
    Larm_m=3995,
    Ri_m=1934,
    Re_m=2245,
    M_kg=0.40,
    Qm=1e2,
    Fm_Hz=1e3,
    Pin_W=1e6,
)


def finesse_FP(params):
    model = Model()
    model.add(fc.Mirror("ITM", T=params.Ti, L=0, Rc=params.Ri_m))
    model.add(fc.Mirror("ETM", T=params.Te, L=0, Rc=params.Re_m))
    model.connect(model.ITM.p1, model.ETM.p1, params.Larm_m)
    model.add(fc.Cavity("cavARM", model.ETM.p1.o))
    model.add(fc.Laser("Laser", P=params.Pin_W))
    model.connect(model.Laser.p1, model.ITM.p2)
    model.add(fc.Pendulum(
        "MechMode", model.ETM.mech, mass=params.M_kg, fz=params.Fm_Hz, Qz=params.Qm,
    ))
    model.add(fd.PowerDetector("PARM", model.ETM.p1.o))
    model.phase_level = 2
    return model


def sflu_FP(F_Hz, params, gouy_rad, overlap, mlib, gfile):
    with open(gfile, "r") as F:
        s = F.read()
    sflu = SFLU.SFLU.convert_yamlstr2self(s)
    sflu.reduce_auto()

    def suscept(_F_Hz):
        den = (
            params.Fm_Hz**2
            - _F_Hz**2
            + 1j * params.Fm_Hz * _F_Hz / params.Qm
        )
        return 1 / (params.M_kg * (2 * np.pi)**2 * den)

    edges = Munch()
    edges.ETM = wield_edges.RPMirrorEdge(
        name="ETM", Thr=params.Te, suscept=suscept, lambda_m=1064e-9,
        overlap=overlap, mlib=mlib,
    )
    edges.ITM = wield_edges.MirrorEdge(
        "ITM", Thr=params.Ti, lambda_m=1064e-9, mlib=mlib,
    )
    edges.ARM = wield_edges.LinkEdge(
        name="ARM", L_m=params.Larm_m, gouy_rad=gouy_rad, mlib=mlib,
    )

    edge_map = {
        "1": mlib.Id,
        "1s": mlib.Id_s,
    }

    # DC calculation
    edgesDC = deepcopy(edge_map)
    for edge in edges.values():
        edgesDC.update(edge.edgesDC())
    tp_dc = {"ETM.fr.o.tp", "ETM.fr.i.tp"}
    Ein_rtW = np.sqrt(params.Pin_W) * mlib.LO(np.pi / 2)
    compDC = sflu.computer(eye=mlib.Id)
    compDC.compute(edge_map=edgesDC)
    resultsDC = compDC.inverse_col(
        tp_dc,
        {
            "ITM.bk.i.exc": Ein_rtW,
        },
    )
    Parm_W = mlib.Vnorm_sq(resultsDC["ETM.fr.o.tp"])

    # AC calculation
    edgesAC = deepcopy(edge_map)
    for edge in edges.values():
        edgesAC.update(edge.edgesAC(F_Hz=F_Hz, resultsDC=resultsDC))
    compAC = sflu.computer(eye=mlib.Id)
    compAC.compute(edge_map=edgesAC)
    resultsAC = compAC.inverse_row(
        {"ETM.pos.tp": None}, {"ETM.pos.exc"},
    )

    return resultsAC["ETM.pos.exc"], Parm_W, suscept


def sflu_FP_graph():
    ifo = wield_optics.GraphElement()
    ifo.subgraph_add(
        "ITM", wield_optics.BasisMirror(),
        translation_xy=(25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "ETM", wield_elements.RPMirrorElement(),
        translation_xy=(55, 0),
        rotation_deg=0,
    )
    ifo["ETM"].locations.update({
        "fr.o.exc": (-3, -10),
    })
    ifo["ETM"].edges.update({
        ("fr.o", "fr.o.exc"): "1",
    })
    ifo["ITM"].locations.update({
        "bk.i.exc": (+8, -7),
    })
    ifo["ITM"].edges.update({
        ("bk.i", "bk.i.exc"): "1",
    })
    ifo.edges.update({
        ("ETM.fr.i", "ITM.fr.o"): "ARM",
        ("ITM.fr.i", "ETM.fr.o"): "ARM",
    })
    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


def sflu_DRFPMI_graph():
    ifo = wield_optics.GraphElement()
    ifo.subgraph_add(
        # "ITMX", wield_elements.RPMirrorElement(),
        "ITMX", wield_optics.BasisMirror(),
        translation_xy=(25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "ETMX", wield_elements.RPMirrorElement(),
        translation_xy=(55, 0),
        rotation_deg=0,
    )
    ifo.subgraph_add(
        # "ITMY", wield_elements.RPMirrorElement(),
        "ITMY", wield_optics.BasisMirror(),
        translation_xy=(0, 25),
        rotation_deg=90+180,
    )
    ifo.subgraph_add(
        # "ETMY", wield_elements.RPMirrorElement(),
        "ETMY", wield_optics.BasisMirror(),
        translation_xy=(0, 55),
        rotation_deg=90,
    )
    ifo.subgraph_add(
        "BS", wield_optics.BeamSplitter(),
        translation_xy=(0, 0),
        rotation_deg=0,
    )
    ifo.subgraph_add(
        "PRM", wield_optics.BasisMirror(),
        translation_xy=(-25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "SEM", wield_optics.BasisMirror(),
        translation_xy=(0, -25),
        rotation_deg=90+180,
    )

    ifo["PRM"].locations.update({
        "bk.i.exc": (15, -10),
        # "bk.o.tp": (15, 10),
    })
    ifo["PRM"].edges.update({
        ("bk.i", "bk.i.exc"): "1",
        # ("bk.o.tp", "bk.o"): "1",
    })

    # ifo["SEM"].locations.update({
    #     "bk.i.exc": (15, -10),
    #     "bk.o.tp": (15, 10),
    # })
    # ifo["SEM"].edges.update({
    #     ("bk.i", "bk.i.exc"): "SEC.to",
    #     ("bk.o.tp", "bk.o"): "SEC.fr",
    # })

    ifo.edges.update({
        ("EX.fr.i", "IX.fr.o"): "XARM.L",
        ("IX.fr.i", "EX.fr.o"): "XARM.L",
        ("IX.bk.i", "BS.bkA.o"): "BSX.L",
        ("BS.bkA.i", "IX.bk.o"): "BSX.L",

        ("EY.fr.i", "IY.fr.o"): "YARM.L",
        ("IY.fr.i", "EY.fr.o"): "YARM.L",
        ("IY.bk.i", "BS.frB.o"): "BSY.L",
        ("BS.frB.i", "IY.bk.o"): "BSY.L",

        ("PRM.fr.i", "BS.frA.o"): "PRC.L",
        ("BS.frA.i", "PRM.fr.o"): "PRC.L",

        ("SEM.fr.i", "BS.bkB.o"): "SEC.L",
        ("BS.bkB.i", "SEM.fr.o"): "SEC.L",
    })
    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


def T_compare_finesse_CM_PI(tpath_join, makegrid, pprint):
    """
    Make sure the CM "PI" is right with HG00 by comparing the gain
    calculated from the mechanical CLG with that calculated by summing
    the optical CLG's for HG00 where we can do optomechanics.
    """
    model = finesse_FP(FP_PARAMS)
    model.ETM.phi = 1
    model.modes("off")
    F_Hz = np.geomspace(10, 15e3, 300)
    fsig = model.fsig.f.ref

    actions = [
        fa.FrequencyResponse3(
            F_Hz,
            [
                (model.ETM.p1.o, +fsig),
                (model.ETM.p1.o, -fsig),
            ],
            [
                (model.ETM.p1.o, +fsig),
                (model.ETM.p1.o, -fsig),
            ],
            name=f"optical",
        ),
        fa.FrequencyResponse(
            F_Hz,
            [model.ETM.mech.z], [model.ETM.mech.z],
            name=f"mech",
        ),
        fa.Noxaxis(name="DC"),
    ]

    with model.temporary_parameters():
        model.Laser.P = 0
        sol_suscept = model.run(
            fa.FrequencyResponse(F_Hz, [model.ETM.mech.F_z], [model.ETM.mech.z])
        )
    sol_RP = model.run(fa.Series(*actions))
    with model.temporary_parameters():
        model.MechMode.mass = np.inf
        sol_inf = model.run(fa.Series(*actions))

    H_sb = sol_inf["optical"].out[..., 0, 0, 0, 0] - sol_inf["optical"].out[..., 1, 1, 0, 0]
    suscept = sol_suscept.out[..., 0, 0]
    par = FP_PARAMS
    suscept2 = 1 / par.M_kg / (2 * np.pi)**2
    suscept2 /= (par.Fm_Hz**2 - F_Hz**2 + 1j * F_Hz * par.Fm_Hz / par.Qm)
    Parm_W = sol_inf["DC"]["PARM"]
    print(Parm_W)
    Re = model.ETM.R.value
    Le = model.ETM.L.value
    B = 1
    CC = 4 * np.pi * (2 * Re + Le) / (model.lambda0 * scc.c) * Parm_W * B**2
    gain1 = CC * np.imag(suscept * H_sb)
    gain2 = CC * np.imag(suscept2 * H_sb)

    fig = plotTF(F_Hz, sol_RP["mech"].out[..., 0, 0])
    plotTF(F_Hz, sol_inf["mech"].out[..., 0, 0], *fig.axes, ls="--")
    fig.savefig(tpath_join("mechanical.pdf"))

    fig = plotTF(F_Hz, sol_RP["optical"].out[..., 0, 0, 0, 0])
    plotTF(F_Hz, sol_inf["optical"].out[..., 0, 0, 0, 0], *fig.axes, ls="--")
    fig.savefig(tpath_join("optical_gains.pdf"))

    clg = sol_RP["mech"].out[..., 0, 0]
    gain_from_clg = np.real(1 - 1 / clg)
    fig, ax = plt.subplots()
    ax.loglog(F_Hz, np.abs(gain_from_clg))
    ax.loglog(F_Hz, np.abs(gain1), ls="--")
    ax.loglog(F_Hz, np.abs(gain2), ls=":")
    makegrid(ax, F_Hz)
    fig.savefig(tpath_join("gain.pdf"))

    fig, ax = plt.subplots()
    ax.semilogx(F_Hz, np.sign(gain_from_clg))
    ax.semilogx(F_Hz, np.sign(gain1), ls="--")
    ax.semilogx(F_Hz, np.sign(gain2), ls=":")
    makegrid(ax, F_Hz)
    fig.savefig(tpath_join("sign.pdf"))

    assert np.allclose(gain_from_clg, gain1)
    assert np.allclose(gain_from_clg, gain2)

    # mass can't change annoyingly

    # sol = model.run(
    #     fa.Series(
    #         *actions("RP"),
    #         fa.Change({"MechMode.mass": 100}),
    #         *actions("inf"),
    #     )
    # )

    # fig = plotTF(F_Hz, sol["RP_mech"].out[..., 0, 0])
    # fig.savefig(tpath_join("mechanical.pdf"))

    # fig = plotTF(F_Hz, sol["RP_optical"].out[..., 0, 0, 0, 0])
    # plotTF(F_Hz, sol["inf_optical"].out[..., 0, 0, 0, 0], *fig.axes, ls="--")
    # fig.savefig(tpath_join("optical_gains.pdf"))


def T_compare_wield_finesse(tpath_join, fpath_join, makegrid, pprint):
    """
    Compare the direct mechanical CLG calculation from wield with summing
    the optical CLG's from finesse for some fake overlaps
    """
    F_Hz = np.geomspace(100, 100e3, 200)

    model = finesse_FP(FP_PARAMS)
    model.MechMode.mass = np.inf
    model.modes(maxtem=4)
    fsig = model.fsig.f.ref
    sol = model.run(fa.Series(
        fa.FrequencyResponse3(
            F_Hz,
            [
                (model.ETM.p1.o, +fsig),
                (model.ETM.p1.o, -fsig),
            ],
            [
                (model.ETM.p1.o, +fsig),
                (model.ETM.p1.o, -fsig),
            ],
            name="fresp",
        ),
        fa.Eigenmodes("cavARM", 0, name="modes"),
        fa.Noxaxis(name="DC"),
    ))

    # one-way Gouy phase
    gouy_rad = np.angle(sol["modes"].eigvalues[1:]) / 2
    print(gouy_rad)
    nmodes = model.Nhoms
    mlib = MatrixLib(nhom=nmodes - 1)
    # makes some random overlaps
    overlap = np.zeros((nmodes, nmodes))
    overlap[0, :] = np.arange(nmodes)
    overlap[:, 0] = np.arange(nmodes)
    print(overlap)
    clg, Parm_W, suscept = sflu_FP(
        F_Hz, FP_PARAMS, gouy_rad=gouy_rad, overlap=overlap, mlib=mlib,
        gfile=fpath_join("sflu_FabryPerot.yaml"),
    )
    print("arm power difference:", Parm_W - sol["DC", "PARM"])

    fig = plotTF(F_Hz, clg)
    fig.savefig(tpath_join("clg.pdf"))

    def transfer_overlap_prod(hom_idx):
        H_sb = (
            sol["fresp"].out[..., 0, 0, hom_idx, hom_idx]
            - sol["fresp"].out[..., 1, 1, hom_idx, hom_idx]
            )
        return H_sb * overlap[0, hom_idx]**2

    CC = 8 * np.pi / (model.lambda0 * scc.c) * Parm_W
    DD = np.array([transfer_overlap_prod(idx) for idx in range(1, nmodes)])
    gain_f = np.imag(CC * suscept(F_Hz) * np.sum(DD, axis=0))

    gain_w = np.real(1 - 1 / clg)
    fig, ax = plt.subplots()
    ax.loglog(F_Hz, np.abs(gain_w), label="wield")
    ax.loglog(F_Hz, np.abs(gain_f), ls="--", label="finesse")
    makegrid(ax, F_Hz)
    ax.set_title("Gain amplitude")
    ax.legend()
    fig.savefig(tpath_join("gain.pdf"))

    fig, ax = plt.subplots()
    ax.semilogx(F_Hz, np.sign(gain_w), label="wield")
    ax.semilogx(F_Hz, np.sign(gain_f), ls="--", label="finesse")
    ax.set_title("Gain sign")
    ax.legend()
    makegrid(ax, F_Hz)
    fig.savefig(tpath_join("sign.pdf"))

    assert np.allclose(gain_w, gain_f)


def T_build_sflu_cavity(tpath_join, fpath_join):
    from wield.control.SFLU import nx2tikz

    sflu = sflu_FP_graph()
    yamlstr = sflu.convert_self2yamlstr()
    with open(fpath_join("sflu_FabryPerot.yaml"), "w") as F:
        F.write(yamlstr)

    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-8, rX=+8, Y=3, dY=-3),
    sflu.reduce_auto()
    sflu.graph_reduce_auto_pos_io(lX=-8, rX=+8, Y=3, dY=-3),
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname=tpath_join("graph.pdf"),
        scale="10pt",
    )


def T_build_sflu_DRFPMI(tpath_join, fpath_join):
    from wield.control.SFLU import nx2tikz

    sflu = sflu_DRFPMI_graph()
    yamlstr = sflu.convert_self2yamlstr()
    with open(fpath_join("sflu_DRFPMI.yaml"), "w") as F:
        F.write(yamlstr)

    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-8, rX=+8, Y=3, dY=-3),
    sflu.reduce_auto()
    sflu.graph_reduce_auto_pos_io(lX=-8, rX=+8, Y=3, dY=-3),
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname=tpath_join("graph.pdf"),
        scale="10pt",
    )
