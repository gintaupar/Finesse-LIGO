import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from munch import Munch
from finesse import Model, BeamParam
from finesse import gaussian
import finesse.components as fc
import finesse.analysis.actions as fa

# from wield import model
import wield.model
from wield.model.system import algo_phys
from wield.LIGO.IFO.Apl import FC
from wield.LIGO.IFO import Apl

from sqz_factory import LIGOwSQZFactory, SQZ_params


def plot_finesse_beamprofile(ps, *args, **kwargs):
    fig, axs = ps.plot(*args, **kwargs)
    for ax in axs:
        ax.grid(True, which="major", alpha=0.5)
        ax.grid(True, which="minor", alpha=0.2)
    return fig


def T_build():
    factory = LIGOwSQZFactory("lho_O4.yaml", SQZ_params)
    factory.options.add_ifo = False
    factory.options.SQZ.add_seed_laser = True
    # factory = ALIGOFactory("lho_O4.yaml")
    model = factory.make()
    sol = model.run(fa.Noxaxis())
    print("")
    print(sol["P_OMC"])


def T_FCtoVIP(tpath_join, makegrid):
    factory = LIGOwSQZFactory("lho_O4.yaml", SQZ_params)
    factory.options.SQZ.add_seed_laser = True
    model = factory.make()
    # model.add(fc.Cavity("cavFC", model.FC2.p1.o))
    direction = "x"
    ps = model.propagate_beam(
        # model.FC1.p2.o,
        model.FC1AR.p2.o,
        model.SFI1.p3.i,
        direction=direction,
    )
    # ps = model.propagate_beam_astig(
    #     model.FC1.p2.o,
    #     model.SFI1.p3.i,
    # )
    print("")
    print(ps)
    fig, axs = ps.plot("all")
    for ax in axs:
        makegrid(ax)
    fig.set_size_inches((6, 7))
    fig.savefig(tpath_join("finesse.pdf"))

    fig, ax = ps.plot_wield(radius_scale=1e3, radius_pref="m")
    fig.savefig(tpath_join("finesse_wield.pdf"), bbox_inches="tight", pad_inches=0.05)

    sys = wield.model.system1064()
    sys["FDS/"] = FC.FDS(IFO="H1")
    pa = algo_phys.PhysicsAlgorithm(sys)
    overlapper = pa.mm.overlap(
        target_to=None,
        target_fr="FDS/FC/cav",
        waypoints=["FDS/+ZM1toVIP"],
        Wk=1064,
    )
    ret = overlapper.gouy_table(axis=direction)
    for q in ret.q_list:
        print(q.value)
    print(ret.Qs_str(side="input"))
    print(ret.Qs_str(side="output"))
    plB = overlapper.plot(tpath_join("wield.pdf"), use_in=False)


def T_SRMtoVIP(tpath_join, makegrid):
    factory = LIGOwSQZFactory("lho_O4.yaml", SQZ_params)
    factory.options.add_ifo = False
    factory.options.SQZ.add_seed_laser = True
    model = factory.make()
    direction = "x"
    qT1200410 = BeamParam(w0=6.06e-4, z=-3.55)
    ps = model.propagate_beam(
        model.SRMAR.p1.o,
        # model.SRMAR.p2.o,
        # model.OFI.p1.i,
        # model.OFI.p2.o,
        # model.ZM6.p1.o,
        model.ZM4.p2.i,
        direction=direction,
        q_in=qT1200410,
        reverse_propagate=True,
    )
    print("")
    print(ps)
    fig, axs = ps.plot("all")
    for ax in axs:
        makegrid(ax)
    fig.set_size_inches((6, 7))
    fig.savefig(tpath_join("finesse.pdf"))

    fig, ax = ps.plot_wield(radius_scale=1e3, radius_pref="m")
    fig.savefig(tpath_join("finesse_wield.pdf"), bbox_inches="tight", pad_inches=0.05)

    sys = wield.model.system1064()
    sys["LIGO/"] = Apl.aLIGOpl_SYS(IFO="H1")
    pa = algo_phys.PhysicsAlgorithm(sys)
    overlapper = pa.mm.overlap(
        target_to="LIGO/IFO/ASqT1200410",
        target_fr=None,
            waypoints=[
                "LIGO/FDS/+ZM4toVIP",
                "LIGO/IFO/SRM/+B1",
            ],
        Wk=1064,
    )
    ret = overlapper.gouy_table(axis=direction)
    # for q in ret.q_list:
    #     print(q.value)
    print(ret.Qs_str(side="input"))
    print(ret.Qs_str(side="output"))
    plB = overlapper.plot(tpath_join("wield.pdf"), use_in=False, reverse=True)


def T_scan(tpath_join, makegrid):
    factory = LIGOwSQZFactory("lho_O4.yaml", SQZ_params)
    factory.options.SQZ.add_seed_laser = True
    model = factory.make()
    model.modes(maxtem=4)
    sol = model.run(
        fa.Series(
            fa.Noxaxis(name="DC"),
            fa.Xaxis(model.OMC_CM1.phi, "lin", -90, 90, 2000, relative=True, name="scan"),
        )
    )
    print(sol["DC", "P_OMC"])
    print(model.trace_forest.draw())
    fig, ax = plt.subplots()
    x_nm = sol["scan"].x[0] * 1064 / 360
    ax.plot(x_nm, sol["scan", "P_OMC"])
    ax.set_yscale("log")
    makegrid(ax, x_nm)
    fig.savefig(tpath_join("scan.pdf"))


def T_q_params(tpath_join):
    factory = LIGOwSQZFactory("lho_O4.yaml", SQZ_params)
    factory.options.SQZ.add_seed_laser = True
    model = factory.make()
    p_fr = model.ZM4.p2.o
    p_to = model.OMC_IC.p1.i
    qx = BeamParam(w=625e-6, Rc=13.81)
    qy = BeamParam(w=685e-6, Rc=-2148)
    ps0_x = model.propagate_beam(p_fr, p_to, direction="x")
    ps0_y = model.propagate_beam(p_fr, p_to, direction="y")
    ps1_x = model.propagate_beam(p_fr, p_to, q_in=qx, direction="x")
    ps1_y = model.propagate_beam(p_fr, p_to, q_in=qy, direction="y")

    def save_bp(ps, direction, num):
        fig = plot_finesse_beamprofile(ps, "all")
        fig.axes[0].set_title(f"{direction} {num}")
        fig.savefig(tpath_join(f"bp{num}_{direction}.pdf"))

    save_bp(ps0_x, "x", 0)
    save_bp(ps0_y, "y", 0)
    save_bp(ps1_x, "x", 1)
    save_bp(ps1_y, "y", 1)


def T_OMCtoVIP(tpath_join):
    factory = LIGOwSQZFactory("lho_O4.yaml", SQZ_params)
    factory.options.add_ifo = False
    factory.options.SQZ.add_seed_laser = False
    model = factory.make()
    p_to = model.OMC_IC.p1.i
    p_fr = model.ZM4.p2.o
    # p_fr = model.SFI2.p3.o
    tf = model.trace_forest
    find_dep = lambda x: tf.find_dependency_from_node(x).name
    ps = model.propagate_beam(p_fr, p_to, reverse_propagate=False)
    print("")
    print("ZM5", find_dep(model.ZM5.p1.o))
    print("OFI p2.i", find_dep(model.OFI.p2.i))
    print("OFI p1.o", find_dep(model.OFI.p1.o))
    print("OFI p2.o", find_dep(model.OFI.p2.o))
    print("SRMAR p1.o", find_dep(model.SRMAR.p1.o))
    print("OFI p1.i", find_dep(model.OFI.p1.i))
    print("OFI p3.o", find_dep(model.OFI.p3.o))
    print("OM1 p1.i", find_dep(model.OM1.p1.i))
    print("OMC_IC p1.i", find_dep(model.OMC_IC.p1.i))

    # fig = plot_finesse_beamprofile(ps)
    # fig.savefig(tpath_join("profile.pdf"))

    fig, ax = ps.plot_wield(
        include="SRMAR",
        ignore="OFI",
        radius_scale=1e3,
        radius_pref="m",
    )
    fig.savefig(tpath_join("finesse_wield.pdf"), bbox_inches="tight", pad_inches=0.05)


def T_propagate_faraday():
    model = Model()
    model.add(fc.Mirror("OMC_IC", Rc=100))
    model.add(fc.Mirror("SRM", Rc=100))
    model.add(fc.Mirror("FC1", Rc=100))
    model.add(fc.DirectionalBeamsplitter("OFI"))
    model.add(fc.DirectionalBeamsplitter("SFI1"))
    model.add(fc.DirectionalBeamsplitter("SFI2"))
    model.add(fc.Beamsplitter("ZM1"))
    model.add(fc.Beamsplitter("ZM4"))
    model.add(fc.Beamsplitter("OM1"))

    # model.connect(model.SRM.p2, model.OFI.p1)
    # model.connect(model.OFI.p3, model.OMC_IC.p2)
    # model.connect(model.SFI1.p3, model.FC1.p2)
    # model.connect(model.SFI1.p4, model.SFI2.p1)
    # model.connect(model.SFI2.p3, model.OFI.p2)

    model.connect(model.SRM.p2, model.OFI.p1)
    model.connect(model.OFI.p3, model.OM1.p1)
    model.connect(model.OM1.p2, model.OMC_IC.p2)
    model.connect(model.SFI1.p3, model.ZM1.p1)
    model.connect(model.ZM1.p2, model.FC1.p2)
    model.connect(model.SFI1.p4, model.ZM4.p1)
    model.connect(model.ZM4.p2, model.SFI2.p1)
    model.connect(model.SFI2.p3, model.OFI.p2)

    model.add(fc.Mirror("ITM", Rc=100))
    model.add(fc.Mirror("OMC_OC", Rc=100))
    model.add(fc.Mirror("FC2", Rc=100))
    model.connect(model.ITM.p2, model.SRM.p1, 50)
    model.connect(model.OMC_IC.p1, model.OMC_OC.p1, 200)
    model.connect(model.FC1.p1, model.FC2.p1, 200)
    model.add(fc.Cavity("cavSRC", model.SRM.p1.o))
    model.add(fc.Cavity("cavOMC", model.OMC_IC.p1.o))
    model.add(fc.Cavity("cavFC", model.FC1.p1.o))
    model.beam_trace()

    SRMtoOMC = [model.SRM.p2.o, model.OMC_IC.p2.i]
    ps = model.propagate_beam(*SRMtoOMC)
    ps = model.propagate_beam(*SRMtoOMC[::-1], reverse_propagate=True)

    FCtoSRM = [model.FC1.p2.o, model.SRM.p2.i]
    ps = model.propagate_beam(*FCtoSRM)
    ps = model.propagate_beam(*FCtoSRM[::-1], reverse_propagate=True)

    FCtoOMC = [model.FC1.p2.o, model.OMC_IC.p2.i]
    ps = model.propagate_beam(*FCtoOMC)
    ps = model.propagate_beam(*FCtoOMC[::-1], reverse_propagate=True)

    # ZM4toSRM = [model.ZM1.p1.o, model.SRM.p2.i]  # this works for both
    ZM4toSRM = [model.ZM4.p2.o, model.SRM.p2.i]
    ps = model.propagate_beam(*ZM4toSRM)
    # ps = model.propagate_beam(*ZM4toSRM[::-1], reverse_propagate=True)
    ps = model.propagate_beam(model.SRM.p2.o, model.ZM4.p2.i, reverse_propagate=True)
