import numpy as np

from finesse import BeamParam
from finesse.ligo.factory import ALIGOFactory


ptype = "pdf"


def find_dep(node, trace_forest):
    return trace_forest.find_dependency_from_node(node).name


def T_plot_IMC_to_ITM(tpath_join, ppath_join):
    print("")
    factory = ALIGOFactory(ppath_join("LLO", "llo_O4.yaml"))
    factory.options.INPUT.add_IMC_and_IM1 = True
    model = factory.make()
    p_to = model.ITMXAR.p1.i
    p_to_zoom = model.PR2.p1.i
    p_fr = model.MC3.p3.o
    # p_fr = model.L0.p1.o
    ps = model.propagate_beam(p_fr, p_to)
    ps_zoom = model.propagate_beam(p_fr, p_to_zoom)
    print(ps)
    print(ps_zoom)
    ps_opts = dict(
        ignore=["*CP*", "*lens"],
        radius_scale=1e3,
        radius_pref="m",
    )
    save_opts = dict(bbox_inches="tight", pad_inches=0.05)
    fig, ax = ps.plot_wield(**ps_opts)
    fig.savefig(tpath_join(f"MC3_to_ITMX.{ptype}"), **save_opts)
    fig, ax = ps_zoom.plot_wield(**ps_opts)
    fig.savefig(tpath_join(f"MC3_to_PR2.{ptype}"), **save_opts)


def T_plot_PRM_to_ITM(tpath_join, ppath_join):
    print("")
    factory = ALIGOFactory(ppath_join("LLO", "llo_O4.yaml"))
    factory.update_parameters(ppath_join("LLO", "llo_addRH.yaml"))
    print(ppath_join("LLO", "llo_O4.yaml"))
    factory.params.INPUT.LASER.power = 2
    factory.reset()
    factory.options.LSC.add_output_detectors = True
    factory.options.ASC.add = True
    factory.options.INPUT.add_IMC_and_IM1 = True
    model = factory.make()
    p_fr = model.PRMAR.p2.i
    # p_to = model.ITMXAR.p1.i
    p_to = model.ETMX.p1.o
    q_in = BeamParam(w0=0.84e-3, z=6.8)
    # ps = model.propagate_beam(p_fr, p_to, q_in=q_in)
    ps = model.propagate_beam(p_fr, p_to, direction="y")
    print(ps)
    ps_opts = dict(
        # ignore=["*CP*", "*lens"],
        radius_scale=1e3,
        radius_pref="m",
    )
    save_opts = dict(bbox_inches="tight", pad_inches=0.05)
    fig, ax = ps.plot_wield(**ps_opts)
    fig.savefig(tpath_join(f"PRM_to_ITMX.{ptype}"), **save_opts)


def T_print_beamsizes(ppath_join):
    print("")
    factory = ALIGOFactory(ppath_join("LLO", "llo_O4.yaml"))
    factory.update_parameters(ppath_join("LLO", "llo_addRH.yaml"))
    factory.options.INPUT.add_IMC_and_IM1 = False
    model = factory.make()
    model.beam_trace()
    tf = model.trace_forest
    p_fr = model.PRMAR.p2.i
    p_to = model.ETMX.p1.i
    ps_x = model.propagate_beam(p_fr, p_to, direction="x")
    ps_y = model.propagate_beam(p_fr, p_to, direction="y")

    print("Dependencies and beamsizes used in model (x direction)")
    nodes = [
        model.ETMX.p1.i,
        model.ITMX.p1.o,
        model.ETMY.p1.i,
        model.ITMY.p1.o,
        model.PRM.p1.o,
        model.L0.p1.o,
        p_fr,
    ]
    for node in nodes:
        print(node.component.name, find_dep(node, tf), f"{node.qx.w * 100:0.2f} cm")
    print("\nBeamsizes from PRMAR beamtrace (x direction)")
    nodes = [
        p_fr,
        model.ETMX.p1.i,
        model.ITMX.p1.o,
    ]
    for node in nodes:
        print(node.component.name, f"{ps_x.qs[node].w * 100:0.2f} cm")
