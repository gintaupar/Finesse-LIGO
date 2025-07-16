# %%
import numpy as np

import finesse.components as fc
from finesse import BeamParam

from sqz_factory import LIGOwSQZFactory, SQZ_params
# %%
def find_dep(node_name, tmodel):
    tf = tmodel.trace_forest
    node = tmodel.get(node_name)
    return tf.find_dependency_from_node(node).name
# %%
def do_trace(q_add_method, gauss_priority=1000):
    """Trace the beam from SFI2 to OMC and print the dependencies of a bunch of nodes

    Inputs
    ------
    q_add_method: method by which the beam parameters are set at SFI2
        None: the beam parameters are calculated using the default cavities
        "node": a BeamParam q is defined and set like model.node.q = [q, q]
        "Gauss": the parameter is set with a Gauss component
    gauss_priority: priority with which to set the Gauss component if that is the method being used
        Default 1000
    """
    qVIP = BeamParam(z=420e-3, zr=500e-3)
    factory = LIGOwSQZFactory("lho_O4.yaml", SQZ_params)
    factory.options.add_ifo = False
    model = factory.make()
    if q_add_method is None:
        pass
    elif q_add_method == "node":
        model.SFI2.p3.o.q = [qVIP, qVIP]
    elif q_add_method == "Gauss":
        model.add(fc.Gauss("gSFI2_p3_o", model.SFI2.p3.o, qx=qVIP, qy=qVIP, priority=gauss_priority))
    else:
        raise ValueError()
    bt = model.beam_trace()

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
        print(node, find_dep(node, model))
# %%
# this one switches intermediate nodes between cavFC and cavOMC
for _ in range(10):
    print("")
    do_trace(None)
# %%
# this one switches intermediate nodes between gSFI2_p3_o and cavOMC
for _ in range(10):
    print("")
    do_trace("node")
# %%
# this one switches intermediate nodes between gSFI2_p3_o and cavOMC
for _ in range(10):
    print("")
    do_trace("Gauss")
# %%
