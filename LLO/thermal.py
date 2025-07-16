from ifo_thermal_state.aligo_3D import (
    make_test_mass_model,
    AdvancedLIGOTestMass3DTime,
)

from ifo_thermal_state.mesh import sweep_cylinder_3D
from ifo_thermal_state.postprocessing import get_deformation, get_opd, get_mask

import matplotlib.pyplot as plt
import numpy as np
import finesse
import finesse.ligo
from types import SimpleNamespace
from finesse.cymath.homs import HGModes
from finesse.knm import Map
from copy import deepcopy
from functools import partial
from tabulate import tabulate


def make_model(radius = 0.17, thickness = 0.2, flat_radius = 0.3265 / 2, heights=[0.95, 1], num_elements=[10, 3]):
    model = make_test_mass_model(
        mesh_function=sweep_cylinder_3D,
        mesh_function_kwargs={
            "radius": radius,
            "thickness": thickness,
            "num_elements": num_elements,
            "heights": heights,
            "add_flats": True,
            "flat_hw": flat_radius, 
            "HR_mesh_size": 0.02,
            "AR_mesh_size": 0.02,
            "mesh_algorithm": 6,
        },
    )
    model.flat_radius = flat_radius
    return model

def zero_initial_condition(x):
    return np.full((x.shape[1],), 0)

def make_solvers(model, dt=30):
    # Make the fea models
    ts_itmx = AdvancedLIGOTestMass3DTime(model, dt)
    ts_etmx = AdvancedLIGOTestMass3DTime(model, dt)
    ts_itmy = AdvancedLIGOTestMass3DTime(model, dt)
    ts_etmy = AdvancedLIGOTestMass3DTime(model, dt)

    ts_itmx.set_initial_condition(zero_initial_condition)
    ts_etmx.set_initial_condition(zero_initial_condition)
    ts_itmy.set_initial_condition(zero_initial_condition)
    ts_etmy.set_initial_condition(zero_initial_condition)

    return ts_itmx, ts_etmx, ts_itmy, ts_etmy

def add_thermal_detectors(model):
    model.parse("""
    fd E_itmx1 ITMX.p1.i f=0
    fd E_itmx2 ITMX.p1.o f=0
    fd E_itmx3 ITMX.p2.i f=0
    fd E_itmx4 ITMX.p2.o f=0
    fd E_etmx  ETMX.p1.i f=0
            
    fd E_itmy1 ITMY.p1.i f=0
    fd E_itmy2 ITMY.p1.o f=0
    fd E_itmy3 ITMY.p2.i f=0
    fd E_itmy4 ITMY.p2.o f=0
    fd E_etmy  ETMY.p1.i f=0
            
    mathd Parm (Px+Py)/2

    fd E_refl_c0  IFI.p4.o f=0
    fd E_refl_u9  IFI.p4.o f=+f1
    fd E_refl_l9  IFI.p4.o f=-f1
    fd E_refl_u45 IFI.p4.o f=+f2
    fd E_refl_l45 IFI.p4.o f=-f2
                        
    fd E_prc_c0  PRM.p1.o f=0
    fd E_prc_u9  PRM.p1.o f=+f1
    fd E_prc_l9  PRM.p1.o f=-f1
    fd E_prc_u45 PRM.p1.o f=+f2
    fd E_prc_l45 PRM.p1.o f=-f2
            
    fd E_src_c0  SRM.p1.o f=0
    fd E_src_u9  SRM.p1.o f=+f1
    fd E_src_l9  SRM.p1.o f=-f1
    fd E_src_u45 SRM.p1.o f=+f2
    fd E_src_l45 SRM.p1.o f=-f2
            
    fd E_x_c0  ETMX.p1.i f=0
    fd E_x_u9  ETMX.p1.i f=+f1
    fd E_x_l9  ETMX.p1.i f=-f1
    fd E_x_u45 ETMX.p1.i f=+f2
    fd E_x_l45 ETMX.p1.i f=-f2
            
    fd E_y_c0  ETMY.p1.i f=0
    fd E_y_u9  ETMY.p1.i f=+f1
    fd E_y_l9  ETMY.p1.i f=-f1
    fd E_y_u45 ETMY.p1.i f=+f2
    fd E_y_l45 ETMY.p1.i f=-f2
            
    fd E_inx_c0  ITMXlens.p1.i f=0
    fd E_inx_u9  ITMXlens.p1.i f=+f1
    fd E_inx_l9  ITMXlens.p1.i f=-f1
    fd E_inx_u45 ITMXlens.p1.i f=+f2
    fd E_inx_l45 ITMXlens.p1.i f=-f2
        
    fd E_iny_c0  ITMYlens.p1.i f=0
    fd E_iny_u9  ITMYlens.p1.i f=+f1
    fd E_iny_l9  ITMYlens.p1.i f=-f1
    fd E_iny_u45 ITMYlens.p1.i f=+f2
    fd E_iny_l45 ITMYlens.p1.i f=-f2

    fd E_c0_as OM1.p1.i f=0
    """)

def print_DC_statue(model):
    DC = model.run()
    data = [
        ("P_x", DC['Px']/1e3, 'kW'),
        ("P_y", DC['Py']/1e3, 'kW'),
        ("PRG", DC['PRG']),
        ("PRG9", DC['PRG9']),
        ("PRG45", DC['PRG45']),
        ("X arm gain", DC['AGX']),
        ("Y arm gain", DC['AGY']),
        ("P_REFL", DC['Prefl'], 'W'),
        ("P_REFL", DC['Prefl'], 'W'),
        ("P_PRC", DC['Pprc'], 'W'),
        ("P_DCPD", DC['Pas']/1e-3, 'mW')
    ]

    print(tabulate(data, headers=["Name", "Value", "Unit"]))

# Get intensity on HR surface:
def I_ITMX_HR(model, values, x):
    HGs = HGModes(model.ITMX.p1.i.q, model.homs)
    a = HGs.compute_points(x[0], x[1]) * values.out["E_itmx1"][:, None]
    E = np.sum(a, axis=0)
    I = E * E.conj()
    return I.real * 0.5e-6
    
def I_ETMX_HR(model, values, x):
    HGs = HGModes(model.ETMX.p1.i.q, model.homs)
    a = HGs.compute_points(x[0], x[1]) * values.out["E_etmx"][:, None]
    E = np.sum(a, axis=0)
    I = E * E.conj()
    return I.real * 0.5e-6 * 3 / 5

def I_ETMY_HR(model, values, x):
    HGs = HGModes(model.ETMY.p1.i.q, model.homs)
    a = HGs.compute_points(x[0], x[1]) * values.out["E_etmy"][:, None]
    E = np.sum(a, axis=0)
    I = E * E.conj()
    return I.real * 0.5e-6 * 3 / 5

def I_ITMY_HR(model, values, x):
    HGs = HGModes(model.ITMY.p1.i.q, model.homs)
    a = HGs.compute_points(x[0], x[1]) * values.out["E_itmy1"][:, None]
    E = np.sum(a, axis=0)
    I = E * E.conj()
    return I.real * 0.5e-6 


def update_maps(base, model, values, ts_itmx, ts_etmx, ts_itmy, ts_etmy):
    x = values.x
    y = values.y
    for ts, TM in zip([ts_itmx, ts_etmx, ts_itmy, ts_etmy], [model.ITMX, model.ETMX, model.ITMY, model.ETMY]):
        TM.surface_map = Map(
            x,
            y,
            amplitude=get_mask(x, y, ts),
            opd=TM.static + get_deformation(x, y, ts),
        )
        a,_ = TM.surface_map.get_radius_of_curvature_reflection(TM.p1.i.qx.w)
        TM.Rc = 2/(2/(base.get(TM.Rcx)) + 2/a)
        TM.surface_map.remove_curvatures(TM.p1.i.qx.w)
        TM.surface_map.remove_tilts(TM.p1.i.qx.w)
        print(f'{TM.name} Rc: {TM.Rc} dioptre: {2/a/1e-6} [uD]')
    

    model.ITMXlens.OPD_map = Map(
        x, y, amplitude=get_mask(x, y, ts_itmx), opd=get_opd(x, y, ts_itmx)
    )
    model.ITMXlens.f.value =  model.ITMXlens.OPD_map.get_thin_lens_f(
        model.ITMXlens.p1.i.qx.w, average=True
    )
    model.ITMXlens.OPD_map.remove_curvatures(model.ITMXlens.p1.i.qx.w, mode="average")
    model.ITMXlens.OPD_map.remove_tilts(model.ITMXlens.p1.i.qx.w)

    model.ITMYlens.OPD_map = Map(
        x, y, amplitude=get_mask(x, y, ts_itmy), opd=get_opd(x, y, ts_itmy)
    )
    model.ITMYlens.f.value =  model.ITMYlens.OPD_map.get_thin_lens_f(
        model.ITMYlens.p1.i.qx.w, average=True
    )
    model.ITMYlens.OPD_map.remove_curvatures(model.ITMYlens.p1.i.qx.w, mode="average")
    model.ITMYlens.OPD_map.remove_tilts(model.ITMYlens.p1.i.qx.w)
    
    model.beam_trace()