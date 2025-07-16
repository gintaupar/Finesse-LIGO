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
import h5py
from finesse.utilities.maps import circular_aperture



def update_rom_operator(rom_dict, rom_file):
    hf = h5py.File(rom_file, 'r')
    # OPD group:
    opd_group = hf.get('opd')
    rom_dict['A_opd'] = opd_group.get('A')[:]
    rom_dict['B_opd'] = opd_group.get('B')[:]
    rom_dict['U_opd'] = opd_group.get('Ux')[:]
    # Deform group
    deform_group = hf.get('deform')
    rom_dict['A_deform'] = deform_group.get('A')[:]
    rom_dict['B_deform'] = deform_group.get('B')[:]
    rom_dict['U_deform'] = deform_group.get('Ux')[:]
    # Control group
    control_group = hf.get('control')
    rom_dict['U_I'] = control_group.get('Ux')[:]

rom_data_file= str(finesse.ligo.git_path() / "LLO"/ "thermal_data" / "dmdc_operators_aligo_v0.h5")
rom_operators = {}
update_rom_operator(rom_operators, rom_data_file)
rom_operators['r'] = np.linspace(0, 0.17, 100)

def make_ts_optics(dt = 20):
    ts_itmx = SimpleNamespace()
    ts_itmy = SimpleNamespace()
    ts_etmx = SimpleNamespace()
    ts_etmy = SimpleNamespace()
    for ts_optic in [ts_itmx, ts_itmy, ts_etmx, ts_etmy]:
        ts_optic.dt = dt
        ts_optic.t = 0
        ts_optic.opd = [np.zeros_like(rom_operators['U_opd'][:, 0])]
        ts_optic.x_opd = [np.zeros((rom_operators['A_opd'].shape[1]))]
        ts_optic.deform = [np.zeros_like(rom_operators['U_deform'][:, 0])]
        ts_optic.x_deform = [np.zeros((rom_operators['A_deform'].shape[1]))]
        ts_optic.I = []
        ts_optic.uI = []
    return ts_itmx, ts_itmy, ts_etmx, ts_etmy

def compute_new_opd_state(optics):
    """
    construct k+1 state of either OPD of eformation with
    state k & control k
    """
    # if dof == 'OPD':
    _A = rom_operators['A_opd']
    _B = rom_operators['B_opd']
    if optics.dt == 20:
        xk = optics.x_opd[-1]
        uk = optics.uI[-1]
        x_k1 = _A.dot(xk) + _B.dot(uk) 
    elif optics.dt > 20:
        if optics.dt % 20 != 0:
            optics.dt = ((optics.dt // 20) + 1) * 20
            print(f'Warning, time step set to the closest multiple of 20: {optics.dt} s')
        _nt = optics.dt // 20
        _xks = [optics.x_opd[-1]]
        uk = optics.uI[-1]
        for i in range(_nt):
            x_k1 = _A.dot(_xks[i]) + _B.dot(uk)
            _xks.append(x_k1)
        x_k1 = _xks[-1]
    optics.x_opd.append(x_k1.real)

def compute_new_deformation_state(optics):
    """
    construct k+1 state of either OPD of eformation with
    state k & control k
    """
    # if dof == 'OPD':
    _A = rom_operators['A_deform']
    _B = rom_operators['B_deform']
    if optics.dt == 20:
        xk = optics.x_deform[-1]
        uk = optics.uI[-1]
        x_k1 = _A.dot(xk) + _B.dot(uk) 
    elif optics.dt > 20:
        if optics.dt % 20 != 0:
            optics.dt = ((optics.dt // 20) + 1) * 20
            print(f'Warning, time step set to the closest multiple of 20: {optics.dt} s')
        _nt = optics.dt // 20
        _xks = [optics.x_deform[-1]]
        uk = optics.uI[-1]
        for i in range(_nt):
            x_k1 = _A.dot(_xks[i]) + _B.dot(uk)
            _xks.append(x_k1)
        x_k1 = _xks[-1]
    optics.x_deform.append(x_k1.real)

def get_opd(x, y, optics):
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx ** 2 + yy ** 2) 
    x_u = optics.x_opd[-1]
    u = rom_operators['U_opd']
    _opd = np.linalg.multi_dot([u,  x_u])
    optics.opd.append(_opd)

    # interp:
    opd2d = np.interp(rr, rom_operators['r'], _opd)
    return opd2d

def get_deformation(x, y, optics):
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx ** 2 + yy ** 2) 
    x_u = optics.x_deform[-1]
    u = rom_operators['U_deform']
    _deform = np.linalg.multi_dot([u,  x_u])
    optics.deform.append(_deform)

    # interp:
    deform2d = np.interp(rr, rom_operators['r'], _deform)
    return deform2d

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

def print_DC_status(model):
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

r_rom = np.linspace(0, 0.17, 100)
phi_rom = np.linspace(0, 2 * np.pi, 100)
_rr, _pp = np.meshgrid(r_rom, phi_rom)
_xx_rom = _rr * np.cos(_pp)
_yy_rom = _rr * np.sin(_pp)
_x_rom  = _xx_rom.flatten()
_y_rom  = _yy_rom.flatten()

def I_ITMX_HR(model, values, absorption = 0.5e-6):
    # HGs = HGModes(model.ITMX.p1.i.q, model.homs)
    # a = HGs.compute_points(_x_rom, _y_rom) * values.out["E_itmx1"][:, None]
    # E = np.sum(a, axis=0)
    # I = E * E.conj()
    # Iabs = (absorption * I.real).reshape(_rr.shape)
    _w = (model.ITMX.p1.o.qx.w + model.ITMX.p1.o.qy.w) / 2
    P = absorption * values.out['Px']
    Iabs = (2 * P) / np.pi / _w ** 2 * np.exp( -2 * r_rom ** 2 / _w ** 2)
    u_k =  np.linalg.multi_dot([rom_operators['U_I'].T, Iabs])
    return u_k, Iabs
    
def I_ETMX_HR(model, values, absorption = 0.3e-6):
    # HGs = HGModes(model.ETMX.p1.i.q, model.homs)
    # a = HGs.compute_points(_x_rom, _y_rom) * values.out["E_etmx"][:, None]
    # E = np.sum(a, axis=0)
    # I = E * E.conj()
    # Iabs = (absorption * I.real).reshape(_rr.shape)
    _w = (model.ETMX.p1.o.qx.w + model.ETMX.p1.o.qy.w) / 2
    P = absorption * values.out['Px']
    Iabs = (2 * P) / np.pi / _w ** 2 * np.exp( -2 * r_rom ** 2 / _w ** 2)
    u_k =  np.linalg.multi_dot([rom_operators['U_I'].T, Iabs])
    return u_k, Iabs

def I_ETMY_HR(model, values,  absorption = 0.3e-6):
    # HGs = HGModes(model.ETMY.p1.i.q, model.homs)
    # a = HGs.compute_points(_x_rom, _y_rom) * values.out["E_etmy"][:, None]
    # E = np.sum(a, axis=0)
    # I = E * E.conj()
    # Iabs = (absorption * I.real).reshape(_rr.shape)
    _w = (model.ETMY.p1.o.qx.w + model.ETMY.p1.o.qy.w) / 2
    P = absorption * values.out['Py']
    Iabs = (2 * P) / np.pi / _w ** 2 * np.exp( -2 * r_rom ** 2 / _w ** 2)
    u_k =  np.linalg.multi_dot([rom_operators['U_I'].T, Iabs])
    return u_k, Iabs

def I_ITMY_HR(model, values, absorption = 0.5e-6):
    # HGs = HGModes(model.ITMY.p1.i.q, model.homs)
    # a = HGs.compute_points(_x_rom, _y_rom) * values.out["E_itmy1"][:, None]
    # E = np.sum(a, axis=0)
    # I = E * E.conj()
    # Iabs = (absorption * I.real).reshape(_rr.shape)
    _w = (model.ITMY.p1.o.qx.w + model.ITMY.p1.o.qy.w) / 2
    P = absorption * values.out['Py']
    Iabs = (2 * P) / np.pi / _w ** 2 * np.exp( -2 * r_rom ** 2 / _w ** 2)
    u_k =  np.linalg.multi_dot([rom_operators['U_I'].T, Iabs])
    return u_k, Iabs

def get_mask(x, y, r_ap = 0.17 ):
    tm_ap = circular_aperture(x, y, r_ap)
    tm_ap[:, abs(y)> 0.163] = 0
    return tm_ap

def update_maps(initial_params, model, values, ts_itmx, ts_etmx, ts_itmy, ts_etmy):
    x = values.x
    y = values.y
    for ts, TM in zip([ts_itmx, ts_etmx, ts_itmy, ts_etmy], [model.ITMX, model.ETMX, model.ITMY, model.ETMY]):
        TM.surface_map = Map(
            x,
            y,
            amplitude = TM.aperture,#get_mask(x, y),
            opd = TM.static + get_deformation(x, y, ts),
        )
        a,_ = TM.surface_map.get_radius_of_curvature_reflection(TM.p1.i.qx.w)
        TM.Rc = 2/(2/(float(initial_params[TM.Rcx.full_name])) + 2/a)
        TM.surface_map.remove_curvatures(TM.p1.i.qx.w)
        TM.surface_map.remove_tilts(TM.p1.i.qx.w)
        print(f'{TM.name} Rc: {TM.Rc} dioptre: {2/TM.Rc/1e-6} [uD]')
    
    for ts, ITMlens in zip([ts_itmx, ts_itmy], [model.ITMXlens, model.ITMYlens]):
        ITMlens.OPD_map = Map(
            x, y, 
            amplitude = ITMlens.aperture, 
            opd = get_opd(x, y, ts)
        )
        a = ITMlens.OPD_map.get_thin_lens_f(ITMlens.p1.i.qx.w, average=True)
        ITMlens.f.value =  1/(1/float(initial_params[ITMlens.f.full_name]) +  1/a)
        ITMlens.OPD_map.remove_curvatures(ITMlens.p1.i.qx.w, mode="average")
        ITMlens.OPD_map.remove_tilts(ITMlens.p1.i.qx.w)
        print(f'{ITMlens.name} Rc: {ITMlens.f.value} dioptre: {1e6/ITMlens.f.value} [uD]')
    model.beam_trace()