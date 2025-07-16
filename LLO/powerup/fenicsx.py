from thermal_state import ThermalState

# from ifo_thermal_state.aligo_3D import (
#     make_test_mass_model,
#     AdvancedLIGOTestMass3DTime,
# )
import ifo_thermal_state.aligo_3D as a3d
import ifo_thermal_state.aligo_2D as a2d

from ifo_thermal_state.mesh import sweep_cylinder_3D, cylinder_2d_axisymetric
from ifo_thermal_state.postprocessing import (get_deformation, get_opd, 
                                              get_mask, get_deformation_2D, 
                                              get_opd_2D)
from finesse.utilities.maps import (
    overlap_1D_curvature_coefficients,
    overlap_1D_piston_coefficient,
)

import numpy as np
from finesse.cymath.homs import HGModes
from functools import partial
import finesse.detectors as fd

def zero_initial_condition(x):
    return np.full((x.shape[1],), 0)


# I_CO2_annular = np.loadtxt("I_CO2_existing_axisymmetric_1W.txt")


def I_CO2X(model, I_CO2, x):
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    return float(model.P_ACO2X.value) * np.interp(
        r,
        I_CO2[:, 0],
        I_CO2[:, 1],
    )


def I_CO2Y(model, I_CO2, x):
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    return float(model.P_ACO2Y.value) * np.interp(
        r,
        I_CO2[:, 0],
        I_CO2[:, 1],
    )


# Get intensity on HR surface:
def I_ITMX_HR(model, values, dim, x):
    HGs = HGModes(model.ITMX.p1.i.q, model.homs)
    # Get dimension
    if dim == 3:
        a = HGs.compute_points(x[0], x[1]) * values.out["E_itmx1"][:, None]
    elif dim == 2:
        a = HGs.compute_points(x[0], np.zeros_like(x[0])) * values.out["E_itmx1"][:, None]
    E = np.sum(a, axis=0)
    intensity = (E * E.conj()).real
    return intensity * float(model.alpha_ITMX)


def I_ETMX_HR(model, values, dim, x):
    HGs = HGModes(model.ETMX.p1.i.q, model.homs)
    if dim == 3:
        a = HGs.compute_points(x[0], x[1]) * values.out["E_etmx"][:, None]
    elif dim == 2:
        a = HGs.compute_points(x[0], np.zeros_like(x[0])) * values.out["E_etmx"][:, None]
    E = np.sum(a, axis=0)
    intensity = (E * E.conj()).real
    return intensity * float(model.alpha_ETMX)


def I_ETMY_HR(model, values, dim, x):
    HGs = HGModes(model.ETMY.p1.i.q, model.homs)
    if dim == 3:
        a = HGs.compute_points(x[0], x[1]) * values.out["E_etmy"][:, None]
    elif dim == 2:
        a = HGs.compute_points(x[0], np.zeros_like(x[0])) * values.out["E_etmy"][:, None]
    E = np.sum(a, axis=0)
    intensity = (E * E.conj()).real
    return intensity * float(model.alpha_ETMY)


def I_ITMY_HR(model, values, dim, x):
    HGs = HGModes(model.ITMY.p1.i.q, model.homs)
    if dim == 3:
        a = HGs.compute_points(x[0], x[1]) * values.out["E_itmy1"][:, None]
    elif dim == 2:
        a = HGs.compute_points(x[0], np.zeros_like(x[0])) * values.out["E_itmy1"][:, None]
    E = np.sum(a, axis=0)
    intensity = (E * E.conj()).real
    return intensity * float(model.alpha_ITMY)


class FENICSXThermalState(ThermalState):
    def __init__(
        self,
        *args,
        model_type = "3D",  
        model_cp = False, 
        radius=0.17,
        thickness=0.2,
        flat_radius=0.3265 / 2,
        heights=(0.95, 1),
        num_elements=(10, 3),
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.model_cp = model_cp
        self.model_type = model_type
        if model_type == "3D":
            self.tm_fea_model = a3d.make_test_mass_model(
                mesh_function=sweep_cylinder_3D,
                mesh_function_kwargs={
                    "radius": radius,
                    "thickness": thickness,
                    "num_elements": num_elements,
                    "heights": heights,
                    "add_flats": False,
                    "flat_hw": flat_radius,
                    "HR_mesh_size": 0.02,
                    "AR_mesh_size": 0.02,
                    "mesh_algorithm": 6,
                },
            )
            self.fea_model.flat_radius = flat_radius
            
            # make fea model
            self.ts_itmx = a3d.AdvancedLIGOTestMass3DTime(self.tm_fea_model, 60)
            self.ts_etmx = a3d.AdvancedLIGOTestMass3DTime(self.tm_fea_model, 60)
            self.ts_itmy = a3d.AdvancedLIGOTestMass3DTime(self.tm_fea_model, 60)
            self.ts_etmy = a3d.AdvancedLIGOTestMass3DTime(self.tm_fea_model, 60)
            
            if model_cp:
                self.cp_fea_model = a3d.make_test_mass_model(
                    mesh_function=sweep_cylinder_3D,
                    mesh_function_kwargs={
                        "radius": radius,
                        "thickness": 0.1,
                        "num_elements": num_elements,
                        "heights": heights,
                        "add_flats": False,
                        "flat_hw": flat_radius,
                        "HR_mesh_size": 0.02,
                        "AR_mesh_size": 0.02,
                        "mesh_algorithm": 6,
                    },
                )
                # Build cp model
                self.cp_fea_model.eps_barrel = 0.03  # polished gold barrel ~0.02-0.04

                # Make the cp fea models
                self.ts_cpx = a3d.AdvancedLIGOTestMass3DTime(self.cp_fea_model, 60)
                self.ts_cpy = a3d.AdvancedLIGOTestMass3DTime(self.cp_fea_model, 60)

        
        # If user choose to make      
        elif model_type == "2D":
            self.tm_fea_model = a2d.make_test_mass_model(
                mesh_function = cylinder_2d_axisymetric,
                mesh_function_kwargs= {
                    "radius": radius,
                    "thickness": thickness,
                    "face_nodes": 50,
                    "barrel_nodes": 30,
                    "face_node_coeff": 1,
                    "barrel_node_coeff": 1.05,
                    }
                )
            
            # make fea model:
            self.ts_itmx = a2d.AdvancedLIGOTestMass2DTime(self.tm_fea_model, 60)
            self.ts_etmx = a2d.AdvancedLIGOTestMass2DTime(self.tm_fea_model, 60)
            self.ts_itmy = a2d.AdvancedLIGOTestMass2DTime(self.tm_fea_model, 60)
            self.ts_etmy = a2d.AdvancedLIGOTestMass2DTime(self.tm_fea_model, 60)
            
            if model_cp:
                self.cp_fea_model = a2d.make_test_mass_model(
                mesh_function = cylinder_2d_axisymetric,
                mesh_function_kwargs= {
                    "radius": radius,
                    "thickness": 0.1,
                    "face_nodes": 50,
                    "barrel_nodes": 20,
                    "face_node_coeff": 1,
                    "barrel_node_coeff": 1.05,
                    }
                )
            
            
        # Apply initial condition:
        self.ts_itmx.set_initial_condition(zero_initial_condition)
        self.ts_etmx.set_initial_condition(zero_initial_condition)
        self.ts_itmy.set_initial_condition(zero_initial_condition)
        self.ts_etmy.set_initial_condition(zero_initial_condition)
        if model_cp:
            self.ts_cpx.set_initial_condition(zero_initial_condition)
            self.ts_cpy.set_initial_condition(zero_initial_condition)
            # Build cp model
            self.cp_fea_model.eps_barrel = 0.03  # polished gold barrel ~0.02-0.04
            # Make the cp fea models
            self.ts_cpx = a2d.AdvancedLIGOTestMass2DTime(self.cp_fea_model, 60)
            self.ts_cpy = a2d.AdvancedLIGOTestMass2DTime(self.cp_fea_model, 60)


    @property
    def t(self):
        return self.ts_itmx.t

    @property
    def dt(self):
        return float(self.ts_etmx.dt.value)

    @dt.setter
    def dt(self, value):
        if value < 0:
            raise ValueError("dt > 0")

        if self.ts_etmx.dt.value != value:
            if self.model_cp:
                self.ts_cpx.dt.value = value
                self.ts_cpy.dt.value = value
            self.ts_itmx.dt.value = value
            self.ts_etmx.dt.value = value
            self.ts_itmy.dt.value = value
            self.ts_etmy.dt.value = value

    def reset(self):
        if self.model_cp:
            self.ts_cpx.t = 0
            self.ts_cpy.t = 0
            self.ts_cpx.set_initial_condition(zero_initial_condition)
            self.ts_cpy.set_initial_condition(zero_initial_condition)
        self.ts_itmx.t = 0
        self.ts_itmy.t = 0
        self.ts_etmx.t = 0
        self.ts_etmy.t = 0
        self.ts_itmx.set_initial_condition(zero_initial_condition)
        self.ts_etmx.set_initial_condition(zero_initial_condition)
        self.ts_itmy.set_initial_condition(zero_initial_condition)
        self.ts_etmy.set_initial_condition(zero_initial_condition)

    def set_intensities(self):
        # Use the model and values.out (the latest FINESSE output of detectors)
        # to compute intensities
        if self.model_type == "3D":
            dim = 3
        elif self.model_type == "2D":
            dim = 2
        if self.model_cp:
            self.ts_cpx.temperature.I_HR.interpolate(
                partial(I_CO2X, self.model, self.values, dim)
            )
            self.ts_cpy.temperature.I_HR.interpolate(
                partial(I_CO2Y, self.model, self.values, dim)
            )
        self.ts_itmx.temperature.I_HR.interpolate(
            partial(I_ITMX_HR, self.model, self.values, dim)
        )
        self.ts_etmx.temperature.I_HR.interpolate(
            partial(I_ETMX_HR, self.model, self.values, dim)
        )
        self.ts_itmy.temperature.I_HR.interpolate(
            partial(I_ITMY_HR, self.model, self.values, dim)
        )
        self.ts_etmy.temperature.I_HR.interpolate(
            partial(I_ETMY_HR, self.model, self.values, dim)
        )

    def step(self, evaluate_deformation=True):
        self.set_intensities()
        if self.model_cp:
            self.ts_cpx.step(evaluate_deformation=False)
            self.ts_cpy.step(evaluate_deformation=False)
        self.ts_itmx.step(evaluate_deformation=evaluate_deformation)
        self.ts_etmx.step(evaluate_deformation=evaluate_deformation)
        self.ts_itmy.step(evaluate_deformation=evaluate_deformation)
        self.ts_etmy.step(evaluate_deformation=evaluate_deformation)

    def get_opd(self, optic):
        # Evaluate 3D model
        if self.model_type == "3D":
            if optic == "ITMX":
                opd = get_opd(self.x, self.y, self.ts_itmx)
                if self.model_cp:
                    opd += get_opd(self.x, self.y, self.ts_cpx, h=0.05)
                return opd
            elif optic == "ITMY":
                opd = get_opd(self.x, self.y, self.ts_itmy)
                if self.model_cp:
                    opd += get_opd(self.x, self.y, self.ts_cpy, h=0.05)
                return opd
            elif optic == "ETMX":
                return get_opd(self.x, self.y, self.ts_etmx)
            elif optic == "ETMY":
                return get_opd(self.x, self.y, self.ts_etmy)
            
        # Evaluate 2D model
        elif self.model_type == "2D":
            if optic == "ITMX":
                opd = get_opd_2D(self.r, self.ts_itmx)
                if self.model_cp:
                    opd += get_opd_2D(self.r, self.ts_cpx, h=0.05)
                opd = np.hstack([np.flip(opd[1:]), opd])
                opd += self.external_OPDs["ITMXlens"]
                self.previous_OPDs["ITMXlens"].append(opd)
                a = overlap_1D_curvature_coefficients(self.x_r, 
                                                      opd, 
                                                      self.model.ITMXlens.p1.i.qx.w)
                self.model.ITMXlens.f = 1/(1/self.initial_parameters["ITMXlens.f"] 
                                        + (-2 * a))
                opd_new = opd - a * self.x_r ** 2
                
                c = overlap_1D_piston_coefficient(self.x_r, 
                                                  opd_new, 
                                                  self.model.ITMXlens.p1.i.qx.w)
                opd_new -= c
                opd_new = np.interp(self.R, self.x_r, opd_new)
                return opd_new
            elif optic == "ITMY":
                opd = get_opd_2D(self.r, self.ts_itmy)
                if self.model_cp:
                    opd += get_opd_2D(self.r, self.ts_cpy, h=0.05)
                opd = np.hstack([np.flip(opd[1:]), opd])
                opd += self.external_OPDs["ITMYlens"]
                self.previous_OPDs["ITMYlens"].append(opd)
                a = overlap_1D_curvature_coefficients(self.x_r, 
                                                      opd, 
                                                      self.model.ITMYlens.p1.i.qx.w)
                self.model.ITMYlens.f = 1/(1/self.initial_parameters["ITMYlens.f"] 
                                        + (-2 * a))
                opd_new = opd - a * self.x_r ** 2
                
                c = overlap_1D_piston_coefficient(self.x_r, 
                                                  opd_new, 
                                                  self.model.ITMYlens.p1.i.qx.w)
                opd_new -= c
                opd_new = np.interp(self.R, self.x_r, opd_new)
                return opd_new
            # elif optic == "ETMX":
            #     opd = get_opd_2D(self.r, self.ts_etmx)
            #     opd = np.interp(self.R, self.r, opd )
            #     return opd
            # elif optic == "ETMY":
            #     opd = get_opd_2D(self.r, self.ts_etmy)
            #     opd = np.interp(self.R, self.r, opd )
            #     return opd

    def get_deformation(self, optic):
        if self.model_type == "3D":
            if optic == "ITMX":
                return get_deformation(self.x, self.y, self.ts_itmx)
            elif optic == "ITMY":
                return get_deformation(self.x, self.y, self.ts_itmy)
            elif optic == "ETMX":
                return get_deformation(self.x, self.y, self.ts_etmx)
            elif optic == "ETMY":
                return get_deformation(self.x, self.y, self.ts_etmy)
        elif self.model_type == "2D":
            if optic == "ITMX":
                w = get_deformation_2D(self.r, self.ts_itmx)
                w = np.hstack([np.flip(w[1:]), w])
                w += self.external_OPDs["ITMX"]
                self.previous_OPDs["ITMX"].append(w)
                a = overlap_1D_curvature_coefficients(self.x_r, 
                                                      w, 
                                                      self.model.ITMX.p1.i.qx.w)
                self.model.ITMX.Rc = 2/(2/self.initial_parameters["ITMX.Rcx"] 
                                        + (2 * a))
                w_new = w - a * self.x_r ** 2
                
                c = overlap_1D_piston_coefficient(self.x_r, 
                                                  w_new, 
                                                  self.model.ITMX.p1.i.qx.w)
                w_new -= c
                w_new = np.interp(self.R, self.x_r, w_new)
                return w_new
            elif optic == "ITMY":
                w = get_deformation_2D(self.r, self.ts_itmy)
                w = np.hstack([np.flip(w[1:]), w])
                w += self.external_OPDs["ITMY"]
                self.previous_OPDs["ITMY"].append(w)
                a = overlap_1D_curvature_coefficients(self.x_r, 
                                                      w, 
                                                      self.model.ITMY.p1.i.qx.w)
                self.model.ITMY.Rc = 2/(2/self.initial_parameters["ITMY.Rcx"] 
                                        + (2 * a))
                w_new = w - a * self.x_r ** 2
                
                c = overlap_1D_piston_coefficient(self.x_r, 
                                                  w_new, 
                                                  self.model.ITMY.p1.i.qx.w)
                w_new -= c
                w_new = np.interp(self.R, self.x_r, w_new)
                return w_new
            elif optic == "ETMX":
                w = get_deformation_2D(self.r, self.ts_etmx)
                w = np.hstack([np.flip(w[1:]), w])
                w += self.external_OPDs["ETMX"]
                self.previous_OPDs["ETMX"].append(w)
                a = overlap_1D_curvature_coefficients(self.x_r, 
                                                      w, 
                                                      self.model.ETMX.p1.i.qx.w)
                self.model.ETMX.Rc = 2/(2/self.initial_parameters["ETMX.Rcx"] 
                                        + (2 * a))
                w_new = w - a * self.x_r ** 2
                
                c = overlap_1D_piston_coefficient(self.x_r, 
                                                  w_new, 
                                                  self.model.ETMX.p1.i.qx.w)
                w_new -= c
                w_new = np.interp(self.R, self.x_r, w_new)
                return w_new
            elif optic == "ETMY":
                w = get_deformation_2D(self.r, self.ts_etmy)
                w = np.hstack([np.flip(w[1:]), w])
                w += self.external_OPDs["ETMY"]
                self.previous_OPDs["ETMY"].append(w)
                a = overlap_1D_curvature_coefficients(self.x_r, 
                                                      w, 
                                                      self.model.ETMY.p1.i.qx.w)
                self.model.ETMY.Rc = 2/(2/self.initial_parameters["ETMY.Rcx"] 
                                        + (2 * a))
                w_new = w - a * self.x_r ** 2
                
                c = overlap_1D_piston_coefficient(self.x_r, 
                                                  w_new, 
                                                  self.model.ETMY.p1.i.qx.w)
                w_new -= c
                w_new = np.interp(self.R, self.x_r, w_new)
                return w_new

    def get_mask(self, optic):
        if optic == "ITMX":
            return get_mask(self.x, self.y, self.ts_itmx)
        elif optic == "ITMY":
            return get_mask(self.x, self.y, self.ts_itmy)
        elif optic == "ETMX":
            return get_mask(self.x, self.y, self.ts_etmx)
        elif optic == "ETMY":
            return get_mask(self.x, self.y, self.ts_etmy)

    def add_detectors(self):
        self.model.add(fd.FieldDetector("E_itmx1", self.model.ITMX.p1.i, f=0))
        self.model.add(fd.FieldDetector("E_etmx", self.model.ETMX.p1.i, f=0))
        self.model.add(fd.FieldDetector("E_itmy1", self.model.ITMY.p1.i, f=0))
        self.model.add(fd.FieldDetector("E_etmy", self.model.ETMY.p1.i, f=0))
