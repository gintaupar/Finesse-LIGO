import finesse
import finesse.ligo
from finesse.ligo.factory import ALIGOFactory
from finesse.knm import Map
import finesse.materials
from tabulate import tabulate
from finesse.ligo.maps import get_test_mass_surface_profile_interpolated, aligo_O4_TM_aperture, aligo_O4_BS_to_ITMX_baffle, aligo_O4_BS_to_ITMY_baffle, aligo_O4_ESD_inner_aperture

def make_lho_quadratic_thermal_apertured(apertured=True):
    # We first make a factory object that can generate an ALIGO model
    # here we do so using the LHO O4 parameter file
    factory = ALIGOFactory(finesse.ligo.git_path() / "LHO" / "yaml" / "lho_O4.yaml")
    factory.update_parameters(finesse.ligo.git_path() / "LHO" / "yaml" / "lho_mcmc_RC_lengths.yaml")
    lho = factory.make()

    lho.parse("fd E_c0_as OM1.p1.i f=0")
    lho.parse("mathd Parm Px+Py")

    R = 0.16
    N = 201

    x, TM_aperture = aligo_O4_TM_aperture(R, N)
    x, X_aperture = aligo_O4_ESD_inner_aperture(R, N)
    x, Y_aperture = aligo_O4_ESD_inner_aperture(R, N)
    y = x

    if apertured:
        # Get surfaces
        ITMX_static = get_test_mass_surface_profile_interpolated(factory.params.X.ITM.ID, make_axisymmetric=True)(x, y)
        ETMX_static = get_test_mass_surface_profile_interpolated(factory.params.X.ETM.ID, make_axisymmetric=True)(x, y)
        ITMY_static = get_test_mass_surface_profile_interpolated(factory.params.Y.ITM.ID, make_axisymmetric=True)(x, y)
        ETMY_static = get_test_mass_surface_profile_interpolated(factory.params.Y.ETM.ID, make_axisymmetric=True)(x, y)

        # For test masses to always recompute, bit of a hack at the moment in FINESSE
        lho.ITMX.misaligned.is_tunable = True
        lho.ETMX.misaligned.is_tunable = True
        lho.ITMY.misaligned.is_tunable = True
        lho.ETMY.misaligned.is_tunable = True

        lho.ITMX.surface_map = Map(x, y, amplitude=TM_aperture, opd=ITMX_static)
        lho.ITMY.surface_map = Map(x, y, amplitude=TM_aperture, opd=ITMY_static)
        lho.ETMX.surface_map = Map(x, y, amplitude=TM_aperture, opd=ETMX_static)
        lho.ETMY.surface_map = Map(x, y, amplitude=TM_aperture, opd=ETMY_static)

        lho.ITMXlens.OPD_map = Map(x, y, amplitude=X_aperture)
        lho.ITMYlens.OPD_map = Map(x, y, amplitude=Y_aperture)

    ITM_sub = lho.add_parameter("ITM_sub", 300.39e-6).ref
    ITM_srf = lho.add_parameter("ITM_srf", -46.52e-6).ref
    ETM_sub = lho.add_parameter("ETM_sub", 209.57e-6).ref
    ETM_srf = lho.add_parameter("ETM_srf", -33.46e-6).ref
    IRH_sub = lho.add_parameter("IRH_sub", -9.92e-6).ref
    IRH_srf = lho.add_parameter("IRH_srf", 0.992e-6).ref
    ERH_sub = lho.add_parameter("ERH_sub", -12.53e-6).ref
    ERH_srf = lho.add_parameter("ERH_srf", 0.841e-6).ref

    PRHX = lho.add_parameter("PRHX", 0).ref
    PRHY = lho.add_parameter("PRHY", 0).ref
    PX = lho.add_parameter("PX", 0e3).ref
    PY = lho.add_parameter("PY", 0e3).ref
    alpha_ITMX = lho.add_parameter("alpha_ITMX", 0.5e-6).ref
    alpha_ITMY = lho.add_parameter("alpha_ITMY", 0.4e-6).ref
    alpha_ETMX = lho.add_parameter("alpha_ETMX", 0.2e-6).ref
    alpha_ETMY = lho.add_parameter("alpha_ETMY", 0.2e-6).ref

    # See https://gitlab.com/ifosim/finesse/finesse3/-/issues/573 for 0+
    lho.ITMXlens.f = 0+1/(1/lho.ITMXlens.f + PX * alpha_ITMX * ITM_sub + PRHX * IRH_sub)
    lho.ITMYlens.f = 0+1/(1/lho.ITMYlens.f + PY * alpha_ITMY * ITM_sub + PRHY * IRH_sub)

    lho.ITMX.Rc = 2/(2/lho.ITMX.Rc + PX * alpha_ITMX * ITM_srf + PRHX * IRH_srf)
    lho.ITMY.Rc = 2/(2/lho.ITMY.Rc + PY * alpha_ITMY * ITM_srf + PRHY * IRH_srf)
    lho.ETMX.Rc = 2/(2/lho.ETMX.Rc + PX * alpha_ETMX * ETM_srf)
    lho.ETMY.Rc = 2/(2/lho.ETMY.Rc + PY * alpha_ETMY * ETM_srf)

    # compute the round trip losses with the maps in and make sure overall loss
    # is reasonable
    lho.L0.P = 2
    lho.modes("even", maxtem=8)
    eigx = lho.run("eigenmodes(cavXARM, 0)")
    eigy = lho.run("eigenmodes(cavYARM, 0)")

    if apertured:
        lho.X_arm_loss = 50e-6
        lho.Y_arm_loss = 50e-6
    else:
        lho.X_arm_loss = 54e-6
        lho.Y_arm_loss = 54e-6

    loss_x = (lho.X_arm_loss + eigx.loss(True)[1][0])
    loss_y = (lho.Y_arm_loss + eigy.loss(True)[1][0])
    print("X arm loss: ", loss_x/1e-6, "ppm")
    print("Y arm loss: ", loss_y/1e-6, "ppm")
    # Apply corrections to get back to original losses
    print("Old X arm plane-wave loss: ", lho.X_arm_loss/1e-6, "ppm")
    print("Old Y arm plane-wave loss: ", lho.Y_arm_loss/1e-6, "ppm")
    lho.X_arm_loss -= eigx.loss(True)[1][0]
    lho.Y_arm_loss -= eigy.loss(True)[1][0]
    print("New X arm plane-wave loss: ", lho.X_arm_loss/1e-6, "ppm")
    print("New Y arm plane-wave loss: ", lho.Y_arm_loss/1e-6, "ppm")

    return lho

def print_DC_state(lho):

    DC = lho.run()

    data = [
        ("P_x", DC['Px']/1e3, 'kW'),
        ("P_y", DC['Py']/1e3, 'kW'),
        ("PRG", DC['PRG']),
        ("PRG9", DC['PRG9']),
        ("PRG45", DC['PRG45']),
        ("X arm gain", DC['AGX']),
        ("Y arm gain", DC['AGY']),
        ("P_IN", DC['Pin'], 'W'),
        ("P_REFL", DC['Prefl'], 'W'),
        ("P_REFL", DC['Prefl'], 'W'),
        ("P_PRC", DC['Pprc'], 'W'),
        ("P_DCPD", DC['Pas']/1e-3, 'mW')
    ]

    print(tabulate(data, headers=["Name", "Value", "Unit"]))