Last Update: May 30, 2025
# These scripts investigate multiple effects on the SRCL detuning. 

functions.py contains several functions to be used in other scripts. 

MM_SRCL_det.py describes the effect of mode mismatch, misalignment, and SRM reflectivity on the detuning of SRCL (in FINESSE that's model.SRCL.DC). 


TMs_lenses_on_SRCL.py investigates the effects of differential and common changes to the ITMs/ETMs RoC changes and differential ITMlenses focal length changes on SRCL DC and the sensing function. 


spherical_aberrations_SRCL_det.py investigates the effects of HOM spherical aberrations on the detuning of SRCL DOF. This does not change the radius of curvatures of ITMs (or any other optic). It only looks at the effect of HOM spherical aberrations by applying an OPD map to the lens using Hello-Vinet equations modeling using FINESSE. https://kevin.kuns.docs.ligo.org/finesse3/physics/thermal_effects/hello_vinet.html?highlight=hv 


spherical_aberrations_SQZ_and_SRCL_det.py investigates the effect of spherical abberations on squeezing and squeezing angle. 

SRC_SRCL_SF.py investigates the effects of SRM misalignment and SR3 RoC changes on the SRCL detuning and on the sensing function. 

SRCL_DC_SF.py looks at the effect of changing SRCL locking point (SRCL.DC) on the sensing function without introducing any imperfections such as misalignment and mode mismatch


