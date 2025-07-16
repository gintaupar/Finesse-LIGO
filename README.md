# LIGO Commissioning Modeling

## Scope

This is a support repository for the activities of the LIGO Commissioning-Modeling working group. Goal to talk about O4 commissioning, and maybe extend in to the design of O5. 

Communications happen throught the IFO modeling mailing list ifosim@ligo.org.

Minutes are uploaded to this repository and to the DCC: https://dcc.ligo.org/LIGO-T2300280

We plan to keep separate LLO and LHO models.

Questions to be asnwered will be managed as GIT issues


# Software instructions

## FINESSE 3

The latest FINESSE documentation can be found at https://finesse.ifosim.org/docs/latest/.

The easiest way to install FINESSE 3 is via conda-forge: https://finesse.ifosim.org/docs/latest/getting_started/install/first.html

FINESSE 3 combines Pykat and FINESSE 2 which used to be separately maintained. It adds numerous features that should make it easier to work with mode complicated models, like LIGO. We no longer support FINESSE 2 and Pykat fully due to time limitations.

There is also a separate package `finesse-ligo` which contains site files and various tools to work with LIGO data, such as maps, QUAD and other suspension models, ASC, etc. It can be installed via pypy `pit install finesse-ligo` after installing FINESSE 3.0 from conda-forge. If you want the latest version from source you can find it at:

https://git.ligo.org/finesse/finesse-ligo/

Or you can use the very latest changes which end up in (Dan Brown's) `ddb` branch before getting merged into the main branch:

https://git.ligo.org/finesse/finesse-ligo/-/tree/ddb/src/finesse_ligo


[NO] - install FEniCSX ( https://github.com/FEniCS/dolfinx#conda)

$ conda create -n fenicsx-env
$ conda activate fenicsx-env
$ conda install -c conda-forge fenics-dolfinx=0.6.0 mpich pyvista

- install test-mass-thermal-state (https://gitlab.com/ifosim/test-mass-thermal-state; pip install -e .)
- install gmsh (conda install -c conda-forge python-gmsh)
-  install scipy (conda install scipy)
- install finesse (conda install -c conda-forge finesse)
