from pathlib import Path
from os.path import join
import sys

def is_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interprete
    
def get_LIGO_COMMISSIONING_DIR():
    return Path(
        join(
            __file__.split('ligo-commissioning-modeling')[0],
            'ligo-commissioning-modeling'
        )
    )

def add_power_up_scripts_to_path():
    LIGO_COMMISSIONING_DIR = get_LIGO_COMMISSIONING_DIR()
    sys.path.append(
        join(
            LIGO_COMMISSIONING_DIR,
            *tuple('analysis/O4/LLO/powerup/'.split('/'))
        )
    )
    return LIGO_COMMISSIONING_DIR