import os
from pathlib import Path
import finesse

from time import sleep
try:
    # Posix based file locking (Linux, Ubuntu, MacOS, etc.)
    #   Only allows locking on writable files, might cause
    #   strange results for reading.
    import fcntl, os
    def lock_file(f):
        if f.writable(): fcntl.lockf(f, fcntl.LOCK_EX)
    def unlock_file(f):
        if f.writable(): fcntl.lockf(f, fcntl.LOCK_UN)
except ModuleNotFoundError:
    # Windows file locking
    import msvcrt, os
    def file_size(f):
        return os.path.getsize( os.path.realpath(f.name) )
    def lock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_RLCK, file_size(f))
    def unlock_file(f):
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, file_size(f))

def isYes(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Please enter ")


def Finesse_Material_Representer(dumper, data):
    keys = [x for x in dir(finesse.materials.Material) if not x.startswith('_')]
    _dict = {}
    for key in keys:
        _dict[key] = getattr(data, key)
    
    return dumper.represent_mapping('!Finesse_Material', _dict)


def Finesse_Material_Constructor(self, node):
    keys = [x for x in dir(finesse.materials.Material) if not x.startswith('_')]
    _dict = {}
    for x in node.value:
        if x[0].value in keys:
            _dict[x[0].value] = float(x[1].value)
    
    return finesse.materials.Material(**_dict)

# https://stackoverflow.com/a/31174427/3157094
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def update_dict(original, p):
    """Updates a dictionary with some new keys and values. This happens recursively, so
    any dicts that are also present will also be updated.

    Parameters
    ----------
    original : dict
        Original to update
    p : dict
        Parameters to update `original` with
    """
    # Copied from Finesse-ligo
    for k, v in p.items():
        if isinstance(v, dict):
            if k in original:
                update_dict(original[k], v)
            else:
                original[k] = v
        else:
            original[k] = v



# Class for ensuring that all file operations are atomic, treat
# initialization like a standard call to 'open' that happens to be atomic.
# This file opener *must* be used in a "with" block.
class AtomicOpen:
    # Open the file with arguments provided by user. Then acquire
    # a lock on that file object (WARNING: Advisory locking).
    def __init__(self, path, *args, **kwargs):
        # Open the file and acquire a lock on the file before operating
        self.path = path
        self.lockpath = Path(str(self.path)+'.lock')
        
        self.file = open(path,*args, **kwargs)
        # Lock the opened file
        lock_file(self.file)

        # The lock proccess only prevents other proccesses writing
        # we also want robustness aginst other proccess reading
        # for this we implement a additional file lock
        i = 0
        try:
            while os.path.isfile(self.lockpath):
                sleep(0.01) # 10ms
                i += 1
                if i % 50==0:
                    print(f'\rWaiting for file lock {i/100:.1f}s',end='')
        except Exception:
            unlock_file(self.file)
        finally:
            if i > 49:
                print('')
            
        with open(self.lockpath, 'w'):
            pass
            
    # Return the opened file object (knowing a lock has been obtained).
    def __enter__(self, *args, **kwargs): return self.file

    # Unlock the file and close the file object.
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):        
        # Flush to make sure all buffered contents are written to file.
        self.file.flush()
        
        # We don't actually need fsync (write to OS memory buff's is okay)
        #os.fsync(self.file.fileno())

        # Release the lock on the file.
        unlock_file(self.file)
        
        # Notify read proccess that reading is okay
        os.remove(self.lockpath)
        
        self.file.close()
        # Handle exceptions that may have come up during execution, by
        # default any exceptions are raised to the user.
        if (exc_type != None): return False
        else:                  return True