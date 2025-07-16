# STD Lib
from copy import deepcopy
from pathlib import Path
import importlib
import os

# SciPy
import numpy as np

# 3rd Party
from munch import Munch
from gpstime import gpsnow
import h5py

# Custom
from gwpy.timeseries import TimeSeries
from scipy.interpolate import interp1d

from follow_the_leader.io import (
    update_dict
)

class llo_data():

    def pure_chan(self, chanin):
        if chanin.endswith('trend'):
            return chanin.split('.')[0]
        else:
            return chanin

    def preprocessYAML(self, file_contents):
        """Convert the raw output from Munch.fromYAML into
        a dictionary. Without this function, one recieves a dictionary
        containing a list of dictionaries. 
        """
        rdict = {}
        for x in file_contents['Channels']:
            # Filter out anything that doesn't have an associated finesse name
            if hasattr(x, 'Finesse'):
                if not x['Finesse'] == '':
                    rdict[x['Name']] = x
        
        return rdict

    def update_mapping(self, chan_mapping=None):
        """ Update the channel mapping

        Parameters
        ----------
        cache : str | Path
            A h5 file that can be used to locally store parameters. 
        chan_mapping : MunchDict | str | None
            Munch dict of parameters, or a filename and path to a yaml.
            If None, it will use the same argument that was supplied to
            __init__ when this instance was created. 
        """
        if chan_mapping is None:
            chan_mapping = self.initial_chan_mapping_arg

        if isinstance(chan_mapping, Munch) or isinstance(chan_mapping, dict):
            update_dict(self.chan_mapping, chan_mapping)

        elif os.path.exists(chan_mapping):
            with open(chan_mapping) as f:
                update_dict(
                    self.chan_mapping, 
                    self.preprocessYAML(
                        Munch.fromYAML(
                            f.read()
                        )
                    )
                )

        elif importlib.resources.is_resource(
            "follow_the_leader.mapping_files", chan_mapping
        ):
            update_dict(
                self.chan_mapping,
                self.preprocessYAML(
                    Munch.fromYAML(
                        importlib.resources.read_text(
                            "follow_the_leader.mapping_files", chan_mapping
                        )
                    ),
                )
            )
        else:
            raise Exception(f"Could not handle parameters: {chan_mapping}")
        
        self.name = {
            ch: self.chan_mapping[ch]['Finesse'] for ch in self.chan_mapping.keys() if self.chan_mapping[ch]['Finesse'] != ''
        }
        self.chan = {
            self.chan_mapping[ch]['Finesse']: ch for ch in self.chan_mapping.keys() if self.chan_mapping[ch]['Finesse'] != ''
        }

        self.purechan = {
            self.chan_mapping[ch]['Finesse']: self.pure_chan(ch) for ch in self.chan_mapping if self.chan_mapping[ch]['Finesse'] != ''
        }

    def __init__(self, cache='parameter_cache.h5', 
                 chan_mapping='L1.yaml'):
        """Container for detector related data. Allows one to use GWpy, local caching and 
        NDscope outputs for the simulation to store and retrieve detector data easily. 

        Parameters
        ----------
        cache : str | Path
            A h5 file that can be used to locally store parameters. 
        chan_mapping : MunchDict | str | None
            Munch dict of parameters, or a filename and path to a yaml.
        """
        self.chan_mapping = Munch()
        self.initial_chan_mapping_arg = chan_mapping
        self.update_mapping(chan_mapping)
        
        self.cache_path = cache
        self.ndscope_outputs = Munch()
        
        try:
            with h5py.File(self.cache_path) as f:
                self.top_keys = [k for k in f.keys()]
                self.attrs = {k: v for k,v in f.attrs.items()}
                print('file found.')
                print('Top level keys:' + str(self.top_keys))
                print('Top level attrs:' + str(self.attrs))
        except FileNotFoundError:
            print('cache not present, creating it')
            with h5py.File(self.cache_path, 'w') as f:
                f.attrs['created'] = gpsnow()

    def attach_ndscope_outputs(self,ndscope_outputs):
        
        for file in ndscope_outputs:
            try:
                with h5py.File(file) as f:
                    self.ndscope_outputs[file] = Munch()
                    self.ndscope_outputs[file].top_keys = [k for k in f.keys()]
                    self.ndscope_outputs[file].attrs = {k: v for k,v in f.attrs.items()}
                    print(f'auxillary file {file}, found.')
                    print('Top level keys:' + str(self.ndscope_outputs[file].top_keys))
                    print('Top level attrs:' + str(self.ndscope_outputs[file].attrs))
            except FileNotFoundError:
                print(f'ndscope output file {file}, not found')
        
    def special_cases(self, data, name):
        ch = self.chan[name]
        
        if hasattr(self.chan_mapping[ch], 'Calibration'):
            _ctype = self.chan_mapping[ch].Calibration.get('type', 'factor')
            if _ctype.lower().strip() == 'factor':
                calib = self.chan_mapping[ch].Calibration.get('value', 1)
                data = float(calib)*data

        if hasattr(self.chan_mapping[ch], 'Limits'):
            high = float(self.chan_mapping[ch].Limits.get('high', np.inf))
            low = float(self.chan_mapping[ch].Limits.get('low', -np.inf))
            data = np.clip(data, low, high)
            
        return data
    
    
    def get_mean(self, gpstime, name, ischannel=False):
        gpstime_s = str(gpstime)
        gpstime_i = int(gpstime)
        
        if ischannel:
            name = self.name[chan]

        if name not in self.chan.keys():
            raise KeyError(name + ' not in '+str(self.chan.keys()))

        with h5py.File(self.cache_path, 'a') as f:
            if gpstime_s not in f.keys():
                f.create_group(gpstime_s)

            if name not in f[gpstime_s].attrs.keys():
                print('Getting data')
                data = TimeSeries.get(self.chan[name], gpstime_i, gpstime_i+1)
                data = self.special_cases(np.mean(data.value), name)

                f[gpstime_s].attrs.create(name, data)

            data = f[gpstime_s].attrs[name]
        
        return data

    def get_timeseries(self, gpstime, name, duration, ischannel=False, 
                       interpolate=False, force_update=False,
                       **kwargs):
        gpstime_s = str(gpstime)
        gpstime_i = int(gpstime)

        verbose=kwargs.get('verbose', False)
        
        if ischannel:
            name = self.name[chan]

        if name not in self.chan.keys():
            raise KeyError(name + ' not in '+str(self.chan.keys()))
        
        chan = self.chan[name]
        pchan = self.purechan[name]
        data = None
        
        for file in self.ndscope_outputs.keys():
            f_t0 = self.ndscope_outputs[file].attrs['t0']
            f_window = self.ndscope_outputs[file].attrs['window']
            
            if ((f_t0 + f_window[0]) <= gpstime) and f_window[1] >= duration:
                if verbose: print(f'Timespan matched to {file}')
                if pchan in self.ndscope_outputs[file].top_keys:
                    if verbose: print(f'Data & timespan matched to {file}')
                    with h5py.File(file, 'r') as f:
                        try:
                            data = f[pchan][:]
                        except TypeError:
                            if verbose: print(f'Looks like low-passed data, trying mean')
                            data = f[pchan]['mean']
                        
                        Lt = f_window[1] - f_window[0]
                        times = np.linspace(0, Lt, len(data))
                        data = data[:]
                        data = self.special_cases(data, name)
                        break
                else:
                    if verbose: print(f'channel {chan} not found in {file}')
                
    
        if data is None:
            with h5py.File(self.cache_path, 'a') as f:
                if gpstime_s not in f.keys():
                    f.create_group(gpstime_s)
    
                if name in f[gpstime_s].keys():
                    dstored = f[gpstime_s][name].attrs['duration']
                    if dstored >= duration:
                        update=False
                    else:
                        update=True
                        print(f'Requested duration {duration}s is less than stored duration {dstored}')
                else:
                    update=True
                    print("Data not found")
                
                if force_update or update:
                    print("getting data")
                    TS = TimeSeries.get(chan, gpstime_i, gpstime_i+duration, **kwargs)
    
                    f[gpstime_s].create_dataset(name, data=TS.value)
                    f[gpstime_s][name].attrs['duration'] = duration 
                    f[gpstime_s][name].attrs['dt'] = TS.dt
                    
                
                data = f[gpstime_s][name][:]
                data = self.special_cases(data, name)
                times = f[gpstime_s][name].attrs['dt'] * np.array(range(len(f[gpstime_s][name])))
            
        if interpolate:
            ret_value = interp1d(times,data)
        else:
            ret_value = data, times
                
        return ret_value

    def reset_cache(self):
        print('Resetting cache')
        Path.unlink(self.cache_path)
        self.__init__(cache=self.cache_path)

    def format_ndscope_yaml(self, start, duration):
        """Print out a yaml file to get required data from ndscope"""
        return f"""
color-mode: dark
font-size: 10
grid-alpha: 0.2
plots:
- channels: 
    {self.purechan['Pin']}:
      color: '#1f77b4'
      label: null
      offset: 0.0
      scale: 1.0
      unit: nulla
      width: 1
  col: 0
  colspan: 1
  row: 0
  rowspan: 1
  y-range: auto
- channels: 
    {self.purechan['Px']}:
      color: '#1f77b4'
      label: null
      offset: 0.0
      scale: 1.0
      unit: nulla
      width: 1
    {self.purechan['Py']}:
      color: '#ff7f0e'
      label: null
      offset: 0.0
      scale: 1.0
      unit: nulla
      width: 1
  col: 1
  colspan: 1
  row: 0
  rowspan: 1
  y-range: auto
- channels: 
    {self.purechan['PRG']}:
      color: '#1f77b4'
      label: null
      offset: 0.0
      scale: 1.0
      unit: nulla
      width: 1
  col: 0
  colspan: 1
  row: 1
  rowspan: 1
  y-range: auto
- channels: 
    {self.purechan['Guard']}:
      color: '#1f77b4'
      label: null
      offset: 0.0
      scale: 1.0
      unit: nulla
      width: 1
  col: 1
  colspan: 1
  row: 1
  rowspan: 1
  y-range: auto
t0: {start}
time-axis-mode: relative
time-window:
- 0
- {duration}
window-title: Follow The Leader
"""
