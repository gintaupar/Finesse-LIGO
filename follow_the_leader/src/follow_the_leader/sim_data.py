# STD Lib
import csv
from collections import OrderedDict
from copy import deepcopy
from warnings import warn
from pathlib import Path

# SciPy
import numpy as np

# 3rd Party
from tqdm import tqdm
import yaml

# Custom
import finesse
from follow_the_leader.io import (
    Finesse_Material_Representer,
    Finesse_Material_Constructor,
    rgetattr,
    AtomicOpen,
    isYes
)


######################################################
# Class for outputting and reading simulation data
# in a thread safe way
######################################################
class SimData():
    
    default_output_model_pars = OrderedDict([
        ('fx', 'ITMXlens.f.value'), 
        ('fy', 'ITMYlens.f.value')
    ])
    
    def __init__(self, comparison_time, path='./',
                 delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, 
                 output_model_pars = 'default'):
        self.fout_data = Path(path) / Path('ftl_'+str(comparison_time)+'.csv')
        self.fout_opts = Path(path) / Path('ftl_'+str(comparison_time)+'_factory_opts.yaml')
        self.fout_pars = Path(path) / Path('ftl_'+str(comparison_time)+'_factory_pars.yaml')
        self.csv_opts = dict(delimiter=delimiter, quotechar=quotechar, quoting=quoting)


        if output_model_pars.lower() == 'default':
            self.output_model_pars = deepcopy(self.default_output_model_pars)
        else:
            self.output_model_pars = output_model_pars

        self.header_written = False
        self.output_names = None

        yaml.SafeDumper.add_representer(finesse.materials.Material, Finesse_Material_Representer)
        yaml.SafeLoader.add_constructor('!Finesse_Material', Finesse_Material_Constructor)
        warn('Modifying YAML SafeDumper to allow R/W finesse material objects')

    def purge(self):
        self.fout_data.unlink(missing_ok=True)
        self.fout_opts.unlink(missing_ok=True)
        self.fout_pars.unlink(missing_ok=True)

    def get_model_params(self,model):
        _dict = OrderedDict()
        for key, value in self.output_model_pars.items():
            _dict[key] = rgetattr(model, value)
        return _dict
        
    
    def write_row(self, time, model, out):
        if not self.header_written:
            raise IOError('Header must be written first `Sim_Data.write_header()`')

        mdl_values = list(self.get_model_params(model).values())
        out_values = [out[variable] for variable in self.output_names]
        row = [time] + out_values + mdl_values
        row = [str(item) for item in row]
        
        with AtomicOpen(self.fout_data, 'a', newline='') as f:
            writer = csv.writer(f, **self.csv_opts)
            writer.writerow(row)

    
    def write_header(self, model, first_out, factory, 
                     output_matrix=False, exist_ok=False):

        if not exist_ok and any([
                self.fout_data.is_file(), 
                self.fout_opts.is_file(), 
                self.fout_pars.is_file()
        ]):
            if not isYes('Previous simulation data exists. Overwrite?'):
                raise ValueError('Cannot overwrite previous data!')

        self.purge()

        with AtomicOpen(self.fout_pars, 'w') as f:
            yaml.safe_dump(factory.params, stream=f)
        
        with AtomicOpen(self.fout_opts, 'w') as f:
            yaml.safe_dump(factory.options, stream=f)

        self.output_names = list(first_out.outputs)
        if not output_matrix:
            self.output_names = [x for x in self.output_names if not x.startswith('E')]
        
        keys = list(self.get_model_params(model).keys())
        header = ['Time'] + self.output_names + keys
        header = [str(item) for item in header]
        
        with AtomicOpen(self.fout_data, 'w', newline='') as f:
            writer = csv.writer(f, **self.csv_opts)
            writer.writerow(header)
        
        self.header_written = True

    def load_sim_data(self, verbose=True):
        failed = []
        
        with AtomicOpen(self.fout_data, newline='') as f:
            reader = csv.reader(f, **self.csv_opts)
            data = OrderedDict()
        
            # Convert header into dictionary
            for column in next(reader):
                data[column] = []
            
            for row in tqdm(reader, disable=not verbose):
                for column, entry in zip(data.keys(), row):
                    try:
                        data[column].append(eval(entry))
                    except SyntaxError:
                        data[column].append('F')
                        if column not in failed:
                            failed.append(column)
                            if verbose: print(f'Failed to load {column}. SyntaxError first occured for this column with this data: {entry}.')
                    except NameError:
                        if entry == 'inf':
                            data[column].append(np.inf)
                        if entry == '-inf':
                            data[column].append(-np.inf)
                        else:
                            data[column].append('F')
                            if column not in failed:
                                failed.append(column)
                                if verbose: print(f'Failed to load {column}. NameError first occured for this column with this data: {entry}.')
                            
        return data, failed







