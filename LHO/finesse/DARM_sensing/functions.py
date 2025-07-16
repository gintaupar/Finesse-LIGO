import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import finesse.detectors as det
from finesse.utilities import maps
import finesse.thermal.hello_vinet as hv
from finesse.materials import FusedSilica
from finesse.knm import Map

def create_data_table(labels, data, name, save_md=True, save_png=True, save_dir='./data/'):
    # Extract the relevant data
    table_data = []
    for idx, label in enumerate(labels['graph_labels']):
        row = {
            'Label': label,
            'XARM YARM MM (%)': f"{(data['XYARM'][idx] * 100):.2f}",
            'SRCL DC (deg)': f"{(data['SRCL_DC'][idx]):.2f}",
            'SRX XARM MM (%)': f"{(data['SRXXARM'][idx] * 100):.2f}",
            'SRY YARM MM (%)': f"{(data['SRYYARM'][idx] * 100):.2f}",
            'ITMX w (cm)': f"{(data['ITMX_w'][idx]/1e-2):.2f}",
            'ITMY w (cm)': f"{(data['ITMY_w'][idx]/1e-2):.2f}",
            'ETMX w (cm)': f"{(data['ETMX_w'][idx]/1e-2):.2f}",
            'ETMY w (cm)': f"{(data['ETMY_w'][idx]/1e-2):.2f}"
        }
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Generate markdown table
    markdown_table = df.to_markdown(index=False, floatfmt='.2f')
    
    if save_md:
        # Save markdown table to file
        save_path = os.path.join(save_dir, f'{name}.md')
        with open(save_path, 'w') as f:
            f.write(markdown_table)
    
    if save_png:
        # Create a visual table using matplotlib
        fig, ax = plt.subplots(figsize=(16, len(table_data)*0.5 + 1))  # Made wider for more columns
        ax.axis('tight')
        ax.axis('off')
        
        # Calculate column widths based on content
        n_cols = len(df.columns)
        col_widths = [0.25 if i == 0 else 0.75/(n_cols-1) for i in range(n_cols)]
        
        # Create table
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=col_widths)
        
        # Adjust font size
        table.auto_set_font_size(False)
        table.set_fontsize(8)  # Slightly smaller font for more columns
        
        # Save the figure
        save_path = os.path.join(save_dir, f'{name}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    return df  # Return the DataFrame for further use if needed

# Example usage:
# create_data_table(labels, data)

def find_dep(node_name, tmodel):
    tf = tmodel1.trace_forest
    node = tmodel1.get(node_name)
    return tf.find_dependency_from_node(node).name


def mismatch_calculator(q1,q2,percentage=True):
    MM=((np.abs(q1-q2))**2)/((np.abs(q1-np.conjugate(q2)))**2)
    if percentage==True:
        return 100*MM
    else:
        return MM


def get_cav_mismatches(model, print_tables=True):
    mismatch_x, mismatch_y = model.cavity_mismatches_table()

    # Extracting the tables from the mismatch data
    cav_mismatches_table_x = mismatch_x.table
    cav_mismatches_table_y = mismatch_y.table 

    arr_float_x = cav_mismatches_table_x[1:-1,1:-1].astype(float)

    arr_float_y = cav_mismatches_table_y[1:-1,1:-1].astype(float)

    n = arr_float_x.shape[0]
    num_elements = n * (n - 1) // 2  # = 21 for a 7x7 matrix

    x_avg = np.sum(np.triu(arr_float_x, k=1)) / num_elements
    y_avg = np.sum(np.triu(arr_float_y, k=1)) / num_elements
    total_avg = (x_avg + y_avg) / 2


    # Specific mismatch values
    XARM_YARM_x = cav_mismatches_table_x[2][3]
    XARM_YARM_y = cav_mismatches_table_y[2][3]
    PRX_XARM_x = cav_mismatches_table_x[4][2]
    PRX_XARM_y = cav_mismatches_table_y[4][2]
    PRY_YARM_x = cav_mismatches_table_x[5][3]
    PRY_YARM_y = cav_mismatches_table_y[5][3]
    SRX_XARM_x = cav_mismatches_table_x[6][2]
    SRX_XARM_y = cav_mismatches_table_y[6][2]
    SRY_YARM_x = cav_mismatches_table_x[7][3]
    SRY_YARM_y = cav_mismatches_table_y[7][3]

    # Optionally print the mismatch tables
    if print_tables==True:
        print(mismatch_x)
        print(mismatch_y)
        print("All cavities average mismatch (x and y) is", total_avg)

    # Return the results as a dictionary
    return {
        "XARM_YARM_x": XARM_YARM_x,
        "XARM_YARM_y": XARM_YARM_y,
        "PRX_XARM_x": PRX_XARM_x,
        "PRX_XARM_y": PRX_XARM_y,
        "PRY_YARM_x": PRY_YARM_x,
        "PRY_YARM_y": PRY_YARM_y,
        "SRX_XARM_x": SRX_XARM_x,
        "SRX_XARM_y": SRX_XARM_y,
        "SRY_YARM_x": SRY_YARM_x,
        "SRY_YARM_y": SRY_YARM_y,
        "x_avg": x_avg,
        "y_avg": y_avg,
        'total_avg': total_avg
    }


def minimize_MM(model):
    model.fsig.f=1
    model.OMC_OC.Rc=np.array([1.43,1.55])
    model.OMC_CM1.Rc=400
    model.ITMX.Rc=2120
    model.ITMY.Rc=2105
    model.PRM.Rc=-11.5
    model.SR2.Rc=-6.4145
    model.SRM.Rc=np.array([-5.623,-5.763])
    return model

def wrap_angle(angle_deg):
    return ((angle_deg+180) % 360) - 180  # Result: [-180, 180)

def add_amplitude_detectors(model):
    model.add(det.AmplitudeDetector("HG00_amp",n=0,m=0,f=0,node=model.SRM.p1.i))
    model.add(det.AmplitudeDetector("HG10_amp",n=1,m=0,f=0,node=model.SRM.p1.i))
    model.add(det.AmplitudeDetector("HG20_amp",n=2,m=0,f=0,node=model.SRM.p1.i))
    model.add(det.AmplitudeDetector("HG01_amp",n=0,m=1,f=0,node=model.SRM.p1.i))
    model.add(det.AmplitudeDetector("HG02_amp",n=0,m=2,f=0,node=model.SRM.p1.i))


def make_thermal_lens(model,arms,rpts=100,s_max=100,r_tm=0.23,r_ap=0.16,h_tm=0.27,plot=False,absorption_x=0.1e-6,absorption_y=0.1e-6):

    outputs={'Px':[],'Py':[]}
    for arm in arms:
        lens = model.get(f"ITM{arm}lens")
        absorption=absorption_x if arm=='X' else absorption_y
        # plt.plot(opd)
        w_itm= model.get(f"ITM{arm}.p1.o.qx.w")

        r_m = np.linspace(0, r_tm, rpts)
        x, y, x_msh, y_msh, r_msh = maps.make_coordinates(rpts, r_tm)

        Z_coat_W_itm, _ = hv.thermal_lenses_HG00(
            r_m, r_tm, h_tm, w_itm, FusedSilica, s_max=s_max,
        )

        aperture = maps.circular_aperture(x, y, r_ap)

        opd_correction=0
        # print(f"adding {lens.name} OPD")
        PRG=53
        AGX=257.5870368662472 
        
        inc_power=model.L0.P*PRG*0.5 #out[f'P{arm.lower()}'] #power[-1][f'Pin{arm.lower()}']
        circ_power=inc_power*AGX 
        # print(inc_power)
        outputs[f'P{arm.lower()}'].append(circ_power)
        opd = Z_coat_W_itm *absorption* circ_power #60 ppm of absorbed power * inc_power # thermal_opts.Pabs_coat  # Will be all zeros
        opd_msh = np.interp(r_msh, r_m, opd)
        opd_map = Map(x, y, amplitude=aperture, opd=opd_msh)
        # plt.imshow(opd_map)

        opd_curvature_D = np.array(opd_map.remove_curvatures(w_itm))
        opd_map.remove_piston(float(w_itm))
        opd_map.opd[:] *= (1 - opd_correction)
        lens.OPD_map = opd_map
        
        if plot==True:
            plt.figure()
            plt.contourf(opd_map.x, opd_map.y, opd_map.opd/1e-6)
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.colorbar(label='OPD [µm]')

            plt.figure()
            plt.contourf(opd_map.x, opd_map.y, opd_map.amplitude)
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.colorbar(label='Amplitude [fractional]')
            plt.show()

            plt.plot(r_m, opd,label='ITMY OPD')
            plt.xlabel("r [m]")
            plt.ylabel("OPD [m]")
            plt.show()
    return outputs

def model_hom_indices(model):
        for idx, hom in enumerate(model.homs):
            print(f"index {idx} corresponds to HG{''.join(hom.astype(str))}")