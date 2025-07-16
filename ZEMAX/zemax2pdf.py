# %% Import the data
# import SCHTUFF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
# %% Find the encoding
IFOname = "LHO"
if IFOname == "LLO":
    fname = 'LLO_zemax.txt'
    DCCnum = 'E2200033-v5'
elif IFOname == "LHO":
    fname = "LHO_zemax_HAM6.txt"
    DCCnum = "E2100383-v4"
else:
    raise ValueError(f"Unrecognized IFO {IFOname}")
latex_fname = f"{IFOname}_coords_{DCCnum}.tex"
import pandas as pd
import chardet
with open(fname, 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large
print('Encoding is ', result['encoding'])
# %%
# string to search in file
with open(fname, 'r', encoding=result['encoding']) as fp:
    # read all lines using readline()
    lines = fp.readlines()
    for row in lines:
        # check if string present on a current line
        word = 'Seg#'
        #print(row.find(word))
        # find() method returns -1 if the value is not found,
        # if found it returns index of the first occurrence of the substring
        if row.find(word) != -1:
            print('string exists in file')
            print('line Number:', lines.index(row))
            tableStart = lines.index(row)
# %% Read the table - skipping the header material
df = pd.read_table(fname, encoding=result['encoding'], skiprows=tableStart)
# %% Manipulate the data
# convert to numpy
for col in df.columns:
    # remove white space from name using strip()
    colName = (col.strip())
    # if colName == 'X':
    #     x = df[col]
    if colName == 'Y':
        Y = df[col].to_numpy()
    if colName == 'Z':
        X = df[col].to_numpy()
    if colName == 'Comment':
        Comment = df[col].to_numpy()
    if colName == 'Intensity':
        intensity = df[col].to_numpy()
# %% Plot the data
plt.plot(X, Y, '-o')
# %% Identify the optics
# optic key: [common optic name, zemax optic name]
if IFOname == "LLO":
    optics = {
        'ITMX': ['ITMX', 'ITMX-RADIUS'],
        'ITMY': ['ITMY', 'ITMY-RADIUS'],
        'BS': ['BS', 'BS A+'],
        'SR3': ['SR3', 'SR3_RADIUS'],
        'SR2': ['SR2', 'SR2_RADIUS'],
        'SRM': ['SRM', 'SRM_RADIUS'],
        'PR3': ['PR3', 'PR3_RADIUS'],
        'PR2': ['PR2', 'PR2_RADIUS'],
        'PRM': ['PRM', 'PRM_RADIUS'],
        'OM1': ['OM1', 'OM1 O4'],
        'OM2': ['OM2', 'OM2'],
        'OM3': ['OM3', 'OM3'],
        'OMC fold': ['OMC fold', 'input mirror'],
        'OMC in': ['OMC input', 'cavity input'],
        'ZM4': ['ZM4', 'ZM4'],
        'ZM5': ['ZM5', 'ZM5'],
        'ZM6': ['ZM6', 'ZM6'],
        'SQZ Inj': ['SQZ Inj', 'SQZ Inj'],
        'SQZ TFP': ['SQZ TFP', 'Thin Film']
    }
elif IFOname == "LHO":
    optics = {
        'ITMX': ['ITMX', 'ITMX_RADIUS'],
        'ITMY': ['ITMY', 'ITMY_RADIUS'],
        'BS': ['BS', 'BS'],
        'SR3': ['SR3', 'SR3_RADIUS'],
        'SR2': ['SR2', 'SR2_RADIUS'],
        'SRM': ['SRM', 'SRM'],
        'PR3': ['PR3', 'PR3_RADIUS'],
        'PR2': ['PR2', 'PR2_RADIUS'],
        'PRM': ['PRM', 'PRM_RADIUS'],
        'OM1': ['OM1', 'OM1'],
        'OM2': ['OM2', 'OM2'],
        'OM3': ['OM3', 'OM3'],
        'OMC fold': ['OMC fold', 'input mirror'],
        'OMC in': ['OMC input', 'cavity input'],
        'ZM4': ['ZM4', 'ZM4'],
        'ZM5': ['ZM5', 'ZM5'],
        'ZM6': ['ZM6', 'ZM6'],
        'SQZ Inj': ['SQZ Inj', 'SQZ Inj'],
        'SQZ TFP': ['SQZ TFP', 'Thin Film']
    }
else:
    raise ValueError(f"Unrecognized IFO {IFOname}")
# %% Find the elements which match the optic names and get the coordinates of those optics
# (subject to discrimination)
not_found = []
for zName in optics:
    cSub = np.array([]).reshape((0,3))
    for ii in range(Comment.size):
        # identify elements that contain the optic name
        if Comment[ii].strip().startswith(optics[zName][1]):
            # add x,y,int data to numpy array
            cSub = np.append(cSub, [[X[ii], Y[ii], intensity[ii]]], axis=0)

    if cSub.size > 0:
        # locate the elements with the highest intenstiy
        maxInt = np.max(cSub[:, 2])
        a = np.argwhere(cSub[:, 2] == maxInt)
        a = a.reshape(a.size)

        # get the coordinates of the highest intensity ray
        coord = cSub[a[0], 0:2]
        # add discriminator for ITMX and ITMY
        maxInd = np.argwhere(cSub[:, 2] > 0.5*maxInt)
        maxInd = maxInd.reshape(maxInd.size)
        coordSub = cSub[maxInd, 0:2]
        if zName=='ITMX':
            # get extreme X most coordinate
            bestInd = np.argwhere(coordSub[:,0]==np.max(coordSub[:,0]))
            bestInd = bestInd.reshape(bestInd.size)[0]
            coord = cSub[bestInd, 0:2]
        if zName=='ITMY':
            # get extreme Y most coordinate
            bestInd = np.argwhere(coordSub[:,1]==np.max(coordSub[:,1]))
            bestInd = bestInd.reshape(bestInd.size)[0]
            coord = cSub[bestInd, 0:2]

        # append the coordinates to the optics list
        optics[zName] = [optics[zName], coord]
    else:
        not_found.append(zName)
print("Optics not found", not_found)
# %% Get the distances between optics
#PL_OM1_OM2 (PL = physical length)
# SRC
if IFOname == "LLO":
    distances = {
        'SR3_SR2': np.linalg.norm(optics['SR3'][1] - optics['SR2'][1]),
        'SR2_SRM': np.linalg.norm(optics['SR2'][1] - optics['SRM'][1]),
        # Output chain
        'SRM_OM1': np.linalg.norm(optics['SRM'][1] - optics['OM1'][1]),
        'OM1_OM2': np.linalg.norm(optics['OM1'][1] - optics['OM2'][1]),
        'OM2_OM3': np.linalg.norm(optics['OM2'][1] - optics['OM3'][1]),
        'OM3_OMCin': np.linalg.norm(optics['OM3'][1] - optics['OMC fold'][1]) + np.linalg.norm(optics['OMC in'][1] - optics['OMC fold'][1]),
        # SQZ distances
        'ZM4_ZM5': np.linalg.norm(optics['ZM4'][1] - optics['ZM5'][1]),
        'ZM5_ZM6': np.linalg.norm(optics['ZM5'][1] - optics['ZM6'][1]),
        'ZM6_OFI': np.linalg.norm(optics['ZM6'][1] - optics['SQZ Inj'][1]) +  np.linalg.norm(optics['SQZ TFP'][1] - optics['SQZ Inj'][1]),
        'OFI_SRM': np.linalg.norm(optics['SQZ TFP'][1] - optics['SRM'][1]),
    }
elif IFOname == "LHO":
    distances = {
        'SR3_SR2': np.linalg.norm(optics['SR3'][1] - optics['SR2'][1]),
        'SR2_SRM': np.linalg.norm(optics['SR2'][1] - optics['SRM'][1]),
        # Output chain
        'SRM_OM1': np.linalg.norm(optics['SRM'][1] - optics['OM1'][1]),
        'OM1_OM2': np.linalg.norm(optics['OM1'][1] - optics['OM2'][1]),
        'OM2_OM3': np.linalg.norm(optics['OM2'][1] - optics['OM3'][1]),
        'OM3_OMCin': 293.2 + 10.8, # from D0901822-v12 (WIP).EASM (31.5 MB) from v11 of that document
        # SQZ distances
        #'ZM4_ZM5': np.linalg.norm(optics['ZM4'][1] - optics['ZM5'][1]),
        'ZM5_ZM6': np.linalg.norm(optics['ZM5'][1] - optics['ZM6'][1]),
        'ZM6_OFI': np.linalg.norm(optics['ZM6'][1] - optics['SQZ Inj'][1]) +  np.linalg.norm(optics['SQZ TFP'][1] - optics['SQZ Inj'][1]),
        'OFI_SRM': np.linalg.norm(optics['SQZ TFP'][1] - optics['SRM'][1]),
    }
else:
    raise ValueError(f"Unrecognized IFO {IFOname}")
#arr = np.round([PL_SR3_SR2, PL_SR2_SRM, PL_SRM_OM1, PL_OM1_OM2, PL_OM2_OM3, PL_OM3_OMC, PL_ZM4_ZM5, PL_ZM5_ZM6, PL_ZM6_OFI, PL_OFI_SRM])
#print(arr)
for dName in distances:
    distances[dName] = [distances[dName], distances[dName]]

# refractive index of glass
n = 1.45

# transmissive optics on HAM5
thkSRM = 100 # mm
thkTFP = 10 # mm
distances['SRM_OM1'][1] = distances['SRM_OM1'][1] + (n-1)*thkSRM + (n-1)*thkTFP
distances['OFI_SRM'][1] = distances['OFI_SRM'][1] + (n-1)*thkSRM

# transmissive optics on HAM6
thkOMCin = 10 #mm
distances['OM3_OMCin'][1] = distances['OM3_OMCin'][1] + (n-1)*thkOMCin
# %% Generate the LaTeX file
now = datetime.now()
dateNow = now.strftime('%d/%b/%Y')
# %%
with open(latex_fname, 'w') as f:
    # header information
    f.write('\\documentclass{article}\n')
    f.write('\\usepackage{graphicx} % Required for inserting images \n')
    f.write('\\usepackage[colorlinks=true,urlcolor=blue]{hyperref} \n')
    f.write(f'\\title{{Table {IFOname}}} \n')
    f.write('\\author{ZEMAX 2 PDF }\n')
    f.write(f'\\date{{{dateNow}}}\n\n')
    f.write('\\begin{document}\n')
    f.write('\\maketitle\n\n\n')

    f.write(f'Data extracted from Ray Database file in {DCCnum} at \\url{{https://dcc.ligo.org/LIGO-{DCCnum}}}\n')
    f.write('\n\n Regarding optical distances:\n')
    f.write(f'\\begin{{itemize}} \n \\item SQZ thin film polarizer assumed to be fused silica and {thkTFP:.0f} mm thick\n')
    f.write(f' \\item OMC input assumed to be fused silica and {thkOMCin:.0f} mm thick\n')
    f.write('\\end{itemize}\n\n')

    f.write('\\begin{table}[h!] \n\\centering \n\\begin{tabular}{| c| c | c |} \n')
    f.write('\\hline \\bf{Optic name} & \\bf{Global X coord (mm)} & \\bf{Global Y coord (mm)} \\\\ \\hline \n')


    # loop through all the elements
    for zName in optics:
        if zName in not_found:
            strOut = f"{zName} & not found & not found \\\\ \n"
        else:
            coords = optics[zName][1]
            strOut = f"{zName} & {coords[0]:.0f} & {coords[1]:.0f} \\\\ \n"
        f.write(strOut)
    
    f.write('\\hline \n\\end{tabular} \n')
    f.write(f'\\caption{{Table of physical {IFOname} coordinates derived from ZEMAX. Generated on {dateNow}}}\n')
    f.write('\\end{table} \n')

    f.write('\n')

    # get the table of distances between optics
    f.write('\\begin{table}[h!] \n\\centering  \n\\begin{tabular}{| c| c | c |} \n')
    f.write('\\hline \\bf{Distance} & \\bf{Physical Length (mm)} & \\bf{Optical Length (mm)} \\\\ \\hline \n')

    # loop through all the elements
    for dName in distances:
        PL = distances[dName][0]
        OL = distances[dName][1]

        strOut = "{:s} & {:0.0f} & {:0.0f} \\\\ \n".format(dName.replace("_", "\_"), PL, OL)
        f.write(strOut)
    
    f.write('\\hline \n\\end{tabular} \n')
    f.write(f'\\caption{{Table of {IFOname} distances derived from ZEMAX. Generated on {dateNow}}}\n')
    f.write('\\end{table} \n')
    f.write('\n\n Generated by \\url{https://git.ligo.org/IFOsim/ligo-commissioning-modeling/-/tree/main/ZEMAX/zemax2pdf.py}\n\n')
    f.write('\n\n\\end{document}')
# %%
os.system('pdflatex ' + latex_fname)
os.system('rm *.dvi *.aux *.log *.out')
# %%
