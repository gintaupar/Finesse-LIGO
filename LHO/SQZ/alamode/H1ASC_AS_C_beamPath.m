function [H1AS_C_Path, rocITM, fITM]=H1ASC_AS_C_beamPath()
%% [H1AS_C_Path, rocITM, fITM] =H1ASC_AS_C_beamPath()
% Returns a beamPath object containing SRC, OM1, lens in front of ASC-AS_C
%  and ASC-AS_C itself for H1 using measured parameters wherever possible
%  and/or meaningful.
% The path is seeded by an averaged beamq from X and Y arm eigenmode
%  measured just downstream of the average ITM position (i.e. outside of the arms).
% ITM in the beamPath object is a transmissive lens with f=fITM. The arm
%  mode inside the arm is correctly calculated using the ROC.
% If you want to calculate e.g. single bounce propagation, replace ITM with
%  a lens-mirror-lens using fITM and -rocITM (note the negative sign).
%  This is already conveniently offered in
%  H1SingleBounce_SRC_OMC_beamPath();
% rocITM and fITM are average of ITMX and ITMY. ITM position is the average
%  position of ITMX and ITMY.
% Depends on H1SRC_beamPath.m (which in turn depends on H1XArm_beamqs.m and
%  H1YArm_beamqs.m).

n=1.4496; %1064nm, fused silica

% Make a copy of H1SRC beamPath object. That object handles SRM as a transmissive lens.
[H1AS_C_Path, rocITM, fITM]=H1SRC_beamPath();
z_SRM=H1AS_C_Path.component('SRM').z;
z_SR2=H1AS_C_Path.component('SR2').z;
%OM1 as a transmissive lens
ROC_OM1=4.600;
f_OM1_trans = -ROC_OM1/(n-1);
d_SRM_OM1=3.646; %T1000247 and T1000317
z_OM1=d_SRM_OM1+z_SRM;
H1AS_C_Path.addComponent(component.lens(f_OM1_trans, z_OM1, 'OM1'));

% lens
f_L1=334e-3; %see D1000342 optics list.
d_OM1_L1= (2.5 + 17+ 4.5 + 5)*25.4e-3;
% this is L1-M7, M7-M5, M5-M4 and M4-OM1 roughly estimated from the above
% picture,
% https://alog.ligo-wa.caltech.edu/aLOG/uploads/40377_20180201215138__wfs_sled1.jpg
% for L1-M7, M7-M5
% https://alog.ligo-wa.caltech.edu/aLOG/uploads/40377_20180201215036__om3.jpg
% for the L1 and OM1 screw holes which helps figuring out M5-M4.
z_L1=z_OM1+d_OM1_L1;
H1AS_C_Path.addComponent(component.lens(f_L1, z_L1, 'L1'));

%ASC_AS_C is OSI FCI-InGaAs-Q3000, diameter= 3mm
%for naming, see D1000342
d_L1_AS_C=10*25.4e-3; % distance in mm, 10" as installed,
%d_L1_AS_C=f_L1;
%d_L1_AS_C=8*25.4e-3; % distance in mm, 10" as installed,
z_AS_C=z_L1+d_L1_AS_C;
%see https://alog.ligo-wa.caltech.edu/aLOG/index.php?callRep=40383
%and especially https://alog.ligo-wa.caltech.edu/aLOG/uploads/40383_20180202094444_IMAG1012.jpg
H1AS_C_Path.addComponent(component.flatMirror(z_AS_C, 'ASC-AS-C'));

return
