clear all
close all

addpath  /ligo/home/sheila.dwyer/PSL/mode_matching/alm-master

%finesse_data = yaml.load('lho_O4.yaml')

finesse_struct = readyaml('../../../../../LHO/yaml/lho_O4.yaml');
n = 1.45;%496;  %index of refraction at 1064nm fused silica
c = 299792458; 

%% IMC eigen mode
imc_struct = finesse_struct.INPUT.IMC;
%define a path that is the IMC, a vertical and a horizontal path are useful
% say that MC3 is the starting point because you want the q at MC3
IMC_v = beamPath;
IMC_h = beamPath;
MC1_z = imc_struct.length_MC3_MC1;
MC2_z = MC1_z+ imc_struct.length_MC1_MC2;
MC3_z = MC2_z + imc_struct.length_MC2_MC3;

%IMC length measured in P1400184
IMC_FSR = 9100235.6;

IMC_length = c/IMC_FSR; %only 2.8mm shorter than 
%another check Paul's numbers from 7973
%it seems that if we include the MC3 substrate path length the distance
%between 

IMC_v.addComponent(component.flatMirror(MC1_z, 'MC1'))
IMC_h.addComponent(component.flatMirror(MC1_z, 'MC1'))
[IMC_v, IMC_h] = add_curved_mirror(IMC_v, IMC_h, imc_struct.MC2.Rc, imc_struct.MC2.AOI, MC2_z, 'MC2');
IMC_v.addComponent(component.flatMirror(MC3_z, 'MC3'))
IMC_h.addComponent(component.flatMirror(MC3_z, 'MC3'))

%solve for the IMC mode
qIMCv_out = IMC_v.eigenMode(0, MC3_z); 
qIMCh_out = IMC_h.eigenMode(0, MC3_z);
%disp(qIMCv_out)
%% Input path

%now make input beams for the IFO with these beam parameters, start as z=0
%at MC3
Input_v = beamPath;
Input_h = beamPath;
%negative 1 is to use the waist after MC3
Input_v.seedWaist(qIMCv_out.waistSize, -1*qIMCv_out.waistZ)
Input_h.seedWaist(qIMCh_out.waistSize, -1*qIMCh_out.waistZ)

%add IMs
IM2_z = finesse_struct.INPUT.length_IMC_IM1 + finesse_struct.INPUT.length_IM1_IM2;
IM3_z = IM2_z + finesse_struct.INPUT.length_IM2_IFI + finesse_struct.INPUT.length_IFI_IM3;
PRM_AR_z = IM3_z + finesse_struct.INPUT.length_IM3_IM4 + finesse_struct.INPUT.length_IM4_PRM_AR;

[Input_v, Input_h] = add_curved_mirror(Input_v, Input_h,finesse_struct.INPUT.IM2.Rc, finesse_struct.INPUT.IM2.AOI,IM2_z,  'IM2' );
[Input_v, Input_h] = add_curved_mirror(Input_v, Input_h,finesse_struct.INPUT.IM3.Rc, finesse_struct.INPUT.IM3.AOI,IM3_z,  'IM3' );

%% refl path
%go through the PRM substrate on the way to HR surface
PRM= finesse_struct.PRC.PRM;
%now branch this path between REFL path and path that propagates on in PRC
Refl_v = Input_v.branchPath(PRM_AR_z );  %take the refl path just before reflecting off the PRM HR surface
Refl_h = Input_h.branchPath(PRM_AR_z );
%add PRM reflection , for the reflected beam this ROC is positive

[Refl_v, Refl_h] = add_curved_mirror(Refl_v, Refl_h, -1*PRM.Rc/n, 0, PRM_AR_z + PRM.thickness/n, 'PRM');
%multiplying ABCD matrix for dielectric refraction, propagation, reflection,
% propgation, dielctric refration, is equivalent to propagate thickness/n,
% cureved mirror ROC/n, propagate thickness/n
%now go back through, IM3, IM2
IM3_refl_z = PRM_AR_z +2* PRM.thickness/n + finesse_struct.INPUT.length_IM4_PRM_AR +finesse_struct.INPUT.length_IM3_IM4;
IM2_refl_z = IM3_refl_z + finesse_struct.INPUT.length_IFI_IM3 + finesse_struct.INPUT.length_IM2_IFI;
[Refl_v, Refl_h] = add_curved_mirror(Refl_v, Refl_h, finesse_struct.INPUT.IM3.Rc, finesse_struct.INPUT.IM3.AOI, IM3_refl_z, 'IM3 refl');
[Refl_v, Refl_h] = add_curved_mirror(Refl_v, Refl_h, finesse_struct.INPUT.IM2.Rc, finesse_struct.INPUT.IM2.AOI, IM2_refl_z, 'IM2 refl');

%get distances to HAM1
refl_telescope = finesse_struct.INPUT.Telescope;
RM1_z = IM2_refl_z + refl_telescope.length_IM2_REFL_LossyMirror + refl_telescope.length_LossyMirror_RM1;
RM2_z = RM1_z + refl_telescope.length_RM1_RM2;
M5_z = RM2_z + refl_telescope.length_RM2_M5;
Refl_v.addComponent(component.curvedMirror(refl_telescope.RM1.Rc, RM1_z, 'RM1'));
Refl_h.addComponent(component.curvedMirror(refl_telescope.RM1.Rc, RM1_z, 'RM1'));
Refl_v.addComponent(component.curvedMirror(refl_telescope.RM2.Rc, RM2_z, 'RM2'));
Refl_h.addComponent(component.curvedMirror(refl_telescope.RM2.Rc, RM2_z, 'RM2'));
Refl_v.addComponent(component.curvedMirror(refl_telescope.M5.Rc, M5_z, 'M5'));
Refl_h.addComponent(component.curvedMirror(refl_telescope.M5.Rc, M5_z, 'M5'));

%measurements from alog 84266
meas_z = [RM1_z - 0.314, M5_z - 0.374, RM2_z - 0.345] ;
meas_v = [3950.6/2, 2335.9/2, 1805.1/2]*1e-6;
meas_h = [4135.0/2, 2304.8/2, 1650.9/2]*1e-6;

%mode master measurements 7950

%from T1300960
% These are the input beam parameters (right before RM1), based on IO as
% built numbers
qH1X_RM1_pred = 2.8071 + 13.3724i; % Lisa/Paul prediction 
qH1Y_RM1_pred = 2.5071 + 13.0988i; % Lisa/Paul prediction 
IO_pred_v = beamPath;
IO_pred_h = beamPath;
IO_pred_v.seedq = beamq(qH1X_RM1_pred);
IO_pred_h.seedq = beamq(qH1Y_RM1_pred);
IO_pred_v.seedz = RM1_z-0.0001;
IO_pred_h.seedz = RM1_z-0.0001;
IO_pred_v.addComponent(component.curvedMirror(refl_telescope.RM1.Rc, RM1_z, 'RM1'));
IO_pred_h.addComponent(component.curvedMirror(refl_telescope.RM1.Rc, RM1_z, 'RM1'));
IO_pred_v.addComponent(component.curvedMirror(refl_telescope.RM2.Rc, RM2_z, 'RM2'));
IO_pred_h.addComponent(component.curvedMirror(refl_telescope.RM2.Rc, RM2_z, 'RM2'));
IO_pred_v.addComponent(component.curvedMirror(refl_telescope.M5.Rc, M5_z, 'M5'));
IO_pred_h.addComponent(component.curvedMirror(refl_telescope.M5.Rc, M5_z, 'M5'));

close all
% plot the results
zplot = 10:.01:14;
Refl_v.components;
figure(1)
clf
hold on
plot(meas_z, meas_v, 'o')
hold on
Refl_v.plotComponents;
Refl_v.plotBeamWidth(zplot, 'b')
%Refl_v.plotWaists(zplot)
IO_pred_v.plotBeamWidth(zplot, 'r')
title('Blue: Refl path vertical using finesse parameters, red: T1300960')
xlim([min(zplot), max(zplot)])

figure(2)
clf
hold on
plot(meas_z, meas_h, 'o')
hold on
Refl_h.plotComponents;
Refl_h.plotBeamWidth(zplot, 'b')
%Refl_v.plotWaists(zplot)
IO_pred_h.plotBeamWidth(zplot, 'r')
plot(meas_z, meas_v, 'x')
title('Refl path horizontal, blue finesse parameters, red T1300960')
xlim([min(zplot), max(zplot)])

zplot = -1:.01:15;
figure(12)
Refl_v.plotComponents;
hold on
Refl_v.plotSummary(zplot)
Refl_h.plotBeamWidth(zplot, 'b')


%% propagate through the PRC to ITMX

PRC_h = Input_h.duplicate;
PRC_v = Input_v.duplicate;
PRC_h.addComponent(component.dielectric(Inf, PRM.Rc, PRM.thickness, n, PRM_AR_z, 'PRM substrate'));
PRC_v.addComponent(component.dielectric(Inf, PRM.Rc, PRM.thickness, n, PRM_AR_z, 'PRM substrate'));
PRC = finesse_struct.PRC;
PR2_z = PRM_AR_z + PRM.thickness + PRC.length_PRM_PR2;  
%Matt and Sheila comment.  Do we need PRM thickness here, or should we skip
%it because it is included in the dielectric ABCD matrix above?
PR3_z = PR2_z + PRC.length_PR2_PR3;
[PRC_v, PRC_h] = add_curved_mirror(PRC_v, PRC_h, PRC.PR2.Rc, PRC.PR2.AOI, PR2_z, 'PR2');
[PRC_v, PRC_h] = add_curved_mirror(PRC_v, PRC_h, PRC.PR3.Rc, PRC.PR3.AOI, PR3_z, 'PR3');

substrate_angle = asin(sin(finesse_struct.BS.AOI*pi/180)/n);

BS_prop = (finesse_struct.BS.thickness/(cos(substrate_angle)));  %physical thickness of propagation in BS
a = cos(substrate_angle)/cos(finesse_struct.BS.AOI*pi/180);
% BS_prop = (finesse_struct.BS.thickness/(cos(substrate_angle)))*(n-1);
%discrepancy between this thickness and finesse
%finesse seems to use thin lens approx for the CPS, so I'll put it's
%location 
BS_z = PR3_z + PRC.length_PR3_BS;
PRC_v.addComponent(component.dielectric(inf, inf, BS_prop, n, BS_z, 'BS v'))
PRC_h.addComponent(component.dielectric(inf, inf, BS_prop/a^2, n, BS_z, 'BS h'))

CPX_z_v =BS_z + BS_prop + finesse_struct.X.length_BS_CP;
CPX_z_h =BS_z + BS_prop/a^2 + finesse_struct.X.length_BS_CP;

PRC_v.addComponent(component.lens(finesse_struct.X.CP.cold_focal_length, CPX_z_v, 'CPx'));
PRC_h.addComponent(component.lens(finesse_struct.X.CP.cold_focal_length, CPX_z_h, 'CPx'));
%because the rayleigh range is large, it doesn't really matter to get the substratte lens for the CP and teh ITM in the right location,
%but it is convient to separate them spatially for putting the branch to the arm in the right place.
PRC_v.addComponent(component.lens(finesse_struct.X.ITM.cold_focal_length, CPX_z_v, 'ITMX lens'));
PRC_h.addComponent(component.lens(finesse_struct.X.ITM.cold_focal_length, CPX_z_h, 'ITMX lens'));

ITMX_HR_z_v = CPX_z_v + finesse_struct.X.CP.thickness/n + finesse_struct.X.length_CP_TM + finesse_struct.X.ITM.thickness/n;
ITMX_HR_z_h = CPX_z_h + finesse_struct.X.CP.thickness/n + finesse_struct.X.length_CP_TM + finesse_struct.X.ITM.thickness/n;

%% branch to arm path, and let PRC beam reflect and return to BS
arm_v = PRC_v.branchPath(ITMX_HR_z_v);
arm_h = PRC_h.branchPath(ITMX_HR_z_h);
%todo, fine arm cavity eigen mode and see how this overlaps with it
%continute to relfect off ITM for PRC_x return beam
[PRC_v, PRC_h] = add_curved_mirror(PRC_v, PRC_h, -1*finesse_struct.X.ITM.Rc/n, 0, [ITMX_HR_z_v, ITMX_HR_z_h], 'ITMX reflection');

CPX_z_return_v =  ITMX_HR_z_v + finesse_struct.X.ITM.thickness/n + finesse_struct.X.length_CP_TM + finesse_struct.X.CP.thickness/n;
CPX_z_return_h =  ITMX_HR_z_h + finesse_struct.X.ITM.thickness/n + finesse_struct.X.length_CP_TM + finesse_struct.X.CP.thickness/n;

PRC_v.addComponent(component.lens(finesse_struct.X.ITM.cold_focal_length, CPX_z_return_v, 'ITMX lens return'));
PRC_h.addComponent(component.lens(finesse_struct.X.ITM.cold_focal_length, CPX_z_return_h, 'ITMX lens return'));
PRC_v.addComponent(component.lens(finesse_struct.X.CP.cold_focal_length, CPX_z_return_v, 'CPx return'));
PRC_h.addComponent(component.lens(finesse_struct.X.CP.cold_focal_length, CPX_z_return_h, 'CPx return'));

%% branch at reflective side of BS, one beam goes to SRC, one returns to POP
BS_AR_z_return_v = CPX_z_return_v  + finesse_struct.X.length_BS_CP ;
BS_AR_z_return_h = CPX_z_return_h  + finesse_struct.X.length_BS_CP ;

PRC_v.addComponent(component.dielectric(inf, inf, BS_prop, n, BS_AR_z_return_v, 'BS v'))
PRC_h.addComponent(component.dielectric(inf, inf, BS_prop/a^2, n, BS_AR_z_return_h, 'BS h'))

BS_z_return_v =BS_AR_z_return_v + BS_prop;  %go back to relfective side of BS, before you branch
BS_z_return_h = BS_AR_z_return_h + BS_prop/a^2; 

AS_v = PRC_v.branchPath(BS_z_return_v);
AS_h = PRC_h.branchPath(BS_z_return_h);
%todo, come back and add SRC, OMs to AS beam path.  When we come back to
%this, the horizontal propagation through the BS should be BS_prop/a^2,
%vertical should be BS_prop

%continue PRC beam to POP return beam
PR3_z_return_v = BS_z_return_v + PRC.length_PR3_BS;
PR3_z_return_h = BS_z_return_h + PRC.length_PR3_BS;
PR2_z_return_v = PR3_z_return_v + PRC.length_PR2_PR3;
PR2_z_return_h = PR3_z_return_h + PRC.length_PR2_PR3;

[PRC_v, PRC_h] = add_curved_mirror(PRC_v, PRC_h, PRC.PR3.Rc, PRC.PR3.AOI, [PR3_z_return_v, PR3_z_return_h], 'PR3');
%branch off POP path here, 
POP_v = PRC_v.branchPath(PR2_z_return_v);
POP_h = PRC_h.branchPath(PR2_z_return_h);
%need to get info for pop path from Elenna
PR2_thickness = 75.1e-3; %number from Elenna
%TODO: this does not correctly handle the AOI, see SEgiman page 586
%correct the thickness to consider the angle
PR2_substrate_angle = asin(sin(PRC.PR2.AOI*pi/180)/n);
POP_v.addComponent(component.dielectric(PRC.PR2.Rc, Inf, PR2_thickness/cos(PR2_substrate_angle), n, PR2_z_return_v, 'PR2 transmission'));
POP_h.addComponent(component.dielectric(PRC.PR2.Rc, Inf, PR2_thickness/cos(PR2_substrate_angle), n, PR2_z_return_h, 'PR2 transmission'));

%HAM3 
%add curved mirror AROMRH3  ROC 1.5m 2.4483 meters from PR2 AR  #from
%E1300207 I'm not dividing the thickness of PR2 by n, since I think that is
%taken care of in the dielectric function
AROM_RH3_z_v = PR2_z_return_v + PR2_thickness/cos(PR2_substrate_angle) + 2.4483; 
AROM_RH3_z_h = PR2_z_return_h + PR2_thickness/cos(PR2_substrate_angle) + 2.4483;

[POP_v, POP_h] = add_curved_mirror(POP_v, POP_h,1.5, 2.9, [AROM_RH3_z_v,AROM_RH3_z_h], 'RH3');
POP_dichroic_z = AROM_RH3_z_v + 19.3277; 
POP_lens_z = POP_dichroic_z + 0.360 + 0.229;
POP_v.addComponent(component.lens(0.334, POP_lens_z, 'POP lens'));
POP_h.addComponent(component.lens(0.334, POP_lens_z, 'POP lens'));
%copied from alog 84307
% reported as distances from the dichroic mirror
pop_meas_zs = [52 147 884 918 893 904 1118] * 1e-3 + POP_dichroic_z;
pop_meas_wx = [6243 6202 870 279 721 483 3280] * 1e-6 / 2 ;
pop_meas_wy = [6404 6365 874 284 726 490 3372] * 1e-6 / 2 ;
%for HAM1 layout Spring 2025 see D10000313-v19 (before updates to match
%photos)
zplot = -1:0.1:126;
figure(3)
clf
hold on
POP_v.plotSummary(zplot)
POP_h.plotBeamWidth(zplot,'b')

zplot = 122:0.01:126;
figure(4)
clf
hold on
plot(pop_meas_zs, pop_meas_wx, 'o')
plot(pop_meas_zs, pop_meas_wy, 'x')
hold on
POP_v.plotBeamWidth(zplot, 'r')
POP_h.plotBeamWidth(zplot,'b')

%% print out some qs to compare with what Elenna gets from finesse
%0 coordinate is the MC3 HR surface, get this starting q
format long
IMC_q = PRC_h.qPropagate(0);

ABCD = Refl_h.getTransferMatrix(PRM_AR_z, PRM_AR_z + 2*PRM.thickness/n);
IM2_q = PRC_h.qPropagate(IM2_z);
%IMC verical I get here matches alm horizontal IMC waist, beam just before
%IM2 matches
IM3_q = PRC_h.qPropagate(IM3_z);
PRM_q = PRC_h.qPropagate(PRM_AR_z+PRM.thickness);  
PR2_q = PRC_h.qPropagate(PR2_z);%match finesse this far
PR3_q = PRC_h.qPropagate(PR3_z);
BS_in_q = PRC_h.qPropagate(BS_z +BS_prop/a^2);
IM3_refl_q = Refl_h.qPropagate(IM3_refl_z - 0.0000001);%match this far
RM1_q =Refl_h.qPropagate(RM1_z - 0.00000001); %match finesse this far, after PR3 on the way in
CPX_return_v = PRC_v.qPropagate(CPX_z_return_v); %agrees well with Finesse
CPX_return_h = PRC_h.qPropagate(CPX_z_return_h);
%ITMX_AR_q = PRC_h.qPropagate(ITMX_AR_z - 1e-6); %
BS_return_q_v = PRC_h.qPropagate(BS_z_return_v);
BS_return_q_h = PRC_h.qPropagate(BS_z_return_h);
PR2_return_q_v = PRC_v.qPropagate(PR2_z_return_v)
PR2_return_q_h = PRC_h.qPropagate(PR2_z_return_h)