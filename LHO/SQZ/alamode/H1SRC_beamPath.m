function [SRC_Path, rocITM, fITM]=H1SRC_beamPath()
%% SRC_Path=H1SRC_beamPath()
% Returns the beamPath object containing BS, SR3, SR2 and SRM (the last one
% is handled as a transmissive lens).
% The path looks like
%   ITM(avg lens)->seedz->BS -> SR3->SR2->SRM(lens),
%   i.e. the seed beam is the arm mode coming out of ITM.
% BS position is 0.
% The path is seeded by an averaged beamq from X and Y arm eigenmode
%  measured at the average ITM position ("downstream of" the average ITM 
%  outside of the average arms).
% X and Y arm beamq just outside of the arm
%  are calculated from H1XArm_beamqs() and H1Yarm_beamqs();
% ITM in the beamPath object is a transmissive lens with f=fITM, though arm
%  mode inside the arm is correctly calculated using the ROC.
%  If you want to calculate e.g. single bounce propagation, replace ITM with
%  a mirror with the ROC of -rocITM (note the negative sign).
% rocITM and fITM are average of ITMX and ITMY. ITM position is the average
%  position of ITMX and ITMY.

[dummy1, Xbeamq, dummy2, dummy3, rocITMX, fITMX]=H1XArm_beamqs();
[dummy1, Ybeamq, dummy2, dummy3, rocITMY, fITMY]=H1YArm_beamqs();

rocITM=(rocITMX+rocITMY)/2;
fITM=(fITMX+fITMY)/2;

BS_IX=5372.66e-3; % from D0901920
BS_IY=5319.73e-3;
zITM=-(BS_IX+BS_IY)/2;
% propagate X and Y beam to this mean position.
dummypath=beamPath();
dummypath.seedq=Xbeamq;
dummypath.seedz=-BS_IX;
newQX=dummypath.qPropagate(zITM+1e-6);

dummypath=beamPath();
dummypath.seedq=Ybeamq;
dummypath.seedz=-BS_IY;
newQY=dummypath.qPropagate(zITM+1e-6);

% seed SRC_path with the newQ=(newQX+newQY)/2 at zITM;
SRC_Path=beamPath();
SRC_Path.seedq=beamq((newQX.q+newQY.q)/2);
SRC_Path.seedz=zITM+1e-6;

% add average ITM as a lens
SRC_Path.addComponent(component.lens(fITM, zITM, 'ITM(avg)'));

% add BS
zBS=0;
SRC_Path.addComponent(component.flatMirror(zBS, 'BS'));

%add SR3
BS_SR3=19469.5e-3; %calculated from D0901920
zSR3=zBS+BS_SR3;
rocSR3=36.013; %galaxy
SRC_Path.addComponent(component.curvedMirror(rocSR3, zSR3, 'SR3'));

% SR2
SR3_SR2=15461.6e-3; %from D0901920
zSR2=zSR3+SR3_SR2;
rocSR2=-6.424; % galaxy
SRC_Path.addComponent(component.curvedMirror(rocSR2, zSR2, 'SR2'));

% SRM as a transmissive lens
SR2_SRM=15724.2E-3; %from D0901920
zSRM=zSR2+SR2_SRM;
rocSRM=-5.678;%galaxy
n=1.4496; %1064nm, fused silica
f_SRM_trans=-rocSRM/(n-1); % transmissive lens
SRC_Path.addComponent(component.lens(f_SRM_trans, zSRM, 'SRM'));
return;
