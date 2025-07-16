function [ITMY_inQ, ITMY_reflQ, ITMY_transQ, ITMY_returnFromETMQ, rocITM, fITM]=H1YArm_beamqs()
%% Returns the beamQ object corresponding the H1 X Arm mode.
% [ITMY_inQ, ITMY_refQ, ITMY_transQ, ITMY_returnFromETMQ]=H1YARM_beamqs()
% ITMY_inQ is the beam coming from BS to ITMY just before ITMY, perfectly
% mode matched to the arm.
% ITMY_reflQ is the beam reflected by ITMY just after reflection, which is
% equal to just conj(-ITMY_inQ).
% ITMY_transQ is the beam transmitted by the ITMY just after ITMY.
% ITMY_returnFromETMQ is the beam coming back from ETMY just before it hits
% ITMY, which is equal to just conj(-ITMY_transQ).

% first calculate the eigen mode beamq.
rocETM = 2246.9; % this is ETM 16. see https://galaxy.ligo.caltech.edu/optics/
rocITM = 1939.2; % ITM 11. see https://galaxy.ligo.caltech.edu/optics/
L=4000;

armpath=beamPath();
armpath.addComponent(component.curvedMirror(rocITM, 0, 'ITM'));
armpath.addComponent(component.curvedMirror(rocETM, L, 'ETM'));
ITMY_returnFromETMQ =armpath.eigenMode(-1e-6, 2*L-1e-6);
%this is beamq of the beam coming from ETM just before hitting ITM.

% calculate the beamq of the ITMX trans.
armpath.seedq=ITMY_returnFromETMQ;
armpath.seedz=-1e-6; %inside the arm.
ITMY_transQ=armpath.qPropagate(1e-6);

%calculate the beamq of the ITMX refl
transpath=beamPath();
n=1.4496; %1064nm, fused silica
fITM=-rocITM/(n-1); % focal length of the transmissive lens. 
transpath.addComponent(component.lens(fITM, 0, 'ITM'));
transpath.seedq=ITMY_returnFromETMQ;
transpath.seedz=-1e-6;
ITMY_reflQ=transpath.qPropagate(1e-6);

%calculate the beamq of the beam coming from BS
transpath.seedq=ITMY_transQ;
transpath.seedz=1e-6;
ITMY_inQ=transpath.qPropagate(-1e-6);

return;