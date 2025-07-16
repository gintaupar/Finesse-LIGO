function [ITMX_inQ, ITMX_reflQ, ITMX_transQ, ITMX_returnFromETMQ, rocITM, fITM]=H1XArm_beamqs()
%% Returns the beamQ object corresponding the H1 X Arm mode.
% [ITMX_inQ, ITMX_refQ, ITMX_transQ, ITMX_returnFromETMQ]=H1XARM_beamqs()
% ITMX_inQ is the beam coming from BS to ITMX just before ITMX, perfectly
% mode matched to the arm.
% ITMX_reflQ is the beam reflected by ITMX just after reflection, which is
% equal to just conj(-ITMX_inQ).
% ITMX_transQ is the beam transmitted by the ITMX just after ITMX.
% ITMX_returnFromETMQ is the beam coming back from ETMX just before it hits
% ITMX, which is equal to just conj(-ITMX_transQ).

% first calculate the eigen mode beamq.
rocETM = 2244.2; % this is ETM 13. see https://galaxy.ligo.caltech.edu/optics/
rocITM = 1940.3; % ITM 07, see https://galaxy.ligo.caltech.edu/optics/
L=4000;

armpath=beamPath();
armpath.addComponent(component.curvedMirror(rocITM, 0, 'ITM'));
armpath.addComponent(component.curvedMirror(rocETM, L, 'ETM'));
ITMX_returnFromETMQ =armpath.eigenMode(-1e-6, 2*L-1e-6);
%this is beamq of the beam coming from ETM just before hitting ITM.

% calculate the beamq of the ITMX trans.
armpath.seedq=ITMX_returnFromETMQ;
armpath.seedz=-1e-6; %inside the arm.
ITMX_transQ=armpath.qPropagate(1e-6);

%calculate the beamq of the ITMX refl
transpath=beamPath();
n=1.4496; %1064nm, fused silica
fITM=-rocITM/(n-1); % focal length of the transmissive lens. 
transpath.addComponent(component.lens(fITM, 0, 'ITM'));
transpath.seedq=ITMX_returnFromETMQ;
transpath.seedz=-1e-6;
ITMX_reflQ=transpath.qPropagate(1e-6);

%calculate the beamq of the beam coming from BS
transpath.seedq=ITMX_transQ;
transpath.seedz=1e-6;
ITMX_inQ=transpath.qPropagate(-1e-6);

return;