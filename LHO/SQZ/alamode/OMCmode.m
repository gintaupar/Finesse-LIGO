function OMC_beamPath=OMCmode()
%% returns a beamPath object containing OM1, OM2, OM3 and OMC.
% mode calculation is done using the OMC's eigen mode, not the arm mode.
% z=0 -> OM1 position
% seedq position is z-1e-6 (i.e. just upstream of OM1.

%first calculate the eigenmode beamq
Rc=2.575; %curved, e.g. T1000276
L1=.2816; % curved-curved and flat-flat length, e.g. T1300189
L2=.2844; % curved-flat length, T1300189
AOI=4.042*pi/180; %4.042 AOI

flat1=component.flatMirror(0, 'in');
flat2=component.flatMirror(L1, 'out');
curved1=component.curvedMirror(Rc, L2+L1, 'curved1');
curved2=component.curvedMirror(Rc, L2+2*L1, 'curved2');
flat0=component.flatMirror(2*L2+2*L1, 'in (again)');

OMCpath=beamPath();
OMCpath.addComponent(flat1);
OMCpath.addComponent(flat2);
OMCpath.addComponent(flat0);
OMCpath.addComponent(curved1);
OMCpath.addComponent(curved2);

OMC_waistq=OMCpath.eigenMode(L1/2, L1/2+2*(L1+L2));
% this gives 4.9e-4 [m] radius waist.
OMCpath.seedq=OMC_waistq;
OMCpath.seedz=L1/2;
OMCpath.plotSummary();
%% back propagate to SRM
OM123path=beamPath;
OMC_OM3=.268;
OM3_OM2=.708;
OM2_OM1=1.395;
R_OM1=4.6;
R_OM2=1.7;

OM123path.addComponent(component.curvedMirror(R_OM1, 0, 'OM1'));
OM123path.addComponent(component.curvedMirror(R_OM2, OM2_OM1, 'OM2'));
OM123path.addComponent(component.flatMirror(OM2_OM1+OM3_OM2, 'OM3'));
OM123path.seedWaist(OMC_waistq.waistSize, OM2_OM1+OM3_OM2+OMC_OM3);
OM123path.addComponent(component.flatMirror(OM2_OM1+OM3_OM2+OMC_OM3, 'OMC waist'))

OM123path.plotSummary();
newseedq=OM123path.qPropagate(-1e-6);

%now that the back propagation is done,set the q just upstream of OM1 as
%the seed, and set the q at OMC waist as the target.
OM123path.seedq=newseedq;
OM123path.seedz=-1e-6;
OM123path.targetq=OMC_waistq;
OM123path.targetz=OM2_OM1+OM3_OM2+OMC_OM3;
disp('AOI=0 for OM1 and OM2');
display(OM123path.targetOverlap);

OMC_beamPath=OM123path;

%% Now change the OM2 pos
%overlap=zeros(size(-10:10));
%for ii=-10:10
%     dispz=ii*1e-2;
%     thispath=OM123path.duplicate();
%     thispath.moveComponent('OM2', dispz);
%     overlap(ii+11)=thispath.targetOverlap;
% end
% 
% figure
% plot((-10:10), 1-overlap);
% xlabel('OM2 displacement [cm]')
% ylabel('Loss due to OMC and IFO mode mismatch (in power)')
% grid on
return