clear all;
close all;
addpath('../../aLIGO_modes/');
OM2Path=H1SingleBounce_SRC_OMC_beamPath().duplicate();
ITMPath=OM2Path.duplicate();

%% MM loss and OM2 ROC, cold and hot
coldLoss=(0.83+1.158)/(15.32+0.17+0.83+1.158); % alog 70502 where 02 and 20 are resolved. 01 mode power (0.17) is in the denominator as the 1st order approximation.
coldLossMax=coldLoss*1.1;
coldLossMin=coldLoss*0.9;

hotLoss= (0.457+0.629)/(16.39+0.16+0.457+0.629); % alog 71100 where 02 and 20 are resolved. 01 mode power (0.16) is in the denominator as the 1st order approximation.
hotLossMax=hotLoss*1.1;
hotLossMin=hotLoss*0.9;

hotROC=1.75;%[m]
coldROC=2;%[m], 
dOM2Range=2*(1/hotROC-1/coldROC); % optical power maximum range for OM2. A factor of 2 as focal length of ROC spherical mirror is ROC/2


%% MM loss with cold OM2, ITMY central CO2 OFF (cold) and ON (hot)
coldITMLoss=(0.575+0.853)/(15.90+0.40+0.575+0.853); % alog 71716 where 02 and 20 are resolved. 
% As a rough approximation, 10 mode power (0.4) is added in the denominator.
% https://alog.ligo-wa.caltech.edu/aLOG/index.php?callRep=71716
coldITMLossMax=coldITMLoss*1.1;
coldITMLossMin=coldITMLoss*0.9;

hotITMLoss= (0.201+0.428)/(17.02+0.40+0.201+0.428); % alog 71100 where 02 and 20 are resolved.
% Again, as a rough approximation, 10 mode power is added in the
% denominator. Seems like the alignment was pretty stable (no change in the
% 1st order mode power).
hotITMLossMax=hotITMLoss*1.1;
hotITMLossMin=hotITMLoss*0.9;

coldITMPowerDP=0; % optical power of the double path ITM central heating in diopter [1/m]. When cold it's 0.
hotITMPower=17.05e-6; % [1/m], this is a single pass number based on H1:TCS-SIM_ITMY_SUB_DEFOCUS_FULL_SINGLE_PASS_OUTPUT.

%% construct a grid for the mismatching parameter space.
% Convention:
%    p: Waist position. 
%   dp: Waist position difference. 
%       Positive = beam waist is downstream of OMC waist.
%    w: Waist radius.
%   dw: Waist radius difference.
%       Positive = beam waist is bigger.
%   dpN=dp/2/reyleighOMC: Normalized waist pos diff.
%   dwN=dw/wOMC: Normalized waist radius diff.
% X axis is the waist position difference dz0 in meters, corresponding to 
%  [-0.6, 0.6] in the normalized papameter dz0/rayleigh/2.
% Y axis is the waist radius difference dw0 in meters, corresponding to
%  [-0.8, 0.8] in the normalized parameter dw0/waistRadius(OMC).

gridN=2*200+1 ;
gridX = ((1:gridN)-(1+gridN)/2) / ((gridN-1)/2); % e.g. (-50:50)/50
gridY = transpose(-gridX); % e.g. transpose(50:-1:-50)/50

targetq=OM2Path.targetq;
wOMC=targetq.waistSize;
pOMC=OM2Path.targetz;
rayleighOMC=targetq.rayleighRange;
dpN=repmat(0.5*gridX, [gridN, 1]); % (pBeam - pOMC)/2/rayleighOMC
dwN=repmat(0.5*gridY, [1, gridN]); % (wBeam - wOMC)/wOMC
dp=dpN*2*rayleighOMC;
dw=dwN*wOMC;

wBeam=dw+wOMC;
rayleighBeam=pi/1064e-9*(wBeam.^2);
qBeam=1.0i*rayleighBeam - dp; % dp positive -> waist is downstream of OMC waist -> still hasn't reached the waist at OMC waist position

loss0=zeros(size(dpN));
for ii=1:gridN
    for jj=1:gridN
        loss0(ii, jj) = 1- targetq.overlap(beamq(qBeam(ii,jj)));
    end
end



% coldMask = mask where cold loss is close to measured.
%  hotMask = mask where hot loss close to measured.
coldMask=ones(size(loss0));
hotMask=ones(size(loss0));
coldMaskITM=ones(size(loss0));
hotMaskITM=ones(size(loss0));

for ii=1:gridN
    for jj=1:gridN
        if loss0(ii, jj) <coldLossMin || loss0(ii, jj) >coldLossMax
            coldMask(ii, jj) = nan;
        end
        if loss0(ii, jj) <hotLossMin || loss0(ii, jj) >hotLossMax
            hotMask(ii, jj) = nan;
        end

        if loss0(ii, jj) <coldITMLossMin || loss0(ii, jj) >coldITMLossMax
            coldMaskITM(ii, jj) = nan;
        end
        if loss0(ii, jj) <hotITMLossMin || loss0(ii, jj) >hotITMLossMax
            hotMaskITM(ii, jj) = nan;
        end

    end
end


%% choose 8 cold points distributed evenly over 360 degrees around [0,0]
% just for plotting.

anglesSample=(0:7)*2*pi/8-pi+1.e-6;
anglesSampleOM2=[anglesSample, atan2(0.16, 0.3775)]; % (add another point that turns out to be useful later.)
xidxSample=nan(size(anglesSampleOM2));
yidxSample=nan(size(anglesSampleOM2));
minLossDSample=ones(size(anglesSampleOM2));

anglesSampleITM=[anglesSample, atan2(0.33, 0.1), atan2(0.1375, -0.31)];% (add 2 other point that turn out to be useful later.)
xidxSampleITM=nan(size(anglesSampleITM));
yidxSampleITM=nan(size(anglesSampleITM));
minLossDSampleITM=ones(size(anglesSampleITM));

for ii=1:gridN
    for jj=1:gridN
        % check if the current grid is closest in angle to one of 12
        % angles.
        if ~isnan(coldMask(ii, jj))
            for kk=1:length(anglesSampleOM2)
                if abs(anglesSampleOM2(kk)-atan2(dwN(ii, jj), dpN(ii, jj))) < 2*pi/360 % within 1 degrees of the reference
                    if abs(loss0(ii, jj)-coldLoss) < minLossDSample(kk)
                        xidxSample(kk)=ii;
                        yidxSample(kk)=jj;
                        minLossDSample(kk)=abs(loss0(ii, jj)-coldLoss);
                    end
                end
            end
        end
        if ~isnan(coldMaskITM(ii, jj))
            for kk=1:length(anglesSampleITM)
                if abs(anglesSampleITM(kk)-atan2(dwN(ii, jj), dpN(ii, jj))) < 2*pi/360 % within 1 degrees of the reference
                    if abs(loss0(ii, jj)-coldITMLoss) < minLossDSampleITM(kk)
                        xidxSampleITM(kk)=ii;
                        yidxSampleITM(kk)=jj;
                        minLossDSampleITM(kk)=abs(loss0(ii, jj)-coldITMLoss);
                    end
                end
            end
        end

    end
end

%% Assume that the component position is close to nominal.
% Start from cold, back-propagate the mode falling on the OMC to the
% upstream of OM2 assuming coldROC.
% Then change OM2 ROC to hotROC.

z_OM2u=OM2Path.component('OM2').z-1e-6; % just upstream of OM2
newPath=OM2Path.duplicate();
newColdLoss0=loss0.*coldMask;
newHotLoss0=nan(size(loss0));
newHotdw=nan(size(loss0));
newHotdp=nan(size(loss0));
newMask=nan(size(loss0));


for ii=1:gridN
    for jj=1:gridN
        if ~isnan(newColdLoss0(ii, jj))
            newPath.replaceComponent('OM2', component.curvedMirror(coldROC, OM2Path.component('OM2').z, 'OM2'));
            newPath.seedq=beamq(qBeam(ii, jj)); % this is at OMC waist location.
            newPath.seedz=OM2Path.targetz;
            newq=newPath.qPropagate(z_OM2u);
            newPath.seedq=newq;
            newPath.seedz=z_OM2u;
            % newPath.seedq and seedz represent the q parameter upstream of
            % the OM2.
            
            % Now we replace OM2 with the one with hotROC.
            newPath.replaceComponent('OM2', component.curvedMirror(hotROC, OM2Path.component('OM2').z, 'OM2'));
            newHotLoss0(ii, jj)=1-newPath.targetOverlap();
            hotq=newPath.qPropagate(newPath.targetz); % propagated to the OMC waist position.
            newHotdp(ii, jj) = -hotq.waistZ; 
            % waist location relative to hot OM2. Negative waistZ would mean 
            % that the beam waist isn't reached yet at OMC waist location, 
            % therefore the beam waist is downsteadm of OMC waist, 
            % therefore dp is positive, i.e. dp = -waistZ.
            newHotdw(ii, jj)=hotq.waistSize-wOMC;
            if newHotLoss0(ii,jj) > hotLossMin && newHotLoss0(ii, jj) <hotLossMax
                newMask(ii,jj)=1;
            else
                newColdLoss0(ii, jj)=nan;
                newHotLoss0(ii, jj)=nan;
            end
        end
    end
end
newHotdpN=newHotdp/2/rayleighOMC; % normalized
newHotdwN=newHotdw/wOMC;

colddpNSampleOM2=zeros(size(anglesSampleOM2));
colddwNSampleOM2=colddpNSampleOM2;
hotdpNSampleOM2=colddpNSampleOM2;
hotdwNSampleOM2=colddpNSampleOM2;
for ii=1:length(colddpNSampleOM2)
    colddpNSampleOM2(ii)=dpN(xidxSample(ii), yidxSample(ii));
    colddwNSampleOM2(ii)=dwN(xidxSample(ii), yidxSample(ii));
    hotdpNSampleOM2(ii)=newHotdpN(xidxSample(ii), yidxSample(ii));
    hotdwNSampleOM2(ii)=newHotdwN(xidxSample(ii), yidxSample(ii));    
end

%% Assume that component positions are all nominal.
% Start from cold ITM, back-propagate the mode upstream of ITM.
% Then change ITM central heating.

z_ITMu=ITMPath.component('l_IN').z-1e-3; % just upstream of the first ITM bulk lens
z_ITMd=ITMPath.component('l_OUT').z+1e-3; % just upstream of the first ITM bulk lens
newPathITM=ITMPath.duplicate();
newPathITM.addComponent(component.lens(0, z_ITMd, 'TCS')); 
% This is a dummy for now, but it will be replaced by ITM TCS 2-pass effect later.
% Since the compensation plate is really close to the ITM, it's OK to add
% the double pass TCS effect as a single lens just upsteam or downstream of
% the ITM substrate lens.

newColdLoss0ITM=loss0.*coldMaskITM;
newHotLoss0ITM=nan(size(loss0));
newHotdwITM=nan(size(loss0));
newHotdpITM=nan(size(loss0));
newMaskITM=nan(size(loss0));


for ii=1:gridN
    for jj=1:gridN
        if ~isnan(newColdLoss0ITM(ii, jj))
            newPathITM.replaceComponent('TCS', component.lens(Inf, z_ITMd, 'TCS')); %restore the cold TCS.
            newPathITM.seedq=beamq(qBeam(ii, jj)); % this is at OMC waist location.
            newPathITM.seedz=ITMPath.targetz;
            newq=newPathITM.qPropagate(z_ITMu);
            newPathITM.seedq=newq;
            newPathITM.seedz=z_ITMu;
            % newPath.seedq and seedz represent the q parameter upstream of
            % the ITM.
            
            % Now we add hot TCS.
            newPathITM.replaceComponent('TCS', component.lens(1/2/hotITMPower, z_ITMd, 'TCS')); % a factor of 2 for 2-pass
            newHotLoss0ITM(ii, jj)=1-newPathITM.targetOverlap();
            hotq=newPathITM.qPropagate(newPathITM.targetz); % propagated to the OMC waist position.
            newHotdpITM(ii, jj) = -hotq.waistZ; 
            % waist location relative to hot OM2. Negative waistZ would mean 
            % that the beam waist isn't reached yet at OMC waist location, 
            % therefore the beam waist is downsteadm of OMC waist, 
            % therefore dp is positive, i.e. dp = -waistZ.
            newHotdwITM(ii, jj)=hotq.waistSize-wOMC;
            if newHotLoss0ITM(ii,jj) > hotITMLossMin && newHotLoss0ITM(ii, jj) <hotITMLossMax
                newMaskITM(ii,jj)=1;
            else
                newColdLoss0ITM(ii, jj)=nan;
                newHotLoss0ITM(ii, jj)=nan;
            end
        end
    end
end
newHotdpNITM=newHotdpITM/2/rayleighOMC; % normalized
newHotdwNITM=newHotdwITM/wOMC;

colddpNSampleITM=zeros(size(anglesSampleITM));
colddwNSampleITM=colddpNSampleITM;
hotdpNSampleITM=colddpNSampleITM;
hotdwNSampleITM=colddpNSampleITM;
for ii=1:length(colddpNSampleITM)
    colddpNSampleITM(ii)=dpN(xidxSampleITM(ii), yidxSampleITM(ii));
    colddwNSampleITM(ii)=dwN(xidxSampleITM(ii), yidxSampleITM(ii));
    hotdpNSampleITM(ii)=newHotdpNITM(xidxSampleITM(ii), yidxSampleITM(ii));
    hotdwNSampleITM(ii)=newHotdwNITM(xidxSampleITM(ii), yidxSampleITM(ii));    
end


%% plot things.
% 
figure
subplot 221
contourf(dpN, dwN, loss0, [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2], 'showtext','on')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
grid on
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
title('MM loss VS waist position and radius difference for the OMC', 'FontSize', 14)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
colorbar

subplot 222
surf(dpN, dwN, loss0.*coldMask, 'EdgeColor','none');
hold on
surf(dpN, dwN, loss0.*hotMask, 'EdgeColor','none');
surf(dpN, dwN, loss0.*coldMaskITM, 'EdgeColor','none');
surf(dpN, dwN, loss0.*hotMaskITM, 'EdgeColor','none');
grid on
view(2)
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
for ii=1:length(anglesSample)
    plot3(colddpNSampleOM2, colddwNSampleOM2, ones(size(colddwNSampleOM2))*0.13, 'x');
    plot3(hotdpNSampleOM2, hotdwNSampleOM2, ones(size(hotdwNSampleOM2))*0.13, 'o');
    plot3(colddpNSampleITM, colddwNSampleITM, ones(size(colddwNSampleITM))*0.13, 'x');
    plot3(hotdpNSampleITM, hotdwNSampleITM, ones(size(hotdwNSampleITM))*0.13, 'o');
end

quiver3(colddpNSampleOM2, colddwNSampleOM2, ones(size(colddwNSampleOM2))*0.13, hotdpNSampleOM2-colddpNSampleOM2, hotdwNSampleOM2-colddwNSampleOM2, zeros(size(colddwNSampleOM2)), 0, 'r')
quiver3(colddpNSampleITM, colddwNSampleITM, ones(size(colddwNSampleITM))*0.13, hotdpNSampleITM-colddpNSampleITM, hotdwNSampleITM-colddwNSampleITM, zeros(size(colddwNSampleITM)), 0, 'r')
caxis([0.05, 0.125])
colorbar

ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
title('Only showing the region close to measured loss', 'FontSize', 14)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
legend('X, OM2 Cold', 'X, OM2 Hot', 'Y, OM2 Cold/TCS OFF', 'Y, OM2 Cold/TCS center')



subplot 224
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
surf(dpN, dwN, newColdLoss0ITM, 'EdgeColor', 'none')
hold on
surf(newHotdpNITM, newHotdwNITM, newHotLoss0ITM, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, hotITMLossMin, hotITMLossMax,  coldLossMin, coldLossMax, coldITMLossMin, coldITMLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

quiver3(colddpNSampleITM([9, 10]), colddwNSampleITM([9, 10]), ones(size(colddwNSampleITM([9, 10])))*0.13, ...
    hotdpNSampleITM([9, 10])-colddpNSampleITM([9, 10]), hotdwNSampleITM([9, 10])-colddwNSampleITM([9, 10]), ...
    zeros(size(colddwNSampleITM([9, 10]))), 0, 'r')

caxis([0.05, 0.125])
colorbar
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
title('Only tiling the region where cold loss and hot loss are compatible with each other', 'FontSize', 14)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)

%% plot the same thing as the (2,2) panel of the previous plot
figure(2000)
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
surf(dpN, dwN, newColdLoss0ITM, 'EdgeColor', 'none')
hold on
surf(newHotdpNITM, newHotdwNITM, newHotLoss0ITM, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, hotITMLossMin, hotITMLossMax,  coldLossMin, coldLossMax, coldITMLossMin, coldITMLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

quiver3(colddpNSampleITM([9, 10]), colddwNSampleITM([9, 10]), ones(size(colddwNSampleITM([9, 10])))*0.13, ...
    hotdpNSampleITM([9, 10])-colddpNSampleITM([9, 10]), hotdwNSampleITM([9, 10])-colddwNSampleITM([9, 10]), ...
    zeros(size(colddwNSampleITM([9, 10]))), 0, 'r')

caxis([0.05, 0.125])
colorbar
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
title('Only tiling the region where cold loss and hot loss are compatible with each other', 'FontSize', 14)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
legend('X, OM2 Cold', 'X, OM2 Hot', 'Y, OM2 Cold/TCS OFF', 'Y, OM2 Cold/TCS center', 'Location', 'SouthEast')



%%
% sanity check
% start with e.g. dz0N=0, dw0N=+0.4225, corresponding to ~11.5% MM loss
anotherPathCold=OM2Path.duplicate();
anotherPathCold.replaceComponent('OM2', component.curvedMirror(coldROC, OM2Path.component('OM2').z, 'OM2cold'));
anotherw0Cold=wOMC*(1+0.4225);

anotherPathColdq=beamq.beamWaistAndZ(anotherw0Cold, 0); % this is at OMC waist location.
anotherPathCold.seedq=anotherPathColdq;
anotherPathCold.seedz=pOMC;

newq=anotherPathCold.qPropagate(z_OM2u);
anotherPathCold.seedq=newq;
anotherPathCold.seedz=z_OM2u;
 
anotherPathHot=anotherPathCold.duplicate();
anotherPathHot.replaceComponent('OM2', component.curvedMirror(hotROC, OM2Path.component('OM2').z, 'OM2hot'))

figure
anotherPathCold.plotSummary;
subplot 211
title(['BeamPath Summary: OM2 Cold, beam too big (wOMC=490um) but pos is perfect, overlap=', num2str(anotherPathCold.targetOverlap)])
figure
anotherPathHot.plotSummary;
title(['BeamPath Summary: OM2 Hot, beam got closer (wOMC=490um), pos downstream of OMC but still good, overlap=', num2str(anotherPathHot.targetOverlap)])

%% Get the single bounce beamPath object containing ITM (as a lens-mirror-lens combo), BS, SRs, OMs and OMC.
singleBouncePath=H1SingleBounce_SRC_OMC_beamPath();
%note that OM1 position is still zero in this model.
disp(['MM loss of single bounce VS OMC w/o no thermal lensing = ', num2str(1-singleBouncePath.targetOverlap())]);

figure
singleBouncePath.plotSummary();
zzz=-1:0.003:3;
figure
singleBouncePath.plotSummary(zzz);

optics={'SR3', 'SR2', 'SRM', 'OM1', 'OM2'};
colors='crbgm';
gouyDiff=zeros(size(optics));

% let's plot 2*gouySeparation. A factor of 2 as this is about MM, not
% misalignment.
figure
for ii=1:length(optics)
    gouyDiff(ii)=singleBouncePath.gouySeparation('ITM(avg)', optics{ii})*pi/180;
    plot([0, cos(2*gouyDiff(ii))], [0, sin(2*gouyDiff(ii))], ['o-', colors(ii)], 'LineWidth', 3);
    hold on;
end
thetas=(0:360)*pi/180;
plot(cos(thetas), sin(thetas), 'k-');
thetas=(0:15:345)*pi/180;
for ii=1:length(thetas)
    plot([0, cos(thetas(ii))], [0, sin(thetas(ii))], ['--k']);
end
legend(optics{:}, '');
title('2*GouySeparation between ITM and other optics')
xlabel('cos(2*Gouy)')
ylabel('cos(2*Gouy)')


%% Next, start with a nominal single bounce path.
% Assume that the input beam is somehow wrong (ROC etc. are all nominal), and
% assume that the beam parameter at the OMC waist position is represented by
% either of the hot patches, i.e. 
% (dpN, dwN) = (hotdpNSample(8), hotdwNSample(8)) for the left half plane
% OM2 scan
% or
% (dpN, dwN) = (hotdpNSample(9), hotdwNSample(9)) for the right,
% or
% (dpN, dwN) = (hotdpNSampleITM(9), hotdwNSampleITM(9)) for the left half
% plane ITMY TCS
% or
% (dpN, dwN) = (hotdpNSampleITM(9), hotdwNSampleITM(9)) for the right.
%
% Change ITM central heating and see what happens.

%first, plot the patches.
figure(100)
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
surf(dpN, dwN, newColdLoss0ITM, 'EdgeColor', 'none')
surf(newHotdpNITM, newHotdwNITM, newHotLoss0ITM, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, hotITMLossMin, hotITMLossMax,  coldLossMin, coldLossMax, coldITMLossMin, coldITMLossMax], '--', 'showtext','off')

xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
%quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
%    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
%    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

caxis([0.05, 0.125])
colorbar

% next, define the beam path corresponding to the hot patch (either in the 
% left or right hand half plane), change the thermal lensing and see how
% the beam parameter at the OMC waist position moves.
for kk=8:9 % Start with OM2 scan data, 8= left half plane, 9= right.
    dpNStart=hotdpNSampleOM2(kk);
    dwNStart=hotdwNSampleOM2(kk);
    dpStart=dpNStart*2*rayleighOMC;
    dwStart=dwNStart*wOMC;
    wBeamStart=dwStart+wOMC;
    rayleighBeamStart=pi/1064e-9*(wBeamStart.^2);
    qBeamStart=1.0i*rayleighBeamStart - dpStart; % dp positive -> waist is downstream of OMC waist -> still hasn't reached the waist at OMC waist position

    hotStartPath=singleBouncePath.duplicate(); %right-hand patch Path
    hotStartPath.seedq=beamq(qBeamStart); % seed beam is in the middle of the hot right hand plane patch.
    hotStartPath.seedz=hotStartPath.targetz; % at the OMC waist position.

    %propagate the seed to upstream of ITM
    newz=hotStartPath.component('ITM(avg)').z-1; %1m upstream of ITM
    newq=hotStartPath.qPropagate(newz);
    hotStartPath.seedq=newq;
    hotStartPath.seedz=newz;

    % hotStartPath has nominally placed components with the wrong beam, and this
    % path represents the hot patch (either left or right hand plane.
    % Change the thermal lens in the ITM and see what happens.
    pITM=hotStartPath.component('ITM(avg)').z; % ITM pos
    dThermal=(-10:10)*1e-5; % diopters, equivalent of up to 10km thermal lensing.
    lossThermal=zeros(size(dThermal));
    dpThermal = zeros(size(lossThermal)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
    dwThermal = zeros(size(lossThermal)); % waist radius difference, positive = beam waist is bigger.

    for ii=1:length(dThermal)
        thermalPath=hotStartPath.duplicate();
        thermalPath.addComponent(component.lens(1/dThermal(ii), pITM+1e-7, 'tl'));
        thermalPath.addComponent(component.lens(1/dThermal(ii), pITM-1e-7, 'tl'));% sincde this is single bounce we need to add two thermal lens elements.
        lossThermal(ii)=1-thermalPath.targetOverlap;
        thermalq=thermalPath.qPropagate(pOMC);
        dpThermal(ii)=-real(thermalq.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
        dwThermal(ii)=thermalq.waistSize-wOMC;
        if ii==11
            figure(100+kk)
            thermalPath.plotSummary();
            subplot 211
            if kk==8 % (left half plane)
                title('back propagating from the hot patch in the left half plane')
            else
                title('back propagating from the hot patch in the right half plane')
            end
        end
    end

    dpNThermal=dpThermal/2/rayleighOMC;
    dwNThermal=dwThermal/wOMC;

    %plot [dpNThermal, dwNThermal] on top of the patches
    figure(100)
    plot3(dpNThermal(1:10), dwNThermal(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
    plot3(dpNThermal(11), dwNThermal(11), 0.15, 'og', 'LineWidth', 2)
    plot3(dpNThermal(12:21), dwNThermal(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)

    figure(101)
    plot(dThermal, lossThermal)
    hold on
end

for kk=9:10 % ITM scan data, ITM cold patch.
    dpNStart=hotdpNSampleITM(kk);
    dwNStart=hotdwNSampleITM(kk);
    dpStart=dpNStart*2*rayleighOMC;
    dwStart=dwNStart*wOMC;
    wBeamStart=dwStart+wOMC;
    rayleighBeamStart=pi/1064e-9*(wBeamStart.^2);
    qBeamStart=1.0i*rayleighBeamStart - dpStart; % dp positive -> waist is downstream of OMC waist -> still hasn't reached the waist at OMC waist position

    hotStartPath=singleBouncePath.duplicate(); %right-hand patch Path
    hotStartPath.seedq=beamq(qBeamStart); % seed beam is in the middle of the hot right hand plane patch.
    hotStartPath.seedz=hotStartPath.targetz; % at the OMC waist position.

    %propagate the seed to upstream of ITM
    newz=hotStartPath.component('ITM(avg)').z-1; %1m upstream of ITM
    newq=hotStartPath.qPropagate(newz);
    hotStartPath.seedq=newq;
    hotStartPath.seedz=newz;

    % hotStartPath has nominally placed components with the wrong beam, and this
    % path represents the hot patch (either left or right hand plane.
    % Change the thermal lens in the ITM and see what happens.
    pITM=hotStartPath.component('ITM(avg)').z; % ITM pos
    dThermal=(-10:10)*1e-5; % diopters, equivalent of up to 10km thermal lensing.
    lossThermal=zeros(size(dThermal));
    dpThermal = zeros(size(lossThermal)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
    dwThermal = zeros(size(lossThermal)); % waist radius difference, positive = beam waist is bigger.

    for ii=1:length(dThermal)
        thermalPath=hotStartPath.duplicate();
        thermalPath.addComponent(component.lens(1/dThermal(ii), pITM+1e-7, 'tl'));
        thermalPath.addComponent(component.lens(1/dThermal(ii), pITM-1e-7, 'tl'));% sincde this is single bounce we need to add two thermal lens elements.
        lossThermal(ii)=1-thermalPath.targetOverlap;
        thermalq=thermalPath.qPropagate(pOMC);
        dpThermal(ii)=-real(thermalq.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
        dwThermal(ii)=thermalq.waistSize-wOMC;
    end

    dpNThermal=dpThermal/2/rayleighOMC;
    dwNThermal=dwThermal/wOMC;

    %plot [dpNThermal, dwNThermal] on top of the patches
    figure(100)
    plot3(dpNThermal(1:10), dwNThermal(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
    plot3(dpNThermal(11), dwNThermal(11), 0.15, 'og', 'LineWidth', 2)
    plot3(dpNThermal(12:21), dwNThermal(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)

    figure(101)
    plot(dThermal, lossThermal)
    hold on
end

% figure(100)
%ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'fontsize', 12)
%xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
%title({'Changing ITM thermal lensing in 1E-5 diopter steps', 'green=zero (hot OM2), blue=negative, red=positive'}, 'fontsize', 14);

figure(101)
title('loss VS ITM central heating (for hot OM2)', 'FontSize', 14)
xlabel('diopter (single path)', 'FontSize', 12)
ylabel('loss', 'FontSize', 12)
xlim([-2e-5, 5e-5])
grid on

% Do the same thing but starting with cold patches.
% Change ITM central heating and see what happens.
% Define the beam path corresponding to the cold patch (either in the 
% left or right hand half plane), change the thermal lensing and see how
% the beam parameter at the OMC waist position moves.
for kk=8:9 % 8= left half plane, 9= right.
    dpNStart=colddpNSampleOM2(kk);
    dwNStart=colddwNSampleOM2(kk);
    dpStart=dpNStart*2*rayleighOMC;
    dwStart=dwNStart*wOMC;
    wBeamStart=dwStart+wOMC;
    rayleighBeamStart=pi/1064e-9*(wBeamStart.^2);
    qBeamStart=1.0i*rayleighBeamStart - dpStart; % dp positive -> waist is downstream of OMC waist -> still hasn't reached the waist at OMC waist position

    coldStartPath=singleBouncePath.duplicate(); %right-hand patch Path
    coldStartPath.seedq=beamq(qBeamStart); % seed beam is in the middle of the hot right hand plane patch.
    coldStartPath.seedz=coldStartPath.targetz; % at the OMC waist position.

    %propagate the seed to upstream of ITM
    newz=coldStartPath.component('ITM(avg)').z-1; %1m upstream of ITM
    newq=coldStartPath.qPropagate(newz);
    coldStartPath.seedq=newq;
    coldStartPath.seedz=newz;

    % coldStartPath has nominally placed components with the wrong beam, and this
    % path represents the cold patch (either left or right hand plane.
    % Change the thermal lens in the ITM and see what happens.
    pITM=coldStartPath.component('ITM(avg)').z; % ITM pos
    dThermal=(-10:10)*1e-5; % diopters, equivalent of up to 10km thermal lensing.
    lossThermal=zeros(size(dThermal));
    dpThermal = zeros(size(lossThermal)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
    dwThermal = zeros(size(lossThermal)); % waist radius difference, positive = beam waist is bigger.

    for ii=1:length(dThermal)
        thermalPath=coldStartPath.duplicate();
        thermalPath.addComponent(component.lens(1/dThermal(ii), pITM+1e-7, 'tl'));
        thermalPath.addComponent(component.lens(1/dThermal(ii), pITM-1e-7, 'tl'));% sincde this is single bounce we need to add two thermal lens elements.
        lossThermal(ii)=1-thermalPath.targetOverlap;
        thermalq=thermalPath.qPropagate(pOMC);
        dpThermal(ii)=-real(thermalq.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
        dwThermal(ii)=thermalq.waistSize-wOMC;
        if ii==11
            figure(200+kk)
            thermalPath.plotSummary();
            subplot 211
            if kk==8 % (left half plane)
                title('back propagating from the cold patch in the left half plane')
            else
                title('back propagating from the cold patch in the right half plane')
            end
        end
    end

    dpNThermal=dpThermal/2/rayleighOMC;
    dwNThermal=dwThermal/wOMC;

    %plot [dpNThermal, dwNThermal] on top of the patches
    figure(100)
    plot3(dpNThermal(1:10), dwNThermal(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
    plot3(dpNThermal(11), dwNThermal(11), 0.15, 'og', 'LineWidth', 2)
    plot3(dpNThermal(12:21), dwNThermal(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)

    figure(102)
    plot(dThermal, lossThermal)
    hold on
end
figure(100)
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'fontsize', 12)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
title({'Changing ITM thermal lensing in 1E-5 diopter (single path) steps', 'green=zero, blue=negative, red=positive'}, 'fontsize', 14);
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])

figure(102)
title('loss VS ITM central heating (for cold OM2)', 'FontSize', 14)
xlabel('diopter (single path)', 'FontSize', 12)
ylabel('loss', 'FontSize', 12)
xlim([-2e-5, 5e-5])
grid on


%%%%
%% Do the same thing, but change OM2 heating instead of ITM.

%first, plot the patches.
figure(1000)
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
surf(dpN, dwN, newColdLoss0ITM, 'EdgeColor', 'none')
surf(newHotdpNITM, newHotdwNITM, newHotLoss0ITM, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, hotITMLossMin, hotITMLossMax,  coldLossMin, coldLossMax, coldITMLossMin, coldITMLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
%quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
%    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
%    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

caxis([0.05, 0.125])
colorbar

% next, define the beam path corresponding to the hot patch (either in the 
% left or right hand half plane), change the thermal lensing and see how
% the beam parameter at the OMC waist position moves.
for kk=8:9 % Start with OM2 scan data, 8= left half plane, 9= right.
    dpNStart=hotdpNSampleOM2(kk);
    dwNStart=hotdwNSampleOM2(kk);
    dpStart=dpNStart*2*rayleighOMC;
    dwStart=dwNStart*wOMC;
    wBeamStart=dwStart+wOMC;
    rayleighBeamStart=pi/1064e-9*(wBeamStart.^2);
    qBeamStart=1.0i*rayleighBeamStart - dpStart; % dp positive -> waist is downstream of OMC waist -> still hasn't reached the waist at OMC waist position

    hotStartPath=singleBouncePath.duplicate(); %right-hand patch Path
    hotStartPath.seedq=beamq(qBeamStart); % seed beam is in the middle of the hot right hand plane patch.
    hotStartPath.seedz=hotStartPath.targetz; % at the OMC waist position.

    %propagate the seed to upstream of ITM
    newz=hotStartPath.component('ITM(avg)').z-1; %1m upstream of ITM
    newq=hotStartPath.qPropagate(newz);
    hotStartPath.seedq=newq;
    hotStartPath.seedz=newz;

    % hotStartPath has nominally placed components with the wrong beam, and this
    % path represents the hot patch (either left or right hand plane.
    % Change the thermal lens in the ITM and see what happens.
    pOM2=hotStartPath.component('OM2').z; % OM2 pos
    dThermal=(-10:10)*dOM2Range; % diopters, equivalent of up to the range of the OM2.
    lossThermal=zeros(size(dThermal));
    dpThermal = zeros(size(lossThermal)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
    dwThermal = zeros(size(lossThermal)); % waist radius difference, positive = beam waist is bigger.

    for ii=1:length(dThermal)
        thermalPath=hotStartPath.duplicate();
        thermalPath.addComponent(component.lens(1/dThermal(ii), pOM2+1e-7, 'thermal'));
        lossThermal(ii)=1-thermalPath.targetOverlap;
        thermalq=thermalPath.qPropagate(pOMC);
        dpThermal(ii)=-real(thermalq.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
        dwThermal(ii)=thermalq.waistSize-wOMC;
    end

    dpNThermal=dpThermal/2/rayleighOMC;
    dwNThermal=dwThermal/wOMC;

    %plot [dpNThermal, dwNThermal] on top of the patches
    figure(1000)
    plot3(dpNThermal(1:10), dwNThermal(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
    plot3(dpNThermal(11), dwNThermal(11), 0.15, 'og', 'LineWidth', 2)
    plot3(dpNThermal(12:21), dwNThermal(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)

end

%hand-selected starting points for ITM scan data
dpNArbitraryStart=[0.105, -0.305, 0.18311];
dwNArbitraryStart=[0.3275, 0.125, 0.0932];
for kk=1:3 % ITM scan data, ITM cold patch.
    dpNStart=dpNArbitraryStart(kk);
    dwNStart=dwNArbitraryStart(kk);
    dpStart=dpNStart*2*rayleighOMC;
    dwStart=dwNStart*wOMC;
    wBeamStart=dwStart+wOMC;
    rayleighBeamStart=pi/1064e-9*(wBeamStart.^2);
    qBeamStart=1.0i*rayleighBeamStart - dpStart; % dp positive -> waist is downstream of OMC waist -> still hasn't reached the waist at OMC waist position

    hotStartPath=singleBouncePath.duplicate(); %right-hand patch Path
    hotStartPath.seedq=beamq(qBeamStart); % seed beam is in the middle of the hot right hand plane patch.
    hotStartPath.seedz=hotStartPath.targetz; % at the OMC waist position.

    %propagate the seed to upstream of ITM
    newz=hotStartPath.component('ITM(avg)').z-1; %1m upstream of ITM
    newq=hotStartPath.qPropagate(newz);
    hotStartPath.seedq=newq;
    hotStartPath.seedz=newz;

    % hotStartPath has nominally placed components with the wrong beam, and this
    % path represents the hot patch (either left or right hand plane.
    % Change the thermal lens in the ITM and see what happens.
    pOM2=hotStartPath.component('OM2').z; % ITM pos
    dThermal=(-10:10)*dOM2Range; % diopters, equivalent of up to 10km thermal lensing.
    lossThermal=zeros(size(dThermal));
    dpThermal = zeros(size(lossThermal)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
    dwThermal = zeros(size(lossThermal)); % waist radius difference, positive = beam waist is bigger.

    for ii=1:length(dThermal)
        thermalPath=hotStartPath.duplicate();
        thermalPath.addComponent(component.lens(1/dThermal(ii), pOM2+1e-7, 'thermal'));
        lossThermal(ii)=1-thermalPath.targetOverlap;
        thermalq=thermalPath.qPropagate(pOMC);
        dpThermal(ii)=-real(thermalq.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
        dwThermal(ii)=thermalq.waistSize-wOMC;
    end

    dpNThermal=dpThermal/2/rayleighOMC;
    dwNThermal=dwThermal/wOMC;

    %plot [dpNThermal, dwNThermal] on top of the patches
    figure(1000)
    plot3(dpNThermal(1:10), dwNThermal(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
    plot3(dpNThermal(11), dwNThermal(11), 0.15, 'og', 'LineWidth', 2)
    plot3(dpNThermal(12:21), dwNThermal(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)

end

figure(1000)
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'fontsize', 12)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
title({'Changing OM2 actuation 1 step=maximum OM2 range', 'green=zero, blue=negative, red=positive'}, 'fontsize', 14);



%% Next, we'll start from a good beam and change a bunch of parameters to see what happens.
%% add extreme thermal lensing for ITM and see what happens
pITM=singleBouncePath.component('ITM(avg)').z; % ITM pos
dThermal=(-10:10)*1e-5; % diopters, equivalent of up to 10km thermal lensing.
lossThermal=zeros(size(dThermal));
dpThermal = zeros(size(lossThermal)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
dwThermal = zeros(size(lossThermal)); % waist radius difference, positive = beam waist is bigger.

for ii=1:length(dThermal)
    thermalPath=singleBouncePath.duplicate();
    thermalPath.addComponent(component.lens(1/dThermal(ii), pITM+1e-7, 'tl'));
    thermalPath.addComponent(component.lens(1/dThermal(ii), pITM-1e-7, 'tl'));% since this is single bounce we need to add two thermal lens elements.
    lossThermal(ii)=1-thermalPath.targetOverlap;
    thermalq=thermalPath.qPropagate(pOMC);
    dpThermal(ii)=-real(thermalq.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
    dwThermal(ii)=thermalq.waistSize-wOMC;
end

dpNThermal=dpThermal/2/rayleighOMC;
dwNThermal=dwThermal/wOMC;

%plot [dpNThermal, dwNThermal] on top of the patches
figure
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, coldLossMin, coldLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

caxis([0.05, 0.125])
colorbar
plot3(dpNThermal(1:10), dwNThermal(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
plot3(dpNThermal(11), dwNThermal(11), 0.15, 'og', 'LineWidth', 2)
plot3(dpNThermal(12:21), dwNThermal(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
title({'Nominal input, changing ITM thermal lensing in 1E-5 diopter steps', 'green=zero, blue=negative, red=positive'}, 'FontSize', 14);

%% Do the same thing but on SR3 ROC
disp(['2x(Gouy phase separation: ITM-SR3) = ', num2str(2*singleBouncePath.gouySeparation('ITM(avg)', 'SR3', 'nowrap')), ' deg'])
pSR3=singleBouncePath.component('SR3').z; % ITM pos
rocSR3=(1+(-10:10)/4000)*singleBouncePath.component('SR3').parameters.ROC; % roc, up to +-0.25% of nominal, 0.025% step.
lossSR3=zeros(size(rocSR3));
dpSR3 = zeros(size(lossSR3)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
dwSR3 = zeros(size(lossSR3)); % waist radius difference, positive = beam waist is bigger.

for ii=1:length(rocSR3)
    sr3Path=singleBouncePath.duplicate();
    newSR3=component.curvedMirror(rocSR3(ii), pSR3, 'SR3');
    sr3Path.replaceComponent('SR3', newSR3);
    lossSR3(ii)=1-sr3Path.targetOverlap;
    sr3q=sr3Path.qPropagate(pOMC);
    dpSR3(ii)=-real(sr3q.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
    dwSR3(ii)=sr3q.waistSize-wOMC;
end

dpNSR3=dpSR3/2/rayleighOMC;
dwNSR3=dwSR3/wOMC;

%plot [dpNThermal, dwNThermal] on top of the patches
figure
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, coldLossMin, coldLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

caxis([0.05, 0.125])
colorbar
plot3(dpNSR3(1:10), dwNSR3(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
plot3(dpNSR3(11), dwNSR3(11), 0.15, 'og', 'LineWidth', 2)
plot3(dpNSR3(12:21), dwNSR3(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
title({'Nominal input, changing SR3 ROC by 0.025% steps', 'green=nominal, blue=ROC smaller, red=ROC larger'}, 'FontSize', 14);
%this is basically the same as ITM thermal, not surprising as gouy
%difference is almost zero.

%% Do the same thing but on SR2 ROC
disp(['2x(Gouy phase separation: ITM-SR2) = ', num2str(2*singleBouncePath.gouySeparation('ITM(avg)', 'SR2', 'nowrap')), ' deg'])
pSR2=singleBouncePath.component('SR2').z; % ITM pos
rocSR2=(1+(-10:10)/1000)*singleBouncePath.component('SR2').parameters.ROC; % roc, up to +-1% of nominal, 0.1% step.
lossSR2=zeros(size(rocSR2));
dpSR2 = zeros(size(lossSR2)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
dwSR2 = zeros(size(lossSR2)); % waist radius difference, positive = beam waist is bigger.

for ii=1:length(rocSR2)
    sr2Path=singleBouncePath.duplicate();
    newSR2=component.curvedMirror(rocSR2(ii), pSR2, 'SR2');
    sr2Path.replaceComponent('SR2', newSR2);
    lossSR2(ii)=1-sr2Path.targetOverlap;
    sr2q=sr2Path.qPropagate(pOMC);
    dpSR2(ii)=-real(sr2q.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
    dwSR2(ii)=sr2q.waistSize-wOMC;
end

dpNSR2=dpSR2/2/rayleighOMC;
dwNSR2=dwSR2/wOMC;

%plot [dpNThermal, dwNThermal] on top of the patches
figure
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, coldLossMin, coldLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

caxis([0.05, 0.125])
colorbar
plot3(dpNSR2(1:10), dwNSR2(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
plot3(dpNSR2(11), dwNSR2(11), 0.15, 'og', 'LineWidth', 2)
plot3(dpNSR2(12:21), dwNSR2(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
title({'Nominal input, changing SR2 ROC by 0.1% steps', 'green=nominal, blue=ROC smaller, red=ROC larger'}, 'FontSize', 14);
%this is basically the same as ITM thermal, not surprising as gouy
%difference is almost zero.


%% Changing SRM ROC.
disp(['2x(Gouy phase separation: ITM-SRM) = ', num2str(2*singleBouncePath.gouySeparation('ITM(avg)', 'SRM', 'nowrap')), ' deg'])
pSRM=singleBouncePath.component('SRM').z; % SRM pos
fSRM=(1+(-10:10)/20)*singleBouncePath.component('SRM').parameters.focalLength; % roc, up to +-50% of nominal, 5% step.
lossSRM=zeros(size(fSRM));
dpSRM = zeros(size(fSRM)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
dwSRM = zeros(size(fSRM)); % waist radius difference, positive = beam waist is bigger.

for ii=1:length(fSRM)
    srmPath=singleBouncePath.duplicate();
    newSRM=component.lens(fSRM(ii), pSRM, 'SRM');
    srmPath.replaceComponent('SRM', newSRM);
    lossSRM(ii)=1-srmPath.targetOverlap;
    srmq=srmPath.qPropagate(pOMC);
    dpSRM(ii)=-real(srmq.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
    dwSRM(ii)=srmq.waistSize-wOMC;
end

dpNSRM=dpSRM/2/rayleighOMC;
dwNSRM=dwSRM/wOMC;

%plot [dpNThermal, dwNThermal] on top of the patches
figure
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, coldLossMin, coldLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

caxis([0.05, 0.125])
colorbar
plot3(dpNSRM(1:10), dwNSRM(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
plot3(dpNSRM(11), dwNSRM(11), 0.15, 'og', 'LineWidth', 2)
plot3(dpNSRM(12:21), dwNSRM(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
title({'Nominal input, changing SRM ROC by 5% steps', 'green=nominal, blue=ROC smaller, red=ROC larger'}, 'FontSize', 14);


%% Move on to OM1 ROC.
disp(['2x(Gouy phase separation: ITM-OM1) = ', num2str(2*singleBouncePath.gouySeparation('ITM(avg)', 'OM1', 'nowrap')), ' deg'])
pOM1=singleBouncePath.component('OM1').z; % OM1 pos
rocOM1=(1+(-10:10)/20)*singleBouncePath.component('OM1').parameters.ROC; % roc, up to +-50% of nominal, 5% step.
lossOM1=zeros(size(rocOM1));
dpOM1 = zeros(size(rocOM1)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
dwOM1 = zeros(size(rocOM1)); % waist radius difference, positive = beam waist is bigger.

for ii=1:length(rocOM1)
    om1Path=singleBouncePath.duplicate();
    newOM1=component.curvedMirror(rocOM1(ii), pOM1, 'OM1');
    om1Path.replaceComponent('OM1', newOM1);
    lossOM1(ii)=1-om1Path.targetOverlap;
    om1q=om1Path.qPropagate(pOMC);
    dpOM1(ii)=-real(om1q.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
    dwOM1(ii)=om1q.waistSize-wOMC;
end

dpNOM1=dpOM1/2/rayleighOMC;
dwNOM1=dwOM1/wOMC;

%plot [dpNThermal, dwNThermal] on top of the patches
figure
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, coldLossMin, coldLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

caxis([0.05, 0.125])
colorbar
plot3(dpNOM1(1:10), dwNOM1(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
plot3(dpNOM1(11), dwNOM1(11), 0.15, 'og', 'LineWidth', 2)
plot3(dpNOM1(12:21), dwNOM1(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
title({'Nominal input, changing OM1 ROC by 5% steps', 'green=nominal, blue=ROC smaller, red=ROC larger'}, 'FontSize', 14);


%% Move on to OM2 ROC.
disp(['2x(Gouy phase separation: ITM-OM2) = ', num2str(2*singleBouncePath.gouySeparation('ITM(avg)', 'OM2', 'nowrap')), ' deg'])
pOM2=singleBouncePath.component('OM2').z; % OM1 pos
rocOM2=(1+(-10:10)/20)*singleBouncePath.component('OM2').parameters.ROC; % roc, up to +-50% of nominal, 5% step.
lossOM2=zeros(size(rocOM2));
dpOM2 = zeros(size(rocOM2)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
dwOM2 = zeros(size(rocOM2)); % waist radius difference, positive = beam waist is bigger.

for ii=1:length(rocOM2)
    om2Path=singleBouncePath.duplicate();
    newOM2=component.curvedMirror(rocOM2(ii), pOM2, 'OM2');
    om2Path.replaceComponent('OM2', newOM2);
    lossOM2(ii)=1-om2Path.targetOverlap;
    om2q=om2Path.qPropagate(pOMC);
    dpOM2(ii)=-real(om2q.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
    dwOM2(ii)=om2q.waistSize-wOMC;
end

dpNOM2=dpOM2/2/rayleighOMC;
dwNOM2=dwOM2/wOMC;

%plot [dpNThermal, dwNThermal] on top of the patches
figure
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, coldLossMin, coldLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

caxis([0.05, 0.125])
colorbar
plot3(dpNOM2(1:10), dwNOM2(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
plot3(dpNOM2(11), dwNOM2(11), 0.15, 'og', 'LineWidth', 2)
plot3(dpNOM2(12:21), dwNOM2(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
title({'Nominal input, changing OM2 ROC by 5% steps', 'green=nominal, blue=ROC smaller, red=ROC larger'}, 'FontSize', 14);

%%%%%%%%%%%%%%%%
%% Now we change the distances between components. I'll start with SR3.
%%%%%%%%%%%%%%%%
dz=(-10:10)/100; %1cm step, +-10cm

lossSR3=zeros(size(dz));
dpSR3 = zeros(size(lossSR3)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
dwSR3 = zeros(size(lossSR3)); % waist radius difference, positive = beam waist is bigger.

for ii=1:length(dz)
    sr3Path=singleBouncePath.duplicate();
    sr3Path.moveComponent('SR3', dz(ii));
    optics={'SR2', 'SRM', 'OM1', 'OM2', 'OMC waist'};
    for jj=1:length(optics)
        sr3Path.moveComponent(optics{jj}, 2*dz(ii)); 
        % if you move SR3, ITM-SR3 and SR3-SR2 separation changes by the same amount
        % while the separation between downstream optics won't change.
    end
    sr3Path.targetz=sr3Path.targetz+2*dz(ii); % moving components won't automatically move the target q.

    lossSR3(ii)=1-sr3Path.targetOverlap;
    sr3q=sr3Path.qPropagate(sr3Path.targetz);
    dpSR3(ii)=-real(sr3q.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
    dwSR3(ii)=sr3q.waistSize-wOMC;
end

dpNSR3=dpSR3/2/rayleighOMC;
dwNSR3=dwSR3/wOMC;

%plot [dpNThermal, dwNThermal] on top of the patches
figure
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, coldLossMin, coldLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

caxis([0.05, 0.125])
colorbar
plot3(dpNSR3(1:10), dwNSR3(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
plot3(dpNSR3(11), dwNSR3(11), 0.15, 'og', 'LineWidth', 2)
plot3(dpNSR3(12:21), dwNSR3(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
title({'Nominal input, moving SR3 by 1cm steps', 'green=nominal, blue=closer to ITM/SR2, red=ROC farther'}, 'FontSize', 14);

%% Move SR2 pos
dz=(-10:10)/100; %1cm step, +-10cm

lossSR2=zeros(size(dz));
dpSR2 = zeros(size(lossSR2)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
dwSR2 = zeros(size(lossSR2)); % waist radius difference, positive = beam waist is bigger.

for ii=1:length(dz)
    sr2Path=singleBouncePath.duplicate();
    sr2Path.moveComponent('SR2', dz(ii));
    optics={'SRM', 'OM1', 'OM2', 'OMC waist'};
    for jj=1:length(optics)
        sr2Path.moveComponent(optics{jj}, 2*dz(ii)); 
        % if you move SR3, ITM-SR3 and SR3-SR2 separation changes by the same amount
        % while the separation between downstream optics won't change.
    end
    sr2Path.targetz=sr2Path.targetz+2*dz(ii); % moving components won't automatically move the target q.

    lossSR2(ii)=1-sr2Path.targetOverlap;
    sr2q=sr2Path.qPropagate(sr2Path.targetz);
    dpSR2(ii)=-real(sr2q.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
    dwSR2(ii)=sr2q.waistSize-wOMC;
end

dpNSR2=dpSR2/2/rayleighOMC;
dwNSR2=dwSR2/wOMC;

%plot [dpNThermal, dwNThermal] on top of the patches
figure
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, coldLossMin, coldLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

caxis([0.05, 0.125])
colorbar
plot3(dpNSR2(1:10), dwNSR2(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
plot3(dpNSR2(11), dwNSR2(11), 0.15, 'og', 'LineWidth', 2)
plot3(dpNSR2(12:21), dwNSR2(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
title({'Nominal input, moving SR2 by 1cm steps', 'green=nominal, blue=closer to SRM/SR3, red=farther'}, 'FontSize', 14);

%% Move SRM pos
% unlike SR3 and SR2, I assumed that moving SRM futher from SR2 would make
% SRM closer to OM1.
dz=(-10:10)/20; %5cm step, +-50cm

lossSRM=zeros(size(dz));
dpSRM = zeros(size(lossSRM)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
dwSRM = zeros(size(lossSRM)); % waist radius difference, positive = beam waist is bigger.

for ii=1:length(dz)
    srmPath=singleBouncePath.duplicate();
    srmPath.moveComponent('SRM', dz(ii));
    % SRM moves but SR2-OM1-OM2-OMC distances won't change, so there's no need to move 
    % targetz either.

    lossSRM(ii)=1-srmPath.targetOverlap;
    srmq=srmPath.qPropagate(srmPath.targetz);
    dpSRM(ii)=-real(srmq.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
    dwSRM(ii)=srmq.waistSize-wOMC;
end

dpNSRM=dpSRM/2/rayleighOMC;
dwNSRM=dwSRM/wOMC;

%plot [dpNThermal, dwNThermal] on top of the patches
figure
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, coldLossMin, coldLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

caxis([0.05, 0.125])
colorbar
plot3(dpNSRM(1:10), dwNSRM(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
plot3(dpNSRM(11), dwNSRM(11), 0.15, 'og', 'LineWidth', 2)
plot3(dpNSRM(12:21), dwNSRM(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
title({'Nominal input, moving SRM by 5cm steps (no change in SR2-OM1 distance)', 'green=nominal, blue=closer to SR2, red=farther'}, 'FontSize', 14);


%% Move OM1 pos
dz=(-10:10)/20; %5cm step, +-50cm

lossOM1=zeros(size(dz));
dpOM1 = zeros(size(lossOM1)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
dwOM1 = zeros(size(lossOM1)); % waist radius difference, positive = beam waist is bigger.

for ii=1:length(dz)
    om1Path=singleBouncePath.duplicate();
    om1Path.moveComponent('OM1', dz(ii));
    optics={'OM2', 'OMC waist'};
    for jj=1:length(optics)
        om1Path.moveComponent(optics{jj}, 2*dz(ii)); 
        % if you move SR3, ITM-SR3 and SR3-SR2 separation changes by the same amount
        % while the separation between downstream optics won't change.
    end
    om1Path.targetz=om1Path.targetz+2*dz(ii); % moving components won't automatically move the target q.

    lossOM1(ii)=1-om1Path.targetOverlap;
    om1q=om1Path.qPropagate(om1Path.targetz);
    dpOM1(ii)=-real(om1q.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
    dwOM1(ii)=om1q.waistSize-wOMC;
end

dpNOM1=dpOM1/2/rayleighOMC;
dwNOM1=dwOM1/wOMC;

%plot [dpNThermal, dwNThermal] on top of the patches
figure
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, coldLossMin, coldLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

caxis([0.05, 0.125])
colorbar
plot3(dpNOM1(1:10), dwNOM1(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
plot3(dpNOM1(11), dwNOM1(11), 0.15, 'og', 'LineWidth', 2)
plot3(dpNOM1(12:21), dwNOM1(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
title({'Nominal input, moving OM1 by 5cm steps', 'green=nominal, blue=closer to OM2/SRM, red=ROC farther'}, 'FontSize', 14);


%% Move OM2 pos
dz=(-10:10)/20; %5cm step, +-50cm

lossOM2=zeros(size(dz));
dpOM2 = zeros(size(lossOM2)); % waist position difference between the beam and OMC, positive = beam waist is downstream.
dwOM2 = zeros(size(lossOM2)); % waist radius difference, positive = beam waist is bigger.

for ii=1:length(dz)
    om2Path=singleBouncePath.duplicate();
    om2Path.moveComponent('OM2', dz(ii));
    om2Path.moveComponent('OMC waist', 2*dz(ii)); 
    % OM2-OM1 and OM2-OMC distances change by the same amount.

    om2Path.targetz=om2Path.targetz+2*dz(ii); % moving components won't automatically move the target q.

    lossOM2(ii)=1-om2Path.targetOverlap;
    om2q=om2Path.qPropagate(om2Path.targetz);
    dpOM2(ii)=-real(om2q.q); % real(q) is positive -> waist is upstream of OMC -> dp is negative
    dwOM2(ii)=om2q.waistSize-wOMC;
end

dpNOM2=dpOM2/2/rayleighOMC;
dwNOM2=dwOM2/wOMC;

%plot [dpNThermal, dwNThermal] on top of the patches
figure
surf(dpN, dwN, newColdLoss0, 'EdgeColor', 'none')
hold on
surf(newHotdpN, newHotdwN, newHotLoss0, 'EdgeColor', 'none')
view(2)
contour(dpN, dwN, loss0, [0, hotLossMin, hotLossMax, coldLossMin, coldLossMax], '--', 'showtext','off')
xlim([-0.5, 0.5])
ylim([-0.4, 0.5])
quiver3(colddpNSampleOM2([8, 9]), colddwNSampleOM2([8, 9]), ones(size(colddwNSampleOM2([8, 9])))*0.13, ...
    hotdpNSampleOM2([8, 9])-colddpNSampleOM2([8, 9]), hotdwNSampleOM2([8, 9])-colddwNSampleOM2([8, 9]), ...
    zeros(size(colddwNSampleOM2([8, 9]))), 0, 'r')

caxis([0.05, 0.125])
colorbar
plot3(dpNOM2(1:10), dwNOM2(1:10), ones([1,10])*0.15, 'xb', 'LineWidth', 2)
plot3(dpNOM2(11), dwNOM2(11), 0.15, 'og', 'LineWidth', 2)
plot3(dpNOM2(12:21), dwNOM2(12:21), ones([1,10])*0.15, 'xr', 'LineWidth', 2)
ylabel(['Normalized waist radius difference, w0(OMC)~', num2str(wOMC*1e6, 3), 'um'], 'FontSize', 12)
xlabel(['Normalized waist position difference, 2*rayleigh(OMC)~', num2str(2*rayleighOMC,3), '[m]'  ], 'FontSize', 12)
title({'Nominal input, moving OM2 by 5cm steps', 'green=nominal, blue=closer to OM1/OMC, red=farther'}, 'FontSize', 14);

