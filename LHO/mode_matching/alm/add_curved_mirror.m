function [path_v, path_h] = add_curved_mirror(path_v, path_h, ROC, AOI, Z, label)
    %adds a  curved mirror to vertical path (path_v) and horizontal path
    %AOI in degrees, ROC in meters, Z is location of optic, can be
    %different for vertical and horizontal (because there is a dielectric
    %at an AOI upstream
    if size(Z)==1
        Z_v = Z;
        Z_h = Z;
    else
        Z_v = Z(1);
        Z_h = Z(2);
    end
    path_v.addComponent(component.curvedMirror(ROC/cos(AOI*pi/180), Z_v, label));
    %vertical is saggital (in the place of incidence)
    Re = ROC*cos(AOI*pi/180);
    path_h.addComponent(component.curvedMirror(ROC*cos(AOI*pi/180), Z_h, label));
end