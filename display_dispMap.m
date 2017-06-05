venus = uint8(dlmread('small_cDisp_venus.dat',' '));
%%
figure
image(venus)
colormap jet
colorbar
%%
if(false);
cones = uint8(dlmread('cDisp_cones.dat',' '));
%%
figure
image(cones)
colormap jet
colorbar
end 