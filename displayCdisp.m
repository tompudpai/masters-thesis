close all

figure
cdisp1 = importdata('cDisp_orig.dat');
image(cdisp1)
colorbar

figure
cdisp2 = importdata('cDisp.dat');
image(cdisp2)
colorbar

sad = sum(sum(abs(cdisp1-cdisp2) > 0));
if(~sad)
    disp('Images are the same.')
else
    disp(['Images are different. Pixels different: ' num2str(sad)]);
    disp(['Error = ' num2str(sad/(383*434)*100) '%']);
    figure
    cdisp12 = abs(cdisp1-cdisp2);
    image(cdisp12)
    colorbar
end