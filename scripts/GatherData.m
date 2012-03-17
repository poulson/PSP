%
% Gather the processed .pvti/.vti files from a PSP run into a single matrix.
%
% Example: running
%     field=GatherData('source_0_real',int32(801),int32(801),int32(187), ...
%                                      int32(16),int32(16),int32(8))
% attempts to gather the (already processed with ProcessVti.sh) .vti files
% 'source_0_real_0.vti', 'source_0_real_1.vti', ..., for each of the 
% 16 x 16 x 8 processes, and load the data for the 801 x 801 x 187 solution 
% into a single 3d array.
%
% It is important that the process grid variables be 32-bit integers so that
% integer division is correctly performed.
%
function[field]=GatherData(fileString,nx,ny,nz,px,py,pz)

xBoxMainSize=idivide(nx,px,'floor');
yBoxMainSize=idivide(ny,py,'floor');
zBoxMainSize=idivide(nz,pz,'floor');
xBoxLastSize=xBoxMainSize+mod(nx,px);
yBoxLastSize=yBoxMainSize+mod(ny,py);
zBoxLastSize=zBoxMainSize+mod(nz,pz);

field=zeros(nx,ny,nz);
for x=0:px-1,
  if x==px-1, xBoxSize=xBoxLastSize; else xBoxSize=xBoxMainSize; end
  xStart=1+x*xBoxMainSize;
  xEnd=xStart+xBoxSize-1;
  for y=0:py-1,
    if y==py-1, yBoxSize=yBoxLastSize; else yBoxSize=yBoxMainSize; end
    yStart=1+y*yBoxMainSize;
    yEnd=yStart+yBoxSize-1;
    for z=0:pz-1,
      if z==pz-1, zBoxSize=zBoxLastSize; else zBoxSize=zBoxMainSize; end
      zStart=1+z*zBoxMainSize;
      zEnd=zStart+zBoxSize-1;

      proc=x+y*px+z*px*py;
      filename=strcat(fileString,sprintf('_%d.vti',proc));
      local=load(filename);
      local=local';
      local=reshape(local,[xBoxSize,yBoxSize,zBoxSize]);
      field(xStart:xEnd,yStart:yEnd,zStart:zEnd)=local;
    end
  end
end

saveFilename=strcat(fileString,'.mat');
save(saveFilename,'field');

return 
