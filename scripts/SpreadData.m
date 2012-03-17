%
% Spread a global 3d array into the local components expected by PSP when 
% the process grid is px x py x pz.
%
% Example: running
%     load('model.mat','model');
%     SpreadData(model,'localModel',16,16,8);
% attempts to spread the 3d array 'model' over a 16 x 16 x 8 process grid by 
% printing out the files 'localModel_r.dat', for r=0,...,(px*py*pz-1).
% For instance, the process with rank 32 should load 'localModel_32.dat'
%
% Ideally, px, py, and pz were created as int32's, e.g.,
%     px=int32(16);
%     py=int32(16);
%     pz=int32(8);
% 
function[]=SpreadData(globalData,localString,px,py,pz)

for x=0:px-1,
  for y=0:py-1,
    for z=0:pz-1,
      proc=x+y*px+z*px*py;
      localData=globalData(1+x:px:end,1+y:py:end,1+z:pz:end);
      filename=strcat(localString,sprintf('_%d.dat',proc));
      file=fopen(filename,'w');
      fwrite(file,localData,'double');
      fclose(file);
    end
  end
end
