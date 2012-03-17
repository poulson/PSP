%
% Interpolates a global 3d array into the local components expected by PSP when 
% the process grid is px x py x pz.
%
% Example: running
%     load('model.mat','model');
%     InterpolateData(model,'localModel',1201,1201,280,32,32,8);
% attempts to spread the 3d array 'model' over a 16 x 16 x 8 process grid by 
% printing out the files 'localModel_r.dat', for r=0,...,(px*py*pz-1).
% For instance, the process with rank 32 should load 'localModel_32.dat'
%
function[]=InterpolateData(globalData,localString,Nx,Ny,Nz,px,py,pz)

[nx,ny,nz]=size(globalData);

xRatio=(nx-1)/(Nx-1);
yRatio=(ny-1)/(Ny-1);
zRatio=(nz-1)/(Nz-1);

box=zeros(2,2,2);
xavg=zeros(2,2);
xyavg=zeros(2,1);
for xProc=0:px-1,
  for yProc=0:py-1,
    for zProc=0:pz-1,
      proc=xProc+yProc*px+zProc*px*py;

      if mod(Nx,px) > xProc,
        xLocalSize=floor(Nx/px)+1;
      else
        xLocalSize=floor(Nx/px); 
      end
      if mod(Ny,py) > yProc,
        yLocalSize=floor(Ny/py)+1;
      else
        yLocalSize=floor(Ny/py);
      end
      if mod(Nz,pz) > zProc,
        zLocalSize=floor(Nz/pz)+1;
      else
        zLocalSize=floor(Nz/pz);
      end

      localData=zeros(xLocalSize,yLocalSize,zLocalSize);

      for zLocal=0:zLocalSize-1,
        zLocal
        z=zProc+zLocal*pz;
        zRel=z*zRatio+1;
        zRelFloor=floor(zRel);
        zRelCeil=ceil(zRel);
        zIndices=[zRelFloor,zRelCeil];
        tz=zRel-zRelFloor;
        for yLocal=0:yLocalSize-1,
          y=yProc+yLocal*py;
          yRel=y*yRatio+1;
          yRelFloor=floor(yRel);
          yRelCeil=ceil(yRel);
          yIndices=[yRelFloor,yRelCeil];
          ty=yRel-yRelFloor;
          for xLocal=0:xLocalSize-1,
            x=xProc+xLocal*px;
            xRel=x*xRatio+1;
            xRelFloor=floor(xRel);
            xRelCeil=ceil(xRel);
            xIndices=[xRelFloor,xRelCeil];
            tx=xRel-xRelFloor;

            % Extract the 2 x 2 x 2 box of relevant coordinates
            box=globalData(xIndices,yIndices,zIndices);
            xavg=(1-tx)*reshape(box(1,:,:),[2,2])+tx*reshape(box(2,:,:),[2,2]);
            xyavg=(1-ty)*xavg(1,:)+ty*xavg(2,:);
            avg=(1-tz)*xyavg(1)+tz*xyavg(2);

            localData(xLocal+1,yLocal+1,zLocal+1)=avg;
          end
        end
      end

      imagesc(reshape(localData(:,:,floor(zLocalSize/2)+1),[xLocalSize,yLocalSize]));
      pause(1);

      filename=strcat(localString,sprintf('_%d.dat',proc));
      file=fopen(filename,'w');
      fwrite(file,localData,'double');
      fclose(file);
    end
  end
end
