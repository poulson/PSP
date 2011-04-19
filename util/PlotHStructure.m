function [] = PlotHStructure(filename)
%PlotHStructure: 
% Generates a plot for an H-structure that was printed using the
% psp::Quasi2dHMatrix::PrintStructure command.

A=importdata(filename);
Z=zeros(25,25);
for j=1:size(A.textdata,1),
  targetOffset=A.data(j,1);
  sourceOffset=A.data(j,2);
  targetSize=A.data(j,3);
  sourceSize=A.data(j,4);
  type=A.textdata(j);
  sourceRange=[(sourceOffset+1):(sourceOffset+sourceSize)];
  targetRange=[(targetOffset+1):(targetOffset+targetSize)];
  if( strcmp(type,'D') )
    Z(targetRange,sourceRange)=Z(targetRange,sourceRange)+1;
  elseif( strcmp(type,'F') )
    Z(targetRange,sourceRange)=Z(targetRange,sourceRange)+2;
  end
end
imagesc(Z);

end

