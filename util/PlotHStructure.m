function [] = PlotHStructure(filename)
%PlotHStructure: 
% Generates a plot for an H-structure that was printed using the
% psp::Quasi2dHMatrix::PrintStructure command.

A=load(filename);

% Scan for the maximum source and target sizes
maxTarget=0;
maxSource=0;
for j=1:size(A,1),
  targetOffset=A(j,2);
  sourceOffset=A(j,3);
  targetSize=A(j,4);
  sourceSize=A(j,5);
  targetEnd=targetOffset+targetSize;
  sourceEnd=sourceOffset+sourceSize;
  maxSource=max(maxSource,sourceEnd);
  maxTarget=max(maxTarget,targetEnd);
end

Z=zeros(maxTarget,maxSource);
for j=1:size(A,1),
  value=A(j,1);
  targetOffset=A(j,2);
  sourceOffset=A(j,3);
  targetSize=A(j,4);
  sourceSize=A(j,5);
  sourceRange=[(sourceOffset+1):(sourceOffset+sourceSize)];
  targetRange=[(targetOffset+1):(targetOffset+targetSize)];
  Z(targetRange,sourceRange)=Z(targetRange,sourceRange)+value;
end
imagesc(Z);

end

