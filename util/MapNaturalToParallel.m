%
%  Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
%  of a sweeping preconditioner for 3d Helmholtz equations.
%
%  Copyright (C) 2011 Jack Poulson, Lexing Ying, and
%  The University of Texas at Austin
%
%  This program is free software: you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation, either version 3 of the License, or
%  (at your option) any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
function index = MapNaturalToParallel( x, y, z, xSize, ySize, zSize, r, c )
% This function maps integer (x,y,z) coordinates in a domain of size 
% (xSize,ySize,zSize) to its parallel ordering index on an r x c grid of 
% processes. It is mainly used within Matlab in order to generate a permutation
% to transform a matrix generated in parallel to the serial ordering. See
% PermuteFromParallelToNatural

xChunkSize=floor(xSize/c);
yChunkSize=floor(ySize/r);
col=min(floor((x-1)/xChunkSize)+1,c);
row=min(floor((y-1)/yChunkSize)+1,r);

if(col==c),
  myXPortion=xChunkSize+mod(xSize,c);
else
  myXPortion=xChunkSize;
end
if(row==r), 
  myYPortion=yChunkSize+mod(ySize,r); 
else
  myYPortion=yChunkSize; 
end

offset=(row-1)*xSize*yChunkSize*zSize+(col-1)*xChunkSize*myYPortion*zSize;
xLocal=x-(col-1)*xChunkSize;
yLocal=y-(row-1)*yChunkSize;
index=offset+xLocal+myXPortion*(yLocal-1)+myXPortion*myYPortion*(z-1);

end

