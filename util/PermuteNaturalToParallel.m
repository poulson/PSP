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
function parL = PermuteNaturalToParallel( natL, xSize, ySize, zSize, r, c )
% This function permutes a matrix L in 'natural' x-y-z major ordering into the
% form expected in parallel on a grid of dimensions r x c.

p=zeros(xSize*ySize*zSize,1);
for z=1:zSize,
  for y=1:ySize,
    for x=1:xSize,
      natIndex=x+xSize*(y-1)+xSize*ySize*(z-1);
      parIndex=MapNaturalToParallel(x,y,z,xSize,ySize,zSize,r,c);
      p(parIndex)=natIndex;
    end
  end
end

% Perform the row permutation
parL=natL(p,:);

% Perform the column permutation
parL=parL(:,p);

end
