/*
   Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
   of a sweeping preconditioner for 3d Helmholtz equations.

   Copyright (C) 2011 Jack Poulson, Lexing Ying, and
   The University of Texas at Austin

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "psp.hpp"

namespace {

void RecurseOnQuadrant
( int* map, int& index, int level, int numLevels,
  int xSize, int ySize, int zSize, int thisXSize, int thisYSize )
{
    if( level == numLevels-1 )
    {
        // Stamp these indices into the buffer 
        for( int k=0; k<zSize; ++k )
        {
            for( int j=0; j<thisYSize; ++j )
            {
                int* thisRow = &map[k*xSize*ySize+j*xSize];
                for( int i=0; i<thisXSize; ++i )
                    thisRow[i] = index++;
            }
        }
    }
    else
    {
        const int leftWidth = thisXSize/2;
        const int rightWidth = thisXSize - leftWidth;
        const int bottomHeight = thisYSize/2;
        const int topHeight = thisYSize - bottomHeight;

        // Recurse on the lower-left quadrant 
        RecurseOnQuadrant
        ( &map[0], index, level+1, numLevels, 
          xSize, ySize, zSize, leftWidth, bottomHeight );
        // Recurse on the lower-right quadrant
        RecurseOnQuadrant
        ( &map[leftWidth], index, level+1, numLevels, 
          xSize, ySize, zSize, rightWidth, bottomHeight );
        // Recurse on the upper-left quadrant
        RecurseOnQuadrant
        ( &map[bottomHeight*xSize], index, level+1, numLevels, 
          xSize, ySize, zSize, leftWidth, topHeight );
        // Recurse on the upper-right quadrant
        RecurseOnQuadrant
        ( &map[bottomHeight*xSize+leftWidth], index, level+1, numLevels,
          xSize, ySize, zSize, rightWidth, topHeight );
    }
}

} // anonymous namespace

// Recursively form the map from natural to hierarchical ordering
void psp::hmatrix_tools::BuildNaturalToHierarchicalMap
( std::vector<int>& naturalToHierarchicalMap, int numLevels, 
  int xSize, int ySize, int zSize )
{
    naturalToHierarchicalMap.resize( xSize*ySize*zSize );

    // Fill the mapping from the 'natural' x-y-z ordering
    int index = 0;
    RecurseOnQuadrant
    ( &naturalToHierarchicalMap[0], index, 0, numLevels,
      xSize, ySize, zSize, xSize, ySize );
#ifndef RELEASE
    if( index != xSize*ySize*zSize )
        throw std::logic_error("Map recursion is incorrect.");
#endif
}

// We can use this to generate the hierarchical -> natural map from the 
// natural -> hierarchical map
void psp::hmatrix_tools::InvertMap
(       std::vector<int>& invertedMap, 
  const std::vector<int>& originalMap )
{
    const int length = originalMap.size();
    invertedMap.resize( length );
    for( int i=0; i<length; ++i )
        invertedMap[originalMap[i]] = i; 
}

