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

template<typename Scalar>
bool
psp::HMatrix_Quasi2d<Scalar>::Admissible
( int xSource, int ySource, int xTarget, int yTarget ) const
{
    if( _stronglyAdmissible )
        return std::max(std::abs(xSource-xTarget),std::abs(ySource-yTarget))>1;
    else
        return xSource != xTarget || ySource != yTarget;
}

template<typename Scalar>
void
psp::HMatrix_Quasi2d<Scalar>::RecursiveConstruction
( MatrixShell& shell, 
  const SparseMatrix<Scalar>& S, 
  int level,
  int xSource, int ySource, 
  int xTarget, int yTarget,
  int sourceOffset, int xSizeSource, int ySizeSource,
  int targetOffset, int xSizeTarget, int ySizeTarget )
{
    if( this->Admissible( xSource, ySource, xTarget, yTarget ) )
    {
        shell.type = FACTOR;
        hmatrix_tools::ConvertSubmatrix
        ( shell.F, S, 
          targetOffset, targetOffset+xSizeTarget*ySizeTarget*_zSize,
          sourceOffset, sourceOffset+xSizeSource*ySizeSource*_zSize );
    }
    else if( level < _numLevels-1 )
    {
        if( _symmetric && sourceOffset == targetOffset )
        {
            shell.type = NODE_SYMMETRIC;
            shell.children.resize( 10 );

            std::vector<int> xSizes(2);
            std::vector<int> ySizes(2);
            xSizes[0] = xSizeSource/2;            // left
            xSizes[1] = xSizeSource - xSizes[0];  // right
            ySizes[0] = ySizeSource/2;            // bottom
            ySizes[1] = ySizeSource - ySizes[0];  // top

            std::vector<int> sizes(4);
            sizes[0] = xSizes[0]*ySizes[0]*_zSize; // bottom-left
            sizes[1] = xSizes[1]*ySizes[0]*_zSize; // bottom-right
            sizes[2] = xSizes[0]*ySizes[1]*_zSize; // top-left
            sizes[3] = xSizes[1]*ySizes[1]*_zSize; // top-right

            int child = 0;
            int jStart = sourceOffset;
            for( int s=0; s<4; ++s )
            {
                int iStart = jStart;
                for( int t=s; t<4; ++t )
                {
                    this->RecursiveConstruction
                    ( shell.children[child++], S, level+1,
                      xSource+(s&1), ySource+(s/2),
                      xTarget+(t&1), yTarget+(t/2),
                      iStart, xSizes[s&1], ySizes[s/2],
                      jStart, xSizes[t&1], ySizes[t/2] );
                    iStart += sizes[t];
                }
                jStart += sizes[s];
            }
        }
        else
        {
            shell.type = NODE;
            shell.children.resize( 16 );
            
            std::vector<int> xSizesSource(2);
            std::vector<int> ySizesSource(2);
            std::vector<int> xSizesTarget(2);
            std::vector<int> ySizesTarget(2);
            xSizesSource[0] = xSizeSource/2;                 // left
            xSizesSource[1] = xSizeSource - xSizesSource[0]; // right
            ySizesSource[0] = ySizeSource/2;                 // bottom
            ySizesSource[1] = ySizeSource - ySizesSource[0]; // top
            xSizesTarget[0] = xSizeTarget/2;                 // left
            xSizesTarget[1] = xSizeTarget - xSizesTarget[0]; // right
            ySizesTarget[0] = ySizeTarget/2;                 // bottom
            ySizesTarget[1] = ySizeTarget - ySizesTarget[0]; // top

            std::vector<int> sourceSizes(4);
            std::vector<int> targetSizes(4);
            sourceSizes[0] = xSizesSource[0]*ySizesSource[0]*_zSize; // BL
            sourceSizes[1] = xSizesSource[1]*ySizesSource[0]*_zSize; // BR
            sourceSizes[2] = xSizesSource[0]*ySizesSource[1]*_zSize; // TL
            sourceSizes[3] = xSizesSource[1]*ySizesSource[1]*_zSize; // TR
            targetSizes[0] = xSizesTarget[0]*ySizesTarget[0]*_zSize; // BL
            targetSizes[1] = xSizesTarget[1]*ySizesTarget[0]*_zSize; // BR
            targetSizes[2] = xSizesTarget[0]*ySizesTarget[1]*_zSize; // TL
            targetSizes[3] = xSizesTarget[1]*ySizesTarget[1]*_zSize; // TR

            int child = 0;
            int jStart = sourceOffset;
            for( int s=0; s<4; ++s )
            {
                int iStart = targetOffset;
                for( int t=0; t<4; ++t )
                {
                    this->RecursiveConstruction
                    ( shell.children[child++], S, level+1,
                      xSource+(s&1), ySource+(s/2),
                      xTarget+(t&1), yTarget+(t/2),
                      iStart, xSizesSource[s&1], ySizesSource[s/2],
                      jStart, xSizesTarget[t&1], ySizesTarget[t/2] );
                    iStart += targetSizes[t];
                }
                jStart += sourceSizes[s];
            }
        }
    }
    else
    {
        if( _symmetric && sourceOffset == targetOffset )
        {
            // TODO: Think about packed storage
            shell.type = DENSE_SYMMETRIC;
            hmatrix_tools::ConvertSubmatrix
            ( shell.D, S,
              targetOffset, targetOffset+xSizeTarget*ySizeTarget*_zSize,
              sourceOffset, sourceOffset+xSizeTarget*ySizeTarget*_zSize );
        }
        else
        {
            shell.type = DENSE;
            hmatrix_tools::ConvertSubmatrix
            ( shell.D, S,
              targetOffset, targetOffset+xSizeTarget*ySizeTarget*_zSize,
              sourceOffset, sourceOffset+xSizeTarget*ySizeTarget*_zSize );
        }
    }
}

template<typename Scalar>
psp::HMatrix_Quasi2d<Scalar>::HMatrix_Quasi2d
( const SparseMatrix<Scalar>& S,
  int xSize, int ySize, int zSize, 
  int numLevels, 
  bool stronglyAdmissible )
: _symmetric(S.symmetric), 
  _xSize(xSize), _ySize(ySize), _zSize(zSize), 
  _numLevels(numLevels),
  _stronglyAdmissible(stronglyAdmissible)
{
    this->RecursiveConstruction
    ( _rootShell, S, 0, 
      0, 0, 
      0, 0, 
      0, xSize, ySize, 
      0, xSize, ySize );
}

template<typename Scalar>
psp::HMatrix_Quasi2d<Scalar>::~HMatrix_Quasi2d()
{
    // Nothing is needed yet
}

template class psp::HMatrix_Quasi2d<float>;
template class psp::HMatrix_Quasi2d<double>;
template class psp::HMatrix_Quasi2d< std::complex<float> >;
template class psp::HMatrix_Quasi2d< std::complex<double> >;
