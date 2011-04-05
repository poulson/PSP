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
        shell.u.factor = new FactorData;
        hmatrix_tools::ConvertSubmatrix
        ( shell.u.factor->F, S, 
          targetOffset, targetOffset+xSizeTarget*ySizeTarget*_zSize,
          sourceOffset, sourceOffset+xSizeSource*ySizeSource*_zSize );
    }
    else if( level < _numLevels-1 )
    {
        if( _symmetric && sourceOffset == targetOffset )
        {
            shell.type = NODE_SYMMETRIC;
            shell.u.nodeSymmetric = new NodeSymmetricData;
            shell.u.nodeSymmetric->children.resize( 10 );

            int* xSizes = shell.u.nodeSymmetric->xSizes;
            xSizes[0] = xSizeSource/2;            // left
            xSizes[1] = xSizeSource - xSizes[0];  // right
            int* ySizes = shell.u.nodeSymmetric->ySizes;
            ySizes[0] = ySizeSource/2;            // bottom
            ySizes[1] = ySizeSource - ySizes[0];  // top

            int* sizes = shell.u.nodeSymmetric->sizes;
            sizes[0] = xSizes[0]*ySizes[0]*_zSize; // bottom-left
            sizes[1] = xSizes[1]*ySizes[0]*_zSize; // bottom-right
            sizes[2] = xSizes[0]*ySizes[1]*_zSize; // top-left
            sizes[3] = xSizes[1]*ySizes[1]*_zSize; // top-right

            int child = 0;
            int iStart = targetOffset;
            for( int t=0; t<4; ++t )
            {
                int jStart = 0;
                for( int s=0; s<=t; ++s )
                {
                    this->RecursiveConstruction
                    ( shell.u.nodeSymmetric->children[child++], 
                      S, level+1,
                      xSource+(s&1), ySource+(s/2),
                      xTarget+(t&1), yTarget+(t/2),
                      iStart, xSizes[s&1], ySizes[s/2],
                      jStart, xSizes[t&1], ySizes[t/2] );
                    jStart += sizes[s];
                }
                iStart += sizes[t];
            }
        }
        else
        {
            shell.type = NODE;
            shell.u.node = new NodeData;
            shell.u.node->children.resize( 16 );
            
            int* xSourceSizes = shell.u.node->xSourceSizes;
            xSourceSizes[0] = xSizeSource/2;                 // left
            xSourceSizes[1] = xSizeSource - xSourceSizes[0]; // right
            int* ySourceSizes = shell.u.node->ySourceSizes;
            ySourceSizes[0] = ySizeSource/2;                 // bottom
            ySourceSizes[1] = ySizeSource - ySourceSizes[0]; // top
            int* xTargetSizes = shell.u.node->xTargetSizes;
            xTargetSizes[0] = xSizeTarget/2;                 // left
            xTargetSizes[1] = xSizeTarget - xTargetSizes[0]; // right
            int* yTargetSizes = shell.u.node->yTargetSizes;
            yTargetSizes[0] = ySizeTarget/2;                 // bottom
            yTargetSizes[1] = ySizeTarget - yTargetSizes[0]; // top

            int* sourceSizes = shell.u.node->sourceSizes;
            sourceSizes[0] = xSourceSizes[0]*ySourceSizes[0]*_zSize; // BL
            sourceSizes[1] = xSourceSizes[1]*ySourceSizes[0]*_zSize; // BR
            sourceSizes[2] = xSourceSizes[0]*ySourceSizes[1]*_zSize; // TL
            sourceSizes[3] = xSourceSizes[1]*ySourceSizes[1]*_zSize; // TR
            int* targetSizes = shell.u.node->targetSizes;
            targetSizes[0] = xTargetSizes[0]*yTargetSizes[0]*_zSize; // BL
            targetSizes[1] = xTargetSizes[1]*yTargetSizes[0]*_zSize; // BR
            targetSizes[2] = xTargetSizes[0]*yTargetSizes[1]*_zSize; // TL
            targetSizes[3] = xTargetSizes[1]*yTargetSizes[1]*_zSize; // TR

            int iStart = targetOffset;
            for( int t=0; t<4; ++t )
            {
                int jStart = sourceOffset;
                for( int s=0; s<4; ++s )
                {
                    this->RecursiveConstruction
                    ( shell.u.node->children[s+4*t], 
                      S, level+1,
                      xSource+(s&1), ySource+(s/2),
                      xTarget+(t&1), yTarget+(t/2),
                      iStart, xSourceSizes[s&1], ySourceSizes[s/2],
                      jStart, xTargetSizes[t&1], yTargetSizes[t/2] );
                    jStart += sourceSizes[s];
                }
                iStart += targetSizes[t];
            }
        }
    }
    else
    {
        shell.type = DENSE;
        shell.u.dense = new DenseData;
        hmatrix_tools::ConvertSubmatrix
        ( shell.u.dense->D, S,
          targetOffset, targetOffset+xSizeTarget*ySizeTarget*_zSize,
          sourceOffset, sourceOffset+xSizeTarget*ySizeTarget*_zSize );
    }
}

template<typename Scalar>
void
psp::HMatrix_Quasi2d<Scalar>::RecursiveMatrixVector
( Scalar alpha, const MatrixShell& shell,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y ) const
{
    if( shell.type == NODE )
    {
        // First scale y so that we can simply sum contributions onto it
        hmatrix_tools::Scale( beta, y );

        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        const int* sourceSizes = shell.u.node->sourceSizes;
        const int* targetSizes = shell.u.node->targetSizes;
        for( int t=0; t<4; ++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, targetOffset, targetSizes[t] );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sourceOffset, sourceSizes[s] );

                RecursiveMatrixVector
                ( alpha, shell.u.node->children[s+4*t], xSub, 1, ySub );

                sourceOffset += sourceSizes[s];
            }
            targetOffset += targetSizes[t];
        }
    }
    else if( shell.type == NODE_SYMMETRIC )
    {
        // First scale y so that we can simply sum contributions onto it
        hmatrix_tools::Scale( beta, y );

        // Loop over the 10 children in the lower triangle, summing in each row
        {
            int child = 0;
            int targetOffset = 0;
            const int* sizes = shell.u.nodeSymmetric->sizes;
            for( int t=0; t<4; ++t )
            {
                Vector<Scalar> ySub;
                ySub.View( y, targetOffset, sizes[t] );

                int sourceOffset = 0;
                for( int s=0; s<=t; ++s )
                {
                    Vector<Scalar> xSub;
                    xSub.LockedView( x, sourceOffset, sizes[s] );

                    RecursiveMatrixVector
                    ( alpha, shell.u.nodeSymmetric->children[child++], xSub, 
                      1, ySub );

                    sourceOffset += sizes[s];
                }
                targetOffset += sizes[t];
            }
        }

        // Loop over the 6 children in the strictly lower triangle, summing in
        // each row
        {
            int targetOffset = 0;
            const int* sizes = shell.u.nodeSymmetric->sizes;
            for( int s=0; s<4; ++s )
            {
                Vector<Scalar> ySub;
                ySub.View( y, targetOffset, sizes[s] );

                int sourceOffset = targetOffset + sizes[s];
                for( int t=s+1; t<4; ++t )
                {
                    Vector<Scalar> xSub;
                    xSub.LockedView( x, sourceOffset, sizes[t] );

                    const int child = (t*(t+1))/2 + s;
                    RecursiveMatrixTransposeVector
                    ( alpha, shell.u.nodeSymmetric->children[child], xSub,
                      1, ySub );

                    sourceOffset += sizes[t];
                }
                targetOffset += sizes[s];
            }
        }
    }
    else if( shell.type == FACTOR )
    {
        hmatrix_tools::MatrixVector( alpha, shell.u.factor->F, x, beta, y );
    }
    else /* shell.type == DENSE */
    {
        hmatrix_tools::MatrixVector( alpha, shell.u.dense->D, x, beta, y );
    }
}

template<typename Scalar>
void
psp::HMatrix_Quasi2d<Scalar>::RecursiveMatrixTransposeVector
( Scalar alpha, const MatrixShell& shell,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y ) const
{
    if( shell.type == NODE )
    {
        // First scale y so that we can simply sum contributions onto it
        hmatrix_tools::Scale( beta, y );

        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        const int* sourceSizes = shell.u.node->sourceSizes;
        const int* targetSizes = shell.u.node->targetSizes;
        for( int t=0; t<4; ++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, targetOffset, sourceSizes[t] );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sourceOffset, targetSizes[s] );

                RecursiveMatrixTransposeVector
                ( alpha, shell.u.node->children[t+4*s], xSub, 1, ySub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
    }
    else if( shell.type == NODE_SYMMETRIC )
    {
        // NOTE: This section is an exact copy of that of the one from 
        //       RecursiveMatrixVector. TODO: Avoid this duplication.

        // First scale y so that we can simply sum contributions onto it
        hmatrix_tools::Scale( beta, y );

        // Loop over the 10 children in the lower triangle, summing in each row
        {
            int child = 0;
            int targetOffset = 0;
            const int* sizes = shell.u.nodeSymmetric->sizes;
            for( int t=0; t<4; ++t )
            {
                Vector<Scalar> ySub;
                ySub.View( y, targetOffset, sizes[t] );

                int sourceOffset = 0;
                for( int s=0; s<=t; ++s )
                {
                    Vector<Scalar> xSub;
                    xSub.LockedView( x, sourceOffset, sizes[s] );

                    RecursiveMatrixVector
                    ( alpha, shell.u.nodeSymmetric->children[child++], xSub, 
                      1, ySub );

                    sourceOffset += sizes[s];
                }
                targetOffset += sizes[t];
            }
        }

        // Loop over the 6 children in the strictly lower triangle, summing in
        // each row
        {
            int targetOffset = 0;
            const int* sizes = shell.u.nodeSymmetric->sizes;
            for( int s=0; s<4; ++s )
            {
                Vector<Scalar> ySub;
                ySub.View( y, targetOffset, sizes[s] );

                int sourceOffset = targetOffset + sizes[s];
                for( int t=s+1; t<4; ++t )
                {
                    Vector<Scalar> xSub;
                    xSub.LockedView( x, sourceOffset, sizes[t] );

                    const int child = (t*(t+1))/2 + s;
                    RecursiveMatrixTransposeVector
                    ( alpha, shell.u.nodeSymmetric->children[child], xSub,
                      1, ySub );

                    sourceOffset += sizes[t];
                }
                targetOffset += sizes[s];
            }
        }
    }
    else if( shell.type == FACTOR )
    {
        hmatrix_tools::MatrixTransposeVector
        ( alpha, shell.u.factor->F, x, beta, y );
    }
    else /* shell.type == DENSE */
    {
        hmatrix_tools::MatrixTransposeVector
        ( alpha, shell.u.dense->D, x, beta, y );
    }
}

template<typename Scalar>
void
psp::HMatrix_Quasi2d<Scalar>::RecursiveMatrixMatrix
( Scalar alpha, const MatrixShell& shell,
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C ) const
{
    if( shell.type == NODE )
    {
        // First scale C so that we can simply sum contributions onto it
        hmatrix_tools::Scale( beta, C );

        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        const int* sourceSizes = shell.u.node->sourceSizes;
        const int* targetSizes = shell.u.node->targetSizes;
        for( int t=0; t<4; ++t )
        {
            DenseMatrix<Scalar> CSub;
            CSub.View( C, targetOffset, targetSizes[t], 0, C.Width() );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                DenseMatrix<Scalar> BSub;
                BSub.LockedView
                ( B, sourceOffset, sourceSizes[s], 0, B.Width() );

                RecursiveMatrixMatrix
                ( alpha, shell.u.node->children[s+4*t], BSub, 1, CSub );

                sourceOffset += sourceSizes[s];
            }
            targetOffset += targetSizes[t];
        }
    }
    else if( shell.type == NODE_SYMMETRIC )
    {
        // First scale C so that we can simply sum contributions onto it
        hmatrix_tools::Scale( beta, C );

        // Loop over the 10 children in the lower triangle, summing in each row
        {
            int child = 0;
            int targetOffset = 0;
            const int* sizes = shell.u.nodeSymmetric->sizes;
            for( int t=0; t<4; ++t )
            {
                DenseMatrix<Scalar> CSub;
                CSub.View( C, targetOffset, sizes[t], 0, C.Width() );

                int sourceOffset = 0;
                for( int s=0; s<=t; ++s )
                {
                    DenseMatrix<Scalar> BSub;
                    BSub.LockedView( B, sourceOffset, sizes[s], 0, B.Width() );

                    RecursiveMatrixMatrix
                    ( alpha, shell.u.nodeSymmetric->children[child++], BSub, 
                      1, CSub );

                    sourceOffset += sizes[s];
                }
                targetOffset += sizes[t];
            }
        }

        // Loop over the 6 children in the strictly lower triangle, summing in
        // each row
        {
            int targetOffset = 0;
            const int* sizes = shell.u.nodeSymmetric->sizes;
            for( int s=0; s<4; ++s )
            {
                DenseMatrix<Scalar> CSub;
                CSub.View( C, targetOffset, sizes[s], 0, C.Width() );

                int sourceOffset = targetOffset + sizes[s];
                for( int t=s+1; t<4; ++t )
                {
                    DenseMatrix<Scalar> BSub;
                    BSub.LockedView( B, sourceOffset, sizes[t], 0, B.Width() );

                    const int child = (t*(t+1))/2 + s;
                    RecursiveMatrixTransposeMatrix
                    ( alpha, shell.u.nodeSymmetric->children[child], BSub,
                      1, CSub );

                    sourceOffset += sizes[t];
                }
                targetOffset += sizes[s];
            }
        }
    }
    else if( shell.type == FACTOR )
    {
        hmatrix_tools::MatrixMatrix( alpha, shell.u.factor->F, B, beta, C );
    }
    else /* shell.type == DENSE */
    {
        hmatrix_tools::MatrixMatrix( alpha, shell.u.dense->D, B, beta, C );
    }
}

template<typename Scalar>
void
psp::HMatrix_Quasi2d<Scalar>::RecursiveMatrixTransposeMatrix
( Scalar alpha, const MatrixShell& shell,
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C ) const
{
    if( shell.type == NODE )
    {
        // First scale C so that we can simply sum contributions onto it
        hmatrix_tools::Scale( beta, C );

        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        const int* sourceSizes = shell.u.node->sourceSizes;
        const int* targetSizes = shell.u.node->targetSizes;
        for( int t=0; t<4; ++t )
        {
            DenseMatrix<Scalar> CSub;
            CSub.View( C, targetOffset, sourceSizes[t], 0, C.Width() );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                DenseMatrix<Scalar> BSub;
                BSub.LockedView
                ( B, sourceOffset, targetSizes[s], 0, B.Width() );

                RecursiveMatrixTransposeMatrix
                ( alpha, shell.u.node->children[t+4*s], BSub, 1, CSub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
    }
    else if( shell.type == NODE_SYMMETRIC )
    {
        // NOTE: This section is an exact copy of that of the one from 
        //       RecursiveMatrixMatrix. TODO: Avoid this duplication.

        // First scale C so that we can simply sum contributions onto it
        hmatrix_tools::Scale( beta, C );

        // Loop over the 10 children in the lower triangle, summing in each row
        {
            int child = 0;
            int targetOffset = 0;
            const int* sizes = shell.u.nodeSymmetric->sizes;
            for( int t=0; t<4; ++t )
            {
                DenseMatrix<Scalar> CSub;
                CSub.View( C, targetOffset, sizes[t], 0, C.Width() );

                int sourceOffset = 0;
                for( int s=0; s<=t; ++s )
                {
                    DenseMatrix<Scalar> BSub;
                    BSub.LockedView( B, sourceOffset, sizes[s], 0, B.Width() );

                    RecursiveMatrixMatrix
                    ( alpha, shell.u.nodeSymmetric->children[child++], BSub, 
                      1, CSub );

                    sourceOffset += sizes[s];
                }
                targetOffset += sizes[t];
            }
        }

        // Loop over the 6 children in the strictly lower triangle, summing in
        // each row
        {
            int targetOffset = 0;
            const int* sizes = shell.u.nodeSymmetric->sizes;
            for( int s=0; s<4; ++s )
            {
                DenseMatrix<Scalar> CSub;
                CSub.View( C, targetOffset, sizes[s], 0, C.Width() );

                int sourceOffset = targetOffset + sizes[s];
                for( int t=s+1; t<4; ++t )
                {
                    DenseMatrix<Scalar> BSub;
                    BSub.LockedView( B, sourceOffset, sizes[t], 0, B.Width() );

                    const int child = (t*(t+1))/2 + s;
                    RecursiveMatrixTransposeMatrix
                    ( alpha, shell.u.nodeSymmetric->children[child], BSub,
                      1, CSub );

                    sourceOffset += sizes[t];
                }
                targetOffset += sizes[s];
            }
        }
    }
    else if( shell.type == FACTOR )
    {
        hmatrix_tools::MatrixTransposeMatrix
        ( alpha, shell.u.factor->F, B, beta, C );
    }
    else /* shell.type == DENSE */
    {
        hmatrix_tools::MatrixTransposeMatrix
        ( alpha, shell.u.dense->D, B, beta, C );
    }
}

//----------------------------------------------------------------------------//
// Public routines                                                            //
//----------------------------------------------------------------------------//

template<typename Scalar>
psp::HMatrix_Quasi2d<Scalar>::HMatrix_Quasi2d
( const SparseMatrix<Scalar>& S,
  int xSize, int ySize, int zSize, 
  int numLevels, 
  bool stronglyAdmissible )
: _m(S.m), _n(S.n), _symmetric(S.symmetric), 
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

template<typename Scalar>
void
psp::HMatrix_Quasi2d<Scalar>::MapVector
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
    RecursiveMatrixVector( alpha, _rootShell, x, beta, y );
}

template<typename Scalar>
void
psp::HMatrix_Quasi2d<Scalar>::MapVector
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
    y.Resize( _m );
    std::memset( y.Buffer(), 0, _m*sizeof(Scalar) );
    RecursiveMatrixVector( alpha, _rootShell, x, 1, y );
}

template<typename Scalar>
void
psp::HMatrix_Quasi2d<Scalar>::TransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
    RecursiveMatrixTransposeVector( alpha, _rootShell, x, beta, y );
}

template<typename Scalar>
void
psp::HMatrix_Quasi2d<Scalar>::TransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
    y.Resize( _n );
    std::memset( y.Buffer(), 0, _n*sizeof(Scalar) );
    RecursiveMatrixTransposeVector( alpha, _rootShell, x, 1, y );
}

template<typename Scalar>
void
psp::HMatrix_Quasi2d<Scalar>::MapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C ) const
{
    RecursiveMatrixMatrix( alpha, _rootShell, B, beta, C );
}

template<typename Scalar>
void
psp::HMatrix_Quasi2d<Scalar>::MapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C ) const
{
    C.Resize( _m, B.Width() );
    std::memset( C.Buffer(), 0, _m*B.Width()*sizeof(Scalar) );
    RecursiveMatrixMatrix( alpha, _rootShell, B, 1, C );
}

template<typename Scalar>
void
psp::HMatrix_Quasi2d<Scalar>::TransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C ) const
{
    RecursiveMatrixTransposeMatrix( alpha, _rootShell, B, beta, C );
}

template<typename Scalar>
void
psp::HMatrix_Quasi2d<Scalar>::TransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C ) const
{
    C.Resize( _n, B.Width() );
    std::memset( C.Buffer(), 0, _n*B.Width()*sizeof(Scalar) );
    RecursiveMatrixTransposeMatrix( alpha, _rootShell, B, 1, C );
}

template class psp::HMatrix_Quasi2d<float>;
template class psp::HMatrix_Quasi2d<double>;
template class psp::HMatrix_Quasi2d< std::complex<float> >;
template class psp::HMatrix_Quasi2d< std::complex<double> >;
