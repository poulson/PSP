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

template<typename Scalar,bool Conjugated>
bool
psp::Quasi2dHMatrix<Scalar,Conjugated>::Admissible
( int xSource, int xTarget, int ySource, int yTarget ) const
{
    if( _stronglyAdmissible )
        return std::max(std::abs(xSource-xTarget),std::abs(ySource-yTarget))>1;
    else
        return xSource != xTarget || ySource != yTarget;
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::ImportSparseMatrix
( const SparseMatrix<Scalar>& S )
{
    if( Admissible( _xSource, _xTarget, _ySource, _yTarget ) )
    {
        _shellType = FACTOR;
        _shell.F = new FactorMatrix<Scalar,Conjugated>;
        hmatrix_tools::ConvertSubmatrix
        ( *_shell.F, S, 
          _targetOffset, _targetOffset+_xSizeTarget*_ySizeTarget*_zSize,
          _sourceOffset, _sourceOffset+_xSizeSource*_ySizeSource*_zSize );
    }
    else if( _numLevels > 1 )
    {
        if( _symmetric && _sourceOffset == _targetOffset )
        {
            _shellType = NODE_SYMMETRIC;
            _shell.nodeSymmetric = new NodeSymmetricData;
            _shell.nodeSymmetric->children.resize( 10 );

            int* xSizes = _shell.nodeSymmetric->xSizes;
            xSizes[0] = _xSizeSource/2;            // left
            xSizes[1] = _xSizeSource - xSizes[0];  // right
            int* ySizes = _shell.nodeSymmetric->ySizes;
            ySizes[0] = _ySizeSource/2;            // bottom
            ySizes[1] = _ySizeSource - ySizes[0];  // top

            int* sizes = _shell.nodeSymmetric->sizes;
            sizes[0] = xSizes[0]*ySizes[0]*_zSize; // bottom-left
            sizes[1] = xSizes[1]*ySizes[0]*_zSize; // bottom-right
            sizes[2] = xSizes[0]*ySizes[1]*_zSize; // top-left
            sizes[3] = xSizes[1]*ySizes[1]*_zSize; // top-right

            int child = 0;
            int targetOffset = _targetOffset;
            for( int t=0; t<4; ++t )
            {
                int sourceOffset = _targetOffset;
                for( int s=0; s<=t; ++s )
                {
                    _shell.nodeSymmetric->children[child++] = 
                      new Quasi2dHMatrix
                      ( S, 
                        _numLevels-1, _stronglyAdmissible,
                        xSizes[s&1], xSizes[t&1],
                        ySizes[s/2], ySizes[t/2],
                        _zSize,
                        _xSource+(s&1), _xTarget+(t&1),
                        _ySource+(s/2), _yTarget+(t/2),
                        sourceOffset, targetOffset );
                    sourceOffset += sizes[s];
                }
                targetOffset += sizes[t];
            }
        }
        else
        {
            _shellType = NODE;
            _shell.node = new NodeData;
            _shell.node->children.resize( 16 );
            
            int* xSourceSizes = _shell.node->xSourceSizes;
            xSourceSizes[0] = _xSizeSource/2;                 // left
            xSourceSizes[1] = _xSizeSource - xSourceSizes[0]; // right
            int* ySourceSizes = _shell.node->ySourceSizes;
            ySourceSizes[0] = _ySizeSource/2;                 // bottom
            ySourceSizes[1] = _ySizeSource - ySourceSizes[0]; // top
            int* xTargetSizes = _shell.node->xTargetSizes;
            xTargetSizes[0] = _xSizeTarget/2;                 // left
            xTargetSizes[1] = _xSizeTarget - xTargetSizes[0]; // right
            int* yTargetSizes = _shell.node->yTargetSizes;
            yTargetSizes[0] = _ySizeTarget/2;                 // bottom
            yTargetSizes[1] = _ySizeTarget - yTargetSizes[0]; // top

            int* sourceSizes = _shell.node->sourceSizes;
            sourceSizes[0] = xSourceSizes[0]*ySourceSizes[0]*_zSize; // BL
            sourceSizes[1] = xSourceSizes[1]*ySourceSizes[0]*_zSize; // BR
            sourceSizes[2] = xSourceSizes[0]*ySourceSizes[1]*_zSize; // TL
            sourceSizes[3] = xSourceSizes[1]*ySourceSizes[1]*_zSize; // TR
            int* targetSizes = _shell.node->targetSizes;
            targetSizes[0] = xTargetSizes[0]*yTargetSizes[0]*_zSize; // BL
            targetSizes[1] = xTargetSizes[1]*yTargetSizes[0]*_zSize; // BR
            targetSizes[2] = xTargetSizes[0]*yTargetSizes[1]*_zSize; // TL
            targetSizes[3] = xTargetSizes[1]*yTargetSizes[1]*_zSize; // TR

            int targetOffset = _targetOffset;
            for( int t=0; t<4; ++t )
            {
                int sourceOffset = _sourceOffset;
                for( int s=0; s<4; ++s )
                {
                    _shell.node->children[s+4*t] = 
                      new Quasi2dHMatrix
                      ( S,
                        _numLevels-1, _stronglyAdmissible,
                        xSourceSizes[s&1], xTargetSizes[t&1],
                        ySourceSizes[s/2], yTargetSizes[t/2],
                        _zSize,
                        _xSource+(s&1), _xTarget+(t&1),
                        _ySource+(s/2), _yTarget+(t/2),
                        sourceOffset, targetOffset );
                    sourceOffset += sourceSizes[s];
                }
                targetOffset += targetSizes[t];
            }
        }
    }
    else
    {
        _shellType = DENSE;
        _shell.D = new DenseMatrix<Scalar>;
        hmatrix_tools::ConvertSubmatrix
        ( *_shell.D, S,
          _targetOffset, _targetOffset+_xSizeTarget*_ySizeTarget*_zSize,
          _sourceOffset, _sourceOffset+_xSizeTarget*_ySizeTarget*_zSize );
    }
}

// y += alpha A x
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::UpdateVectorWithNodeSymmetric
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
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

                const Quasi2dHMatrix& ASub = 
                    *_shell.nodeSymmetric->children[child];
                ASub.MapVector( alpha, xSub, 1, ySub );
                ++child;

                sourceOffset += sizes[s];
            }
            targetOffset += sizes[t];
        }
    }

    // Loop over the 6 children in the strictly lower triangle, summing in
    // each row
    {
        int targetOffset = 0;
        const int* sizes = _shell.nodeSymmetric->sizes;
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
                const Quasi2dHMatrix& ASub = 
                    *_shell.nodeSymmetric->children[child];
                ASub.TransposeMapVector( alpha, xSub, 1, ySub );

                sourceOffset += sizes[t];
            }
            targetOffset += sizes[s];
        }
    }
}

// C += alpha A B
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::UpdateMatrixWithNodeSymmetric
( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) const
{
    // Loop over the 10 children in the lower triangle, summing in each row
    {
        int child = 0;
        int targetOffset = 0;
        const int* sizes = _shell.nodeSymmetric->sizes;
        for( int t=0; t<4; ++t )
        {
            DenseMatrix<Scalar> CSub;
            CSub.View( C, targetOffset, sizes[t], 0, C.Width() );

            int sourceOffset = 0;
            for( int s=0; s<=t; ++s )
            {
                DenseMatrix<Scalar> BSub;
                BSub.LockedView( B, sourceOffset, sizes[s], 0, B.Width() );

                const Quasi2dHMatrix& ASub = 
                    *_shell.nodeSymmetric->children[child];
                ASub.MapMatrix( alpha, BSub, 1, CSub );
                ++child;

                sourceOffset += sizes[s];
            }
            targetOffset += sizes[t];
        }
    }

    // Loop over the 6 children in the strictly lower triangle, summing in
    // each row
    {
        int targetOffset = 0;
        const int* sizes = _shell.nodeSymmetric->sizes;
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
                const Quasi2dHMatrix& ASub =  
                    *_shell.nodeSymmetric->children[child];
                ASub.TransposeMapMatrix( alpha, BSub, 1, CSub );

                sourceOffset += sizes[t];
            }
            targetOffset += sizes[s];
        }
    }
}

//----------------------------------------------------------------------------//
// Public routines                                                            //
//----------------------------------------------------------------------------//

// Create a square top-level H-matrix
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix
( const SparseMatrix<Scalar>& S,
  int numLevels, bool stronglyAdmissible,
  int xSize, int ySize, int zSize )
: _m(S.m), _n(S.n), _symmetric(S.symmetric), 
  _numLevels(numLevels), _stronglyAdmissible(stronglyAdmissible),
  _xSizeSource(xSize), _xSizeTarget(xSize),
  _ySizeSource(ySize), _ySizeTarget(ySize),
  _zSize(zSize),
  _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0),
  _sourceOffset(0), _targetOffset(0)
{
    ImportSparseMatrix( S );
}

// Create a potentially non-square non-top-level H-matrix
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix
( const SparseMatrix<Scalar>& S,
  int numLevels, bool stronglyAdmissible,
  int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int zSize,
  int xSource, int xTarget,
  int ySource, int yTarget,
  int sourceOffset, int targetOffset )
: _m(S.m), _n(S.n), _symmetric(S.symmetric), 
  _numLevels(numLevels), _stronglyAdmissible(stronglyAdmissible),
  _xSizeSource(xSizeSource), _xSizeTarget(xSizeTarget),
  _ySizeSource(ySizeSource), _ySizeTarget(ySizeTarget), 
  _zSize(zSize),
  _xSource(xSource), _xTarget(xTarget),
  _ySource(ySource), _yTarget(yTarget),
  _sourceOffset(sourceOffset), _targetOffset(targetOffset)
{
    ImportSparseMatrix( S );
}

template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::~Quasi2dHMatrix()
{
    switch( _shellType )
    {
        case NODE:           delete _shell.node;          break;
        case NODE_SYMMETRIC: delete _shell.nodeSymmetric; break;
        case DENSE:          delete _shell.D;             break;
        case FACTOR:         delete _shell.F;             break;
    }
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapVector
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
    hmatrix_tools::Scale( beta, y );
    if( _shellType == NODE )
    {
        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        const int* sourceSizes = _shell.node->sourceSizes;
        const int* targetSizes = _shell.node->targetSizes;
        for( int t=0; t<4; ++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, targetOffset, targetSizes[t] );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sourceOffset, sourceSizes[s] );

                const Quasi2dHMatrix& ASub = *_shell.node->children[s+4*t];
                ASub.MapVector( alpha, xSub, 1, ySub );

                sourceOffset += sourceSizes[s];
            }
            targetOffset += targetSizes[t];
        }
    }
    else if( _shellType == NODE_SYMMETRIC )
    {
        UpdateVectorWithNodeSymmetric( alpha, x, y );
    }
    else if( _shellType == FACTOR )
    {
        hmatrix_tools::MatrixVector( alpha, *_shell.F, x, 1, y );
    }
    else /* _shellType == DENSE */
    {
        hmatrix_tools::MatrixVector( alpha, *_shell.D, x, 1, y );
    }
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapVector
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
    y.Resize( _m );
    MapVector( alpha, x, 0, y );
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::TransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
    hmatrix_tools::Scale( beta, y );
    if( _shellType == NODE )
    {
        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        const int* sourceSizes = _shell.node->sourceSizes;
        const int* targetSizes = _shell.node->targetSizes;
        for( int t=0; t<4; ++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, targetOffset, sourceSizes[t] );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sourceOffset, targetSizes[s] );

                const Quasi2dHMatrix& ASub = *_shell.node->children[t+4*s];
                ASub.TransposeMapVector( alpha, xSub, 1, ySub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
    }
    else if( _shellType == NODE_SYMMETRIC )
    {
        UpdateVectorWithNodeSymmetric( alpha, x, y );
    }
    else if( _shellType == FACTOR )
    {
        hmatrix_tools::MatrixTransposeVector( alpha, *_shell.F, x, 1, y );
    }
    else /* _shellType == DENSE */
    {
        hmatrix_tools::MatrixTransposeVector( alpha, *_shell.D, x, 1, y );
    }
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::TransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
    y.Resize( _n );
    TransposeMapVector( alpha, x, 0, y );
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
    hmatrix_tools::Scale( beta, y );
    if( _shellType == NODE )
    {
        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        const int* sourceSizes = _shell.node->sourceSizes;
        const int* targetSizes = _shell.node->targetSizes;
        for( int t=0; t<4; ++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, targetOffset, sourceSizes[t] );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sourceOffset, targetSizes[s] );

                const Quasi2dHMatrix& ASub = *_shell.node->children[t+4*s];
                ASub.HermitianTransposeMapVector( alpha, xSub, 1, ySub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
    }
    else if( _shellType == NODE_SYMMETRIC )
    {
        Vector<Scalar> xConj;
        hmatrix_tools::Conjugate( x, xConj );
        hmatrix_tools::Conjugate( y );
        UpdateVectorWithNodeSymmetric( Conj(alpha), xConj, y ); 
        hmatrix_tools::Conjugate( y );
    }
    else if( _shellType == FACTOR )
    {
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, *_shell.F, x, 1, y );
    }
    else /* _shellType == DENSE */
    {
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, *_shell.D, x, 1, y );
    }
}

// Having a non-const x allows us to conjugate x in place for the 
// NODE_SYMMETRIC updates.
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
    hmatrix_tools::Scale( beta, y );
    if( _shellType == NODE )
    {
        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        const int* sourceSizes = _shell.node->sourceSizes;
        const int* targetSizes = _shell.node->targetSizes;
        for( int t=0; t<4; ++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, targetOffset, sourceSizes[t] );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sourceOffset, targetSizes[s] );

                const Quasi2dHMatrix& ASub = *_shell.node->children[t+4*s];
                ASub.HermitianTransposeMapVector( alpha, xSub, 1, ySub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
    }
    else if( _shellType == NODE_SYMMETRIC )
    {
        Conjugate( x );
        Conjugate( y );
        UpdateVectorWithNodeSymmetric( Conj(alpha), x, y ); 
        Conjugate( x );
        Conjugate( y );
    }
    else if( _shellType == FACTOR )
    {
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, *_shell.F, x, 1, y );
    }
    else /* _shellType == DENSE */
    {
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, *_shell.D, x, 1, y );
    }
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
    y.Resize( _n );
    HermitianTransposeMapVector( alpha, x, 0 y );
}

// This version allows for temporary in-place conjugation of x
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, Vector<Scalar>& x, Vector<Scalar>& y ) const
{
    y.Resize( _n );
    HermitianTransposeMapVector( alpha, x, 0, y );
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C ) const
{
    hmatrix_tools::Scale( beta, C );
    if( _shellType == NODE )
    {
        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        const int* sourceSizes = _shell.node->sourceSizes;
        const int* targetSizes = _shell.node->targetSizes;
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

                const Quasi2dHMatrix& ASub = *_shell.node->children[s+4*t];
                ASub.MapMatrix( alpha, BSub, 1, CSub );

                sourceOffset += sourceSizes[s];
            }
            targetOffset += targetSizes[t];
        }
    }
    else if( _shellType == NODE_SYMMETRIC )
    {
        UpdateMatrixWithNodeSymmetric( alpha, B, C );
    }
    else if( _shellType == FACTOR )
    {
        hmatrix_tools::MatrixMatrix( alpha, *_shell.F, B, 1, C );
    }
    else /* _shellType == DENSE */
    {
        hmatrix_tools::MatrixMatrix( alpha, *_shell.D, B, 1, C );
    }
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C ) const
{
    C.Resize( _m, B.Width() );
    MapMatrix( alpha, B, 0, C );
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C ) const
{
    hmatrix_tools::Scale( beta, C );
    if( _shellType == NODE )
    {
        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        const int* sourceSizes = _shell.node->sourceSizes;
        const int* targetSizes = _shell.node->targetSizes;
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

                const Quasi2dHMatrix& ASub = *_shell.node->children[t+4*s];
                ASub.TransposeMapMatrix( alpha, BSub, 1, CSub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
    }
    else if( _shellType == NODE_SYMMETRIC )
    {
        UpdateMatrixWithNodeSymmetric( alpha, B, C );
    }
    else if( _shellType == FACTOR )
    {
        hmatrix_tools::MatrixTransposeMatrix( alpha, *_shell.F, B, 1, C );
    }
    else /* _shellType == DENSE */
    {
        hmatrix_tools::MatrixTransposeMatrix( alpha, *_shell.D, B, 1, C );
    }
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C ) const
{
    C.Resize( _n, B.Width() );
    TransposeMapMatrix( alpha, B, 0, C );
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C ) const
{
    hmatrix_tools::Scale( beta, C );
    if( _shellType == NODE )
    {
        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        const int* sourceSizes = _shell.node->sourceSizes;
        const int* targetSizes = _shell.node->targetSizes;
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

                const Quasi2dHMatrix& ASub = *_shell.node->children[t+4*s];
                ASub.HermitianTransposeMapMatrix( alpha, BSub, 1, CSub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
    }
    else if( _shellType == NODE_SYMMETRIC )
    {
        DenseMatrix<Scalar> BConj;
        Conjugate( B, BConj );
        Conjugate( C );
        UpdateMatrixWithNodeSymmetric( alpha, B, C );
        Conjugate( C );
    }
    else if( _shellType == FACTOR )
    {
        hmatrix_tools::MatrixHermitianTransposeMatrix
        ( alpha, *_shell.F, B, 1, C );
    }
    else /* _shellType == DENSE */
    {
        hmatrix_tools::MatrixHermitianTransposeMatrix
        ( alpha, *_shell.D, B, 1, C );
    }
}

// This version allows for temporary in-place conjugation of B
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrix
( Scalar alpha, DenseMatrix<Scalar>& B,
  Scalar beta,  DenseMatrix<Scalar>& C ) const
{
    hmatrix_tools::Scale( beta, C );
    if( _shellType == NODE )
    {
        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        const int* sourceSizes = _shell.node->sourceSizes;
        const int* targetSizes = _shell.node->targetSizes;
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

                const Quasi2dHMatrix& ASub = *_shell.node->children[t+4*s];
                ASub.HermitianTransposeMapMatrix( alpha, BSub, 1, CSub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
    }
    else if( _shellType == NODE_SYMMETRIC )
    {
        Conjugate( B );
        Conjugate( C );
        UpdateMatrixWithNodeSymmetric( alpha, B, C );
        Conjugate( B );
        Conjugate( C );
    }
    else if( _shellType == FACTOR )
    {
        hmatrix_tools::MatrixHermitianTransposeMatrix
        ( alpha, *_shell.F, B, 1, C );
    }
    else /* _shellType == DENSE */
    {
        hmatrix_tools::MatrixHermitianTransposeMatrix
        ( alpha, *_shell.D, B, 1, C );
    }
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C ) const
{
    C.Resize( _n, B.Width() );
    HermitianTransposeMapMatrix( alpha, B, 0, C );
}

// This version allows for temporary in-place conjugation of B
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrix
( Scalar alpha, DenseMatrix<Scalar>& B,
                DenseMatrix<Scalar>& C ) const
{
    C.Resize( _n, B.Width() );
    HermitianTransposeMapMatrix( alpha, B, 0, C );
}

template class psp::Quasi2dHMatrix<float,false>;
template class psp::Quasi2dHMatrix<float,true>;
template class psp::Quasi2dHMatrix<double,false>;
template class psp::Quasi2dHMatrix<double,true>;
template class psp::Quasi2dHMatrix<std::complex<float>,false>;
template class psp::Quasi2dHMatrix<std::complex<float>,true>;
template class psp::Quasi2dHMatrix<std::complex<double>,false>;
template class psp::Quasi2dHMatrix<std::complex<double>,true>;
