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
void BuildMapOnQuadrant
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
        BuildMapOnQuadrant
        ( &map[0], index, level+1, numLevels,
          xSize, ySize, zSize, leftWidth, bottomHeight );
        // Recurse on the lower-right quadrant
        BuildMapOnQuadrant
        ( &map[leftWidth], index, level+1, numLevels,
          xSize, ySize, zSize, rightWidth, bottomHeight );
        // Recurse on the upper-left quadrant
        BuildMapOnQuadrant
        ( &map[bottomHeight*xSize], index, level+1, numLevels,
          xSize, ySize, zSize, leftWidth, topHeight );
        // Recurse on the upper-right quadrant
        BuildMapOnQuadrant
        ( &map[bottomHeight*xSize+leftWidth], index, level+1, numLevels,
          xSize, ySize, zSize, rightWidth, topHeight );
    }
}
} // anonymous namespace

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::BuildNaturalToHierarchicalMap
( std::vector<int>& map, int xSize, int ySize, int zSize, int numLevels )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::BuildNaturalToHierarchicalMap");
#endif
    map.resize( xSize*ySize*zSize );

    // Fill the mapping from the 'natural' x-y-z ordering
    int index = 0;
    BuildMapOnQuadrant
    ( &map[0], index, 0, numLevels, xSize, ySize, zSize, xSize, ySize );
#ifndef RELEASE
    if( index != xSize*ySize*zSize )
        throw std::logic_error("Map recursion is incorrect.");
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Public routines                                                            //
//----------------------------------------------------------------------------//

// Create an empty H-matrix
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix()
: AbstractHMatrix<Scalar>(),
  _xSizeSource(0), _xSizeTarget(0),
  _ySizeSource(0), _ySizeTarget(0),
  _zSize(0),
  _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0)
{ }

// Create a square top-level H-matrix
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix
( const LowRankMatrix<Scalar,Conjugated>& F,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSize, int ySize, int zSize )
: AbstractHMatrix<Scalar>
  (F.Height(),F.Width(),numLevels,maxRank,0,0,false,stronglyAdmissible),
  _xSizeSource(xSize), _xSizeTarget(xSize),
  _ySizeSource(ySize), _ySizeTarget(ySize),
  _zSize(zSize),
  _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0)
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Quasi2dHMatrix");
#endif
    ImportLowRankMatrix( F );
#ifndef RELEASE
    PopCallStack();
#endif
}
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix
( const SparseMatrix<Scalar>& S,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSize, int ySize, int zSize )
: AbstractHMatrix<Scalar>
  (S.height,S.width,numLevels,maxRank,0,0,S.symmetric,stronglyAdmissible),
  _xSizeSource(xSize), _xSizeTarget(xSize),
  _ySizeSource(ySize), _ySizeTarget(ySize),
  _zSize(zSize),
  _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0)
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Quasi2dHMatrix");
#endif
    ImportSparseMatrix( S );
#ifndef RELEASE
    PopCallStack();
#endif
}

// Create a potentially non-square non-top-level H-matrix
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix
( const LowRankMatrix<Scalar,Conjugated>& F,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int zSize,
  int xSource, int xTarget,
  int ySource, int yTarget,
  int sourceOffset, int targetOffset )
: AbstractHMatrix<Scalar>
  (xSizeTarget*ySizeTarget*zSize,xSizeSource*ySizeSource*zSize,
   numLevels,maxRank,sourceOffset,targetOffset,false,stronglyAdmissible),
  _xSizeSource(xSizeSource), _xSizeTarget(xSizeTarget),
  _ySizeSource(ySizeSource), _ySizeTarget(ySizeTarget),
  _zSize(zSize),
  _xSource(xSource), _xTarget(xTarget),
  _ySource(ySource), _yTarget(yTarget)
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Quasi2dHMatrix");
#endif
    ImportLowRankMatrix( F, targetOffset, sourceOffset );
#ifndef RELEASE
    PopCallStack();
#endif
}
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix
( const SparseMatrix<Scalar>& S,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int zSize,
  int xSource, int xTarget,
  int ySource, int yTarget,
  int sourceOffset, int targetOffset )
: AbstractHMatrix<Scalar>
  (xSizeTarget*ySizeTarget*zSize,xSizeSource*ySizeSource*zSize,
   numLevels,maxRank,sourceOffset,targetOffset,
   (S.symmetric && sourceOffset==targetOffset),stronglyAdmissible),
  _xSizeSource(xSizeSource), _xSizeTarget(xSizeTarget),
  _ySizeSource(ySizeSource), _ySizeTarget(ySizeTarget), 
  _zSize(zSize),
  _xSource(xSource), _xTarget(xTarget),
  _ySource(ySource), _yTarget(yTarget)
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Quasi2dHMatrix");
#endif
    ImportSparseMatrix( S, targetOffset, sourceOffset );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::~Quasi2dHMatrix()
{ }

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapVector
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::MapVector (y := H x + y)");
#endif
    hmatrix_tools::Scale( beta, y );
    switch( _shell.type )
    {
    case NODE:
    {
        Node& node = *_shell.data.node;
        const int* sourceSizes = node.sourceSizes;
        const int* targetSizes = node.targetSizes;

        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        for( int t=0; t<4; ++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, targetOffset, targetSizes[t] );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sourceOffset, sourceSizes[s] );

                const Quasi2dHMatrix& ASub = node.Child(t,s);
                ASub.MapVector( alpha, xSub, (Scalar)1, ySub );

                sourceOffset += sourceSizes[s];
            }
            targetOffset += targetSizes[t];
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateVectorWithNodeSymmetric( alpha, x, y );
        break;
    case LOW_RANK:
        hmatrix_tools::MatrixVector( alpha, *_shell.data.F, x, (Scalar)1, y );
        break;
    case DENSE:
        hmatrix_tools::MatrixVector( alpha, *_shell.data.D, x, (Scalar)1, y );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapVector
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::MapVector (y := H x)");
#endif
    y.Resize( this->_height );
    MapVector( alpha, x, 0, y );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::TransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::TransposeMapVector (y := H^T x + y)");
#endif
    hmatrix_tools::Scale( beta, y );
    switch( _shell.type )
    {
    case NODE:
    {
        Node& node = *_shell.data.node;
        const int* sourceSizes = node.sourceSizes;
        const int* targetSizes = node.targetSizes;

        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        for( int t=0; t<4; ++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, targetOffset, sourceSizes[t] );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sourceOffset, targetSizes[s] );

                const Quasi2dHMatrix& ASub = node.Child(s,t);
                ASub.TransposeMapVector( alpha, xSub, (Scalar)1, ySub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateVectorWithNodeSymmetric( alpha, x, y );
        break;
    case LOW_RANK:
        hmatrix_tools::MatrixTransposeVector
        ( alpha, *_shell.data.F, x, (Scalar)1, y );
        break;
    case DENSE:
        hmatrix_tools::MatrixTransposeVector
        ( alpha, *_shell.data.D, x, (Scalar)1, y );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::TransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::TransposeMapVector (y := H^T x)");
#endif
    y.Resize( this->_width );
    TransposeMapVector( alpha, x, 0, y );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack
    ("Quasi2dHMatrix::HermitianTransposeMapVector (y := H^H x + y)");
#endif
    hmatrix_tools::Scale( beta, y );
    switch( _shell.type )
    {
    case NODE:
    {
        Node& node = *_shell.data.node;
        const int* sourceSizes = node.sourceSizes;
        const int* targetSizes = node.targetSizes;

        // Loop over all 16 children, summing in each row
        int targetOffset = 0;

        for( int t=0; t<4; ++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, targetOffset, sourceSizes[t] );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sourceOffset, targetSizes[s] );

                const Quasi2dHMatrix& ASub = node.Child(s,t);
                ASub.HermitianTransposeMapVector
                ( alpha, xSub, (Scalar)1, ySub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        Vector<Scalar> xConj;
        hmatrix_tools::Conjugate( x, xConj );
        hmatrix_tools::Conjugate( y );
        UpdateVectorWithNodeSymmetric( Conj(alpha), xConj, y ); 
        hmatrix_tools::Conjugate( y );
        break;
    }
    case LOW_RANK:
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, *_shell.data.F, x, (Scalar)1, y );
        break;
    case DENSE:
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, *_shell.data.D, x, (Scalar)1, y );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Having a non-const x allows us to conjugate x in place for the 
// NODE_SYMMETRIC updates.
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack
    ("Quasi2dHMatrix::HermitianTransposeMapVector (y := H^H x + y, non-const)");
#endif
    hmatrix_tools::Scale( beta, y );
    switch( _shell.type )
    {
    case NODE:
    {
        Node& node = *_shell.data.node;
        const int* sourceSizes = node.sourceSizes;
        const int* targetSizes = node.targetSizes;

        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        for( int t=0; t<4; ++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, targetOffset, sourceSizes[t] );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sourceOffset, targetSizes[s] );

                const Quasi2dHMatrix& ASub = node.Child(s,t);
                ASub.HermitianTransposeMapVector
                ( alpha, xSub, (Scalar)1, ySub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
        break;
    }
    case NODE_SYMMETRIC:
        hmatrix_tools::Conjugate( x );
        hmatrix_tools::Conjugate( y );
        UpdateVectorWithNodeSymmetric( Conj(alpha), x, y ); 
        hmatrix_tools::Conjugate( x );
        hmatrix_tools::Conjugate( y );
        break;
    case LOW_RANK:
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, *_shell.data.F, x, (Scalar)1, y );
        break;
    case DENSE:
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, *_shell.data.D, x, (Scalar)1, y );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::HermitianTransposeMapVector (y := H^H x)");
#endif
    y.Resize( this->_width );
    HermitianTransposeMapVector( alpha, x, 0, y );
#ifndef RELEASE
    PopCallStack();
#endif
}

// This version allows for temporary in-place conjugation of x
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack
    ("Quasi2dHMatrix::HermitianTransposeMapVector (y := H^H x, non-const)");
#endif
    y.Resize( this->_width );
    HermitianTransposeMapVector( alpha, x, 0, y );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::MapMatrix (D := H D + D)");
#endif
    hmatrix_tools::Scale( beta, C );
    switch( _shell.type )
    {
    case NODE:
    {
        Node& node = *_shell.data.node;
        const int* sourceSizes = node.sourceSizes;
        const int* targetSizes = node.targetSizes;

        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        for( int t=0; t<4; ++t )
        {
            DenseMatrix<Scalar> CSub;
            CSub.View( C, targetOffset, 0, targetSizes[t], C.Width() );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                DenseMatrix<Scalar> BSub;
                BSub.LockedView
                ( B, sourceOffset, 0, sourceSizes[s], B.Width() );

                const Quasi2dHMatrix& ASub = node.Child(t,s);
                ASub.MapMatrix( alpha, BSub, (Scalar)1, CSub );

                sourceOffset += sourceSizes[s];
            }
            targetOffset += targetSizes[t];
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateMatrixWithNodeSymmetric( alpha, B, C );
        break;
    case LOW_RANK:
        hmatrix_tools::MatrixMatrix( alpha, *_shell.data.F, B, (Scalar)1, C );
        break;
    case DENSE:
        hmatrix_tools::MatrixMatrix( alpha, *_shell.data.D, B, (Scalar)1, C );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::MapMatrix (D := H D)");
#endif
    C.SetType( GENERAL );
    C.Resize( this->_height, B.Width() );
    MapMatrix( alpha, B, 0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::TransposeMapMatrix (D := H^T D + D)");
#endif
    hmatrix_tools::Scale( beta, C );
    switch( _shell.type )
    {
    case NODE:
    {
        Node& node = *_shell.data.node;
        const int* sourceSizes = node.sourceSizes;
        const int* targetSizes = node.targetSizes;

        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        for( int t=0; t<4; ++t )
        {
            DenseMatrix<Scalar> CSub;
            CSub.View( C, targetOffset, 0, sourceSizes[t], C.Width() );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                DenseMatrix<Scalar> BSub;
                BSub.LockedView
                ( B, sourceOffset, 0, targetSizes[s], B.Width() );

                const Quasi2dHMatrix& ASub = node.Child(s,t);
                ASub.TransposeMapMatrix( alpha, BSub, (Scalar)1, CSub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateMatrixWithNodeSymmetric( alpha, B, C );
        break;
    case LOW_RANK:
        hmatrix_tools::MatrixTransposeMatrix
        ( alpha, *_shell.data.F, B, (Scalar)1, C );
        break;
    case DENSE:
        hmatrix_tools::MatrixTransposeMatrix
        ( alpha, *_shell.data.D, B, (Scalar)1, C );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::TransposeMapMatrix (D := H^T D)");
#endif
    C.SetType( GENERAL );
    C.Resize( this->_width, B.Width() );
    TransposeMapMatrix( alpha, B, 0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack
    ("Quasi2dHMatrix::HermitianTransposeMapMatrix (D := H^H D + D)");
#endif
    hmatrix_tools::Scale( beta, C );
    switch( _shell.type )
    {
    case NODE:
    {
        Node& node = *_shell.data.node;
        const int* sourceSizes = node.sourceSizes;
        const int* targetSizes = node.targetSizes;

        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        for( int t=0; t<4; ++t )
        {
            DenseMatrix<Scalar> CSub;
            CSub.View( C, targetOffset, 0, sourceSizes[t], C.Width() );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                DenseMatrix<Scalar> BSub;
                BSub.LockedView
                ( B, sourceOffset, 0, targetSizes[s], B.Width() );

                const Quasi2dHMatrix& ASub = node.Child(s,t);
                ASub.HermitianTransposeMapMatrix
                ( alpha, BSub, (Scalar)1, CSub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        DenseMatrix<Scalar> BConj;
        hmatrix_tools::Conjugate( B, BConj );
        hmatrix_tools::Conjugate( C );
        UpdateMatrixWithNodeSymmetric( alpha, B, C );
        hmatrix_tools::Conjugate( C );
        break;
    }
    case LOW_RANK:
        hmatrix_tools::MatrixHermitianTransposeMatrix
        ( alpha, *_shell.data.F, B, (Scalar)1, C );
        break;
    case DENSE:
        hmatrix_tools::MatrixHermitianTransposeMatrix
        ( alpha, *_shell.data.D, B, (Scalar)1, C );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// This version allows for temporary in-place conjugation of B
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrix
( Scalar alpha, DenseMatrix<Scalar>& B,
  Scalar beta,  DenseMatrix<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack
    ("Quasi2dHMatrix::HermitianTransposeMapMatrix (D := H^H D + D, non-const)");
#endif
    hmatrix_tools::Scale( beta, C );
    switch( _shell.type )
    {
    case NODE:
    {
        Node& node = *_shell.data.node;
        const int* sourceSizes = node.sourceSizes;
        const int* targetSizes = node.targetSizes;

        // Loop over all 16 children, summing in each row
        int targetOffset = 0;
        for( int t=0; t<4; ++t )
        {
            DenseMatrix<Scalar> CSub;
            CSub.View( C, targetOffset, 0, sourceSizes[t], C.Width() );

            int sourceOffset = 0;
            for( int s=0; s<4; ++s )
            {
                DenseMatrix<Scalar> BSub;
                BSub.LockedView
                ( B, sourceOffset, 0, targetSizes[s], B.Width() );

                const Quasi2dHMatrix& ASub = node.Child(s,t);
                ASub.HermitianTransposeMapMatrix
                ( alpha, BSub, (Scalar)1, CSub );

                sourceOffset += targetSizes[s];
            }
            targetOffset += sourceSizes[t];
        }
        break;
    }
    case NODE_SYMMETRIC:
        hmatrix_tools::Conjugate( B );
        hmatrix_tools::Conjugate( C );
        UpdateMatrixWithNodeSymmetric( alpha, B, C );
        hmatrix_tools::Conjugate( B );
        hmatrix_tools::Conjugate( C );
        break;
    case LOW_RANK:
        hmatrix_tools::MatrixHermitianTransposeMatrix
        ( alpha, *_shell.data.F, B, (Scalar)1, C );
        break;
    case DENSE:
        hmatrix_tools::MatrixHermitianTransposeMatrix
        ( alpha, *_shell.data.D, B, (Scalar)1, C );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrix
( Scalar alpha, const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::HermitianTransposeMapMatrix (D := H^H D)");
#endif
    C.SetType( GENERAL );
    C.Resize( this->_width, B.Width() );
    HermitianTransposeMapMatrix( alpha, B, 0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

// This version allows for temporary in-place conjugation of B
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapMatrix
( Scalar alpha, DenseMatrix<Scalar>& B,
                DenseMatrix<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack
    ("Quasi2dHMatrix::HermitianTransposeMapMatrix (D := H^H D, non-const)");
#endif
    C.SetType( GENERAL );
    C.Resize( this->_width, B.Width() );
    HermitianTransposeMapMatrix( alpha, B, 0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

/*\
|*| Computational routines specific to Quasi2dHMatrix
\*/

// A := B
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::CopyFrom
( const Quasi2dHMatrix<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::CopyFrom");
#endif
    this->_height = B.Height();
    this->_width = B.Width();
    this->_numLevels = B.NumLevels();
    this->_maxRank = B.MaxRank();
    this->_sourceOffset = B.SourceOffset();
    this->_targetOffset = B.TargetOffset();
    this->_symmetric = B.Symmetric();
    this->_stronglyAdmissible = B.StronglyAdmissible();
    _xSizeSource = B.XSizeSource();
    _xSizeTarget = B.XSizeTarget();
    _ySizeSource = B.YSizeSource();
    _ySizeTarget = B.YSizeTarget();
    _zSize = B.ZSize();
    _xSource = B.XSource();
    _xTarget = B.XTarget();
    _ySource = B.YSource();
    _yTarget = B.YTarget();

    // Delete the old type and switch
    switch( _shell.type )
    {
    case NODE:           delete _shell.data.node;          break;
    case NODE_SYMMETRIC: delete _shell.data.nodeSymmetric; break;
    case LOW_RANK:       delete _shell.data.F;             break;
    case DENSE:          delete _shell.data.D;             break;
    }
    _shell.type = B._shell.type;

    switch( _shell.type )
    {
    case NODE:
    {
        _shell.data.node = 
            new Node
            ( _xSizeSource, _xSizeTarget,
              _ySizeSource, _ySizeTarget, _zSize );
        Node& nodeA = *_shell.data.node;
        const Node& nodeB = *B._shell.data.node;
        for( int i=0; i<16; ++i )
        {
            nodeA.children[i] = new Quasi2dHMatrix<Scalar,Conjugated>;
            nodeA.children[i]->CopyFrom( *nodeB.children[i] );
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        _shell.data.nodeSymmetric = 
            new NodeSymmetric( _xSizeSource, _ySizeSource, _zSize );
        NodeSymmetric& nodeA = *_shell.data.nodeSymmetric;
        const NodeSymmetric& nodeB = *B._shell.data.nodeSymmetric;
        for( int i=0; i<10; ++i )
        {
            nodeA.children[i] = new Quasi2dHMatrix<Scalar,Conjugated>;
            nodeA.children[i]->CopyFrom( *nodeB.children[i] );
        }
        break;
    }
    case LOW_RANK:
        _shell.data.F = new LowRankMatrix<Scalar,Conjugated>;
        hmatrix_tools::Copy( *B._shell.data.F, *_shell.data.F );
        break;
    case DENSE:
        _shell.data.D = new DenseMatrix<Scalar>;
        hmatrix_tools::Copy( *B._shell.data.D, *_shell.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := alpha A
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::Scale( Scalar alpha )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Scale");
#endif
    switch( _shell.type )
    {
    case NODE:
    {
        Node& nodeA = *_shell.data.node;
        for( int i=0; i<16; ++i )
            nodeA.children[i]->Scale( alpha );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *_shell.data.nodeSymmetric;
        for( int i=0; i<10; ++i )
            nodeA.children[i]->Scale( alpha );
        break;
    }
    case LOW_RANK:
        hmatrix_tools::Scale( alpha, *_shell.data.F );
        break;
    case DENSE:
        hmatrix_tools::Scale( alpha, *_shell.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := I
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::SetToIdentity()
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::SetToIdentity");
#endif
    switch( _shell.type )
    {
    case NODE:
    {
        Node& nodeA = *_shell.data.node;
        for( int i=0; i<4; ++i )
        {
            for( int j=0; j<4; ++j )
            {
                if( i == j )
                    nodeA.Child(i,j).SetToIdentity();
                else
                    nodeA.Child(i,j).Scale( (Scalar)0 );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *_shell.data.nodeSymmetric;
        for( int i=0; i<4; ++i )
        {
            for( int j=0; j<i; ++j )
                nodeA.Child(i,j).Scale( (Scalar)0 );
            nodeA.Child(i,i).SetToIdentity();
        }
        break;
    }
    case LOW_RANK:
    {
#ifndef RELEASE
        throw std::logic_error("Error in SetToIdentity logic.");
#endif
        break;
    }
    case DENSE:
    {
        DenseMatrix<Scalar>& D = *_shell.data.D;
        hmatrix_tools::Scale( (Scalar)0, D );
        for( int j=0; j<D.Width(); ++j )
        {
            if( j < D.Height() )
                D.Set( j, j, (Scalar)1 );
            else
                break;
        }
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := alpha B + A
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::UpdateWith
( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::UpdateWith");
#endif
    switch( _shell.type )
    {
    case NODE:
    {
        Node& nodeA = *_shell.data.node;
        const Node& nodeB = *B._shell.data.node;
        for( int i=0; i<16; ++i )
            nodeA.children[i]->UpdateWith( alpha, *nodeB.children[i] );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *_shell.data.nodeSymmetric;
        const NodeSymmetric& nodeB = *B._shell.data.nodeSymmetric;
        for( int i=0; i<10; ++i )
            nodeA.children[i]->UpdateWith( alpha, *nodeB.children[i] );
        break;
    }
    case LOW_RANK:
        hmatrix_tools::MatrixUpdateRounded
        ( this->_maxRank, 
          alpha, *B._shell.data.F, (Scalar)1, *_shell.data.F );
        break;
    case DENSE:
        hmatrix_tools::MatrixUpdate
        ( alpha, *B._shell.data.D, (Scalar)1, *_shell.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// C := alpha A B
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B,
                      Quasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::MapMatrix H := H H");
    if( this->Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( this->NumLevels() != B.NumLevels() )
        throw std::logic_error
        ("Can only multiply H-matrices with same number of levels.");
    if( this->StronglyAdmissible() != B.StronglyAdmissible() )
        throw std::logic_error
        ("Can only multiply H-matrices with same admissiblity.");
#endif
    C._height = this->Height();
    C._width = B.Width();
    C._numLevels = this->NumLevels();
    C._maxRank = this->MaxRank();
    C._sourceOffset = B.SourceOffset();
    C._targetOffset = this->TargetOffset();
    C._symmetric = false;
    C._stronglyAdmissible = this->StronglyAdmissible();
    C._xSizeSource = B.XSizeSource();
    C._xSizeTarget = this->XSizeTarget();
    C._ySizeSource = B.YSizeSource();
    C._ySizeTarget = this->YSizeTarget();
    C._zSize = this->ZSize();
    C._xSource = B.XSource();
    C._xTarget = this->XTarget();
    C._ySource = B.YSource();
    C._yTarget = this->YTarget();

    // Delete the old type
    switch( C._shell.type )
    {
    case NODE:           delete C._shell.data.node;          break;
    case NODE_SYMMETRIC: delete C._shell.data.nodeSymmetric; break;
    case LOW_RANK:       delete C._shell.data.F;             break;
    case DENSE:          delete C._shell.data.D;             break;
    }

    if( C.Admissible( C._xSource, C._xTarget, C._ySource, C._yTarget ) )
    {
        C._shell.type = LOW_RANK;
        C._shell.data.F = new LowRankMatrix<Scalar,Conjugated>;
        if( this->IsLowRank() && B.IsLowRank() )
        {
            // C.F := alpha A.F B.F
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.F, *C._shell.data.F );
        }
        else if( this->IsLowRank() && B.IsHierarchical() )
        {
            // C.F.U := A.F.U
            hmatrix_tools::Copy( _shell.data.F->U, C._shell.data.F->U );
            // C.F.V := scale B^H A.F.V
            const Scalar scale = ( Conjugated ? Conj(alpha) : alpha );
            B.HermitianTransposeMapMatrix
            ( scale, _shell.data.F->V, C._shell.data.F->V );
        }
        else if( this->IsLowRank() && B.IsDense() )
        {
            // C.F.U := A.F.U
            hmatrix_tools::Copy( _shell.data.F->U, C._shell.data.F->U );
            // C.F.V := scale B^H A.F.V
            const Scalar scale = ( Conjugated ? Conj(alpha) : alpha );
            hmatrix_tools::MatrixHermitianTransposeMatrix
            ( scale, *B._shell.data.D, _shell.data.F->V, C._shell.data.F->V );
        }
        else if( this->IsHierarchical() && B.IsLowRank() )
        {
            // C.F.U := alpha A B.F.U
            this->MapMatrix( alpha, B._shell.data.F->U, C._shell.data.F->U );
            // C.F.V := B.F.V
            hmatrix_tools::Copy( B._shell.data.F->V, C._shell.data.F->V );
        }
        else if( this->IsHierarchical() && B.IsHierarchical() )
        {
            // C.F := alpha H H
            const int oversampling = 4; // lift this definition
            hmatrix_tools::MatrixMatrix
            ( oversampling, alpha, *this, B, *C._shell.data.F );
        }
        else if( this->IsDense() && B.IsLowRank() )
        {
            // C.F.U := alpha A B.F.U
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.D, B._shell.data.F->U, C._shell.data.F->U );
            // C.F.V := B.F.V
            hmatrix_tools::Copy( B._shell.data.F->V, C._shell.data.F->V );
        }
        else if( this->IsDense() && B.IsDense() )
        {
            hmatrix_tools::MatrixMatrix
            ( C.MaxRank(),
              alpha, *_shell.data.D, *B._shell.data.D, *C._shell.data.F );
        }
#ifndef RELEASE
        else
        {
            std::logic_error("Invalid H-matrix combination.");
        }
#endif
    }
    else if( C.NumLevels() > 1 )
    {
        // A product of two matrices will be assumed non-symmetric.
        C._shell.type = NODE;
        C._shell.data.node = 
            new Node
            ( C._xSizeSource, C._xSizeTarget, C._ySizeSource, C._ySizeTarget,
              C._zSize );

#ifndef RELEASE
        if( this->IsDense() || B.IsDense() )
            throw std::logic_error("Invalid H-matrix combination");
#endif
        if( this->IsLowRank() && B.IsLowRank() )
        {
            // Form W := alpha A B
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.F, W );

            // Form C :~= W
            C.ImportLowRankMatrix( W );
        }
        else if( this->IsLowRank() && B.IsHierarchical() )
        {
            // Form W := alpha A B
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::Copy( _shell.data.F->U, W.U );
            if( Conjugated )
            {
                B.HermitianTransposeMapMatrix
                ( Conj(alpha), _shell.data.F->V, W.V );
            }
            else
            {
                B.TransposeMapMatrix
                ( alpha, _shell.data.F->V, W.V );
            }

            // Form C :=~ W
            C.ImportLowRankMatrix( W );
        }
        else if( this->IsHierarchical() && B.IsLowRank() )
        {
            // Form W := alpha A B    
            LowRankMatrix<Scalar,Conjugated> W;
            this->MapMatrix( alpha, B._shell.data.F->U, W.U );
            hmatrix_tools::Copy( B._shell.data.F->V, W.V );

            // Form C :=~ W
            C.ImportLowRankMatrix( W );
        }
        else
        {
            if( this->Symmetric() && B.Symmetric() )
            {
                throw std::logic_error("Unsupported h-matrix multipy case.");
            }
            else if( this->Symmetric() && !B.Symmetric() )
            {
                throw std::logic_error("Unsupported h-matrix multiply case.");
            }
            else if( !this->Symmetric() && B.Symmetric() )
            {
                throw std::logic_error("Unsupported h-matrix multiply case.");
            }
            else /* !this->Symmetric() && !B.Symmetric() */
            {
                const Node& nodeA = *this->_shell.data.node;
                const Node& nodeB = *B._shell.data.node;
                Node& nodeC = *C._shell.data.node;
#ifndef RELEASE
                if( nodeA.children.size() != 16 )
                    throw std::logic_error("nodeA not properly initialized");
                if( nodeB.children.size() != 16 )
                    throw std::logic_error("nodeB not properly initialized");
                if( nodeC.children.size() != 16 )
                    throw std::logic_error("nodeC not properly initialized");
#endif
                for( int t=0; t<4; ++t )
                {
                    for( int s=0; s<4; ++s )
                    {
                        // Create the H-matrix here
                        nodeC.children[s+4*t] = 
                            new Quasi2dHMatrix<Scalar,Conjugated>;

                        // Initialize the [t,s] box of C with the first product
                        nodeA.Child(t,0).MapMatrix
                        ( alpha, nodeB.Child(0,s), nodeC.Child(t,s) );
        
                        // Add the other three products onto it
                        for( int u=1; u<4; ++u )
                        {
                            nodeA.Child(t,u).MapMatrix
                            ( alpha, nodeB.Child(u,s), 1, nodeC.Child(t,s) );
                        }
                    }
                }
            }
        }
    }
    else /* C is dense */
    {
#ifndef RELEASE
        if( this->IsHierarchical() || B.IsHierarchical() )
            throw std::logic_error("Invalid combination of H-matrices.");
#endif
        C._shell.type = DENSE;
        C._shell.data.D = new DenseMatrix<Scalar>;

        if( this->IsDense() && B.IsDense() )
        {
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.D, *B._shell.data.D, *C._shell.data.D );
        }
        else if( this->IsDense() && B.IsLowRank() )
        {
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.D, *B._shell.data.F, *C._shell.data.D );
        }
        else if( this->IsLowRank() && B.IsDense() )
        {
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.D, *C._shell.data.D );
        }
        else /* both low-rank */
        {
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.F, *C._shell.data.D );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// C := alpha A B + beta C
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B,
  Scalar beta,        Quasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::MapMatrix (H := H H + H)");
    if( this->Width() != B.Height() || 
        this->Height() != C.Height() || B.Width() != C.Width() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( this->NumLevels() != B.NumLevels() || 
        this->NumLevels() != C.NumLevels() )
        throw std::logic_error
        ("Can only multiply H-matrices with same number of levels.");
    if( this->StronglyAdmissible() != B.StronglyAdmissible() ||
        this->StronglyAdmissible() != C.StronglyAdmissible() )
        throw std::logic_error
        ("Can only multiply H-matrices with same admissiblity.");
    if( C.Symmetric() )
        throw std::logic_error("Symmetric updates not yet supported.");
#endif
    if( C.Admissible( C._xSource, C._xTarget, C._ySource, C._yTarget ) )
    {
        if( this->IsLowRank() && B.IsLowRank() )
        {
            // W := alpha A.F B.F
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.F, W );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._shell.data.F );
        }
        else if( this->IsLowRank() && B.IsHierarchical() )
        {
            // W := alpha A.F B
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::Copy( _shell.data.F->U, W.U );
            const Scalar scale = ( Conjugated ? Conj(alpha) : alpha );
            B.HermitianTransposeMapMatrix
            ( scale, _shell.data.F->V, W.V );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._shell.data.F );
        }
        else if( this->IsLowRank() && B.IsDense() )
        {
            // W := alpha A.F B.D
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::Copy( _shell.data.F->U, W.U );
            const Scalar scale = ( Conjugated ? Conj(alpha) : alpha );
            hmatrix_tools::MatrixHermitianTransposeMatrix
            ( scale, *B._shell.data.D, _shell.data.F->V, W.V );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._shell.data.F );
        }
        else if( this->IsHierarchical() && B.IsLowRank() )
        {
            // W := alpha A B.F
            LowRankMatrix<Scalar,Conjugated> W;
            this->MapMatrix( alpha, B._shell.data.F->U, W.U );
            hmatrix_tools::Copy( B._shell.data.F->V, W.V );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._shell.data.F );
        }
        else if( this->IsHierarchical() && B.IsHierarchical() )
        {
            // W := alpha A B
            LowRankMatrix<Scalar,Conjugated> W;
            const int oversampling = 4; // lift this definition
            hmatrix_tools::MatrixMatrix
            ( oversampling, alpha, *this, B, W );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._shell.data.F );
        }
        else if( this->IsDense() && B.IsLowRank() )
        {
            // W := alpha A.D B.F
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.D, B._shell.data.F->U, W.U );
            hmatrix_tools::Copy( B._shell.data.F->V, W.V );

            // C.F :=~ W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._shell.data.F );
        }
        else if( this->IsDense() && B.IsDense() )
        {
            hmatrix_tools::MatrixMatrix
            ( C.MaxRank(),
              alpha, *_shell.data.D, *B._shell.data.D, beta, *C._shell.data.F );
        }
#ifndef RELEASE
        else
        {
            std::logic_error("Invalid H-matrix combination.");
        }
#endif
    }
    else if( C.NumLevels() > 1 )
    {

#ifndef RELEASE
        if( this->IsDense() || B.IsDense() )
            throw std::logic_error("Invalid H-matrix combination");
#endif
        if( this->IsLowRank() && B.IsLowRank() )
        {
            // HERE        
        }
        else if( this->IsLowRank() && B.IsHierarchical() )
        {

        }
        else if( this->IsHierarchical() && B.IsLowRank() )
        {

        }
        else
        {
            if( this->Symmetric() && B.Symmetric() )
            {
                throw std::logic_error("Unsupported h-matrix multipy case.");
            }
            else if( this->Symmetric() && !B.Symmetric() )
            {
                throw std::logic_error("Unsupported h-matrix multiply case.");
            }
            else if( !this->Symmetric() && B.Symmetric() )
            {
                throw std::logic_error("Unsupported h-matrix multiply case.");
            }
            else /* !this->Symmetric() && !B.Symmetric() */
            {
                const Node& nodeA = *this->_shell.data.node;
            const Node& nodeB = *B._shell.data.node;
                Node& nodeC = *C._shell.data.node;
#ifndef RELEASE
                if( nodeA.children.size() != 16 )
                    throw std::logic_error("nodeA not properly initialized");
                if( nodeB.children.size() != 16 )
                    throw std::logic_error("nodeB not properly initialized");
                if( nodeC.children.size() != 16 )
                    throw std::logic_error("nodeC not properly initialized");
#endif
                for( int t=0; t<4; ++t )
                {
                    for( int s=0; s<4; ++s )
                    {
                        // Scale the [t,s] box of C in the first product
                        nodeA.Child(t,0).MapMatrix
                        ( alpha, nodeB.Child(0,s), beta, nodeC.Child(t,s) ); 
        
                        // Add the other three products onto it
                        for( int u=1; u<4; ++u )
                        {
                            nodeA.Child(t,u).MapMatrix
                            ( alpha, nodeB.Child(u,s), 
                              (Scalar)1, nodeC.Child(t,s) ); 
                        }
                    }
                }
            }
        }
    }
    else /* C is dense */
    {
#ifndef RELEASE
        if( this->IsHierarchical() || B.IsHierarchical() )
            throw std::logic_error("Invalid combination of H-matrices.");
#endif
        if( this->IsDense() && B.IsDense() )
        {
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.D, *B._shell.data.D, beta, *C._shell.data.D );
        }
        else if( this->IsDense() && B.IsLowRank() )
        {
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.D, *B._shell.data.F, beta, *C._shell.data.D );
        }
        else if( this->IsLowRank() && B.IsDense() )
        {
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.D, beta, *C._shell.data.D );
        }
        else /* both low-rank */
        {
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.F, beta, *C._shell.data.D );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := inv(A)
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::Invert()
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Invert");
    if( this->Height() != this->Width() )
        throw std::logic_error("Cannot invert non-square matrices");
    if( this->IsLowRank() )
        throw std::logic_error("Cannot invert low-rank matrices");
#endif
    switch( _shell.type )
    {
    case NODE:
    {
        // We will form the inverse in the original matrix, so we only need to
        // create a temporary matrix.
        Quasi2dHMatrix<Scalar,Conjugated> B; 
        B.CopyFrom( *this );

        // Initialize our soon-to-be inverse as the identity
        this->SetToIdentity();

        Node& nodeA = *_shell.data.node;
        Node& nodeB = *B._shell.data.node;
#ifndef RELEASE
        if( nodeA.children.size() != 16 )
            throw std::logic_error("nodeA not properly initialized");
        if( nodeB.children.size() != 16 )
            throw std::logic_error("nodeB not properly initialized");
#endif
        for( int l=0; l<4; ++l )
        {
            // A_ll := inv(B_ll)
            nodeA.Child(l,l).CopyFrom( nodeB.Child(l,l) );
            nodeA.Child(l,l).Invert();

            // NOTE: Can be skipped for upper-triangular matrices
            for( int j=0; j<l; ++j )
            {
                // A_lj := A_ll A_lj
                Quasi2dHMatrix<Scalar,Conjugated> C;
                C.CopyFrom( nodeA.Child(l,j) );
                nodeA.Child(l,l).MapMatrix( (Scalar)1, C, nodeA.Child(l,j) );
            }

            // NOTE: Can be skipped for lower-triangular matrices
            for( int j=l+1; j<4; ++j )
            {
                // B_lj := A_ll B_lj
                Quasi2dHMatrix<Scalar,Conjugated> C;
                C.CopyFrom( nodeB.Child(l,j) );
                nodeA.Child(l,l).MapMatrix( (Scalar)1, C, nodeB.Child(l,j) );
            }

            for( int i=l+1; i<4; ++i )
            {
                // NOTE: Can be skipped for upper triangular matrices.
                for( int j=0; j<=l; ++j )
                {
                    // A_ij -= B_il A_lj
                    nodeB.Child(i,l).MapMatrix
                    ( (Scalar)-1, nodeA.Child(l,j), 
                      (Scalar)1,  nodeA.Child(i,j) );
                }
                // NOTE: Can be skipped for either lower or upper-triangular
                //       matrices, effectively decoupling the diagonal block
                //       inversions.
                for( int j=l+1; j<4; ++j )
                {
                    // B_ij -= B_il B_lj
                    nodeB.Child(i,l).MapMatrix
                    ( (Scalar)-1, nodeB.Child(l,j),
                      (Scalar)1,  nodeB.Child(i,j) );
                }
            }
        }

        // NOTE: Can be skipped for lower-triangular matrices.
        for( int l=3; l>=0; --l )
        {
            for( int i=l-1; i>=0; --i )
            {
                // NOTE: For upper-triangular matrices, change the loop to
                //       for( int j=l; j<4; ++j )
                for( int j=0; j<4; ++j )
                {
                    // A_ij -= B_il A_lj
                    nodeB.Child(i,l).MapMatrix
                    ( (Scalar)-1, nodeA.Child(l,j),
                      (Scalar)1,  nodeA.Child(i,j) );
                }
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        throw std::logic_error("Symmetric inversion not yet supported.");
        break;
    }
    case DENSE:
        hmatrix_tools::Invert( *_shell.data.D );
        break;
    case LOW_RANK:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Private routines                                                           //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
bool
psp::Quasi2dHMatrix<Scalar,Conjugated>::Admissible
( int xSource, int xTarget, int ySource, int yTarget ) const
{
    if( this->_stronglyAdmissible )
        return std::max(std::abs(xSource-xTarget),std::abs(ySource-yTarget))>1;
    else
        return xSource != xTarget || ySource != yTarget;
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::ImportLowRankMatrix
( const LowRankMatrix<Scalar,Conjugated>& F, int iOffset, int jOffset )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::ImportLowRankMatrix");
#endif
    // View our portions of F
    DenseMatrix<Scalar> FUSub, FVSub;
    FUSub.LockedView( F.U, iOffset, 0, this->_height, F.Rank() );
    FVSub.LockedView( F.V, jOffset, 0, this->_width,  F.Rank() );

    // Delete the old type
    switch( _shell.type )
    {
    case NODE:           delete _shell.data.node;          break;
    case NODE_SYMMETRIC: delete _shell.data.nodeSymmetric; break;
    case LOW_RANK:       delete _shell.data.F;             break;
    case DENSE:          delete _shell.data.D;             break;
    }

    if( Admissible( _xSource, _xTarget, _ySource, _yTarget ) )
    {
        _shell.type = LOW_RANK;
        _shell.data.F = new LowRankMatrix<Scalar,Conjugated>;
        hmatrix_tools::Copy( FUSub, _shell.data.F->U );
        hmatrix_tools::Copy( FVSub, _shell.data.F->V );
    }
    else if( this->_numLevels > 1 )
    {
        if( this->_symmetric && this->_sourceOffset == this->_targetOffset )
        {
            _shell.type = NODE_SYMMETRIC;
            _shell.data.nodeSymmetric = 
                new NodeSymmetric( _xSizeSource, _ySizeSource, _zSize );

            NodeSymmetric& node = *_shell.data.nodeSymmetric;
            const int* xSizes = node.xSizes;
            const int* ySizes = node.ySizes;
            const int* sizes = node.sizes;

            int child = 0;
            int targetOffset = this->_targetOffset;
            for( int t=0; t<4; ++t )
            {
                int sourceOffset = this->_targetOffset;
                for( int s=0; s<=t; ++s )
                {
                    node.children[child++] = 
                      new Quasi2dHMatrix
                      ( F, 
                        this->_numLevels-1, this->_maxRank, 
                        this->_stronglyAdmissible,
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
            _shell.type = NODE;
            _shell.data.node = 
                new Node
                ( _xSizeSource, _xSizeTarget,
                  _ySizeSource, _ySizeTarget, _zSize );

            Node& node = *_shell.data.node;
            const int* xSourceSizes = node.xSourceSizes;
            const int* xTargetSizes = node.xTargetSizes;
            const int* ySourceSizes = node.ySourceSizes;
            const int* yTargetSizes = node.yTargetSizes;
            const int* sourceSizes  = node.sourceSizes;
            const int* targetSizes  = node.targetSizes;

            int targetOffset = this->_targetOffset;
            for( int t=0; t<4; ++t )
            {
                int sourceOffset = this->_sourceOffset;
                for( int s=0; s<4; ++s )
                {
                    node.children[s+4*t] = 
                      new Quasi2dHMatrix
                      ( F,
                        this->_numLevels-1, this->_maxRank,
                        this->_stronglyAdmissible,
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
        _shell.type = DENSE;
        _shell.data.D = new DenseMatrix<Scalar>( this->_height, this->_width );
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, this->_height, this->_width, F.Rank(),
          1, FUSub.LockedBuffer(), FUSub.LDim(),
             FVSub.LockedBuffer(), FVSub.LDim(),
          0, _shell.data.D->Buffer(), _shell.data.D->LDim() );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::UpdateWithLowRankMatrix
( Scalar alpha, 
  const LowRankMatrix<Scalar,Conjugated>& F, int iOffset, int jOffset )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::UpdateWithLowRankMatrix");
#endif
    // View our portions of F
    LowRankMatrix<Scalar,Conjugated> FSub;
    FSub.U.LockedView( F.U, iOffset, 0, this->_height, F.Rank() );
    FSub.V.LockedView( F.V, jOffset, 0, this->_width,  F.Rank() );

    if( Admissible( _xSource, _xTarget, _ySource, _yTarget ) )
    {
        hmatrix_tools::MatrixUpdateRounded
        ( this->MaxRank(), alpha, FSub, (Scalar)1, *_shell.data.F );
    }
    else if( this->_numLevels > 1 )
    {
        if( this->_symmetric && this->_sourceOffset == this->_targetOffset )
        {
            NodeSymmetric& node = *_shell.data.nodeSymmetric;
            const int* sizes = node.sizes;

            int targetOffset = this->_targetOffset;
            for( int t=0; t<4; ++t )
            {
                int sourceOffset = this->_targetOffset;
                for( int s=0; s<=t; ++s )
                {
                    node.Child(t,s).UpdateWithLowRankMatrix
                    ( alpha, FSub, targetOffset, sourceOffset );

                    sourceOffset += sizes[s];
                }
                targetOffset += sizes[t];
            }
        }
        else
        {
            Node& node = *_shell.data.node;
            const int* sourceSizes  = node.sourceSizes;
            const int* targetSizes  = node.targetSizes;

            int targetOffset = this->_targetOffset;
            for( int t=0; t<4; ++t )
            {
                int sourceOffset = this->_sourceOffset;
                for( int s=0; s<4; ++s )
                {
                    node.Child(t,s).UpdateWithLowRankMatrix
                    ( alpha, FSub, targetOffset, sourceOffset );

                    sourceOffset += sourceSizes[s];
                }
                targetOffset += targetSizes[t];
            }
        }
    }
    else
    {
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, this->_height, this->_width, F.Rank(),
          alpha, FSub.U.LockedBuffer(), FSub.U.LDim(),
                 FSub.V.LockedBuffer(), FSub.V.LDim(),
          1, _shell.data.D->Buffer(), _shell.data.D->LDim() );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}


template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::ImportSparseMatrix
( const SparseMatrix<Scalar>& S, int iOffset, int jOffset )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::ImportSparseMatrix");
#endif
    // Delete the old type
    switch( _shell.type )
    {
    case NODE:           delete _shell.data.node;          break;
    case NODE_SYMMETRIC: delete _shell.data.nodeSymmetric; break;
    case LOW_RANK:       delete _shell.data.F;             break;
    case DENSE:          delete _shell.data.D;             break;
    }

    if( Admissible( _xSource, _xTarget, _ySource, _yTarget ) )
    {
        _shell.type = LOW_RANK;
        _shell.data.F = new LowRankMatrix<Scalar,Conjugated>;
        hmatrix_tools::ConvertSubmatrix
        ( *_shell.data.F, S, iOffset, jOffset, this->_height, this->_width );
#ifndef RELEASE
        std::cout << "Converted sparse " 
                  << this->_height << " x " << this->_width 
                  << " matrix with offsets (" << iOffset << ","
                  << jOffset << ") to rank "
                  << _shell.data.F->Rank() << std::endl;
#endif
    }
    else if( this->NumLevels() > 1 )
    {
        if( this->_symmetric && this->_sourceOffset == this->_targetOffset )
        {
            _shell.type = NODE_SYMMETRIC;
            _shell.data.nodeSymmetric = 
                new NodeSymmetric( _xSizeSource, _ySizeSource, _zSize );

            NodeSymmetric& node = *_shell.data.nodeSymmetric;
            int* xSizes = node.xSizes;
            int* ySizes = node.ySizes;
            int* sizes = node.sizes;

            int child = 0;
            int targetOffset = this->_targetOffset;
            for( int t=0; t<4; ++t )
            {
                int sourceOffset = this->_targetOffset;
                for( int s=0; s<=t; ++s )
                {
                    node.children[child++] = 
                      new Quasi2dHMatrix
                      ( S, 
                        this->_numLevels-1, this->_maxRank,
                        this->_stronglyAdmissible,
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
            _shell.type = NODE;
            _shell.data.node = 
                new Node
                ( _xSizeSource, _xSizeTarget,
                  _ySizeSource, _ySizeTarget, _zSize );
            
            Node& node = *_shell.data.node;
            int* xSourceSizes = node.xSourceSizes;
            int* ySourceSizes = node.ySourceSizes;
            int* xTargetSizes = node.xTargetSizes;
            int* yTargetSizes = node.yTargetSizes;
            int* sourceSizes = node.sourceSizes;
            int* targetSizes = node.targetSizes;

            int targetOffset = this->_targetOffset;
            for( int t=0; t<4; ++t )
            {
                int sourceOffset = this->_sourceOffset;
                for( int s=0; s<4; ++s )
                {
                    node.children[s+4*t] = 
                      new Quasi2dHMatrix
                      ( S,
                        this->_numLevels-1, this->_maxRank,
                        this->_stronglyAdmissible,
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
        _shell.type = DENSE;
        _shell.data.D = new DenseMatrix<Scalar>;
        hmatrix_tools::ConvertSubmatrix
        ( *_shell.data.D, S, iOffset, jOffset, this->_height, this->_width );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// y += alpha A x
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::UpdateVectorWithNodeSymmetric
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::UpdateVectorWithNodeSymmetric");
#endif
    NodeSymmetric& node = *_shell.data.nodeSymmetric;

    // Loop over the 10 children in the lower triangle, summing in each row
    {
        int targetOffset = 0;
        const int* sizes = node.sizes;
        for( int t=0; t<4; ++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, targetOffset, sizes[t] );

            int sourceOffset = 0;
            for( int s=0; s<=t; ++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sourceOffset, sizes[s] );

                const Quasi2dHMatrix& ASub = node.Child(t,s);
                ASub.MapVector( alpha, xSub, (Scalar)1, ySub );

                sourceOffset += sizes[s];
            }
            targetOffset += sizes[t];
        }
    }

    // Loop over the 6 children in the strictly lower triangle, summing in
    // each row
    {
        int targetOffset = 0;
        const int* sizes = node.sizes;
        for( int s=0; s<4; ++s )
        {
            Vector<Scalar> ySub;
            ySub.View( y, targetOffset, sizes[s] );

            int sourceOffset = targetOffset + sizes[s];
            for( int t=s+1; t<4; ++t )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sourceOffset, sizes[t] );

                const Quasi2dHMatrix& ASub = node.Child(t,s);
                ASub.TransposeMapVector( alpha, xSub, (Scalar)1, ySub );

                sourceOffset += sizes[t];
            }
            targetOffset += sizes[s];
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// C += alpha A B
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::UpdateMatrixWithNodeSymmetric
( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::UpdateMatrixWithNodeSymmetric");
#endif
    NodeSymmetric& node = *_shell.data.nodeSymmetric;

    // Loop over the 10 children in the lower triangle, summing in each row
    {
        int targetOffset = 0;
        const int* sizes = node.sizes;
        for( int t=0; t<4; ++t )
        {
            DenseMatrix<Scalar> CSub;
            CSub.View( C, targetOffset, 0, sizes[t], C.Width() );

            int sourceOffset = 0;
            for( int s=0; s<=t; ++s )
            {
                DenseMatrix<Scalar> BSub;
                BSub.LockedView( B, sourceOffset, 0, sizes[s], B.Width() );

                const Quasi2dHMatrix& ASub = node.Child(t,s);
                ASub.MapMatrix( alpha, BSub, (Scalar)1, CSub );

                sourceOffset += sizes[s];
            }
            targetOffset += sizes[t];
        }
    }

    // Loop over the 6 children in the strictly lower triangle, summing in
    // each row
    {
        int targetOffset = 0;
        const int* sizes = node.sizes;
        for( int s=0; s<4; ++s )
        {
            DenseMatrix<Scalar> CSub;
            CSub.View( C, targetOffset, 0, sizes[s], C.Width() );

            int sourceOffset = targetOffset + sizes[s];
            for( int t=s+1; t<4; ++t )
            {
                DenseMatrix<Scalar> BSub;
                BSub.LockedView( B, sourceOffset, 0, sizes[t], B.Width() );

                const Quasi2dHMatrix& ASub = node.Child(t,s);
                ASub.TransposeMapMatrix( alpha, BSub, (Scalar)1, CSub );

                sourceOffset += sizes[t];
            }
            targetOffset += sizes[s];
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template class psp::Quasi2dHMatrix<float,false>;
template class psp::Quasi2dHMatrix<float,true>;
template class psp::Quasi2dHMatrix<double,false>;
template class psp::Quasi2dHMatrix<double,true>;
template class psp::Quasi2dHMatrix<std::complex<float>,false>;
template class psp::Quasi2dHMatrix<std::complex<float>,true>;
template class psp::Quasi2dHMatrix<std::complex<double>,false>;
template class psp::Quasi2dHMatrix<std::complex<double>,true>;
