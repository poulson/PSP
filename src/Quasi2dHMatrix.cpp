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

//----------------------------------------------------------------------------//
// Public static routines                                                     //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void 
psp::Quasi2dHMatrix<Scalar,Conjugated>::BuildMapOnQuadrant
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
// Private static routines                                                    //
//----------------------------------------------------------------------------//
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::PackedSizeRecursion
( std::size_t& packedSize, const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    // Make space for the abstract H-matrix information
    packedSize += 6*sizeof(int) + 2*sizeof(bool);

    // Quasi2dHMatrix-specific information
    packedSize += 9*sizeof(int);

    const Shell& shell = H._shell;
    packedSize += sizeof(ShellType);
    switch( shell.type )
    {
    case NODE:
        for( int i=0; i<16; ++i )
            PackedSizeRecursion( packedSize, *shell.data.node->children[i] );
        break;
    case NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
        {
            PackedSizeRecursion
            ( packedSize, *shell.data.nodeSymmetric->children[i] );
        }
        break;
    case LOW_RANK:
    {
        const DenseMatrix<Scalar>& U = shell.data.F->U;
        const DenseMatrix<Scalar>& V = shell.data.F->V;
        const int m = U.Height();
        const int n = V.Height();
        const int r = U.Width();

        // The height and width are already known, we just need the rank
        packedSize += sizeof(int);

        // Make space for U and V
        packedSize += (m+n)*r*sizeof(Scalar);

        break;
    }
    case DENSE:
    {
        const DenseMatrix<Scalar>& D = *shell.data.D;
        const int m = D.Height();
        const int n = D.Width();
        const MatrixType type = D.Type();

        // Make space for the matrix type and data
        packedSize += sizeof(MatrixType);
        if( type == GENERAL )
            packedSize += m*n*sizeof(Scalar);
        else
            packedSize += ((m*m+m)/2)*sizeof(Scalar);

        break;
    }
    }
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::PackRecursion
( byte*& head, const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    // Write out the abstract H-matrix information
    *((int*)head)  = H._height;             head += sizeof(int);
    *((int*)head)  = H._width;              head += sizeof(int);
    *((int*)head)  = H._numLevels;          head += sizeof(int);
    *((int*)head)  = H._maxRank;            head += sizeof(int);
    *((int*)head)  = H._sourceOffset;       head += sizeof(int);
    *((int*)head)  = H._targetOffset;       head += sizeof(int);
    *((bool*)head) = H._symmetric;          head += sizeof(bool);
    *((bool*)head) = H._stronglyAdmissible; head += sizeof(bool);

    // Write out the Quasi2dHMatrix-specific information
    *((int*)head) = H._xSizeSource; head += sizeof(int);
    *((int*)head) = H._xSizeTarget; head += sizeof(int);
    *((int*)head) = H._ySizeSource; head += sizeof(int);
    *((int*)head) = H._ySizeTarget; head += sizeof(int);
    *((int*)head) = H._zSize;       head += sizeof(int);
    *((int*)head) = H._xSource;     head += sizeof(int);
    *((int*)head) = H._xTarget;     head += sizeof(int);
    *((int*)head) = H._ySource;     head += sizeof(int);
    *((int*)head) = H._yTarget;     head += sizeof(int);

    const Shell& shell = H._shell;
    *((ShellType*)head) = shell.type; head += sizeof(ShellType);
    switch( shell.type )
    {
    case NODE:
        for( int i=0; i<16; ++i )
            PackRecursion( head, *shell.data.node->children[i] );
        break;
    case NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
            PackRecursion( head, *shell.data.nodeSymmetric->children[i] );
        break;
    case LOW_RANK:
    {
        const DenseMatrix<Scalar>& U = shell.data.F->U;
        const DenseMatrix<Scalar>& V = shell.data.F->V;
        const int m = U.Height();
        const int n = V.Height();
        const int r = U.Width();

        // Write out the rank
        *((int*)head) = r; head += sizeof(int);

        // Write out U
        for( int j=0; j<r; ++j )
        {
            std::memcpy( head, U.LockedBuffer(0,j), m*sizeof(Scalar) );
            head += m*sizeof(Scalar);
        }

        // Write out V
        for( int j=0; j<r; ++j )
        {
            std::memcpy( head, V.LockedBuffer(0,j), n*sizeof(Scalar) );
            head += n*sizeof(Scalar);
        }

        break;
    }
    case DENSE:
    {
        const DenseMatrix<Scalar>& D = *shell.data.D;
        const int m = D.Height();
        const int n = D.Width();
        const MatrixType type = D.Type();

        // Write out the matrix type and data
        *((MatrixType*)head) = type; head += sizeof(MatrixType);
        if( type == GENERAL )
        {
            for( int j=0; j<n; ++j )
            {
                std::memcpy( head, D.LockedBuffer(0,j), m*sizeof(Scalar) );
                head += m*sizeof(Scalar);
            }
        }
        else
        {
            for( int j=0; j<n; ++j )
            {
                std::memcpy( head, D.LockedBuffer(j,j), (m-j)*sizeof(Scalar) );
                head += (m-j)*sizeof(Scalar);
            }
        }

        break;
    }
    }
}

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
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
    ImportLowRankMatrix( F );
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
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix
( const std::vector<byte>& packedHMatrix )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Quasi2dHMatrix");
#endif
    Unpack( packedHMatrix );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::~Quasi2dHMatrix()
{ }

template<typename Scalar,bool Conjugated>
std::size_t
psp::Quasi2dHMatrix<Scalar,Conjugated>::PackedSize() const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::PackedSize");
#endif
    std::size_t packedSize = 0;
    PackedSizeRecursion( packedSize, *this );
#ifndef RELEASE
    PopCallStack();
#endif
    return packedSize;
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::Pack
( byte* packedHMatrix ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Pack");
#endif
    PackRecursion( packedHMatrix, *this );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::Pack
( std::vector<byte>& packedHMatrix ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Pack");
#endif
    // Create the storage and extract the buffer
    const std::size_t packedSize = PackedSize();
    packedHMatrix.resize( packedSize );
    byte* head = &packedHMatrix[0];

    PackRecursion( head, *this );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::Unpack
( const byte* packedHMatrix )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Unpack");
#endif
    UnpackRecursion( packedHMatrix, *this );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::Unpack
( const std::vector<byte>& packedHMatrix )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Unpack");
#endif
    const byte* head = &packedHMatrix[0];
    UnpackRecursion( head, *this );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::Print
( const std::string& tag ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Print");
#endif
    DenseMatrix<Scalar> I( this->Width(), this->Width() );
    std::memset( I.Buffer(), 0, I.LDim()*I.Width() );
    for( int j=0; j<this->Width(); ++j )
        I.Set(j,j,(Scalar)1);

    DenseMatrix<Scalar> HFull;
    this->MapMatrix( (Scalar)1, I, HFull );
    HFull.Print( tag );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::WriteStructure
( const std::string& filename ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::WriteStructure");
#endif
    std::ofstream file( filename.c_str() );
    WriteStructureRecursion( file );
#ifndef RELEASE
    PopCallStack();
#endif
}

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
    C.Resize( this->Height(), B.Width() );
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

    // Delete the old type 
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

// A := Conj(A)
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::Conjugate()
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Conjugate");
#endif
    switch( _shell.type )
    {
    case NODE:
        for( int i=0; i<16; ++i )
            _shell.data.node->children[i]->Conjugate();
        break;
    case NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
            _shell.data.nodeSymmetric->children[i]->Conjugate();
        break;
    case LOW_RANK:
        hmatrix_tools::Conjugate( _shell.data.F->U );
        hmatrix_tools::Conjugate( _shell.data.F->V );
        break;
    case DENSE:
        hmatrix_tools::Conjugate( *_shell.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := Conj(B)
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::ConjugateFrom
( const Quasi2dHMatrix<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::ConjugateFrom");
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

    // Delete the old type 
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
        for( int i=0; i<16; ++i )
        {
            _shell.data.node->children[i]->ConjugateFrom
            ( *B._shell.data.node->children[i] );
        }
        break;
    case NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
        {
            _shell.data.nodeSymmetric->children[i]->ConjugateFrom
            ( *B._shell.data.nodeSymmetric->children[i] );
        }
        break;
    case LOW_RANK:
        hmatrix_tools::Conjugate( B._shell.data.F->U, _shell.data.F->U );
        hmatrix_tools::Conjugate( B._shell.data.F->V, _shell.data.F->V );
        break;
    case DENSE:
        hmatrix_tools::Conjugate( *B._shell.data.D, *_shell.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := B^T
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::TransposeFrom
( const Quasi2dHMatrix<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::TransposeFrom");
#endif
    this->_height = B.Width();
    this->_width = B.Height();
    this->_numLevels = B.NumLevels();
    this->_maxRank = B.MaxRank();
    this->_sourceOffset = B.TargetOffset();
    this->_targetOffset = B.SourceOffset();
    this->_symmetric = B.Symmetric();
    this->_stronglyAdmissible = B.StronglyAdmissible();
    _xSizeSource = B.XSizeTarget();
    _xSizeTarget = B.XSizeSource();
    _ySizeSource = B.YSizeTarget();
    _ySizeTarget = B.YSizeSource();
    _zSize = B.ZSize();
    _xSource = B.XTarget();
    _xTarget = B.XSource();
    _ySource = B.YTarget();
    _yTarget = B.YSource();

    // Delete the old type
    switch( _shell.type )
    {
    case NODE:           delete _shell.data.node;          break;
    case NODE_SYMMETRIC: delete _shell.data.nodeSymmetric; break;
    case LOW_RANK:       delete _shell.data.F;             break;
    case DENSE:          delete _shell.data.D;             break;
    }
    _shell.type = B._shell.type;

    switch( B._shell.type )
    {
    case NODE:
    {
        _shell.data.node = 
            new Node
            ( _xSizeSource, _xSizeTarget,
              _ySizeSource, _ySizeTarget, _zSize );
        Node& nodeA = *_shell.data.node;
        const Node& nodeB = *B._shell.data.node;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<4; ++s )
            {
                nodeA.children[s+4*t] = 
                    new Quasi2dHMatrix<Scalar,Conjugated>;
                nodeA.Child(t,s).TransposeFrom( nodeB.Child(s,t) );
            }
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
    {
        _shell.data.F = new LowRankMatrix<Scalar,Conjugated>;
        hmatrix_tools::Transpose( *B._shell.data.F, *_shell.data.F );
        break;
    }
    case DENSE:
    {
        _shell.data.D = new DenseMatrix<Scalar>;
        hmatrix_tools::Transpose( *B._shell.data.D, *_shell.data.D );
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := B^H
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeFrom
( const Quasi2dHMatrix<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::HermitianTransposeFrom");
#endif
    this->_height = B.Width();
    this->_width = B.Height();
    this->_numLevels = B.NumLevels();
    this->_maxRank = B.MaxRank();
    this->_sourceOffset = B.TargetOffset();
    this->_targetOffset = B.SourceOffset();
    this->_symmetric = B.Symmetric();
    this->_stronglyAdmissible = B.StronglyAdmissible();
    _xSizeSource = B.XSizeTarget();
    _xSizeTarget = B.XSizeSource();
    _ySizeSource = B.YSizeTarget();
    _ySizeTarget = B.YSizeSource();
    _zSize = B.ZSize();
    _xSource = B.XTarget();
    _xTarget = B.XSource();
    _ySource = B.YTarget();
    _yTarget = B.YSource();

    // Delete the old type
    switch( _shell.type )
    {
    case NODE:           delete _shell.data.node;          break;
    case NODE_SYMMETRIC: delete _shell.data.nodeSymmetric; break;
    case LOW_RANK:       delete _shell.data.F;             break;
    case DENSE:          delete _shell.data.D;             break;
    }
    _shell.type = B._shell.type;

    switch( B._shell.type )
    {
    case NODE:
    {
        _shell.data.node = 
            new Node
            ( _xSizeSource, _xSizeTarget,
              _ySizeSource, _ySizeTarget, _zSize );
        Node& nodeA = *_shell.data.node;
        const Node& nodeB = *B._shell.data.node;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<4; ++s )
            {
                nodeA.children[s+4*t] = 
                    new Quasi2dHMatrix<Scalar,Conjugated>;
                nodeA.Child(t,s).HermitianTransposeFrom( nodeB.Child(s,t) );
            }
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
            nodeA.children[i]->ConjugateFrom( *nodeB.children[i] );
        }
        break;
    }
    case LOW_RANK:
    {
        _shell.data.F = new LowRankMatrix<Scalar,Conjugated>;
        hmatrix_tools::HermitianTranspose( *B._shell.data.F, *_shell.data.F );
        break;
    }
    case DENSE:
    {
        _shell.data.D = new DenseMatrix<Scalar>;
        hmatrix_tools::HermitianTranspose( *B._shell.data.D, *_shell.data.D );
        break;
    }
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

// A := A + alpha I
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::AddConstantToDiagonal
( Scalar alpha )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::AddConstantToDiagonal");
#endif
    switch( _shell.type )
    {
    case NODE:
    {
        Node& nodeA = *_shell.data.node;
        for( int i=0; i<4; ++i )
            nodeA.Child(i,i).AddConstantToDiagonal( alpha );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *_shell.data.nodeSymmetric;
        for( int i=0; i<4; ++i )
            nodeA.Child(i,i).AddConstantToDiagonal( alpha );
        break;
    }
    case LOW_RANK:
#ifndef RELEASE
        throw std::logic_error("Mistake in logic");
#endif
        break;
    case DENSE:
    {
        Scalar* DBuffer = _shell.data.D->Buffer();
        const int m = _shell.data.D->Height();
        const int n = _shell.data.D->Width();
        const int DLDim = _shell.data.D->LDim();
        for( int j=0; j<std::min(m,n); ++j )
            DBuffer[j+j*DLDim] += alpha;
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
    C._ySizeSource = B.YSizeSource();
    C._xSizeTarget = this->XSizeTarget();
    C._ySizeTarget = this->YSizeTarget();
    C._zSize = this->ZSize();
    C._xSource = B.XSource();
    C._ySource = B.YSource();
    C._xTarget = this->XTarget();
    C._yTarget = this->YTarget();

    // Delete the old type
    switch( C._shell.type )
    {
    case NODE:           delete C._shell.data.node;          break;
    case NODE_SYMMETRIC: delete C._shell.data.nodeSymmetric; break;
    case LOW_RANK:       delete C._shell.data.F;             break;
    case DENSE:          delete C._shell.data.D;             break;
    }

    if( C.Admissible() )
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
            hmatrix_tools::Copy( _shell.data.F->U, C._shell.data.F->U );
            if( Conjugated )
            {
                B.HermitianTransposeMapMatrix
                ( Conj(alpha), _shell.data.F->V, C._shell.data.F->V );
            }
            else
            {
                B.TransposeMapMatrix
                ( alpha, _shell.data.F->V, C._shell.data.F->V );
            }
        }
        else if( this->IsLowRank() && B.IsDense() )
        {
            hmatrix_tools::Copy( _shell.data.F->U, C._shell.data.F->U );
            if( Conjugated )
            {
                hmatrix_tools::MatrixHermitianTransposeMatrix
                ( Conj(alpha), *B._shell.data.D, _shell.data.F->V, 
                  C._shell.data.F->V );
            }
            else
            {
                hmatrix_tools::MatrixTransposeMatrix
                ( alpha, *B._shell.data.D, _shell.data.F->V,
                  C._shell.data.F->V );
            }
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
    if( C.Admissible() )
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

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._shell.data.F );
        }
        else if( this->IsLowRank() && B.IsDense() )
        {
            // W := alpha A.F B.D
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::Copy( _shell.data.F->U, W.U );
            if( Conjugated )
            {
                hmatrix_tools::MatrixHermitianTransposeMatrix
                ( Conj(alpha), *B._shell.data.D, _shell.data.F->V, W.V );
            }
            else
            {
                hmatrix_tools::MatrixTransposeMatrix
                ( alpha, *B._shell.data.D, _shell.data.F->V, W.V );
            }

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
            // Form W := alpha A B 
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.F, W );

            // C :~= W + beta C
            C.Scale( beta );
            C.UpdateWithLowRankMatrix( (Scalar)1, W );
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

            // C :~= W + beta C
            C.Scale( beta );
            C.UpdateWithLowRankMatrix( (Scalar)1, W );
        }
        else if( this->IsHierarchical() && B.IsLowRank() )
        {
            // Form W := alpha A B    
            LowRankMatrix<Scalar,Conjugated> W;
            this->MapMatrix( alpha, B._shell.data.F->U, W.U );
            hmatrix_tools::Copy( B._shell.data.F->V, W.V );

            // Form C :~= W + beta C
            C.Scale( beta );
            C.UpdateWithLowRankMatrix( (Scalar)1, W );
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

// A := inv(A) using recursive Schur complements
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::DirectInvert()
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::DirectInvert");
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

        for( int l=0; l<4; ++l )
        {
            // A_ll := inv(B_ll)
            nodeA.Child(l,l).CopyFrom( nodeB.Child(l,l) );
            nodeA.Child(l,l).DirectInvert();

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
#ifndef RELEASE
        throw std::logic_error("Symmetric inversion not yet supported.");
#endif
        break;
    }
    case DENSE:
        hmatrix_tools::Invert( *_shell.data.D );
        break;
    case LOW_RANK:
    {
#ifndef RELEASE
        throw std::logic_error("Mistake in inversion code.");
#endif
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := inv(A) using Schulz iterations, X_k+1 := (2I - X_k A) X_k
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::SchulzInvert( int maxIts )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::SchulzInvert");
    if( this->Height() != this->Width() )
        throw std::logic_error("Cannot invert non-square matrices");
    if( this->IsLowRank() )
        throw std::logic_error("Cannot invert low-rank matrices");
#endif
    // Need to find alpha s.t. 0 < alpha < 2/||A||_2^2, but we do not have
    // a cheap way of estimating ||A||_2, so instead pick alpha very small.
    //
    // TODO: Allow alpha as a parameter
    std::cout << "TODO: Implement a 2-norm estimator for choosing alpha" 
              << std::endl;
    Scalar alpha = ((Scalar)1)/((Scalar)1.e7);

    // Initialize X_0 := alpha A^H
    Quasi2dHMatrix<Scalar,Conjugated> X;
    X.HermitianTransposeFrom( *this );
    X.Scale( alpha );

    for( int k=0; k<maxIts; ++k )
    {
        // Form Z := 2I - X_k A
        Quasi2dHMatrix<Scalar,Conjugated> Z;        
        X.MapMatrix( (Scalar)-1, *this, Z );
        Z.AddConstantToDiagonal( (Scalar)2 );

        // Form X_k+1 := Z X_k = (2I - X_k A) X_k
        Quasi2dHMatrix<Scalar,Conjugated> XCopy;
        XCopy.CopyFrom( X );
        Z.MapMatrix( (Scalar)1, XCopy, X );
    }

    this->CopyFrom( X );
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
bool
psp::Quasi2dHMatrix<Scalar,Conjugated>::Admissible() const
{
    return Admissible( _xSource, _xTarget, _ySource, _yTarget );
}

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
( const LowRankMatrix<Scalar,Conjugated>& F )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::ImportLowRankMatrix");
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
        hmatrix_tools::Copy( F.U, _shell.data.F->U );
        hmatrix_tools::Copy( F.V, _shell.data.F->V );
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
            LowRankMatrix<Scalar,Conjugated> FSub;
            for( int t=0; t<4; ++t )
            {
                FSub.U.LockedView
                ( F.U, 
                  targetOffset-this->_targetOffset, 0, 
                  sizes[t], F.Rank() );

                int sourceOffset = this->_targetOffset;
                for( int s=0; s<=t; ++s )
                {
                    FSub.V.LockedView
                    ( F.V, 
                      sourceOffset-this->_sourceOffset, 0, 
                      sizes[s], F.Rank() );

                    node.children[child++] = 
                      new Quasi2dHMatrix
                      ( FSub, 
                        this->_numLevels-1, this->_maxRank, 
                        this->_stronglyAdmissible,
                        xSizes[s&1], xSizes[t&1],
                        ySizes[s/2], ySizes[t/2],
                        _zSize,
                        2*_xSource+(s&1), 2*_xTarget+(t&1),
                        2*_ySource+(s/2), 2*_yTarget+(t/2),
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
            LowRankMatrix<Scalar,Conjugated> FSub;
            for( int t=0; t<4; ++t )
            {
                FSub.U.LockedView
                ( F.U, 
                  targetOffset-this->_targetOffset, 0, 
                  targetSizes[t], F.Rank() );

                int sourceOffset = this->_sourceOffset;
                for( int s=0; s<4; ++s )
                {
                    FSub.V.LockedView
                    ( F.V, 
                      sourceOffset-this->_sourceOffset, 0, 
                      sourceSizes[s], F.Rank() );

                    node.children[s+4*t] = 
                      new Quasi2dHMatrix
                      ( FSub,
                        this->_numLevels-1, this->_maxRank,
                        this->_stronglyAdmissible,
                        xSourceSizes[s&1], xTargetSizes[t&1],
                        ySourceSizes[s/2], yTargetSizes[t/2],
                        _zSize,
                        2*_xSource+(s&1), 2*_xTarget+(t&1),
                        2*_ySource+(s/2), 2*_yTarget+(t/2),
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
          1, F.U.LockedBuffer(), F.U.LDim(),
             F.V.LockedBuffer(), F.V.LDim(),
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
  const LowRankMatrix<Scalar,Conjugated>& F )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::UpdateWithLowRankMatrix");
#endif
    if( Admissible( _xSource, _xTarget, _ySource, _yTarget ) )
    {
        hmatrix_tools::MatrixUpdateRounded
        ( this->MaxRank(), alpha, F, (Scalar)1, *_shell.data.F );
    }
    else if( this->_numLevels > 1 )
    {
        if( this->_symmetric )
        {
            NodeSymmetric& node = *_shell.data.nodeSymmetric;
            const int* sizes = node.sizes;

            int targetOffset = 0;
            LowRankMatrix<Scalar,Conjugated> FSub;
            for( int t=0; t<4; ++t )
            {
                FSub.U.LockedView
                ( F.U, targetOffset, 0, sizes[t], F.Rank() );

                int sourceOffset = 0;
                for( int s=0; s<=t; ++s )
                {
                    FSub.V.LockedView
                    ( F.V, sourceOffset, 0, sizes[s],  F.Rank() );

                    node.Child(t,s).UpdateWithLowRankMatrix( alpha, FSub );

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

            int targetOffset = 0;
            LowRankMatrix<Scalar,Conjugated> FSub;
            for( int t=0; t<4; ++t )
            {
                FSub.U.LockedView
                ( F.U, targetOffset, 0, targetSizes[t], F.Rank() );

                int sourceOffset = 0;
                for( int s=0; s<4; ++s )
                {
                    FSub.V.LockedView
                    ( F.V, sourceOffset, 0, sourceSizes[s],  F.Rank() );

                    node.Child(t,s).UpdateWithLowRankMatrix( alpha, FSub );

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
          alpha, F.U.LockedBuffer(), F.U.LDim(),
                 F.V.LockedBuffer(), F.V.LDim(),
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
                        2*_xSource+(s&1), 2*_xTarget+(t&1),
                        2*_ySource+(s/2), 2*_yTarget+(t/2),
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
                        2*_xSource+(s&1), 2*_xTarget+(t&1),
                        2*_ySource+(s/2), 2*_yTarget+(t/2),
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



template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::UnpackRecursion
( const byte*& head, Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    // Set the abstract H-matrix data
    H._height             = *((int*)head);  head += sizeof(int);
    H._width              = *((int*)head);  head += sizeof(int);
    H._numLevels          = *((int*)head);  head += sizeof(int);
    H._maxRank            = *((int*)head);  head += sizeof(int);
    H._sourceOffset       = *((int*)head);  head += sizeof(int);
    H._targetOffset       = *((int*)head);  head += sizeof(int);
    H._symmetric          = *((bool*)head); head += sizeof(bool);
    H._stronglyAdmissible = *((bool*)head); head += sizeof(bool);

    // Set the Quasi2dHMatrix-specific information
    H._xSizeSource = *((int*)head); head += sizeof(int);
    H._xSizeTarget = *((int*)head); head += sizeof(int);
    H._ySizeSource = *((int*)head); head += sizeof(int);
    H._ySizeTarget = *((int*)head); head += sizeof(int);
    H._zSize       = *((int*)head); head += sizeof(int);
    H._xSource     = *((int*)head); head += sizeof(int);
    H._xTarget     = *((int*)head); head += sizeof(int);
    H._ySource     = *((int*)head); head += sizeof(int);
    H._yTarget     = *((int*)head); head += sizeof(int);

    // If data has been allocated, delete it
    Shell& shell = H._shell;
    switch( shell.type )
    {
    case NODE:           delete shell.data.node; break;
    case NODE_SYMMETRIC: delete shell.data.nodeSymmetric; break;
    case LOW_RANK:       delete shell.data.F; break;
    case DENSE:          delete shell.data.D; break;
    }

    // Create this layer of the H-matrix from the packed information
    shell.type = *((ShellType*)head); head += sizeof(ShellType);
    switch( shell.type )
    {
    case NODE:
    {
        shell.data.node = 
            new Node
            ( H._xSizeSource, H._xSizeTarget, 
              H._ySizeSource, H._ySizeTarget, H._zSize );
        Node& node = *shell.data.node;
        for( int i=0; i<16; ++i )
        {
            node.children[i] = new Quasi2dHMatrix<Scalar,Conjugated>;
            UnpackRecursion( head, *node.children[i] );
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        shell.data.nodeSymmetric = 
            new NodeSymmetric( H._xSizeSource, H._ySizeSource, H._zSize );
        NodeSymmetric& node = *shell.data.nodeSymmetric;
        for( int i=0; i<10; ++i )
        {
            node.children[i] = new Quasi2dHMatrix<Scalar,Conjugated>;
            UnpackRecursion( head, *node.children[i] );
        }
        break;
    }
    case LOW_RANK:
    {
        shell.data.F = new LowRankMatrix<Scalar,Conjugated>;
        DenseMatrix<Scalar>& U = shell.data.F->U;
        DenseMatrix<Scalar>& V = shell.data.F->V;
        const int m = H._height;
        const int n = H._width;

        // Read in the matrix rank
        const int r = *((int*)head); head += sizeof(int);
        U.SetType( GENERAL ); U.Resize( m, r );
        V.SetType( GENERAL ); V.Resize( n, r );

        // Read in U
        for( int j=0; j<r; ++j )
        {
            std::memcpy( U.Buffer(0,j), head, m*sizeof(Scalar) );
            head += m*sizeof(Scalar);
        }

        // Read in V
        for( int j=0; j<r; ++j )
        {
            std::memcpy( V.Buffer(0,j), head, n*sizeof(Scalar) );
            head += n*sizeof(Scalar);
        }

        break;
    }
    case DENSE:
        shell.data.D = new DenseMatrix<Scalar>;
        DenseMatrix<Scalar>& D = *shell.data.D;
        const int m = H._height;
        const int n = H._width;

        const MatrixType type = *((MatrixType*)head); 
        head += sizeof(MatrixType);
        D.SetType( type ); 
        D.Resize( m, n );

        // Read in the matrix
        if( type == GENERAL )
        {
            for( int j=0; j<n; ++j )
            {
                std::memcpy( D.Buffer(0,j), head, m*sizeof(Scalar) );
                head += m*sizeof(Scalar);
            }
        }
        else
        {
            for( int j=0; j<n; ++j )
            {
                std::memcpy( D.Buffer(j,j), head, (m-j)*sizeof(Scalar) );
                head += (m-j)*sizeof(Scalar);
            }
        }

        break;
    }
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

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::WriteStructureRecursion
( std::ofstream& file ) const
{
    switch( _shell.type )
    {
    case NODE:
    {
        file << "1 " 
             << this->TargetOffset() << " " << this->SourceOffset() << " "
             << this->TargetSize() << " " << this->SourceSize() 
             << std::endl;
        const Node& node = *_shell.data.node;
        for( unsigned child=0; child<node.children.size(); ++child )
            node.children[child]->WriteStructureRecursion( file );
        break;
    }
    case NODE_SYMMETRIC:
    {
        file << "1 " 
             << this->TargetOffset() << " " << this->SourceOffset() << " "
             << this->TargetSize() << " " << this->SourceSize() 
             << std::endl;
        const NodeSymmetric& node = *_shell.data.nodeSymmetric;
        for( unsigned child=0; child<node.children.size(); ++child )
            node.children[child]->WriteStructureRecursion( file );
        break;
    }
    case LOW_RANK:
        file << "5 " 
             << this->TargetOffset() << " " << this->SourceOffset() << " "
             << this->TargetSize() << " " << this->SourceSize() 
             << std::endl;
        break;
    case DENSE:
        file << "20 " 
             << this->TargetOffset() << " " << this->SourceOffset() << " "
             << this->TargetSize() << " " << this->SourceSize() 
             << std::endl;
        break;
    }
}

template class psp::Quasi2dHMatrix<float,false>;
template class psp::Quasi2dHMatrix<float,true>;
template class psp::Quasi2dHMatrix<double,false>;
template class psp::Quasi2dHMatrix<double,true>;
template class psp::Quasi2dHMatrix<std::complex<float>,false>;
template class psp::Quasi2dHMatrix<std::complex<float>,true>;
template class psp::Quasi2dHMatrix<std::complex<double>,false>;
template class psp::Quasi2dHMatrix<std::complex<double>,true>;
