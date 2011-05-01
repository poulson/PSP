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
std::pair<std::size_t,std::size_t>
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::PackedSizes
( const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::PackedSizes");
#endif
    // Recurse on this shell to compute the packed sizes
    std::size_t sourceSize=0, targetSize=0;
    PackedSizesRecursion( sourceSize, targetSize, H );
#ifndef RELEASE
    PopCallStack();
#endif
    return std::pair<std::size_t,std::size_t>(sourceSize,targetSize);
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::PackedSourceSize
( const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::PackedSourceSize");
#endif
    std::size_t sourceSize=0, targetSize=0;
    PackedSizesRecursion( sourceSize, targetSize, H );
#ifndef RELEASE
    PopCallStack();
#endif
    return sourceSize;
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::PackedTargetSize
( const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::PackedTargetSize");
#endif
    std::size_t sourceSize=0, targetSize=0;
    PackedSizesRecursion( sourceSize, targetSize, H );
#ifndef RELEASE
    PopCallStack();
#endif
    return targetSize;
}

template<typename Scalar,bool Conjugated>
std::pair<std::size_t,std::size_t>
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::Pack
( byte* packedSourceSide, byte* packedTargetSide,
  int sourceRank, int targetRank,
  const Quasi2dHMatrix<Scalar,Conjugated>& H ) 
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::Pack");
#endif
    byte* sourceHead = packedSourceSide;
    byte* targetHead = packedTargetSide;
    PackRecursion( sourceHead, targetHead, sourceRank, targetRank, H );
    const std::size_t sourceSize = sourceHead-packedSourceSide;
    const std::size_t targetSize = targetHead-packedTargetSide;
#ifndef RELEASE
    PopCallStack();
#endif
    return std::pair<std::size_t,std::size_t>(sourceSize,targetSize);
}

//----------------------------------------------------------------------------//
// Private static routines                                                    //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::PackedSizesRecursion
( std::size_t& sourceSize, std::size_t& targetSize,
  const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    typedef Quasi2dHMatrix<Scalar,Conjugated> Quasi2d;
    typedef SplitQuasi2dHMatrix<Scalar,Conjugated> SplitQuasi2d;

    // Make space for the SplitQuasi2dHMatrix member variables
    {
        const std::size_t headerSize = 
            16*sizeof(int) + 3*sizeof(bool) + sizeof(ShellType);
        sourceSize += headerSize;
        targetSize += headerSize;
    }

    const typename Quasi2d::Shell& shell = H._shell;
    switch( shell.type )
    {
    case Quasi2d::NODE:
        for( int i=0; i<16; ++i )
        {
            PackedSizesRecursion
            ( sourceSize, targetSize, *shell.data.node->children[i] );
        }
        break;
    case Quasi2d::NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
        {
            PackedSizesRecursion
            ( sourceSize, targetSize, *shell.data.nodeSymmetric->children[i] );
        }
        break;
    case Quasi2d::LOW_RANK:
    {
        const std::size_t headerSize = sizeof(int);
        sourceSize += headerSize;
        targetSize += headerSize;

        const LowRankMatrix<Scalar,Conjugated>& F = *shell.data.F;
        const int m = F.Height();
        const int n = F.Width();
        const int r = F.Rank();
        sourceSize += n*r*sizeof(Scalar);
        targetSize += m*r*sizeof(Scalar);
        break;
    }
    case Quasi2d::DENSE:
    {
        const DenseMatrix<Scalar>& D = *shell.data.D;
        const int m = D.Height();
        const int n = D.Width();
        sourceSize += sizeof(MatrixType);
        if( D.Type() == GENERAL )
            sourceSize += m*n*sizeof(Scalar);
        else
            sourceSize += ((m*m+m)/2)*sizeof(Scalar);
        break;
    }
    }
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::PackRecursion
( byte*& sourceHead, byte*& targetHead, 
  int sourceRank, int targetRank,
  const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    typedef Quasi2dHMatrix<Scalar,Conjugated> Quasi2d;
    typedef SplitQuasi2dHMatrix<Scalar,Conjugated> SplitQuasi2d;

    // Write out the source member variables
    *((int*)sourceHead) = H._height;              sourceHead += sizeof(int);
    *((int*)sourceHead) = H._width;               sourceHead += sizeof(int);
    *((int*)sourceHead) = H._numLevels;           sourceHead += sizeof(int);
    *((int*)sourceHead) = H._maxRank;             sourceHead += sizeof(int);
    *((int*)sourceHead) = H._sourceOffset;        sourceHead += sizeof(int);
    *((int*)sourceHead) = H._targetOffset;        sourceHead += sizeof(int);
    *((bool*)sourceHead) = H._symmetric;          sourceHead += sizeof(bool);
    *((bool*)sourceHead) = H._stronglyAdmissible; sourceHead += sizeof(bool);
    *((int*)sourceHead) = H._xSizeSource;         sourceHead += sizeof(int);
    *((int*)sourceHead) = H._xSizeTarget;         sourceHead += sizeof(int);
    *((int*)sourceHead) = H._ySizeSource;         sourceHead += sizeof(int);
    *((int*)sourceHead) = H._ySizeTarget;         sourceHead += sizeof(int);
    *((int*)sourceHead) = H._zSize;               sourceHead += sizeof(int);
    *((int*)sourceHead) = H._xSource;             sourceHead += sizeof(int);
    *((int*)sourceHead) = H._xTarget;             sourceHead += sizeof(int);
    *((int*)sourceHead) = H._ySource;             sourceHead += sizeof(int);
    *((int*)sourceHead) = H._yTarget;             sourceHead += sizeof(int);
    *((bool*)sourceHead) = true;                  sourceHead += sizeof(bool);
    *((int*)sourceHead) = targetRank;             sourceHead += sizeof(int);
    
    // Write out the target member variables
    *((int*)targetHead) = H._height;              targetHead += sizeof(int);
    *((int*)targetHead) = H._width;               targetHead += sizeof(int);
    *((int*)targetHead) = H._numLevels;           targetHead += sizeof(int);
    *((int*)targetHead) = H._maxRank;             targetHead += sizeof(int);
    *((int*)targetHead) = H._sourceOffset;        targetHead += sizeof(int);
    *((int*)targetHead) = H._targetOffset;        targetHead += sizeof(int);
    *((bool*)targetHead) = H._symmetric;          targetHead += sizeof(bool);
    *((bool*)targetHead) = H._stronglyAdmissible; targetHead += sizeof(bool);
    *((int*)targetHead) = H._xSizeSource;         targetHead += sizeof(int);
    *((int*)targetHead) = H._xSizeTarget;         targetHead += sizeof(int);
    *((int*)targetHead) = H._ySizeSource;         targetHead += sizeof(int);
    *((int*)targetHead) = H._ySizeTarget;         targetHead += sizeof(int);
    *((int*)targetHead) = H._zSize;               targetHead += sizeof(int);
    *((int*)targetHead) = H._xSource;             targetHead += sizeof(int);
    *((int*)targetHead) = H._xTarget;             targetHead += sizeof(int);
    *((int*)targetHead) = H._ySource;             targetHead += sizeof(int);
    *((int*)targetHead) = H._yTarget;             targetHead += sizeof(int);
    *((bool*)targetHead) = false;                 targetHead += sizeof(bool);
    *((int*)targetHead) = sourceRank;             targetHead += sizeof(int);

    const typename Quasi2d::Shell& shell = H._shell;
    switch( shell.type )
    {
    case Quasi2d::NODE:
        *((ShellType*)sourceHead) = NODE;
        *((ShellType*)targetHead) = NODE;
        sourceHead += sizeof(ShellType);
        targetHead += sizeof(ShellType);
        for( int i=0; i<16; ++i )        
        {
            PackRecursion
            ( sourceHead, targetHead, sourceRank, targetRank, 
              *shell.data.node->children[i] );
        }
        break;
    case Quasi2d::NODE_SYMMETRIC:
        *((ShellType*)sourceHead) = NODE_SYMMETRIC;
        *((ShellType*)targetHead) = NODE_SYMMETRIC;
        sourceHead += sizeof(ShellType);
        targetHead += sizeof(ShellType);
        for( int i=0; i<10; ++i )
        {
            PackRecursion
            ( sourceHead, targetHead, sourceRank, targetRank, 
              *shell.data.nodeSymmetric->children[i] );
        }
        break;
    case Quasi2d::LOW_RANK:
    {
        *((ShellType*)sourceHead) = SPLIT_LOW_RANK;
        *((ShellType*)targetHead) = SPLIT_LOW_RANK;
        sourceHead += sizeof(ShellType);
        targetHead += sizeof(ShellType);

        const DenseMatrix<Scalar>& U = shell.data.F->U;
        const DenseMatrix<Scalar>& V = shell.data.F->V;
        const int m = U.Height();
        const int n = V.Height();
        const int r = U.Width();

        // Write out the source information
        *((int*)sourceHead) = r; sourceHead += sizeof(int);
        for( int j=0; j<r; ++j )
        {
            std::memcpy( sourceHead, V.LockedBuffer(0,j), n*sizeof(Scalar) );
            sourceHead += n*sizeof(Scalar);
        }

        // Write out the target information
        *((int*)targetHead) = r; targetHead += sizeof(int);
        for( int j=0; j<r; ++j )
        {
            std::memcpy( targetHead, U.LockedBuffer(0,j), m*sizeof(Scalar) );
            targetHead += m*sizeof(Scalar);
        }

        break;
    }
    case Quasi2d::DENSE:
    {
        *((ShellType*)sourceHead) = SPLIT_DENSE;
        *((ShellType*)targetHead) = SPLIT_DENSE;
        sourceHead += sizeof(ShellType);
        targetHead += sizeof(ShellType);

        const DenseMatrix<Scalar>& D = *shell.data.D;
        const int m = D.Height();
        const int n = D.Width();
        const MatrixType type = D.Type();

        // Write out the source information
        *((MatrixType*)sourceHead) = type; sourceHead += sizeof(MatrixType);
        if( type == GENERAL )
        {
            for( int j=0; j<n; ++j )
            {
                std::memcpy
                ( sourceHead, D.LockedBuffer(0,j), m*sizeof(Scalar) );
                sourceHead += m*sizeof(Scalar);
            }
        }
        else
        {
            for( int j=0; j<n; ++j )
            {
                std::memcpy
                ( sourceHead, D.LockedBuffer(j,j), (m-j)*sizeof(Scalar) );
                sourceHead += (m-j)*sizeof(Scalar);
            }
        }

        // There is no target information to write

        break;
    }
    }
}

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

// Create an empty split H-matrix
template<typename Scalar,bool Conjugated>
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::SplitQuasi2dHMatrix
( MPI_Comm comm )
: _height(0), _width(0), _numLevels(0), _maxRank(0), 
  _sourceOffset(0), _targetOffset(0), 
  _symmetric(false), _stronglyAdmissible(false),
  _xSizeSource(0), _xSizeTarget(0), _ySizeSource(0), _ySizeTarget(0),
  _zSize(0), _xSource(0), _xTarget(0), _ySource(0), _yTarget(0),
  _ownSourceSide(false), _comm(comm), _partner(0)
{ }

// Create an H-matrix from packed data
template<typename Scalar,bool Conjugated>
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::SplitQuasi2dHMatrix
( const byte* packedHalf, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::SplitQuasi2dHMatrix");
#endif
    Unpack( packedHalf, comm );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::~SplitQuasi2dHMatrix()
{ }

template<typename Scalar,bool Conjugated>
std::size_t
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::Unpack
( const byte* packedHalf, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::Unpack");
#endif
    _comm = comm;

    const byte* head = packedHalf;
    UnpackRecursion( head, *this );
#ifndef RELEASE
    PopCallStack();
#endif
    return (head-packedHalf);
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::UnpackRecursion
( const byte*& head, SplitQuasi2dHMatrix<Scalar,Conjugated>& H )
{
    MPI_Comm comm = H._comm;

    H._height             = *((int*)head);  head += sizeof(int);
    H._width              = *((int*)head);  head += sizeof(int);
    H._numLevels          = *((int*)head);  head += sizeof(int);
    H._maxRank            = *((int*)head);  head += sizeof(int);
    H._sourceOffset       = *((int*)head);  head += sizeof(int);
    H._targetOffset       = *((int*)head);  head += sizeof(int);
    H._symmetric          = *((bool*)head); head += sizeof(bool);
    H._stronglyAdmissible = *((bool*)head); head += sizeof(bool);
    H._xSizeSource        = *((int*)head);  head += sizeof(int);
    H._xSizeTarget        = *((int*)head);  head += sizeof(int);
    H._ySizeSource        = *((int*)head);  head += sizeof(int);
    H._ySizeTarget        = *((int*)head);  head += sizeof(int);
    H._zSize              = *((int*)head);  head += sizeof(int);
    H._xSource            = *((int*)head);  head += sizeof(int);
    H._xTarget            = *((int*)head);  head += sizeof(int);
    H._ySource            = *((int*)head);  head += sizeof(int);
    H._yTarget            = *((int*)head);  head += sizeof(int);
    H._ownSourceSide     = *((bool*)head);  head += sizeof(bool);
    H._partner            = *((int*)head);  head += sizeof(int);

    // Delete the old shell information if it exists
    Shell& shell = H._shell;
    switch( shell.type )
    {
    case NODE:           delete shell.data.node; break;
    case NODE_SYMMETRIC: delete shell.data.nodeSymmetric; break;
    case SPLIT_LOW_RANK: delete shell.data.SF; break;
    case SPLIT_DENSE:    delete shell.data.SD; break;
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
            node.children[i] = 
                new SplitQuasi2dHMatrix<Scalar,Conjugated>( comm );
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
            node.children[i] = 
                new SplitQuasi2dHMatrix<Scalar,Conjugated>( comm );
            UnpackRecursion( head, *node.children[i] );
        }
        break;
    }
    case SPLIT_LOW_RANK:
    {
        shell.data.SF = new SplitLowRankMatrix<Scalar,Conjugated>;
        SplitLowRankMatrix<Scalar,Conjugated>& SF = *shell.data.SF;

        const int m = H._height;
        const int n = H._width;
        const int r = *((int*)head); head += sizeof(int);
        SF.height        = m;
        SF.width         = n;
        SF.rank          = r;
        SF.comm          = comm;
        SF.ownSourceSide = H._ownSourceSide;
        SF.partner       = H._partner;

        SF.D.SetType( GENERAL );
        if( SF.ownSourceSide )
        {
            SF.D.Resize( n, r );
            for( int j=0; j<r; ++j )
            {
                std::memcpy( SF.D.Buffer(0,j), head, n*sizeof(Scalar) );
                head += n*sizeof(Scalar);
            }
        }
        else
        {
            SF.D.Resize( m, r );
            for( int j=0; j<r; ++j )
            {
                std::memcpy( SF.D.Buffer(0,j), head, m*sizeof(Scalar) );
                head += m*sizeof(Scalar);
            }
        }
        
        break;
    }
    case SPLIT_DENSE:
    {
        shell.data.SD = new SplitDenseMatrix<Scalar>;
        SplitDenseMatrix<Scalar>& SD = *shell.data.SD;

        const int m = H._height;
        const int n = H._width;
        SD.height        = m;
        SD.width         = n;
        SD.comm          = comm;
        SD.partner       = H._partner;
        SD.ownSourceSide = H._ownSourceSide;

        if( SD.ownSourceSide )
        {
            const MatrixType type = *((MatrixType*)head); 
            head += sizeof(MatrixType);

            SD.D.SetType( type );
            SD.D.Resize( m, n );
            if( type == GENERAL )
            {
                for( int j=0; j<n; ++j )
                {
                    std::memcpy( SD.D.Buffer(0,j), head, m*sizeof(Scalar) );
                    head += m*sizeof(Scalar);
                }
            }
            else
            {
                for( int j=0; j<n; ++j )
                {
                    std::memcpy( SD.D.Buffer(j,j), head, (m-j)*sizeof(Scalar) );
                    head += (m-j)*sizeof(Scalar);
                }
            }
        }

        break;
    }
    }
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::MapVectorPrecompute
( Scalar alpha, const Vector<Scalar>& xLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::MapVectorPrecompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;
        for( int t=0; t<4; ++t )
        {
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                Vector<Scalar> xLocalSub;
                xLocalSub.LockedView( xLocal, sOffset, node.sourceSizes[s] );

                node.Child(t,s).MapVectorPrecompute( alpha, xLocalSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix<Scalar,Conjugated>& SF = *shell.data.SF;
        if( SF.ownSourceSide )
        {
            if( Conjugated )
            {
                hmatrix_tools::MatrixHermitianTransposeVector
                ( alpha, SF.D, xLocal, SF.z );
            }
            else
            {
                hmatrix_tools::MatrixTransposeVector
                ( alpha, SF.D, xLocal, SF.z );
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix<Scalar>& SD = *shell.data.SD;
        if( SD.ownSourceSide )
            hmatrix_tools::MatrixVector( alpha, SD.D, xLocal, SD.z );
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::MapVectorNaivePassData() const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::MapVectorNaivePassData");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorNaivePassData();
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix<Scalar,Conjugated>& SF = *shell.data.SF;
        if( SF.ownSourceSide )
            mpi::Send( SF.z.LockedBuffer(), SF.rank, SF.partner, 0, SF.comm );
        else
        {
            SF.z.Resize( SF.rank );
            mpi::Recv( SF.z.Buffer(), SF.rank, SF.partner, 0, SF.comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix<Scalar>& SD = *shell.data.SD;
        if( SD.ownSourceSide )
            mpi::Send( SD.z.LockedBuffer(), SD.height, SD.partner, 0, SD.comm );
        else
        {
            SD.z.Resize( SD.height );
            mpi::Recv( SD.z.Buffer(), SD.height, SD.partner, 0, SD.comm );
        }
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::MapVectorPostcompute
( Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::MapVectorPostcompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.node;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
            for( int s=0; s<4; ++s )
                node.Child(t,s).MapVectorPostcompute( yLocalSub );
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix<Scalar,Conjugated>& SF = *shell.data.SF;
        if( !SF.ownSourceSide )
        {
            hmatrix_tools::MatrixVector
            ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocal );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix<Scalar>& SD = *shell.data.SD;
        if( !SD.ownSourceSide )
        {
            const int localHeight = SD.height;
            const Scalar* zBuffer = SD.z.LockedBuffer();
            Scalar* yLocalBuffer = yLocal.Buffer();
            for( int i=0; i<localHeight; ++i )
                yLocalBuffer[i] += zBuffer[i];
        }
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template class psp::SplitQuasi2dHMatrix<float,false>;
template class psp::SplitQuasi2dHMatrix<float,true>;
template class psp::SplitQuasi2dHMatrix<double,false>;
template class psp::SplitQuasi2dHMatrix<double,true>;
template class psp::SplitQuasi2dHMatrix<std::complex<float>,false>;
template class psp::SplitQuasi2dHMatrix<std::complex<float>,true>;
template class psp::SplitQuasi2dHMatrix<std::complex<double>,false>;
template class psp::SplitQuasi2dHMatrix<std::complex<double>,true>;
