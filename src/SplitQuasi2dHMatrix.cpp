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
            16*sizeof(int) + 2*sizeof(bool) + sizeof(ShellType);
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
            ( sourceSize, targetSize, *shell.data.N->children[i] );
        }
        break;
    case Quasi2d::NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
        {
            PackedSizesRecursion
            ( sourceSize, targetSize, *shell.data.NS->children[i] );
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
    Write( sourceHead, H._height );
    Write( sourceHead, H._width );
    Write( sourceHead, H._numLevels );
    Write( sourceHead, H._maxRank );
    Write( sourceHead, H._sourceOffset );
    Write( sourceHead, H._targetOffset );
    //Write( sourceHead, H._type );
    Write( sourceHead, H._stronglyAdmissible );
    Write( sourceHead, H._xSizeSource );
    Write( sourceHead, H._xSizeTarget );
    Write( sourceHead, H._ySizeSource );
    Write( sourceHead, H._ySizeTarget );
    Write( sourceHead, H._zSize );
    Write( sourceHead, H._xSource );
    Write( sourceHead, H._xTarget );
    Write( sourceHead, H._ySource );
    Write( sourceHead, H._yTarget );
    Write( sourceHead, true );
    Write( sourceHead, targetRank );
    
    // Write out the target member variables
    Write( targetHead, H._height );
    Write( targetHead, H._width );
    Write( targetHead, H._numLevels );
    Write( targetHead, H._maxRank );
    Write( targetHead, H._sourceOffset );
    Write( targetHead, H._targetOffset );
    //Write( targetHead, H._type );
    Write( targetHead, H._stronglyAdmissible );
    Write( targetHead, H._xSizeSource );
    Write( targetHead, H._xSizeTarget );
    Write( targetHead, H._ySizeSource );
    Write( targetHead, H._ySizeTarget );
    Write( targetHead, H._zSize );
    Write( targetHead, H._xSource );
    Write( targetHead, H._xTarget );
    Write( targetHead, H._ySource );
    Write( targetHead, H._yTarget );
    Write( targetHead, false );
    Write( targetHead, sourceRank );

    const typename Quasi2d::Shell& shell = H._shell;
    switch( shell.type )
    {
    case Quasi2d::NODE:
        Write( sourceHead, NODE );
        Write( targetHead, NODE );
        for( int i=0; i<16; ++i )        
        {
            PackRecursion
            ( sourceHead, targetHead, sourceRank, targetRank, 
              *shell.data.N->children[i] );
        }
        break;
    case Quasi2d::NODE_SYMMETRIC:
        Write( sourceHead, NODE_SYMMETRIC );
        Write( targetHead, NODE_SYMMETRIC );
        for( int i=0; i<10; ++i )
        {
            PackRecursion
            ( sourceHead, targetHead, sourceRank, targetRank, 
              *shell.data.NS->children[i] );
        }
        break;
    case Quasi2d::LOW_RANK:
    {
        Write( sourceHead, SPLIT_LOW_RANK );
        Write( targetHead, SPLIT_LOW_RANK );

        const DenseMatrix<Scalar>& U = shell.data.F->U;
        const DenseMatrix<Scalar>& V = shell.data.F->V;
        const int m = U.Height();
        const int n = V.Height();
        const int r = U.Width();

        // Write out the source information
        Write( sourceHead, r );
        for( int j=0; j<r; ++j )
        {
            std::memcpy( sourceHead, V.LockedBuffer(0,j), n*sizeof(Scalar) );
            sourceHead += n*sizeof(Scalar);
        }

        // Write out the target information
        Write( targetHead, r );
        for( int j=0; j<r; ++j )
        {
            std::memcpy( targetHead, U.LockedBuffer(0,j), m*sizeof(Scalar) );
            targetHead += m*sizeof(Scalar);
        }

        break;
    }
    case Quasi2d::DENSE:
    {
        Write( sourceHead, SPLIT_DENSE );
        Write( targetHead, SPLIT_DENSE );

        const DenseMatrix<Scalar>& D = *shell.data.D;
        const int m = D.Height();
        const int n = D.Width();
        const MatrixType type = D.Type();

        // Write out the source information
        Write( sourceHead, type );
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
  /*_type(GENERAL),*/ _stronglyAdmissible(false),
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

    H._height             = Read<int>( head );
    H._width              = Read<int>( head );
    H._numLevels          = Read<int>( head );
    H._maxRank            = Read<int>( head );
    H._sourceOffset       = Read<int>( head );
    H._targetOffset       = Read<int>( head );
    //H._type = Read<MatrixType>( head );
    H._stronglyAdmissible = Read<bool>( head );
    H._xSizeSource        = Read<int>( head );
    H._xSizeTarget        = Read<int>( head );
    H._ySizeSource        = Read<int>( head );
    H._ySizeTarget        = Read<int>( head );
    H._zSize              = Read<int>( head );
    H._xSource            = Read<int>( head );
    H._xTarget            = Read<int>( head );
    H._ySource            = Read<int>( head );
    H._yTarget            = Read<int>( head );
    H._ownSourceSide      = Read<bool>( head );
    H._partner            = Read<int>( head );

    // Delete the old shell information if it exists
    Shell& shell = H._shell;
    switch( shell.type )
    {
    case NODE:           delete shell.data.N;  break;
    case NODE_SYMMETRIC: delete shell.data.NS; break;
    case SPLIT_LOW_RANK: delete shell.data.SF; break;
    case SPLIT_DENSE:    delete shell.data.SD; break;
    }

    // Create this layer of the H-matrix from the packed information
    shell.type = Read<ShellType>( head );
    switch( shell.type )
    {
    case NODE:
    {
        shell.data.N = 
            new Node
            ( H._xSizeSource, H._xSizeTarget,
              H._ySizeSource, H._ySizeTarget, H._zSize );
        Node& node = *shell.data.N;
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
        shell.data.NS = 
            new NodeSymmetric( H._xSizeSource, H._ySizeSource, H._zSize );
        NodeSymmetric& node = *shell.data.NS;
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
        shell.data.SF = new SplitLowRankMatrix;
        SplitLowRankMatrix& SF = *shell.data.SF;
        const int m = H._height;
        const int n = H._width;

        SF.rank = Read<int>( head );

        SF.D.SetType( GENERAL );
        if( _ownSourceSide )
        {
            SF.D.Resize( n, SF.rank );
            for( int j=0; j<SF.rank; ++j )
            {
                std::memcpy( SF.D.Buffer(0,j), head, n*sizeof(Scalar) );
                head += n*sizeof(Scalar);
            }
        }
        else
        {
            SF.D.Resize( m, SF.rank );
            for( int j=0; j<SF.rank; ++j )
            {
                std::memcpy( SF.D.Buffer(0,j), head, m*sizeof(Scalar) );
                head += m*sizeof(Scalar);
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        shell.data.SD = new SplitDenseMatrix;
        SplitDenseMatrix& SD = *shell.data.SD;

        const int m = H._height;
        const int n = H._width;

        if( _ownSourceSide )
        {
            const MatrixType type = Read<MatrixType>( head );

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
        const Node& node = *shell.data.N;
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
        if( _ownSourceSide )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
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
    case SPLIT_DENSE:
        if( _ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            hmatrix_tools::MatrixVector( alpha, SD.D, xLocal, SD.z );
        }
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorPrecompute
( Scalar alpha, const Vector<Scalar>& xLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::TransposeMapVectorPrecompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            Vector<Scalar> xLocalSub;
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorPrecompute
                ( alpha, xLocalSub );
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
        if( !_ownSourceSide )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            hmatrix_tools::MatrixTransposeVector( alpha, SF.D, xLocal, SF.z );
        }
        break;
    case SPLIT_DENSE:
        break;
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
        const Node& node = *shell.data.N;
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
        const SplitLowRankMatrix& SF = *shell.data.SF;
        if( _ownSourceSide )
            mpi::Send( SF.z.LockedBuffer(), SF.rank, _partner, 0, _comm );
        else
        {
            SF.z.Resize( SF.rank );
            mpi::Recv( SF.z.Buffer(), SF.rank, _partner, 0, _comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        if( _ownSourceSide )
            mpi::Send( SD.z.LockedBuffer(), _height, _partner, 0, _comm );
        else
        {
            SD.z.Resize( _height );
            mpi::Recv( SD.z.Buffer(), _height, _partner, 0, _comm );
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
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorNaivePassData
( const Vector<Scalar>& xLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::TransposeMapVectorNaivePassData");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).TransposeMapVectorNaivePassData( xLocal );
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        if( !_ownSourceSide )
            mpi::Send( SF.z.LockedBuffer(), SF.rank, _partner, 0, _comm );
        else
        {
            SF.z.Resize( SF.rank );
            mpi::Recv( SF.z.Buffer(), SF.rank, _partner, 0, _comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        if( !_ownSourceSide )
            mpi::Send( xLocal.LockedBuffer(), _height, _partner, 0, _comm );
        else
        {
            SD.z.Resize( _height );
            mpi::Recv( SD.z.Buffer(), _height, _partner, 0, _comm );
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
        const Node& node = *shell.data.N;
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
        if( !_ownSourceSide )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            hmatrix_tools::MatrixVector
            ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocal );
        }
        break;
    case SPLIT_DENSE:
        if( !_ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            const int localHeight = _height;
            const Scalar* zBuffer = SD.z.LockedBuffer();
            Scalar* yLocalBuffer = yLocal.Buffer();
            for( int i=0; i<localHeight; ++i )
                yLocalBuffer[i] += zBuffer[i];
        }
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorPostcompute
( Scalar alpha, Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::MapVectorPostcompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            Vector<Scalar> yLocalSub;
            yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
            for( int t=0; t<4; ++t )
                node.Child(t,s).TransposeMapVectorPostcompute
                ( alpha, yLocalSub );
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
        if( _ownSourceSide )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            if( Conjugated )
            {
                // yLocal += conj(V) z
                hmatrix_tools::Conjugate( SF.z );
                hmatrix_tools::Conjugate( yLocal );
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocal );
                hmatrix_tools::Conjugate( yLocal );
            }
            else
            {
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocal );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            hmatrix_tools::MatrixTransposeVector
            ( alpha, SD.D, SD.z, (Scalar)1, yLocal );
        }
        break;
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
