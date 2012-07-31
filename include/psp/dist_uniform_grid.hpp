/*
   Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
   of a sweeping preconditioner for 3d Helmholtz equations.

   Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and
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
#ifndef PSP_DIST_UNIFORM_GRID_HPP
#define PSP_DIST_UNIFORM_GRID_HPP 1

namespace psp {

enum GridDataOrder {
    XYZ,
    XZY,
    YXZ,
    YZX,
    ZXY,
    ZYX
};

enum PlaneType {
  XY,
  XZ,
  YZ
};

// The control structure for passing in the distributed data on a 3d grid.
// 
//                 _______________ (wx,wy,0)
//                /              /|
//            x  /              / |
//              /              /  |
//             /______________/   |
//             |              |   |
//             |              |   / (wx,wy,wz)
//           z |              |  /  
//             |              | /  
//             |______________|/
//          (0,0,wz)    y    (0,wy,wz)
//
// The communicator is decomposed into a px x py x pz grid, and the data is 
// cyclically distributed over each of the three dimensions, x first, y second,
// and z third.
//
template<typename F>
class DistUniformGrid
{
public:

    // Generate an nx x ny x nz grid, where each node contains 'numScalars' 
    // entries of type 'F' and the grid is distributed over a px x py x pz grid 
    // over the specified communicator.
    DistUniformGrid
    ( int numScalars,
      int nx, int ny,int nz, GridDataOrder order,
      int px, int py, int pz, mpi::Comm comm );

    // Same as above, but automatically determine the process grid dimensions
    DistUniformGrid
    ( int numScalars,
      int nx, int ny,int nz, GridDataOrder order, 
      mpi::Comm comm );

    int XSize() const;
    int YSize() const;
    int ZSize() const;
    mpi::Comm Comm() const;
    int OwningProcess( int naturalIndex ) const;
    int OwningProcess( int x, int y, int z ) const;

    int NumScalars() const;
    int LocalIndex( int naturalIndex ) const;
    int LocalIndex( int x, int y, int z ) const;
    int NaturalIndex( int x, int y, int z ) const;
    F* LocalBuffer();
    const F* LockedLocalBuffer() const;

    int XShift() const;
    int YShift() const;
    int ZShift() const;
    int XStride() const;
    int YStride() const;
    int ZStride() const;
    int XLocalSize() const;
    int YLocalSize() const;
    int ZLocalSize() const;
    GridDataOrder Order() const;

    // Load the distributed data from a sequential file
    void SequentialLoad( std::string filename );

    // Linearly interpolate the grid to the specified dimensions
    void InterpolateTo( int nx, int ny, int nz );

    void WritePlane
    ( PlaneType planeType, int whichPlane, const std::string basename ) const;
    template<typename R>
    struct WritePlaneHelper
    {
        static void Func
        ( const DistUniformGrid<R>& parent, 
          PlaneType planeType, int whichPlane, const std::string basename );
    };
    template<typename R>
    struct WritePlaneHelper<Complex<R> >
    {
        static void Func
        ( const DistUniformGrid<Complex<R> >& parent, 
          PlaneType planeType, int whichPlane, const std::string basename );
    };
    template<typename R> friend struct WritePlaneHelper;

    void WriteVolume( const std::string basename ) const;
    template<typename R>
    struct WriteVolumeHelper
    {
        static void Func
        ( const DistUniformGrid<R>& parent, const std::string basename );
    };
    template<typename R>
    struct WriteVolumeHelper<Complex<R> >
    {
        static void Func
        ( const DistUniformGrid<Complex<R> >& parent, 
          const std::string basename );
    };
    template<typename R> friend struct WriteVolumeHelper;

private:
    int numScalars_;
    int nx_, ny_, nz_;
    GridDataOrder order_;

    int px_, py_, pz_;
    mpi::Comm comm_;

    int xShift_, yShift_, zShift_;
    int xLocalSize_, yLocalSize_, zLocalSize_;
    std::vector<F> localData_;

    void RedistributeForVtk( std::vector<F>& localBox ) const;
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F>
inline 
DistUniformGrid<F>::DistUniformGrid
( int numScalars,
  int nx, int ny, int nz, GridDataOrder order,
  int px, int py, int pz, mpi::Comm comm )
: numScalars_(numScalars), 
  nx_(nx), ny_(ny), nz_(nz), order_(order),
  px_(px), py_(py), pz_(pz), comm_(comm)
{
    const int commRank = mpi::CommRank( comm );
    const int commSize = mpi::CommSize( comm );
    if( commSize != px*py*pz )
        throw std::logic_error("px*py*pz != commSize");
    if( px < 0 || py < 0 || pz < 0 )
        throw std::logic_error("process dimensions must be non-negative");

    xShift_ = commRank % px;
    yShift_ = (commRank/px) % py;
    zShift_ = commRank/(px*py);
    xLocalSize_ = LocalLength( nx, xShift_, px );
    yLocalSize_ = LocalLength( ny, yShift_, py );
    zLocalSize_ = LocalLength( nz, zShift_, pz );
    localData_.resize( numScalars*xLocalSize_*yLocalSize_*zLocalSize_ );
}

template<typename F>
inline 
DistUniformGrid<F>::DistUniformGrid
( int numScalars,
  int nx, int ny, int nz, GridDataOrder order,
  mpi::Comm comm )
: numScalars_(numScalars), 
  nx_(nx), ny_(ny), nz_(nz), order_(order),
  comm_(comm)
{
#ifndef RELEASE
    PushCallStack("DistUniformGrid::DistUniformGrid");
#endif
    const int commRank = mpi::CommRank( comm );
    const int commSize = mpi::CommSize( comm );

    const int cubeRoot = 
        std::max(1,(int)std::floor(pow((double)commSize,1./3.)));
    px_ = cubeRoot;
    while( commSize % px_ != 0 )
        ++px_;
    const int yzCommSize = commSize / px_;
    const int squareRoot = 
        std::max(1,(int)std::floor(sqrt((double)yzCommSize)));
    py_ = squareRoot;
    while( yzCommSize % py_ != 0 )
        ++py_;
    pz_ = yzCommSize / py_;

    if( commSize != px_*py_*pz_ )
        throw std::logic_error("px*py*pz != commSize");

    xShift_ = commRank % px_;
    yShift_ = (commRank/px_) % py_;
    zShift_ = commRank/(px_*py_);
    xLocalSize_ = LocalLength( nx, xShift_, px_ );
    yLocalSize_ = LocalLength( ny, yShift_, py_ );
    zLocalSize_ = LocalLength( nz, zShift_, pz_ );
    localData_.resize( numScalars*xLocalSize_*yLocalSize_*zLocalSize_ );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline int 
DistUniformGrid<F>::XSize() const
{ return nx_; }

template<typename F>
inline int 
DistUniformGrid<F>::YSize() const
{ return ny_; }

template<typename F>
inline int 
DistUniformGrid<F>::ZSize() const
{ return nz_; }

template<typename F>
inline mpi::Comm 
DistUniformGrid<F>::Comm() const
{ return comm_; }

template<typename F>
inline int 
DistUniformGrid<F>::XShift() const
{ return xShift_; }

template<typename F>
inline int 
DistUniformGrid<F>::YShift() const
{ return yShift_; }

template<typename F>
inline int 
DistUniformGrid<F>::ZShift() const
{ return zShift_; }

template<typename F>
inline int 
DistUniformGrid<F>::XStride() const
{ return px_; }

template<typename F>
inline int 
DistUniformGrid<F>::YStride() const
{ return py_; }

template<typename F>
inline int 
DistUniformGrid<F>::ZStride() const
{ return pz_; }

template<typename F>
inline int 
DistUniformGrid<F>::XLocalSize() const
{ return xLocalSize_; }

template<typename F>
inline int 
DistUniformGrid<F>::YLocalSize() const
{ return yLocalSize_; }

template<typename F>
inline int 
DistUniformGrid<F>::ZLocalSize() const
{ return zLocalSize_; }

template<typename F>
inline F* 
DistUniformGrid<F>::LocalBuffer()
{ return &localData_[0]; }

template<typename F>
inline const F* 
DistUniformGrid<F>::LockedLocalBuffer() const
{ return &localData_[0]; }

template<typename F>
inline int 
DistUniformGrid<F>::NumScalars() const
{ return numScalars_; }

template<typename F>
inline int 
DistUniformGrid<F>::OwningProcess( int naturalIndex ) const
{
    const int x = naturalIndex % nx_;
    const int y = (naturalIndex/nx_) % ny_;
    const int z = naturalIndex/(nx_*ny_);
    return OwningProcess( x, y, z );
}

template<typename F>
inline int 
DistUniformGrid<F>::OwningProcess( int x, int y, int z ) const
{
    const int xProc = x % px_;
    const int yProc = y % py_;
    const int zProc = z % pz_;
    return xProc + yProc*px_ + zProc*px_*py_;
}

template<typename F>
inline int 
DistUniformGrid<F>::LocalIndex( int naturalIndex ) const
{
    const int x = naturalIndex % nx_;
    const int y = (naturalIndex/nx_) % ny_;
    const int z = naturalIndex/(nx_*ny_);
    return LocalIndex( x, y, z );
}

template<typename F>
inline int 
DistUniformGrid<F>::LocalIndex( int x, int y, int z ) const
{ 
#ifndef RELEASE
    PushCallStack("DistUniformGrid::LocalIndex");
    if( x % px_ != xShift_ )
        throw std::logic_error("x coordinate not owned by this process");
    if( y % py_ != yShift_ )
        throw std::logic_error("y coordinate not owned by this process");
    if( z % pz_ != zShift_ )
        throw std::logic_error("z coordinate not owned by this process");
    PopCallStack();
#endif
    const int xLocal = (x-xShift_) / px_;
    const int yLocal = (y-yShift_) / py_;
    const int zLocal = (z-zShift_) / pz_;

    int index=-1;
    switch( order_ )
    {
    case XYZ:
        index = xLocal + yLocal*xLocalSize_ + zLocal*xLocalSize_*yLocalSize_;
        break;
    case XZY:
        index = xLocal + zLocal*xLocalSize_ + yLocal*xLocalSize_*zLocalSize_;
        break;
    case YXZ:
        index = yLocal + xLocal*yLocalSize_ + zLocal*yLocalSize_*xLocalSize_;
        break;
    case YZX:
        index = yLocal + zLocal*yLocalSize_ + xLocal*yLocalSize_*zLocalSize_;
        break;
    case ZXY:
        index = zLocal + xLocal*zLocalSize_ + yLocal*zLocalSize_*yLocalSize_;
        break;
    case ZYX:
        index = zLocal + yLocal*zLocalSize_ + xLocal*zLocalSize_*yLocalSize_;
        break;
    }
    return index*numScalars_;
}

template<typename F>
inline int
DistUniformGrid<F>::NaturalIndex( int x, int y, int z ) const
{ return x + y*nx_ + z*nx_*ny_; }

template<typename F>
inline void
DistUniformGrid<F>::SequentialLoad( std::string filename )
{
#ifndef RELEASE
    PushCallStack("DistUniformGrid::SequentialLoad");
#endif
    std::ifstream is;
    is.open( filename.c_str(), std::ios::in|std::ios::binary );
    if( !is.is_open() )
    {
        std::ostringstream os;
        os << "Could not open " << filename;
        throw std::logic_error( os.str().c_str() );
    }

    switch( order_ )
    {
    case XYZ:
        for( int zLocal=0; zLocal<zLocalSize_; ++zLocal )
        {
            const int z = zShift_ + zLocal*pz_;
            for( int yLocal=0; yLocal<yLocalSize_; ++yLocal )
            {
                const int y = yShift_ + yLocal*py_;
                for( int xLocal=0; xLocal<xLocalSize_; ++xLocal )
                {
                    const int x = xShift_ + xLocal*px_;
                    const int i = x + y*nx_ + z*nx_*ny_;
                    const std::streamoff pos = i*numScalars_*sizeof(F);
                    is.seekg( pos );             
                    const int localIndex = LocalIndex( x, y, z );
                    is.read
                    ( (char*)&localData_[localIndex], numScalars_*sizeof(F) );
                }
            }
        }
        break;
    case XZY:
        for( int yLocal=0; yLocal<yLocalSize_; ++yLocal )
        {
            const int y = yShift_ + yLocal*py_;
            for( int zLocal=0; zLocal<zLocalSize_; ++zLocal )
            {
                const int z = zShift_ + zLocal*pz_;
                for( int xLocal=0; xLocal<xLocalSize_; ++xLocal )
                {
                    const int x = xShift_ + xLocal*px_;
                    const int i = x + y*nx_ + z*nx_*ny_;
                    const std::streamoff pos = i*numScalars_*sizeof(F);
                    is.seekg( pos );             
                    const int localIndex = LocalIndex( x, y, z );
                    is.read
                    ( (char*)&localData_[localIndex], numScalars_*sizeof(F) );
                }
            }
        }
        break;
    case YXZ:
        for( int zLocal=0; zLocal<zLocalSize_; ++zLocal )
        {
            const int z = zShift_ + zLocal*pz_;
            for( int xLocal=0; xLocal<xLocalSize_; ++xLocal )
            {
                const int x = xShift_ + xLocal*px_;
                for( int yLocal=0; yLocal<yLocalSize_; ++yLocal )
                {
                    const int y = yShift_ + yLocal*py_;
                    const int i = x + y*nx_ + z*nx_*ny_;
                    const std::streamoff pos = i*numScalars_*sizeof(F);
                    is.seekg( pos );             
                    const int localIndex = LocalIndex( x, y, z );
                    is.read
                    ( (char*)&localData_[localIndex], numScalars_*sizeof(F) );
                }
            }
        }
        break;
    case YZX:
        for( int xLocal=0; xLocal<xLocalSize_; ++xLocal )
        {
            const int x = xShift_ + xLocal*px_;
            for( int zLocal=0; zLocal<zLocalSize_; ++zLocal )
            {
                const int z = zShift_ + zLocal*pz_;
                for( int yLocal=0; yLocal<yLocalSize_; ++yLocal )
                {
                    const int y = yShift_ + yLocal*py_;
                    const int i = x + y*nx_ + z*nx_*ny_;
                    const std::streamoff pos = i*numScalars_*sizeof(F);
                    is.seekg( pos );             
                    const int localIndex = LocalIndex( x, y, z );
                    is.read
                    ( (char*)&localData_[localIndex], numScalars_*sizeof(F) );
                }
            }
        }
        break;
    case ZXY:
        for( int yLocal=0; yLocal<yLocalSize_; ++yLocal )
        {
            const int y = yShift_ + yLocal*py_;
            for( int xLocal=0; xLocal<xLocalSize_; ++xLocal )
            {
                const int x = xShift_ + xLocal*px_;
                for( int zLocal=0; zLocal<zLocalSize_; ++zLocal )
                {
                    const int z = zShift_ + zLocal*pz_;
                    const int i = x + y*nx_ + z*nx_*ny_;
                    const std::streamoff pos = i*numScalars_*sizeof(F);
                    is.seekg( pos );             
                    const int localIndex = LocalIndex( x, y, z );
                    is.read
                    ( (char*)&localData_[localIndex], numScalars_*sizeof(F) );
                }
            }
        }
        break;
    case ZYX:
        for( int xLocal=0; xLocal<xLocalSize_; ++xLocal )
        {
            const int x = xShift_ + xLocal*px_;
            for( int yLocal=0; yLocal<yLocalSize_; ++yLocal )
            {
                const int y = yShift_ + yLocal*py_;
                for( int zLocal=0; zLocal<zLocalSize_; ++zLocal )
                {
                    const int z = zShift_ + zLocal*pz_;
                    const int i = x + y*nx_ + z*nx_*ny_;
                    const std::streamoff pos = i*numScalars_*sizeof(F);
                    is.seekg( pos );             
                    const int localIndex = LocalIndex( x, y, z );
                    is.read
                    ( (char*)&localData_[localIndex], numScalars_*sizeof(F) );
                }
            }
        }
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void
DistUniformGrid<F>::InterpolateTo( int nx, int ny, int nz )
{
#ifndef RELEASE
    PushCallStack("DistUniformGrid::InterpolateTo");
#endif
    const int nxOld = nx_;
    const int nyOld = ny_;
    const int nzOld = nz_;

    // Compute the new local dimensions
    const int xLocalNewSize = LocalLength( nx, xShift_, px_ );
    const int yLocalNewSize = LocalLength( ny, yShift_, py_ );
    const int zLocalNewSize = LocalLength( nz, zShift_, pz_ );
    const double xRatio = (1.*(nx-1)) / (nxOld-1);
    const double yRatio = (1.*(ny-1)) / (nyOld-1);
    const double zRatio = (1.*(nz-1)) / (nzOld-1);

    // Compute the list of natural indices of the old model which we will
    // interpolate between
    std::set<int> surroundingSet;
    for( int zLocal=0; zLocal<zLocalNewSize; ++zLocal )
    {
        const int z = zShift_ + zLocal*pz_;
        const double zOld = z / zRatio;
        const int zLower = floor( zOld );
        const int zUpper = ( zLower!=nzOld-1 ? zLower+1 : zLower );
#ifndef RELEASE
        if( zLower < 0 || zUpper >= nzOld )
            throw std::logic_error("z interpolation out of bounds");
#endif
        for( int yLocal=0; yLocal<yLocalNewSize; ++yLocal )
        {
            const int y = yShift_ + yLocal*py_;
            const double yOld = y / yRatio;
            const int yLower = floor( yOld );
            const int yUpper = ( yLower!=nyOld-1 ? yLower+1 : yLower );
#ifndef RELEASE
            if( yLower < 0 || yUpper >= nyOld )
                throw std::logic_error("y interpolation out of bounds");
#endif
            for( int xLocal=0; xLocal<xLocalNewSize; ++xLocal )
            {
                const int x = xShift_ + xLocal*px_;
                const double xOld = x / xRatio;
                const int xLower = floor( xOld );
                const int xUpper = ( xLower!=nxOld-1 ? xLower+1 : xLower );
#ifndef RELEASE
                if( xLower < 0 || xUpper >= nxOld )
                    throw std::logic_error("x interpolation out of bounds");
#endif
                surroundingSet.insert( NaturalIndex(xLower,yLower,zLower) );
                surroundingSet.insert( NaturalIndex(xLower,yLower,zUpper) );
                surroundingSet.insert( NaturalIndex(xLower,yUpper,zLower) );
                surroundingSet.insert( NaturalIndex(xLower,yUpper,zUpper) );
                surroundingSet.insert( NaturalIndex(xUpper,yLower,zLower) );
                surroundingSet.insert( NaturalIndex(xUpper,yLower,zUpper) );
                surroundingSet.insert( NaturalIndex(xUpper,yUpper,zLower) );
                surroundingSet.insert( NaturalIndex(xUpper,yUpper,zUpper) );
            }
        }
    }

    // Extract the vector of surrounding natural indices
    const int numSurrounding = surroundingSet.size();
    std::vector<int> surrounding( numSurrounding );
    std::copy
    ( surroundingSet.begin(), surroundingSet.end(), surrounding.begin() );

    // Map the surrounding indices to their owning processes
    const int commSize = mpi::CommSize( comm_ );
    std::vector<int> recvSizes( commSize, 0 );
    std::vector<int> owningProcs( numSurrounding );
    for( int s=0; s<numSurrounding; ++s )
    {
        const int q = OwningProcess( surrounding[s] );
        owningProcs[s] = q;
        ++recvSizes[q];
    }
    int numRecvs=0;
    std::vector<int> recvOffsets( commSize );
    for( int q=0; q<commSize; ++q )
    {
        recvOffsets[q] = numRecvs;
        numRecvs += recvSizes[q];
    }

    // Run an AllToAll to coordinate the send sizes
    std::vector<int> sendSizes( commSize );
    mpi::AllToAll( &recvSizes[0], 1, &sendSizes[0], 1, comm_ );

    // Pack the indices from the original grid which we need
    std::vector<int> recvIndices( numRecvs );
    std::vector<int> offsets = recvOffsets;
    for( int s=0; s<numSurrounding; ++s )
        recvIndices[offsets[owningProcs[s]]++] = surrounding[s];

    // Exchange the indices
    int numSends=0;
    std::vector<int> sendOffsets( commSize );
    for( int q=0; q<commSize; ++q )
    {
        sendOffsets[q] = numSends;
        numSends += sendSizes[q];
    }
    std::vector<int> sendIndices( numSends );
    mpi::AllToAll
    ( &recvIndices[0], &recvSizes[0], &recvOffsets[0],
      &sendIndices[0], &sendSizes[0], &sendOffsets[0], comm_ );

    // Pack the requested indices
    std::vector<F> sendValues( numSends*numScalars_ );
    for( int s=0; s<numSends; ++s )
    {
        const int localIndex = LocalIndex( sendIndices[s] );
        for( int t=0; t<numScalars_; ++t )
            sendValues[s*numScalars_+t] = localData_[localIndex+t];
    }
    for( int q=0; q<commSize; ++q )
    {
        sendSizes[q] *= numScalars_;
        recvSizes[q] *= numScalars_;
        sendOffsets[q] *= numScalars_;
        recvOffsets[q] *= numScalars_;
    }

    // Perform the exchange
    std::vector<F> recvValues( numRecvs*numScalars_ );
    mpi::AllToAll
    ( &sendValues[0], &sendSizes[0], &sendOffsets[0],
      &recvValues[0], &recvSizes[0], &recvOffsets[0], comm_ );
    sendValues.clear();

    // Interpolate to generate the new grid
    nx_ = nx;
    ny_ = ny;
    nz_ = nz;
    xLocalSize_ = xLocalNewSize;
    yLocalSize_ = yLocalNewSize;
    zLocalSize_ = zLocalNewSize;
    localData_.resize( xLocalNewSize*yLocalNewSize*zLocalNewSize*numScalars_ );
    F values[2][2][2];
    int indices[2][2][2], offs[2][2][2];
    for( int zLocal=0; zLocal<zLocalNewSize; ++zLocal )
    {
        const int z = zShift_ + zLocal*pz_;
        const double zOld = z / zRatio;
        const int zLower = floor( zOld );
        const int zUpper = ( zLower!=nzOld-1 ? zLower+1 : zLower );
        for( int yLocal=0; yLocal<yLocalNewSize; ++yLocal )
        {
            const int y = yShift_ + yLocal*py_;
            const double yOld = y / yRatio;
            const int yLower = floor( yOld );
            const int yUpper = ( yLower!=nyOld-1 ? yLower+1 : yLower );
            for( int xLocal=0; xLocal<xLocalNewSize; ++xLocal )
            {
                const int x = xShift_ + xLocal*px_;
                const double xOld = x / xRatio;
                const int xLower = floor( xOld );
                const int xUpper = ( xLower!=nxOld-1 ? xLower+1 : xLower);

                indices[0][0][0] = xLower + yLower*nxOld + zLower*nxOld*nyOld;
                indices[0][0][1] = xLower + yLower*nxOld + zUpper*nxOld*nyOld;
                indices[0][1][0] = xLower + yUpper*nxOld + zLower*nxOld*nyOld;
                indices[0][1][1] = xLower + yUpper*nxOld + zUpper*nxOld*nyOld;
                indices[1][0][0] = xUpper + yLower*nxOld + zLower*nxOld*nyOld;
                indices[1][0][1] = xUpper + yLower*nxOld + zUpper*nxOld*nyOld;
                indices[1][1][0] = xUpper + yUpper*nxOld + zLower*nxOld*nyOld;
                indices[1][1][1] = xUpper + yUpper*nxOld + zUpper*nxOld*nyOld;

                // Find the location of the surrounding indices
                const int* pointer;
                for( int i=0; i<2; ++i )
                {
                    for( int j=0; j<2; ++j )
                    {
                        for( int k=0; k<2; ++k )
                        {
                            const int index = indices[i][j][k];
                            const int xIndex = index % nxOld;
                            const int yIndex = (index/nxOld) % nyOld;
                            const int zIndex = index/(nxOld*nyOld);
                            const int q = OwningProcess( xIndex, yIndex, zIndex );
                            const int* beg = 
                                &recvIndices[recvOffsets[q]/numScalars_];
                            const int* end = 
                                beg + recvSizes[q]/numScalars_;
                            pointer = std::lower_bound( beg, end, index );
#ifndef RELEASE
                            if( pointer == end )
                                throw std::logic_error
                                ("Could not find surrounding index");
#endif
                            const int offset = pointer - &recvIndices[0];
                            offs[i][j][k] = offset;
                        }
                    }
                }

                const int localIndex = LocalIndex( x, y, z );
                for( int s=0; s<numScalars_; ++s )
                {
                    for( int i=0; i<2; ++i )
                        for( int j=0; j<2; ++j )
                            for( int k=0; k<2; ++k )
                                values[i][j][k] = 
                                    recvValues[offs[i][j][k]*numScalars_+s];

                    // Interpolate in the z dimension
                    F t = zOld-floor(zOld);
                    for( int i=0; i<2; ++i )
                        for( int j=0; j<2; ++j )
                            values[i][j][0] = 
                                (1.-t)*values[i][j][0]+t*values[i][j][1];

                    // Interpolate in the y dimension
                    t = yOld-floor(yOld);
                    for( int i=0; i<2; ++i )
                        values[i][0][0] = 
                            (1.-t)*values[i][0][0]+t*values[i][1][0];

                    // Interpolate in the x dimension
                    t = xOld-floor(xOld);
                    const F average = (1.-t)*values[0][0][0]+t*values[1][0][0];
                    localData_[localIndex+s] = average;
                }
            }
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void 
DistUniformGrid<F>::WritePlane
( PlaneType planeType, int whichPlane, const std::string basename ) const
{ return WritePlaneHelper<F>::Func( *this, planeType, whichPlane, basename ); }

template<typename F>
template<typename R>
inline void 
DistUniformGrid<F>::WritePlaneHelper<R>::Func
( const DistUniformGrid<R>& parent, 
  PlaneType planeType, int whichPlane, const std::string basename )
{
#ifndef RELEASE
    PushCallStack("DistUniformGrid::WritePlaneHelper");
#endif
    const int commRank = mpi::CommRank( parent.comm_ );
    const int commSize = mpi::CommSize( parent.comm_ );
    const int nx = parent.nx_;
    const int ny = parent.ny_;
    const int nz = parent.nz_;
    const int px = parent.px_;
    const int py = parent.py_;
    const int pz = parent.pz_;
    const int xLocalSize = parent.xLocalSize_;
    const int yLocalSize = parent.yLocalSize_;
    const int zLocalSize = parent.zLocalSize_;
    const int numScalars = parent.numScalars_;

    // TODO: Use a 2d subcommunicator to speed up the gather
    if( planeType == XY )
    {
        if( whichPlane < 0 || whichPlane >= nz )
            throw std::logic_error("Invalid plane");

        // Compute the number of entries to send to the root
        const int zProc = whichPlane % pz;
        int sendCount = 
            ( zProc==parent.zShift_ ? xLocalSize*yLocalSize*numScalars : 0 );

        int totalRecvSize=0;
        std::vector<int> recvCounts, recvDispls;
        if( commRank == 0 )
        {
            // Compute the number of entries to receive from each process
            recvCounts.resize( commSize, 0 );
            for( int yProc=0; yProc<py; ++yProc )
            {
                const int yLength = LocalLength( ny, yProc, 0, py );
                for( int xProc=0; xProc<px; ++xProc )
                {
                    const int xLength = LocalLength( nx, xProc, 0, px );
                    const int proc = xProc + yProc*px + zProc*px*py;

                    recvCounts[proc] += xLength*yLength*numScalars;
                }
            }

            // Create the send and recv displacements, and the total sizes
            recvDispls.resize( commSize );
            for( int proc=0; proc<commSize; ++proc )
            {
                recvDispls[proc] = totalRecvSize;
                totalRecvSize += recvCounts[proc];
            }
        }

        // Pack the send buffer
        std::vector<F> sendBuffer( std::max(sendCount,1) );
        if( sendCount != 0 )
        {
            int offset=0;
            for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
            {
                const int y = parent.yShift_ + yLocal*py;
                for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                {
                    const int x = parent.xShift_ + xLocal*px;
                    const int localIndex = 
                        parent.LocalIndex( x, y, whichPlane );
                    for( int k=0; k<numScalars; ++k )
                        sendBuffer[offset+k] = parent.localData_[localIndex+k];
                    offset += numScalars;
                }
            }
        }

        std::vector<F> recvBuffer( std::max(totalRecvSize,1) );
        mpi::Gather
        ( &sendBuffer[0], sendCount,
          &recvBuffer[0], &recvCounts[0], &recvDispls[0], 0, parent.comm_ );
        sendBuffer.clear();

        if( commRank == 0 )
        {
            // Unpack the data
            std::vector<F> planes( totalRecvSize );
            const int planeSize = nx*ny;
            for( int yProc=0; yProc<py; ++yProc )
            {
                const int yLength = LocalLength( ny, yProc, 0, py );
                for( int xProc=0; xProc<px; ++xProc )
                {
                    const int xLength = LocalLength( nx, xProc, 0, px );
                    const int proc = xProc + yProc*px + zProc*px*py;

                    for( int jLocal=0; jLocal<yLength; ++jLocal )
                    {
                        const int j = yProc + jLocal*py;
                        for( int iLocal=0; iLocal<xLength; ++iLocal )
                        {
                            const int i = xProc + iLocal*px;
                            for( int k=0; k<numScalars; ++k )
                                planes[i+j*nx+k*planeSize] = 
                                    recvBuffer[recvDispls[proc]++];
                        }
                    }
                }
            }
            recvBuffer.clear();
            
            // Write the data to file
            for( int k=0; k<numScalars; ++k )
            {
                const F* plane = &planes[k*planeSize];

                // For writing a VTK file
                const int maxPoints = std::max(nx,ny);
                const R h = 1./(maxPoints+1.0);
                std::ostringstream os;
                os << basename << "_" << k << ".vti";
                std::ofstream file( os.str().c_str(), std::ios::out );

                os.clear(); os.str("");
                os << "<?xml version=\"1.0\"?>\n"
                   << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
                   << " <ImageData WholeExtent=\""
                   << "0 " << nx << " 0 " << ny << " 0 1\" "
                   << "Origin=\"0 0 " << h*whichPlane << "\" "
                   << "Spacing=\"" << h << " " << h << " " << h << "\">\n"
                   << "  <Piece Extent=\"0 " << nx << " 0 " << ny << " 0 1"
                   << "\">\n"
                   << "    <CellData Scalars=\"cell_scalars\">\n"
                   << "     <DataArray type=\"Float64\" Name=\"cell_scalars\" "
                   << "format=\"ascii\">\n";
                file << os.str();
                for( int j=0; j<ny; ++j ) 
                {
                    for( int i=0; i<nx; ++i )
                    {
                        double value = plane[i+j*nx];
                        if( Abs(value) < 1.0e-300 )
                            value = 0;
                        file << value << " ";
                    }
                    file << "\n";
                }
                os.clear(); os.str("");
                os << "    </DataArray>\n"
                   << "   </CellData>\n"
                   << "  </Piece>\n"
                   << " </ImageData>\n"
                   << "</VTKFile>" << std::endl;
                file << os.str();
            }
        }
    }
    else if( planeType == XZ )
    {
        if( whichPlane < 0 || whichPlane >= ny )
            throw std::logic_error("Invalid plane");

        // Compute the number of entries to send to the root
        const int yProc = whichPlane % py;
        int sendCount = 
            ( yProc==parent.yShift_ ? xLocalSize*zLocalSize*numScalars : 0 );

        int totalRecvSize=0;
        std::vector<int> recvCounts, recvDispls;
        if( commRank == 0 )
        {
            // Compute the number of entries to receive from each process
            recvCounts.resize( commSize, 0 );
            for( int zProc=0; zProc<pz; ++zProc )
            {
                const int zLength = LocalLength( nz, zProc, 0, pz );
                for( int xProc=0; xProc<px; ++xProc )
                {
                    const int xLength = LocalLength( nx, xProc, 0, px );
                    const int proc = xProc + yProc*px + zProc*px*py;

                    recvCounts[proc] += xLength*zLength*numScalars;
                }
            }

            // Create the send and recv displacements, and the total sizes
            recvDispls.resize( commSize );
            for( int proc=0; proc<commSize; ++proc )
            {
                recvDispls[proc] = totalRecvSize;
                totalRecvSize += recvCounts[proc];
            }
        }

        // Pack the send buffer
        std::vector<F> sendBuffer( std::max(sendCount,1) );
        if( sendCount != 0 )
        {
            int offset=0;
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = parent.zShift_ + zLocal*pz;
                for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                {
                    const int x = parent.xShift_ + xLocal*px;
                    const int localIndex = 
                        parent.LocalIndex( x, whichPlane, z );
                    for( int k=0; k<numScalars; ++k )
                        sendBuffer[offset+k] = parent.localData_[localIndex+k];
                    offset += numScalars;
                }
            }
        }

        std::vector<F> recvBuffer( std::max(totalRecvSize,1) );
        mpi::Gather
        ( &sendBuffer[0], sendCount,
          &recvBuffer[0], &recvCounts[0], &recvDispls[0], 0, parent.comm_ );
        sendBuffer.clear();

        if( commRank == 0 )
        {
            // Unpack the data
            std::vector<F> planes( totalRecvSize );
            const int planeSize = nx*nz;
            for( int zProc=0; zProc<pz; ++zProc )
            {
                const int zLength = LocalLength( nz, zProc, 0, pz );
                for( int xProc=0; xProc<px; ++xProc )
                {
                    const int xLength = LocalLength( nx, xProc, 0, px );
                    const int proc = xProc + yProc*px + zProc*px*py;

                    for( int jLocal=0; jLocal<zLength; ++jLocal )
                    {
                        const int j = zProc + jLocal*pz;
                        for( int iLocal=0; iLocal<xLength; ++iLocal )
                        {
                            const int i = xProc + iLocal*px;
                            for( int k=0; k<numScalars; ++k )
                                planes[i+j*nx+k*planeSize] = 
                                    recvBuffer[recvDispls[proc]++];
                        }
                    }
                }
            }
            recvBuffer.clear();
            
            // Write the data to file
            for( int k=0; k<numScalars; ++k )
            {
                const F* plane = &planes[k*planeSize];

                // For writing a VTK file
                const int maxPoints = std::max(nx,nz);
                const R h = 1./(maxPoints+1.0);
                std::ostringstream os;
                os << basename << "_" << k << ".vti";
                std::ofstream file( os.str().c_str(), std::ios::out );
                os.clear(); os.str("");
                os << "<?xml version=\"1.0\"?>\n"
                   << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
                   << " <ImageData WholeExtent=\""
                   << "0 " << nx << " 0 1 0 " << nz << "\" "
                   << "Origin=\"0 " << h*whichPlane << " 0\" "
                   << "Spacing=\"" << h << " " << h << " " << h << "\">\n"
                   << "  <Piece Extent=\"0 " << nx << " 0 1 0 " << nz << "\">\n"
                   << "    <CellData Scalars=\"cell_scalars\">\n"
                   << "     <DataArray type=\"Float64\" Name=\"cell_scalars\" "
                   << "format=\"ascii\">\n";
                file << os.str();
                for( int j=0; j<nz; ++j ) 
                {
                    for( int i=0; i<nx; ++i )
                    {
                        double value = plane[i+j*nx];
                        if( Abs(value) < 1.0e-300 )
                            value = 0;
                        file << value << " ";
                    }
                    file << "\n";
                }
                os.clear(); os.str("");
                os << "    </DataArray>\n"
                   << "   </CellData>\n"
                   << "  </Piece>\n"
                   << " </ImageData>\n"
                   << "</VTKFile>" << std::endl;
                file << os.str();
            }
        }
    }
    else if( planeType == YZ )
    {
        if( whichPlane < 0 || whichPlane >= nx )
            throw std::logic_error("Invalid plane");

        // Compute the number of entries to send to the root
        const int xProc = whichPlane % px;
        int sendCount = 
            ( xProc==parent.xShift_ ? yLocalSize*zLocalSize*numScalars : 0 );

        int totalRecvSize=0;
        std::vector<int> recvCounts, recvDispls;
        if( commRank == 0 )
        {
            // Compute the number of entries to receive from each process
            recvCounts.resize( commSize, 0 );
            for( int zProc=0; zProc<pz; ++zProc )
            {
                const int zLength = LocalLength( nz, zProc, 0, pz );
                for( int yProc=0; yProc<py; ++yProc )
                {
                    const int yLength = LocalLength( ny, yProc, 0, py );
                    const int proc = xProc + yProc*px + zProc*px*py;

                    recvCounts[proc] += yLength*zLength*numScalars;
                }
            }

            // Create the send and recv displacements, and the total sizes
            recvDispls.resize( commSize );
            for( int proc=0; proc<commSize; ++proc )
            {
                recvDispls[proc] = totalRecvSize;
                totalRecvSize += recvCounts[proc];
            }
        }

        // Pack the send buffer
        std::vector<F> sendBuffer( std::max(sendCount,1) );
        if( sendCount != 0 )
        {
            int offset=0;
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = parent.zShift_ + zLocal*pz;
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = parent.yShift_ + yLocal*py;
                    const int localIndex = 
                        parent.LocalIndex( whichPlane, y, z );
                    for( int k=0; k<numScalars; ++k )
                        sendBuffer[offset+k] = parent.localData_[localIndex+k];
                    offset += numScalars;
                }
            }
        }

        std::vector<F> recvBuffer( std::max(totalRecvSize,1) );
        mpi::Gather
        ( &sendBuffer[0], sendCount,
          &recvBuffer[0], &recvCounts[0], &recvDispls[0], 0, parent.comm_ );
        sendBuffer.clear();

        if( commRank == 0 )
        {
            // Unpack the data
            std::vector<F> planes( totalRecvSize );
            const int planeSize = ny*nz;
            for( int zProc=0; zProc<pz; ++zProc )
            {
                const int zLength = LocalLength( nz, zProc, 0, pz );
                for( int yProc=0; yProc<py; ++yProc )
                {
                    const int yLength = LocalLength( ny, yProc, 0, py );
                    const int proc = xProc + yProc*px + zProc*px*py;

                    for( int jLocal=0; jLocal<zLength; ++jLocal )
                    {
                        const int j = zProc + jLocal*pz;
                        for( int iLocal=0; iLocal<yLength; ++iLocal )
                        {
                            const int i = yProc + iLocal*py;
                            for( int k=0; k<numScalars; ++k )
                                planes[i+j*ny+k*planeSize] = 
                                    recvBuffer[recvDispls[proc]++];
                        }
                    }
                }
            }
            recvBuffer.clear();
            
            // Write the data to file
            for( int k=0; k<numScalars; ++k )
            {
                const F* plane = &planes[k*planeSize];

                // For writing a VTK file
                const int maxPoints = std::max(ny,nz);
                const R h = 1./(maxPoints+1.0);
                std::ostringstream os;
                os << basename << "_" << k << ".vti";
                std::ofstream file( os.str().c_str(), std::ios::out );
                os.clear(); os.str("");
                os << "<?xml version=\"1.0\"?>\n"
                   << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
                   << " <ImageData WholeExtent=\""
                   << "0 1 0 " << ny << " 0 " << nz << "\" "
                   << "Origin=\"" << h*whichPlane << " 0 0\" "
                   << "Spacing=\"" << h << " " << h << " " << h << "\">\n"
                   << "  <Piece Extent=\"0 1 0 " << ny << " 0 " << nz << "\">\n"
                   << "    <CellData Scalars=\"cell_scalars\">\n"
                   << "     <DataArray type=\"Float64\" Name=\"cell_scalars\" "
                   << "format=\"ascii\">\n";
                file << os.str();
                for( int j=0; j<nz; ++j ) 
                {
                    for( int i=0; i<ny; ++i )
                    {
                        double value = plane[i+j*ny];
                        if( Abs(value) < 1.0e-300 )
                            value = 0;
                        file << value << " ";
                    }
                    file << "\n";
                }
                os.clear(); os.str("");
                os << "    </DataArray>\n"
                   << "   </CellData>\n"
                   << "  </Piece>\n"
                   << " </ImageData>\n"
                   << "</VTKFile>" << std::endl;
                file << os.str();
            }
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
template<typename R>
inline void 
DistUniformGrid<F>::WritePlaneHelper<Complex<R> >::Func
( const DistUniformGrid<Complex<R> >& parent, 
  PlaneType planeType, int whichPlane, const std::string basename )
{
    const int commRank = mpi::CommRank( parent.comm_ );
    const int commSize = mpi::CommSize( parent.comm_ );
    const int nx = parent.nx_;
    const int ny = parent.ny_;
    const int nz = parent.nz_;
    const int px = parent.px_;
    const int py = parent.py_;
    const int pz = parent.pz_;
    const int xLocalSize = parent.xLocalSize_;
    const int yLocalSize = parent.yLocalSize_;
    const int zLocalSize = parent.zLocalSize_;
    const int numScalars = parent.numScalars_;

    // TODO: Use a 2d subcommunicator to speed up the gather

    if( planeType == XY )
    {
        if( whichPlane < 0 || whichPlane >= nz )
            throw std::logic_error("Invalid plane");

        // Compute the number of entries to send to the root
        const int zProc = whichPlane % pz;
        int sendCount = 
            ( zProc==parent.zShift_ ? xLocalSize*yLocalSize*numScalars : 0 );

        int totalRecvSize=0;
        std::vector<int> recvCounts, recvDispls;
        if( commRank == 0 )
        {
            // Compute the number of entries to receive from each process
            recvCounts.resize( commSize, 0 );
            for( int yProc=0; yProc<py; ++yProc )
            {
                const int yLength = LocalLength( ny, yProc, 0, py );
                for( int xProc=0; xProc<px; ++xProc )
                {
                    const int xLength = LocalLength( nx, xProc, 0, px );
                    const int proc = xProc + yProc*px + zProc*px*py;

                    recvCounts[proc] += xLength*yLength*numScalars;
                }
            }

            // Create the send and recv displacements, and the total sizes
            recvDispls.resize( commSize );
            for( int proc=0; proc<commSize; ++proc )
            {
                recvDispls[proc] = totalRecvSize;
                totalRecvSize += recvCounts[proc];
            }
        }

        // Pack the send buffer
        std::vector<F> sendBuffer( std::max(sendCount,1) );
        if( sendCount != 0 )
        {
            int offset=0;
            for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
            {
                const int y = parent.yShift_ + yLocal*py;
                for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                {
                    const int x = parent.xShift_ + xLocal*px;
                    const int localIndex = 
                        parent.LocalIndex( x, y, whichPlane );
                    for( int k=0; k<numScalars; ++k )
                        sendBuffer[offset+k] = parent.localData_[localIndex+k];
                    offset += numScalars;
                }
            }
        }

        std::vector<F> recvBuffer( std::max(totalRecvSize,1) );
        mpi::Gather
        ( &sendBuffer[0], sendCount,
          &recvBuffer[0], &recvCounts[0], &recvDispls[0], 0, parent.comm_ );
        sendBuffer.clear();

        if( commRank == 0 )
        {
            // Unpack the data
            std::vector<F> planes( totalRecvSize );
            const int planeSize = nx*ny;
            for( int yProc=0; yProc<py; ++yProc )
            {
                const int yLength = LocalLength( ny, yProc, 0, py );
                for( int xProc=0; xProc<px; ++xProc )
                {
                    const int xLength = LocalLength( nx, xProc, 0, px );
                    const int proc = xProc + yProc*px + zProc*px*py;

                    for( int jLocal=0; jLocal<yLength; ++jLocal )
                    {
                        const int j = yProc + jLocal*py;
                        for( int iLocal=0; iLocal<xLength; ++iLocal )
                        {
                            const int i = xProc + iLocal*px;
                            for( int k=0; k<numScalars; ++k )
                                planes[i+j*nx+k*planeSize] = 
                                    recvBuffer[recvDispls[proc]++];
                        }
                    }
                }
            }
            recvBuffer.clear();
            
            // Write the data to file
            for( int k=0; k<numScalars; ++k )
            {
                const F* plane = &planes[k*planeSize];

                // For writing a VTK file
                const int maxPoints = std::max(nx,ny);
                const R h = 1./(maxPoints+1.0);
                std::ostringstream os;
                os << basename << "_" << k << "_real.vti";
                std::ofstream realFile( os.str().c_str(), std::ios::out );
                os.clear(); os.str("");
                os << basename << "_" << k << "_imag.vti";
                std::ofstream imagFile( os.str().c_str(), std::ios::out );
                os.clear(); os.str("");
                os << "<?xml version=\"1.0\"?>\n"
                   << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
                   << " <ImageData WholeExtent=\""
                   << "0 " << nx << " 0 " << ny << " 0 1\" "
                   << "Origin=\"0 0 " << h*whichPlane << "\" "
                   << "Spacing=\"" << h << " " << h << " " << h << "\">\n"
                   << "  <Piece Extent=\"0 " << nx << " 0 " << ny << " 0 1"
                   << "\">\n"
                   << "    <CellData Scalars=\"cell_scalars\">\n"
                   << "     <DataArray type=\"Float64\" Name=\"cell_scalars\" "
                   << "format=\"ascii\">\n";
                realFile << os.str();
                imagFile << os.str();
                for( int j=0; j<ny; ++j ) 
                {
                    for( int i=0; i<nx; ++i )
                    {
                        double value = plane[i+j*nx].real;
                        if( Abs(value) < 1.0e-300 )
                            value = 0;
                        realFile << value << " ";
                    }
                    realFile << "\n";
                }
                for( int j=0; j<ny; ++j )
                {
                    for( int i=0; i<nx; ++i )
                    {
                        double value = plane[i+j*nx].imag;
                        if( Abs(value) < 1.0e-300 )
                            value = 0;
                        imagFile << value << " ";
                    }
                    imagFile << "\n";
                }
                os.clear(); os.str("");
                os << "    </DataArray>\n"
                   << "   </CellData>\n"
                   << "  </Piece>\n"
                   << " </ImageData>\n"
                   << "</VTKFile>" << std::endl;
                realFile << os.str();
                imagFile << os.str();
            }
        }
    }
    else if( planeType == XZ )
    {
        if( whichPlane < 0 || whichPlane >= ny )
            throw std::logic_error("Invalid plane");

        // Compute the number of entries to send to the root
        const int yProc = whichPlane % py;
        int sendCount = 
            ( yProc==parent.yShift_ ? xLocalSize*zLocalSize*numScalars : 0 );

        int totalRecvSize=0;
        std::vector<int> recvCounts, recvDispls;
        if( commRank == 0 )
        {
            // Compute the number of entries to receive from each process
            recvCounts.resize( commSize, 0 );
            for( int zProc=0; zProc<pz; ++zProc )
            {
                const int zLength = LocalLength( nz, zProc, 0, pz );
                for( int xProc=0; xProc<px; ++xProc )
                {
                    const int xLength = LocalLength( nx, xProc, 0, px );
                    const int proc = xProc + yProc*px + zProc*px*py;

                    recvCounts[proc] += xLength*zLength*numScalars;
                }
            }

            // Create the send and recv displacements, and the total sizes
            recvDispls.resize( commSize );
            for( int proc=0; proc<commSize; ++proc )
            {
                recvDispls[proc] = totalRecvSize;
                totalRecvSize += recvCounts[proc];
            }
        }

        // Pack the send buffer
        std::vector<F> sendBuffer( std::max(sendCount,1) );
        if( sendCount != 0 )
        {
            int offset=0;
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = parent.zShift_ + zLocal*pz;
                for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                {
                    const int x = parent.xShift_ + xLocal*px;
                    const int localIndex = 
                        parent.LocalIndex( x, whichPlane, z );
                    for( int k=0; k<numScalars; ++k )
                        sendBuffer[offset+k] = parent.localData_[localIndex+k];
                    offset += numScalars;
                }
            }
        }

        std::vector<F> recvBuffer( std::max(totalRecvSize,1) );
        mpi::Gather
        ( &sendBuffer[0], sendCount,
          &recvBuffer[0], &recvCounts[0], &recvDispls[0], 0, parent.comm_ );
        sendBuffer.clear();

        if( commRank == 0 )
        {
            // Unpack the data
            std::vector<F> planes( totalRecvSize );
            const int planeSize = nx*nz;
            for( int zProc=0; zProc<pz; ++zProc )
            {
                const int zLength = LocalLength( nz, zProc, 0, pz );
                for( int xProc=0; xProc<px; ++xProc )
                {
                    const int xLength = LocalLength( nx, xProc, 0, px );
                    const int proc = xProc + yProc*px + zProc*px*py;

                    for( int jLocal=0; jLocal<zLength; ++jLocal )
                    {
                        const int j = zProc + jLocal*pz;
                        for( int iLocal=0; iLocal<xLength; ++iLocal )
                        {
                            const int i = xProc + iLocal*px;
                            for( int k=0; k<numScalars; ++k )
                                planes[i+j*nx+k*planeSize] = 
                                    recvBuffer[recvDispls[proc]++];
                        }
                    }
                }
            }
            recvBuffer.clear();
            
            // Write the data to file
            for( int k=0; k<numScalars; ++k )
            {
                const F* plane = &planes[k*planeSize];

                // For writing a VTK file
                const int maxPoints = std::max(nx,nz);
                const R h = 1./(maxPoints+1.0);
                std::ostringstream os;
                os << basename << "_" << k << "_real.vti";
                std::ofstream realFile( os.str().c_str(), std::ios::out );
                os.clear(); os.str("");
                os << basename << "_" << k << "_imag.vti";
                std::ofstream imagFile( os.str().c_str(), std::ios::out );
                os.clear(); os.str("");
                os << "<?xml version=\"1.0\"?>\n"
                   << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
                   << " <ImageData WholeExtent=\""
                   << "0 " << nx << " 0 1 0 " << nz << "\" "
                   << "Origin=\"0 " << h*whichPlane << " 0\" "
                   << "Spacing=\"" << h << " " << h << " " << h << "\">\n"
                   << "  <Piece Extent=\"0 " << nx << " 0 1 0 " << nz << "\">\n"
                   << "    <CellData Scalars=\"cell_scalars\">\n"
                   << "     <DataArray type=\"Float64\" Name=\"cell_scalars\" "
                   << "format=\"ascii\">\n";
                realFile << os.str();
                imagFile << os.str();
                for( int j=0; j<nz; ++j ) 
                {
                    for( int i=0; i<nx; ++i )
                    {
                        double value = plane[i+j*nx].real;
                        if( Abs(value) < 1.0e-300 )
                            value = 0;
                        realFile << value << " ";
                    }
                    realFile << "\n";
                }
                for( int j=0; j<nz; ++j ) 
                {
                    for( int i=0; i<nx; ++i )
                    {
                        double value = plane[i+j*nx].imag;
                        if( Abs(value) < 1.0e-300 )
                            value = 0;
                        imagFile << value << " ";
                    }
                    imagFile << "\n";
                }
                os.clear(); os.str("");
                os << "    </DataArray>\n"
                   << "   </CellData>\n"
                   << "  </Piece>\n"
                   << " </ImageData>\n"
                   << "</VTKFile>" << std::endl;
                realFile << os.str();
                imagFile << os.str();
            }
        }
    }
    else if( planeType == YZ )
    {
        if( whichPlane < 0 || whichPlane >= nx )
            throw std::logic_error("Invalid plane");

        // Compute the number of entries to send to the root
        const int xProc = whichPlane % px;
        int sendCount = 
            ( xProc==parent.xShift_ ? yLocalSize*zLocalSize*numScalars : 0 );

        int totalRecvSize=0;
        std::vector<int> recvCounts, recvDispls;
        if( commRank == 0 )
        {
            // Compute the number of entries to receive from each process
            recvCounts.resize( commSize, 0 );
            for( int zProc=0; zProc<pz; ++zProc )
            {
                const int zLength = LocalLength( nz, zProc, 0, pz );
                for( int yProc=0; yProc<py; ++yProc )
                {
                    const int yLength = LocalLength( ny, yProc, 0, py );
                    const int proc = xProc + yProc*px + zProc*px*py;

                    recvCounts[proc] += yLength*zLength*numScalars;
                }
            }

            // Create the send and recv displacements, and the total sizes
            recvDispls.resize( commSize );
            for( int proc=0; proc<commSize; ++proc )
            {
                recvDispls[proc] = totalRecvSize;
                totalRecvSize += recvCounts[proc];
            }
        }

        // Pack the send buffer
        std::vector<F> sendBuffer( std::max(sendCount,1) );
        if( sendCount != 0 )
        {
            int offset=0;
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = parent.zShift_ + zLocal*pz;
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = parent.yShift_ + yLocal*py;
                    const int localIndex = 
                        parent.LocalIndex( whichPlane, y, z );
                    for( int k=0; k<numScalars; ++k )
                        sendBuffer[offset+k] = parent.localData_[localIndex+k];
                    offset += numScalars;
                }
            }
        }

        std::vector<F> recvBuffer( std::max(totalRecvSize,1) );
        mpi::Gather
        ( &sendBuffer[0], sendCount,
          &recvBuffer[0], &recvCounts[0], &recvDispls[0], 0, parent.comm_ );
        sendBuffer.clear();

        if( commRank == 0 )
        {
            // Unpack the data
            std::vector<F> planes( totalRecvSize );
            const int planeSize = ny*nz;
            for( int zProc=0; zProc<pz; ++zProc )
            {
                const int zLength = LocalLength( nz, zProc, 0, pz );
                for( int yProc=0; yProc<py; ++yProc )
                {
                    const int yLength = LocalLength( ny, yProc, 0, py );
                    const int proc = xProc + yProc*px + zProc*px*py;

                    for( int jLocal=0; jLocal<zLength; ++jLocal )
                    {
                        const int j = zProc + jLocal*pz;
                        for( int iLocal=0; iLocal<yLength; ++iLocal )
                        {
                            const int i = yProc + iLocal*py;
                            for( int k=0; k<numScalars; ++k )
                                planes[i+j*ny+k*planeSize] = 
                                    recvBuffer[recvDispls[proc]++];
                        }
                    }
                }
            }
            recvBuffer.clear();
            
            // Write the data to file
            for( int k=0; k<numScalars; ++k )
            {
                const F* plane = &planes[k*planeSize];

                // For writing a VTK file
                const int maxPoints = std::max(ny,nz);
                const R h = 1./(maxPoints+1.0);
                std::ostringstream os;
                os << basename << "_" << k << "_real.vti";
                std::ofstream realFile( os.str().c_str(), std::ios::out );
                os.clear(); os.str("");
                os << basename << "_" << k << "_imag.vti";
                std::ofstream imagFile( os.str().c_str(), std::ios::out );
                os.clear(); os.str("");
                os << "<?xml version=\"1.0\"?>\n"
                   << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
                   << " <ImageData WholeExtent=\""
                   << "0 1 0 " << ny << " 0 " << nz << "\" "
                   << "Origin=\"" << h*whichPlane << " 0 0\" "
                   << "Spacing=\"" << h << " " << h << " " << h << "\">\n"
                   << "  <Piece Extent=\"0 1 0 " << ny << " 0 " << nz << "\">\n"
                   << "    <CellData Scalars=\"cell_scalars\">\n"
                   << "     <DataArray type=\"Float64\" Name=\"cell_scalars\" "
                   << "format=\"ascii\">\n";
                realFile << os.str();
                imagFile << os.str();
                for( int j=0; j<nz; ++j ) 
                {
                    for( int i=0; i<ny; ++i )
                    {
                        double value = plane[i+j*ny].real;
                        if( Abs(value) < 1.0e-300 )
                            value = 0;
                        realFile << value << " ";
                    }
                    realFile << "\n";
                }
                for( int j=0; j<nz; ++j ) 
                {
                    for( int i=0; i<ny; ++i )
                    {
                        double value = plane[i+j*ny].imag;
                        if( Abs(value) < 1.0e-300 )
                            value = 0;
                        imagFile << value << " ";
                    }
                    imagFile << "\n";
                }
                os.clear(); os.str("");
                os << "    </DataArray>\n"
                   << "   </CellData>\n"
                   << "  </Piece>\n"
                   << " </ImageData>\n"
                   << "</VTKFile>" << std::endl;
                realFile << os.str();
                imagFile << os.str();
            }
        }
    }
}

template<typename F>
inline void 
DistUniformGrid<F>::RedistributeForVtk( std::vector<F>& localBox ) const
{
#ifndef RELEASE
    PushCallStack("DistUniformGrid::RedistributeForVtk");
#endif
    const int commSize = mpi::CommSize( comm_ );

    // Compute our local box
    const int xMainSize = nx_ / px_;
    const int yMainSize = ny_ / py_;
    const int zMainSize = nz_ / pz_;
    const int xLeftoverSize = xMainSize + (nx_ % px_);
    const int yLeftoverSize = yMainSize + (ny_ % py_);
    const int zLeftoverSize = zMainSize + (nz_ % pz_);
    const int xBoxStart = xMainSize*xShift_;
    const int yBoxStart = yMainSize*yShift_;
    const int zBoxStart = zMainSize*zShift_;
    const int xBoxSize = ( xShift_==px_-1 ? xLeftoverSize : xMainSize );
    const int yBoxSize = ( yShift_==py_-1 ? yLeftoverSize : yMainSize );
    const int zBoxSize = ( zShift_==pz_-1 ? zLeftoverSize : zMainSize );

    // Compute the number of entries to send to each process
    std::vector<int> sendCounts( commSize, 0 );
    for( int zLocal=0; zLocal<zLocalSize_; ++zLocal )
    {
        const int z = zShift_ + zLocal*pz_;
        const int zProc = std::min(pz_-1,z/zMainSize);
        for( int yLocal=0; yLocal<yLocalSize_; ++yLocal )
        {
            const int y = yShift_ + yLocal*py_;
            const int yProc = std::min(py_-1,y/yMainSize);
            for( int xLocal=0; xLocal<xLocalSize_; ++xLocal )
            {
                const int x = xShift_ + xLocal*px_;
                const int xProc = std::min(px_-1,x/xMainSize);
                const int proc = xProc + yProc*px_ + zProc*px_*py_;
                sendCounts[proc] += numScalars_;
            }
        }
    }
    
    // Compute the number of entries to receive from each process
    std::vector<int> recvCounts( commSize, 0 );
    const int xAlign = xBoxStart % px_;
    const int yAlign = yBoxStart % py_;
    const int zAlign = zBoxStart % pz_;
    for( int zProc=0; zProc<pz_; ++zProc )
    {
        const int zLength = LocalLength( zBoxSize, zProc, zAlign, pz_ );
        for( int yProc=0; yProc<py_; ++yProc )
        {
            const int yLength = LocalLength( yBoxSize, yProc, yAlign, py_ );
            for( int xProc=0; xProc<px_; ++xProc )
            {
                const int xLength = LocalLength( xBoxSize, xProc, xAlign, px_ );
                const int proc = xProc + yProc*px_ + zProc*px_*py_;

                recvCounts[proc] += xLength*yLength*zLength*numScalars_;
            }
        }
    }

    // Create the send and recv displacements, and the total sizes
    int totalSendSize=0, totalRecvSize=0;
    std::vector<int> sendDispls( commSize ), recvDispls( commSize );
    for( int proc=0; proc<commSize; ++proc )
    {
        sendDispls[proc] = totalSendSize; 
        recvDispls[proc] = totalRecvSize;
        totalSendSize += sendCounts[proc];
        totalRecvSize += recvCounts[proc];
    }
#ifndef RELEASE
    if( totalRecvSize != xBoxSize*yBoxSize*zBoxSize*numScalars_ )
        throw std::logic_error("Incorrect total recv size");
#endif

    // Pack the send buffer
    std::vector<F> sendBuffer( totalSendSize );
    std::vector<int> offsets = sendDispls;
    for( int zLocal=0; zLocal<zLocalSize_; ++zLocal )
    {
        const int z = zShift_ + zLocal*pz_;
        const int zProc = std::min(pz_-1,z/zMainSize);
        for( int yLocal=0; yLocal<yLocalSize_; ++yLocal )
        {
            const int y = yShift_ + yLocal*py_;
            const int yProc = std::min(py_-1,y/yMainSize);
            for( int xLocal=0; xLocal<xLocalSize_; ++xLocal )
            {
                const int x = xShift_ + xLocal*px_;
                const int xProc = std::min(px_-1,x/xMainSize);
                const int proc = xProc + yProc*px_ + zProc*px_*py_;

                const int localIndex = LocalIndex( x, y, z );
                for( int k=0; k<numScalars_; ++k )
                    sendBuffer[offsets[proc]+k] = localData_[localIndex+k];
                offsets[proc] += numScalars_;
            }
        }
    }

    // Perform AllToAllv
    std::vector<F> recvBuffer( totalRecvSize );
    mpi::AllToAll
    ( &sendBuffer[0], &sendCounts[0], &sendDispls[0],
      &recvBuffer[0], &recvCounts[0], &recvDispls[0], comm_ );
    sendBuffer.clear();

    // Unpack the recv buffer
    localBox.resize( totalRecvSize );
    for( int zProc=0; zProc<pz_; ++zProc )
    {
        const int zOffset = Shift( zProc, zAlign, pz_ );
        const int zLength = LocalLength( zBoxSize, zOffset, pz_ );
        for( int yProc=0; yProc<py_; ++yProc )
        {
            const int yOffset = Shift( yProc, yAlign, py_ );
            const int yLength = LocalLength( yBoxSize, yOffset, py_ );
            for( int xProc=0; xProc<px_; ++xProc )
            {
                const int xOffset = Shift( xProc, xAlign, px_ );
                const int xLength = LocalLength( xBoxSize, xOffset, px_ );
                const int proc = xProc + yProc*px_ + zProc*px_*py_;

                // Unpack all of the data from this process
                const int localOffset = 
                    (xOffset + yOffset*xBoxSize + zOffset*xBoxSize*yBoxSize)*
                    numScalars_;
                F* offsetLocal = &localBox[localOffset];
                const F* procRecv = &recvBuffer[recvDispls[proc]];
                for( int zLocal=0; zLocal<zLength; ++zLocal )
                {
                    for( int yLocal=0; yLocal<yLength; ++yLocal )
                    {
                        const int localRowIndex = 
                            (yLocal*py_ + zLocal*pz_*yBoxSize)*xBoxSize*
                            numScalars_;
                        const int procRowIndex = 
                            (yLocal + zLocal*yLength)*xLength*numScalars_;
                        F* localRow = &offsetLocal[localRowIndex];
                        const F* procRow = &procRecv[procRowIndex];
                        for( int xLocal=0; xLocal<xLength; ++xLocal )
                        {
                            elem::MemCopy
                            ( &localRow[xLocal*px_*numScalars_],
                              &procRow[xLocal*numScalars_], numScalars_ );
                        }
                    }
                }
            }
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void 
DistUniformGrid<F>::WriteVolume( const std::string basename ) const
{ return WriteVolumeHelper<F>::Func( *this, basename ); }

template<typename F>
template<typename R>
inline void 
DistUniformGrid<F>::WriteVolumeHelper<R>::Func
( const DistUniformGrid<R>& parent, const std::string basename )
{
#ifndef RELEASE
    PushCallStack("DistUniformGrid::WriteVolumeHelper");
#endif
    const int commRank = mpi::CommRank( parent.comm_ );
    const int px = parent.px_;
    const int py = parent.py_;
    const int pz = parent.pz_;
    const int nx = parent.nx_;
    const int ny = parent.ny_;
    const int nz = parent.nz_;
    const int numScalars = parent.numScalars_;

    // Compute our local box
    const int xMainSize = nx / px;
    const int yMainSize = ny / py;
    const int zMainSize = nz / pz;
    const int xLeftoverSize = xMainSize + (nx % px);
    const int yLeftoverSize = yMainSize + (ny % py);
    const int zLeftoverSize = zMainSize + (nz % pz);

    // For display purposes, set the width of the box to one in the dimension
    // with the largest number of grid points, and then scale the other 
    // dimensions proportionally.
    int maxPoints = std::max(nx,ny);
    maxPoints = std::max(nz,maxPoints);
    const R h = 1.0/(maxPoints+1.0);

    // Form the local box
    std::vector<R> localBox;
    parent.RedistributeForVtk( localBox );

    // Have the root process create the parallel descriptions and the 
    // appropriate subdirectories (if possible)
    if( commRank == 0 )
    {
        std::vector<std::ofstream*> files(numScalars);
        for( int k=0; k<numScalars; ++k )
        {
            std::ostringstream os;
            os << basename << "_" << k << ".pvti";
            files[k] = new std::ofstream;
            files[k]->open( os.str().c_str() );
#ifdef HAVE_MKDIR
            os.clear(); os.str("");
            os << basename << "_" << k;
            EnsureDirExists( os.str().c_str() );
#endif
        }
        std::ostringstream os;
        os << "<?xml version=\"1.0\"?>\n"
           << "<VTKFile type=\"PImageData\" version=\"0.1\">\n"
           << " <PImageData WholeExtent=\""
           << "0 " << nx << " "
           << "0 " << ny << " "
           << "0 " << nz << "\" "
           << "Origin=\"0 0 0\" "
           << "Spacing=\"" << h << " " << h << " " << h << "\" "
           << "GhostLevel=\"0\">\n"
           << "  <PCellData Scalars=\"cell_scalars\">\n"
           << "    <PDataArray type=\"Float64\" Name=\"cell_scalars\"/>\n"
           << "  </PCellData>\n";
        for( int zProc=0; zProc<pz; ++zProc )
        {
            int zBoxSize = ( zProc==pz-1 ? zLeftoverSize : zMainSize );
            int zStart = zProc*zMainSize;
            for( int yProc=0; yProc<py; ++yProc )
            {
                int yBoxSize = ( yProc==py-1 ? yLeftoverSize : yMainSize );
                int yStart = yProc*yMainSize;
                for( int xProc=0; xProc<px; ++xProc )
                {
                    int xBoxSize = ( xProc==px-1 ? xLeftoverSize : xMainSize );
                    int xStart = xProc*xMainSize;

                    int proc = xProc + yProc*px + zProc*px*py;

                    os << "  <Piece Extent=\""
                       << xStart << " " << xStart+xBoxSize << " "
                       << yStart << " " << yStart+yBoxSize << " "
                       << zStart << " " << zStart+zBoxSize << "\" "
                       << "Source=\"" << basename << "_";
                    for( int k=0; k<numScalars; ++k )
                        *files[k] << os.str();
                    os.clear(); os.str("");
                    for( int k=0; k<numScalars; ++k )
                    {
#ifdef HAVE_MKDIR
                        *files[k] << k << "/" << proc << ".vti\"/>\n";
#else
                        *files[k] << k << "_" << proc << ".vti\"/>\n";
#endif
                    }
                }
            }
        }
        os << " </PImageData>\n"
           << "</VTKFile>" << std::endl;
        for( int k=0; k<numScalars; ++k )
        {
            *files[k] << os.str();
            files[k]->close();
            delete files[k];
        }
    }
    mpi::Barrier( parent.comm_ );

    // Have each process create their individual data file
    const int xShift = parent.xShift_;
    const int yShift = parent.yShift_;
    const int zShift = parent.zShift_;
    const int xBoxStart = xMainSize*xShift;
    const int yBoxStart = yMainSize*yShift;
    const int zBoxStart = zMainSize*zShift;
    const int xBoxSize = ( xShift==px-1 ? xLeftoverSize : xMainSize );
    const int yBoxSize = ( yShift==py-1 ? yLeftoverSize : yMainSize );
    const int zBoxSize = ( zShift==pz-1 ? zLeftoverSize : zMainSize );
    std::vector<std::ofstream*> files(numScalars);
    for( int k=0; k<numScalars; ++k )
    {
        std::ostringstream os;    
#ifdef HAVE_MKDIR
        os << basename << "_" << k << "/" << commRank << ".vti";
#else
        os << basename << "_" << k << "_" << commRank << ".vti";
#endif
        files[k] = new std::ofstream;
        files[k]->open( os.str().c_str() );
    }
    std::ostringstream os;
    os << "<?xml version=\"1.0\"?>\n"
       << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
       << " <ImageData WholeExtent=\""
       << "0 " << nx << " 0 " << ny << " 0 " << nz << "\" "
       << "Origin=\"0 0 0\" "
       << "Spacing=\"" << h << " " << h << " " << h << "\">\n"
       << "  <Piece Extent=\"" 
       << xBoxStart << " " << xBoxStart+xBoxSize << " "
       << yBoxStart << " " << yBoxStart+yBoxSize << " "
       << zBoxStart << " " << zBoxStart+zBoxSize << "\">\n"
       << "    <CellData Scalars=\"cell_scalars\">\n"
       << "     <DataArray type=\"Float64\" Name=\"cell_scalars\" "
       << "format=\"ascii\">\n";
    for( int k=0; k<numScalars; ++k )
        *files[k] << os.str();
    os.clear(); os.str("");
    for( int zLocal=0; zLocal<zBoxSize; ++zLocal )
    {
        for( int yLocal=0; yLocal<yBoxSize; ++yLocal )
        {
            for( int xLocal=0; xLocal<xBoxSize; ++xLocal )
            {
                const int offset = 
                    xLocal + yLocal*xBoxSize + zLocal*xBoxSize*yBoxSize;
                for( int k=0; k<numScalars; ++k )
                {
                    double alpha = localBox[offset*numScalars+k];
                    if( Abs(alpha) < 1.0e-300 )
                        alpha = 0;
                    *files[k] << alpha << " ";
                }
            }
            for( int k=0; k<numScalars; ++k )
                *files[k] << "\n";
        }
    }
    os << "    </DataArray>\n"
       << "   </CellData>\n"
       << "  </Piece>\n"
       << " </ImageData>\n"
       << "</VTKFile>" << std::endl;
    for( int k=0; k<numScalars; ++k )
    {
        *files[k] << os.str();
        files[k]->close();
        delete files[k];
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
template<typename R>
inline void 
DistUniformGrid<F>::WriteVolumeHelper<Complex<R> >::Func
( const DistUniformGrid<Complex<R> >& parent, const std::string basename )
{
#ifndef RELEASE
    PushCallStack("DistUniformGrid::WriteVolumeHelper");
#endif
    const int commRank = mpi::CommRank( parent.comm_ );
    const int px = parent.px_;
    const int py = parent.py_;
    const int pz = parent.pz_;
    const int nx = parent.nx_;
    const int ny = parent.ny_;
    const int nz = parent.nz_;
    const int numScalars = parent.numScalars_;
    
    // Compute our local box
    const int xMainSize = nx / px;
    const int yMainSize = ny / py;
    const int zMainSize = nz / pz;
    const int xLeftoverSize = xMainSize + (nx % px);
    const int yLeftoverSize = yMainSize + (ny % py);
    const int zLeftoverSize = zMainSize + (nz % pz);

    // For display purposes, set the width of the box to one in the dimension
    // with the largest number of grid points, and then scale the other 
    // dimensions proportionally.
    int maxPoints = std::max(nx,ny);
    maxPoints = std::max(nz,maxPoints);
    const R h = 1.0/(maxPoints+1.0);

    // Form the local box
    std::vector<Complex<R> > localBox;
    parent.RedistributeForVtk( localBox );

    // Have the root process create the parallel description
    if( commRank == 0 )
    {
        std::vector<std::ofstream*> realFiles(numScalars), 
                                    imagFiles(numScalars);
        for( int k=0; k<numScalars; ++k )
        {
            std::ostringstream os;
            os << basename << "_" << k << "_real.pvti";
            realFiles[k] = new std::ofstream;
            realFiles[k]->open( os.str().c_str() );
#ifdef HAVE_MKDIR
            os.clear(); os.str("");
            os << basename << "_" << k << "_real";
            EnsureDirExists( os.str().c_str() );
#endif
        }
        for( int k=0; k<numScalars; ++k )
        {
            std::ostringstream os;
            os << basename << "_" << k << "_imag.pvti";
            imagFiles[k] = new std::ofstream;
            imagFiles[k]->open( os.str().c_str() );
#ifdef HAVE_MKDIR
            os.clear(); os.str("");
            os << basename << "_" << k << "_imag";
            EnsureDirExists( os.str().c_str() );
#endif
        }
        std::ostringstream os;
        os << "<?xml version=\"1.0\"?>\n"
           << "<VTKFile type=\"PImageData\" version=\"0.1\">\n"
           << " <PImageData WholeExtent=\""
           << "0 " << nx << " "
           << "0 " << ny << " "
           << "0 " << nz << "\" "
           << "Origin=\"0 0 0\" "
           << "Spacing=\"" << h << " " << h << " " << h << "\" "
           << "GhostLevel=\"0\">\n"
           << "  <PCellData Scalars=\"cell_scalars\">\n"
           << "    <PDataArray type=\"Float64\" Name=\"cell_scalars\"/>\n"
           << "  </PCellData>\n";
        for( int zProc=0; zProc<pz; ++zProc )
        {
            int zBoxSize = ( zProc==pz-1 ? zLeftoverSize : zMainSize );
            int zStart = zProc*zMainSize;
            for( int yProc=0; yProc<py; ++yProc )
            {
                int yBoxSize = ( yProc==py-1 ? yLeftoverSize : yMainSize );
                int yStart = yProc*yMainSize;
                for( int xProc=0; xProc<px; ++xProc )
                {
                    int xBoxSize = ( xProc==px-1 ? xLeftoverSize : xMainSize );
                    int xStart = xProc*xMainSize;

                    int proc = xProc + yProc*px + zProc*px*py;

                    os << "  <Piece Extent=\""
                       << xStart << " " << xStart+xBoxSize << " "
                       << yStart << " " << yStart+yBoxSize << " "
                       << zStart << " " << zStart+zBoxSize << "\" "
                       << "Source=\"" << basename << "_";
                    for( int k=0; k<numScalars; ++k )
                    {
                        *realFiles[k] << os.str();
                        *imagFiles[k] << os.str();
                    }
                    os.clear(); os.str("");
                    for( int k=0; k<numScalars; ++k )
                    {
#ifdef HAVE_MKDIR
                        *realFiles[k] << k << "_real/" << proc << ".vti\"/>\n";
                        *imagFiles[k] << k << "_imag/" << proc << ".vti\"/>\n";
#else
                        *realFiles[k] << k << "_real_" << proc << ".vti\"/>\n";
                        *imagFiles[k] << k << "_imag_" << proc << ".vti\"/>\n";
#endif
                    }
                }
            }
        }
        os << " </PImageData>\n"
           << "</VTKFile>" << std::endl;
        for( int k=0; k<numScalars; ++k )
        {
            *realFiles[k] << os.str();
            *imagFiles[k] << os.str();
            realFiles[k]->close();
            imagFiles[k]->close();
            delete realFiles[k];
            delete imagFiles[k];
        }
    }
    mpi::Barrier( parent.comm_ );

    // Have each process create their individual data file
    const int xShift = parent.xShift_;
    const int yShift = parent.yShift_;
    const int zShift = parent.zShift_;
    const int xBoxStart = xMainSize*xShift;
    const int yBoxStart = yMainSize*yShift;
    const int zBoxStart = zMainSize*zShift;
    const int xBoxSize = ( xShift==px-1 ? xLeftoverSize : xMainSize );
    const int yBoxSize = ( yShift==py-1 ? yLeftoverSize : yMainSize );
    const int zBoxSize = ( zShift==pz-1 ? zLeftoverSize : zMainSize );
    std::vector<std::ofstream*> realFiles(numScalars), imagFiles(numScalars);
    for( int k=0; k<numScalars; ++k )
    {
        std::ostringstream os;    
#ifdef HAVE_MKDIR
        os << basename << "_" << k << "_real/" << commRank << ".vti";
#else
        os << basename << "_" << k << "_real_" << commRank << ".vti";
#endif
        realFiles[k] = new std::ofstream;
        realFiles[k]->open( os.str().c_str() );
    }
    for( int k=0; k<numScalars; ++k )
    {
        std::ostringstream os;
#ifdef HAVE_MKDIR
        os << basename << "_" << k << "_imag/" << commRank << ".vti";
#else
        os << basename << "_" << k << "_imag_" << commRank << ".vti";
#endif
        imagFiles[k] = new std::ofstream;
        imagFiles[k]->open( os.str().c_str() );
    }
    std::ostringstream os;
    os << "<?xml version=\"1.0\"?>\n"
       << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
       << " <ImageData WholeExtent=\""
       << "0 " << nx << " 0 " << ny << " 0 " << nz << "\" "
       << "Origin=\"0 0 0\" "
       << "Spacing=\"" << h << " " << h << " " << h << "\">\n"
       << "  <Piece Extent=\"" 
       << xBoxStart << " " << xBoxStart+xBoxSize << " "
       << yBoxStart << " " << yBoxStart+yBoxSize << " "
       << zBoxStart << " " << zBoxStart+zBoxSize << "\">\n"
       << "    <CellData Scalars=\"cell_scalars\">\n"
       << "     <DataArray type=\"Float64\" Name=\"cell_scalars\" "
       << "format=\"ascii\">\n";
    for( int k=0; k<numScalars; ++k )
    {
        *realFiles[k] << os.str();
        *imagFiles[k] << os.str();
    }
    os.clear(); os.str("");
    for( int zLocal=0; zLocal<zBoxSize; ++zLocal )
    {
        for( int yLocal=0; yLocal<yBoxSize; ++yLocal )
        {
            for( int xLocal=0; xLocal<xBoxSize; ++xLocal )
            {
                const int offset = 
                    xLocal + yLocal*xBoxSize + zLocal*xBoxSize*yBoxSize;
                for( int k=0; k<numScalars; ++k )
                {
                    const Complex<double> alpha = localBox[offset*numScalars+k];
                    double realAlpha = alpha.real;
                    double imagAlpha = alpha.imag;
                    if( Abs(realAlpha) < 1.0e-300 )
                        realAlpha = 0;
                    if( Abs(imagAlpha) < 1.0e-300 )
                        imagAlpha = 0;
                    *realFiles[k] << realAlpha << " ";
                    *imagFiles[k] << imagAlpha << " ";
                }
            }
            for( int k=0; k<numScalars; ++k )
            {
                *realFiles[k] << "\n";
                *imagFiles[k] << "\n";
            }
        }
    }
    os << "    </DataArray>\n"
       << "   </CellData>\n"
       << "  </Piece>\n"
       << " </ImageData>\n"
       << "</VTKFile>" << std::endl;
    for( int k=0; k<numScalars; ++k )
    {
        *realFiles[k] << os.str();
        *imagFiles[k] << os.str();
        realFiles[k]->close();
        imagFiles[k]->close();
        delete realFiles[k];
        delete imagFiles[k];
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_DIST_UNIFORM_GRID_HPP
