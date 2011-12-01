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
#ifndef PSP_GRID_DATA_HPP
#define PSP_GRID_DATA_HPP 1

namespace psp {

enum GridDataOrder {
    XYZ,
    XZY,
    YXZ,
    YZX,
    ZXY,
    ZYX
};

// The control structure for passing in the distributed data on a 3d grid.
// 
//                 _______________ (wx,wy,0)
//                /              /|
//            x  /              / |
// sweep dir    /              /  |
//     /\      /______________/   |
//     ||      |              |   |
//     ||      |              |   / (wx,wy,wz)
//     ||    z |              |  /  
//     ||      |              | /  
//             |______________|/
//          (0,0,wz)    y    (0,wy,wz)
//
// The communicator is decomposed into a px x py x pz grid, and the data is 
// cyclically distributed over each of the three dimensions, x first, y second,
// and z third.
//
template<typename T>
class GridData
{
public:

    // Generate an nx x ny x nz grid, where each node contains 'numScalars' 
    // entries of type 'T' and the grid is distributed over a px x py x pz grid 
    // over the specified communicator.
    GridData
    ( int numScalars,
      int nx, int ny,int nz, GridDataOrder order,
      int px, int py, int pz, elemental::mpi::Comm comm );

    int XSize() const;
    int YSize() const;
    int ZSize() const;
    elemental::mpi::Comm Comm() const;
    int OwningProcess( int naturalIndex ) const;
    int OwningProcess( int x, int y, int z ) const;

    int NumScalars() const;
    int LocalIndex( int naturalIndex ) const;
    int LocalIndex( int x, int y, int z ) const;
    T* LocalBuffer();
    const T* LockedLocalBuffer() const;

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

    void WriteVtkFiles( const std::string baseName ) const;

    template<typename R>
    struct WriteVtkFilesHelper
    {
        static void Func
        ( const GridData<R>& parent, const std::string baseName );
    };
    template<typename R>
    struct WriteVtkFilesHelper<std::complex<R> >
    {
        static void Func
        ( const GridData<std::complex<R> >& parent, 
          const std::string baseName );
    };
    template<typename R> friend struct WriteVtkFilesHelper;

private:
    int numScalars_;
    int nx_, ny_, nz_;
    GridDataOrder order_;

    int px_, py_, pz_;
    elemental::mpi::Comm comm_;

    int xShift_, yShift_, zShift_;
    int xLocalSize_, yLocalSize_, zLocalSize_;
    std::vector<T> localData_;

    void RedistributeForVtk( std::vector<T>& localBox ) const;
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename T>
inline GridData<T>::GridData
( int numScalars,
  int nx, int ny, int nz, GridDataOrder order,
  int px, int py, int pz, elemental::mpi::Comm comm )
: numScalars_(numScalars), 
  nx_(nx), ny_(ny), nz_(nz), order_(order),
  px_(px), py_(py), pz_(pz), comm_(comm)
{
    const int commRank = elemental::mpi::CommRank( comm );
    const int commSize = elemental::mpi::CommSize( comm );
    if( commSize != px*py*pz )
        throw std::logic_error("px*py*pz != commSize");
    if( px < 0 || py < 0 || pz < 0 )
        throw std::logic_error("process dimensions must be non-negative");

    xShift_ = commRank % px;
    yShift_ = (commRank/px) % py;
    zShift_ = commRank/(px*py);
    xLocalSize_ = elemental::LocalLength( nx, xShift_, px );
    yLocalSize_ = elemental::LocalLength( ny, yShift_, py );
    zLocalSize_ = elemental::LocalLength( nz, zShift_, pz );
    localData_.resize( numScalars*xLocalSize_*yLocalSize_*zLocalSize_ );
}

template<typename T>
inline int GridData<T>::XSize() const
{ return nx_; }

template<typename T>
inline int GridData<T>::YSize() const
{ return ny_; }

template<typename T>
inline int GridData<T>::ZSize() const
{ return nz_; }

template<typename T>
inline elemental::mpi::Comm GridData<T>::Comm() const
{ return comm_; }

template<typename T>
inline int GridData<T>::XShift() const
{ return xShift_; }

template<typename T>
inline int GridData<T>::YShift() const
{ return yShift_; }

template<typename T>
inline int GridData<T>::ZShift() const
{ return zShift_; }

template<typename T>
inline int GridData<T>::XStride() const
{ return px_; }

template<typename T>
inline int GridData<T>::YStride() const
{ return py_; }

template<typename T>
inline int GridData<T>::ZStride() const
{ return pz_; }

template<typename T>
inline int GridData<T>::XLocalSize() const
{ return xLocalSize_; }

template<typename T>
inline int GridData<T>::YLocalSize() const
{ return yLocalSize_; }

template<typename T>
inline int GridData<T>::ZLocalSize() const
{ return zLocalSize_; }

template<typename T>
inline T* GridData<T>::LocalBuffer()
{ return &localData_[0]; }

template<typename T>
inline const T* GridData<T>::LockedLocalBuffer() const
{ return &localData_[0]; }

template<typename T>
inline int GridData<T>::NumScalars() const
{ return numScalars_; }

template<typename T>
inline int GridData<T>::OwningProcess( int naturalIndex ) const
{
    const int x = naturalIndex % nx_;
    const int y = (naturalIndex/nx_) % ny_;
    const int z = naturalIndex/(nx_*ny_);
    return OwningProcess( x, y, z );
}

template<typename T>
inline int GridData<T>::OwningProcess( int x, int y, int z ) const
{
    const int xProc = x % px_;
    const int yProc = y % py_;
    const int zProc = z % pz_;
    return xProc + yProc*px_ + zProc*px_*py_;
}

template<typename T>
inline int GridData<T>::LocalIndex( int naturalIndex ) const
{
    const int x = naturalIndex % nx_;
    const int y = (naturalIndex/nx_) % ny_;
    const int z = naturalIndex/(nx_*ny_);
    return LocalIndex( x, y, z );
}

template<typename T>
inline int GridData<T>::LocalIndex( int x, int y, int z ) const
{ 
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

template<typename T>
inline void GridData<T>::RedistributeForVtk( std::vector<T>& localBox ) const
{
    const int commSize = elemental::mpi::CommSize( comm_ );

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
        const int zLength = 
            elemental::LocalLength( zBoxSize, zProc, zAlign, pz_ );
        for( int yProc=0; yProc<py_; ++yProc )
        {
            const int yLength = 
                elemental::LocalLength( yBoxSize, yProc, yAlign, py_ );
            for( int xProc=0; xProc<px_; ++xProc )
            {
                const int xLength = 
                    elemental::LocalLength( xBoxSize, xProc, xAlign, px_ );
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
    std::vector<T> sendBuffer( totalSendSize );
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
    std::vector<T> recvBuffer( totalRecvSize );
    elemental::mpi::AllToAll
    ( &sendBuffer[0], &sendCounts[0], &sendDispls[0],
      &recvBuffer[0], &recvCounts[0], &recvDispls[0], comm_ );
    sendBuffer.clear();

    // Unpack the recv buffer
    localBox.resize( totalRecvSize );
    for( int zProc=0; zProc<pz_; ++zProc )
    {
        const int zOffset = elemental::Shift( zProc, zAlign, pz_ );
        const int zLength = 
            elemental::LocalLength( zBoxSize, zOffset, pz_ );
        for( int yProc=0; yProc<py_; ++yProc )
        {
            const int yOffset = elemental::Shift( yProc, yAlign, py_ );
            const int yLength = 
                elemental::LocalLength( yBoxSize, yOffset, py_ );
            for( int xProc=0; xProc<px_; ++xProc )
            {
                const int xOffset = elemental::Shift( xProc, xAlign, px_ );
                const int xLength = 
                    elemental::LocalLength( xBoxSize, xOffset, px_ );
                const int proc = xProc + yProc*px_ + zProc*px_*py_;

                // Unpack all of the data from this process
                const int localOffset = 
                    (xOffset + yOffset*xBoxSize + zOffset*xBoxSize*yBoxSize)*
                    numScalars_;
                T* offsetLocal = &localBox[localOffset];
                const T* procRecv = &recvBuffer[recvDispls[proc]];
                for( int zLocal=0; zLocal<zLength; ++zLocal )
                {
                    for( int yLocal=0; yLocal<yLength; ++yLocal )
                    {
                        const int localRowIndex = 
                            (yLocal*py_ + zLocal*pz_*yBoxSize)*xBoxSize*
                            numScalars_;
                        const int procRowIndex = 
                            (yLocal + zLocal*yLength)*xLength*numScalars_;
                        T* localRow = &offsetLocal[localRowIndex];
                        const T* procRow = &procRecv[procRowIndex];
                        for( int xLocal=0; xLocal<xLength; ++xLocal )
                        {
                            std::memcpy
                            ( &localRow[xLocal*px_*numScalars_],
                              &procRow[xLocal*numScalars_],
                              numScalars_*sizeof(T) );
                        }
                    }
                }
            }
        }
    }
}

template<typename T>
inline void GridData<T>::WriteVtkFiles( const std::string baseName ) const
{ return WriteVtkFilesHelper<T>::Func( *this, baseName ); }

template<typename T>
template<typename R>
inline void GridData<T>::WriteVtkFilesHelper<R>::Func
( const GridData<R>& parent, const std::string baseName )
{
    const int commRank = elemental::mpi::CommRank( parent.comm_ );
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

    // Form the local box
    std::vector<R> localBox;
    parent.RedistributeForVtk( localBox );

    // Have the root process create the parallel description
    if( commRank == 0 )
    {
        std::vector<std::ofstream*> files(numScalars);
        for( int k=0; k<numScalars; ++k )
        {
            std::ostringstream os;
            os << baseName << "_" << k << ".pvti";
            files[k] = new std::ofstream;
            files[k]->open( os.str().c_str() );
        }
        std::ostringstream os;
        os << "<?xml version=\"1.0\"?>\n"
           << "<VTKFile type=\"PImageData\" version=\"0.1\">\n"
           << " <PImageData WholeExtent=\""
           << "0 " << nx << " "
           << "0 " << ny << " "
           << "0 " << nz << "\" "
           << "Origin=\"0 0 0\" "
           << "Spacing=\"" << 1.0/nx << " " << 1.0/ny << " " << 1.0/nz << "\" "
           << "GhostLevel=\"0\">\n"
           << "  <PCellData Scalars=\"cell_scalars\">\n"
           << "    <PDataArray type=\"Float32\" Name=\"cell_scalars\"/>\n"
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
                       << "Source=\"" << baseName << "_";
                    for( int k=0; k<numScalars; ++k )
                        *files[k] << os.str();
                    os.clear(); os.str("");
                    for( int k=0; k<numScalars; ++k )
                        *files[k] << k << "_" << proc << ".vti\"/>\n";
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
        os << baseName << "_" << k << "_" << commRank << ".vti";
        files[k] = new std::ofstream;
        files[k]->open( os.str().c_str() );
    }
    std::ostringstream os;
    os << "<?xml version=\"1.0\"?>\n"
       << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
       << " <ImageData WholeExtent=\""
       << "0 " << nx << " 0 " << ny << " 0 " << nz << "\" "
       << "Origin=\"0 0 0\" "
       << "Spacing=\"" << 1.0/nx << " " << 1.0/ny << " " << 1.0/nz << "\">\n"
       << "  <Piece Extent=\"" 
       << xBoxStart << " " << xBoxStart+xBoxSize << " "
       << yBoxStart << " " << yBoxStart+yBoxSize << " "
       << zBoxStart << " " << zBoxStart+zBoxSize << "\">\n"
       << "    <CellData Scalars=\"cell_scalars\">\n"
       << "     <DataArray type=\"Float32\" Name=\"cell_scalars\" "
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
                    const float alpha = localBox[offset*numScalars+k];
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
}

template<typename T>
template<typename R>
inline void 
GridData<T>::WriteVtkFilesHelper<std::complex<R> >::Func
( const GridData<std::complex<R> >& parent, const std::string baseName )
{
    const int commRank = elemental::mpi::CommRank( parent.comm_ );
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

    // Form the local box
    std::vector<std::complex<R> > localBox;
    parent.RedistributeForVtk( localBox );

    // Have the root process create the parallel description
    if( commRank == 0 )
    {
        std::vector<std::ofstream*> realFiles(numScalars), 
                                    imagFiles(numScalars);
        for( int k=0; k<numScalars; ++k )
        {
            std::ostringstream os;
            os << baseName << "_" << k << "_real.pvti";
            realFiles[k] = new std::ofstream;
            realFiles[k]->open( os.str().c_str() );
        }
        for( int k=0; k<numScalars; ++k )
        {
            std::ostringstream os;
            os << baseName << "_" << k << "_imag.pvti";
            imagFiles[k] = new std::ofstream;
            imagFiles[k]->open( os.str().c_str() );
        }
        std::ostringstream os;
        os << "<?xml version=\"1.0\"?>\n"
           << "<VTKFile type=\"PImageData\" version=\"0.1\">\n"
           << " <PImageData WholeExtent=\""
           << "0 " << nx << " "
           << "0 " << ny << " "
           << "0 " << nz << "\" "
           << "Origin=\"0 0 0\" "
           << "Spacing=\""
           << 1.0/nx << " "
           << 1.0/ny << " "
           << 1.0/nz << "\" "
           << "GhostLevel=\"0\">\n"
           << "  <PCellData Scalars=\"cell_scalars\">\n"
           << "    <PDataArray type=\"Float32\" Name=\"cell_scalars\"/>\n"
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
                       << "Source=\"" << baseName << "_";
                    for( int k=0; k<numScalars; ++k )
                    {
                        *realFiles[k] << os.str();
                        *imagFiles[k] << os.str();
                    }
                    os.clear(); os.str("");
                    for( int k=0; k<numScalars; ++k )
                    {
                        *realFiles[k] << k << "_real_" << proc << ".vti\"/>\n";
                        *imagFiles[k] << k << "_imag_" << proc << ".vti\"/>\n";
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
        os << baseName << "_" << k << "_real_" << commRank << ".vti";
        realFiles[k] = new std::ofstream;
        realFiles[k]->open( os.str().c_str() );
    }
    for( int k=0; k<numScalars; ++k )
    {
        std::ostringstream os;
        os << baseName << "_" << k << "_imag_" << commRank << ".vti";
        imagFiles[k] = new std::ofstream;
        imagFiles[k]->open( os.str().c_str() );
    }
    std::ostringstream os;
    os << "<?xml version=\"1.0\"?>\n"
       << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
       << " <ImageData WholeExtent=\""
       << "0 " << nx << " 0 " << ny << " 0 " << nz << "\" "
       << "Origin=\"0 0 0\" "
       << "Spacing=\"" << 1.0/nx << " " << 1.0/ny << " " << 1.0/nz << "\">\n"
       << "  <Piece Extent=\"" 
       << xBoxStart << " " << xBoxStart+xBoxSize << " "
       << yBoxStart << " " << yBoxStart+yBoxSize << " "
       << zBoxStart << " " << zBoxStart+zBoxSize << "\">\n"
       << "    <CellData Scalars=\"cell_scalars\">\n"
       << "     <DataArray type=\"Float32\" Name=\"cell_scalars\" "
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
                    std::complex<R> alpha = localBox[offset*numScalars+k];
                    *realFiles[k] << (float)real(alpha) << " ";
                    *imagFiles[k] << (float)imag(alpha) << " ";
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
}

} // namespace psp

#endif // PSP_GRID_DATA_HPP
