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

    elemental::mpi::Comm Comm() const;
    int OwningProcess( int x, int y, int z ) const;

    int NumScalars() const;
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

private:
    int numScalars_;
    int nx_, ny_, nz_;
    GridDataOrder order_;

    int px_, py_, pz_;
    elemental::mpi::Comm comm_;

    int xShift_, yShift_, zShift_;
    int xLocalSize_, yLocalSize_, zLocalSize_;
    std::vector<T> localData_;
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
    localData_.resize( xLocalSize_*yLocalSize_*zLocalSize_ );
}

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
inline int GridData<T>::OwningProcess( int x, int y, int z ) const
{
    const int xProc = x % px_;
    const int yProc = y % py_;
    const int zProc = z % pz_;
    return xProc + yProc*px_ + zProc*px_*py_;
}

template<typename T>
inline int GridData<T>::LocalIndex( int x, int y, int z ) const
{ 
    const int xLocal = (x-xShift_) / px_;
    const int yLocal = (y-yShift_) / py_;
    const int zLocal = (z-zShift_) / pz_;

    int index;
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

} // namespace psp

#endif // PSP_GRID_DATA_HPP
