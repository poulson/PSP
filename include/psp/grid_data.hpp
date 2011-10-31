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

// The control structure for passing in the distributed data on a 3d grid.
// 
//                 _______________ (wx,wy,0)
//                /              /|
//            x  /              / |
//              /              /  |
// sweep dir.  /______________/   |
//     ||      |              |   |
//     ||      |              |   / (wx,wy,wz)
//     ||    z |              |  /  
//     ||      |              | /  
//     \/      |______________|/
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
    GridData
    ( int nx, int ny,int nz, int px, int py, int pz, 
      elemental::mpi::Comm comm );

    int XShift() const;
    int YShift() const;
    int ZShift() const;
    int XStride() const;
    int YStride() const;
    int ZStride() const;
    int XLocalSize() const;
    int YLocalSize() const;
    int ZLocalSize() const;

    T* LocalBuffer
    ( int xLocal=0, int yLocal=0, int zLocal=0 );
    const T* LockedLocalBuffer
    ( int xLocal=0, int yLocal=0, int zLocal=0 ) const;

private:
    elemental::mpi::Comm comm_;
    int nx_, ny_, nz_;
    int px_, py_, pz_;
    int xShift_, yShift_, zShift_;
    int xLocalSize_, yLocalSize_, zLocalSize_;
    std::vector<T> localData_;
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename T>
inline GridData<T>::GridData
( int nx, int ny, int nz, int px, int py, int pz, elemental::mpi::Comm comm )
: nx_(nx), ny_(ny), nz_(nz), px_(px), py_(py), pz_(pz)
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
inline T* GridData<T>::LocalBuffer
( int xLocal, int yLocal, int zLocal )
{ 
    const int index = 
        xLocal + yLocal*xLocalSize_ + zLocal*xLocalSize_*yLocalSize_;
    return &localData_[index];
}

template<typename T>
inline const T* GridData<T>::LockedLocalBuffer
( int xLocal, int yLocal, int zLocal ) const
{ 
    const int index = 
        xLocal + yLocal*xLocalSize_ + zLocal*xLocalSize_*yLocalSize_;
    return &localData_[index];
}

} // namespace psp

#endif // PSP_GRID_DATA_HPP
