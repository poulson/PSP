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
#ifndef PSP_DIST_VECTOR_HPP
#define PSP_DIST_VECTOR_HPP 1

#include "psp/vector.hpp"
#include <algorithm>

namespace psp {

template<typename Scalar>
class DistVector
{
    int _height;
    int _localOffset;
    Vector<Scalar> _localVector;
    MPI_Comm _comm;

    int _rank;
    std::vector<int> _offsets;

    void CheckLocalHeights();
    void FillOffsets();

public:
    DistVector( MPI_Comm comm );
    DistVector( int height, int localOffset, int localHeight, MPI_Comm comm );
    DistVector
    ( int height, int localOffset, int localHeight, Scalar* localBuffer,
      MPI_Comm comm );
    DistVector
    ( int height, int localOffset, int localHeight, const Scalar* localBuffer,
      MPI_Comm comm );
    DistVector
    ( int height, int localOffset, const Vector<Scalar>& localVector, 
      MPI_Comm comm );
    ~DistVector();

    int Height() const;
    void Resize( int height, int localOffset, int localHeight );

    void Set( int i, Scalar value );
    Scalar Get( int i ) const;

    void SetLocal( int iLocal, Scalar value );
    Scalar GetLocal( int iLocal ) const;

    void Print( const std::string& tag ) const;

    Scalar* LocalBuffer( int iLocal=0 );
    const Scalar* LocalBuffer( int iLocal=0 ) const;

    void View( DistVector<Scalar>& x );
    void View( DistVector<Scalar>& x, int i, int height );

    void LockedView( const DistVector<Scalar>& x );
    void LockedView( const DistVector<Scalar>& x, int i, int height );
};

} // namespace psp

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

/*\
|*| Private routines
\*/

template<typename Scalar>
inline void 
psp::DistVector<Scalar>::CheckLocalHeights()
{
    int localHeight = _localVector.Height();
    int sumOfLocalHeights;
    mpi::AllSum( &localHeight, &sumOfLocalHeights, 1, _comm );
    if( sumOfLocalHeights != _height )
        throw std::logic_error("Incompatible local heights");
}

template<typename Scalar>
inline void 
psp::DistVector<Scalar>::FillOffsets()
{
    int p = mpi::CommSize( _comm );
    _offsets.resize( p+1 );
    mpi::AllGather( &p, 1, &_offsets[0], 1, _comm );
    _offsets[p] = _height;
}

/*\
|*| Public routines
\*/

template<typename Scalar>
inline
psp::DistVector<Scalar>::DistVector( MPI_Comm comm )
: _height(0), _localOffset(0), _localVector(), _comm(comm), _offsets()
{ 
    _rank = mpi::CommRank( comm );
} 

template<typename Scalar>
inline
psp::DistVector<Scalar>::DistVector
( int height, int localOffset, int localHeight, Scalar* localBuffer,
  MPI_Comm comm )
: _height(height), _localOffset(localOffset), 
  _localVector(localHeight,localBuffer), _comm(comm)
{
#ifndef RELEASE
    PushCallStack("DistVector::DistVector");
    CheckLocalHeights();
#endif
    _rank = mpi::CommRank( comm );
    FillOffsets();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline
psp::DistVector<Scalar>::DistVector
( int height, int localOffset, int localHeight, const Scalar* localBuffer,
  MPI_Comm comm )
: _height(height), _localOffset(localOffset), 
  _localVector(localHeight,localBuffer), _comm(comm)
{
#ifndef RELEASE
    PushCallStack("DistVector::DistVector");
    CheckLocalHeights();
#endif
    _rank = mpi::CommRank( comm );
    FillOffsets();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline
psp::DistVector<Scalar>::DistVector
( int height, int localOffset, int localHeight, MPI_Comm comm )
: _height(height), _localOffset(localOffset), _localVector(localHeight), 
  _comm(comm)
{ 
#ifndef RELEASE
    PushCallStack("DistVector::DistVector");
    CheckLocalHeights();
#endif
    _rank = mpi::CommRank( comm );
    FillOffsets();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline
psp::DistVector<Scalar>::DistVector
( int height, int localOffset, const Vector<Scalar>& localVector, 
  MPI_Comm comm )
: _height(height), _localOffset(localOffset), _localVector(localVector),
  _comm(comm)
{
#ifndef RELEASE
    PushCallStack("DistVector::DistVector");
    CheckLocalHeights();
#endif
    _rank = mpi::CommRank( comm );
    FillOffsets();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline
psp::DistVector<Scalar>::~DistVector()
{ }


template<typename Scalar>
inline int
psp::DistVector<Scalar>::Height() const
{ return _height; }

template<typename Scalar>
inline void
psp::DistVector<Scalar>::Resize
( int height, int localOffset, int localHeight )
{
#ifndef RELEASE
    PushCallStack("DistVector::Resize");
#endif
    _height = height;
    _localOffset = localOffset;
    _localVector.Resize( localHeight );
#ifndef RELEASE
    CheckLocalHeights();
#endif
    FillOffsets();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline void
psp::DistVector<Scalar>::Set
( int i, Scalar value )
{
#ifndef RELEASE
    PushCallStack("DistVector::Set");
    if( i < 0 || i >= _height )
        throw std::logic_error("Index out of bounds");
#endif
    if( i >= _localOffset && i < _localOffset+_localVector.Height() )
        _localVector.Set( i-_localOffset, value );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar>
inline Scalar
psp::DistVector<Scalar>::Get( int i ) const
{
#ifndef RELEASE
    PushCallStack("DistVector::Get");
    if( i < 0 || i >= _height )
        throw std::logic_error("Index out of bounds");
#endif
    std::vector<int>::iterator upperBound = 
        std::upper_bound( _offsets.begin(), _offsets.end(), i );
    const int owner = int(upperBound-_offsets.begin()) - 1;

    Scalar value;
    if( _rank == owner )
        value = _localVector.Get( i-_localOffset );
    mpi::Broadcast( &value, 1, owner, _comm );
#ifndef RELEASE
    PopCallStack();
#endif
}

// HERE
/*
    void SetLocal( int iLocal, Scalar value );
    Scalar GetLocal( int iLocal ) const;

    void Print( const std::string& tag ) const;

    Scalar* LocalBuffer( int iLocal=0 );
    const Scalar* LocalBuffer( int iLocal=0 ) const;

    void View( DistVector<Scalar>& x );
    void View( DistVector<Scalar>& x, int i, int height );

    void LockedView( const DistVector<Scalar>& x );
    void LockedView( const DistVector<Scalar>& x, int i, int height );
 */

#endif // PSP_DIST_VECTOR_HPP
