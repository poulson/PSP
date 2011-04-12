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
#ifndef PSP_VECTOR_HPP
#define PSP_VECTOR_HPP 1

#include <cstring>
#include <stdexcept>
#include <vector>

namespace psp {

// A vector implementation that allows O(1) creation of subvectors. 
// The tradeoff versus std::vector is that introducing (locked) views makes 
// operator[] usage impractical, so we instead require Set() and Get().
template<typename Scalar>
class Vector
{
    int _height;
    bool _viewing;
    bool _lockedView;
    std::vector<Scalar> _memory;
    Scalar* _buffer;
    const Scalar* _lockedBuffer;

public:
    Vector();
    Vector( int height );
    Vector( Scalar* buffer, int height );
    Vector( const Scalar* lockedBuffer, int height );
    ~Vector();

    int Height() const;
    void Resize( int height );

    void Set( int i, Scalar value );
    Scalar Get( int i ) const;
    void Print( const std::string& tag ) const;

    Scalar* Buffer( int i=0 );
    const Scalar* LockedBuffer( int i=0 ) const;

    void View( Vector<Scalar>& x );
    void View( Vector<Scalar>& x, int i, int height );

    void LockedView( const Vector<Scalar>& x );
    void LockedView( const Vector<Scalar>& x, int i, int height );
};

} // namespace psp

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar>
inline
psp::Vector<Scalar>::Vector()
: _height(0), _viewing(false), _lockedView(false),
  _memory(), _buffer(0), _lockedBuffer(0)
{ }

template<typename Scalar>
inline
psp::Vector<Scalar>::Vector( int height )
: _height(height), _viewing(false), _lockedView(false),
  _memory(height), _buffer(&_memory[0]), _lockedBuffer(0)
{ }

template<typename Scalar>
inline
psp::Vector<Scalar>::Vector( Scalar* buffer, int height )
: _height(height), _viewing(true), _lockedView(false),
  _memory(), _buffer(buffer), _lockedBuffer(0)
{ }

template<typename Scalar>
inline
psp::Vector<Scalar>::Vector( const Scalar* lockedBuffer, int height )
: _height(height), _viewing(true), _lockedView(true),
  _memory(), _buffer(0), _lockedBuffer(lockedBuffer)
{ }

template<typename Scalar>
inline
psp::Vector<Scalar>::~Vector()
{ }

template<typename Scalar>
inline int
psp::Vector<Scalar>::Height() const
{ 
    return _height;
}

template<typename Scalar>
inline void
psp::Vector<Scalar>::Resize( int height )
{
#ifndef RELEASE
    if( _viewing || _lockedView )
        throw std::logic_error("Cannot resize a Vector that is a view.");
#endif
    _height = height;
    _memory.resize( height );
    _buffer = &_memory[0];
}

template<typename Scalar>
inline void
psp::Vector<Scalar>::Set( int i, Scalar value )
{
#ifndef RELEASE
    if( _lockedView )
        throw std::logic_error("Cannot modify locked views");
    if( i < 0 )
        throw std::logic_error("Negative buffer offsets are nonsensical");
    if( i >= _height )
        throw std::logic_error("Vector::Set is out of bounds");
#endif
    _buffer[i] = value;
}

template<typename Scalar>
inline Scalar
psp::Vector<Scalar>::Get( int i ) const
{
#ifndef RELEASE
    if( i < 0 )
        throw std::logic_error("Negative buffer offsets are nonsensical");
    if( i >= _height )
        throw std::logic_error("Vector::Get is out of bounds");
#endif
    if( _lockedView )
        return _lockedBuffer[i];
    else
        return _buffer[i];
}

template<typename Scalar>
inline void
psp::Vector<Scalar>::Print( const std::string& tag ) const
{
    std::cout << tag << "\n";
    if( _lockedView )
    {
        for( int i=0; i<_height; ++i )
            std::cout << WrapScalar(_lockedBuffer[i]) << "\n";
    }
    else
    {
        for( int i=0; i<_height; ++i )
            std::cout << WrapScalar(_buffer[i]) << "\n";
    }
    std::cout << std::endl;
}

template<typename Scalar>
inline Scalar*
psp::Vector<Scalar>::Buffer( int i )
{
#ifndef RELEASE
    if( _lockedView )
        throw std::logic_error("Cannot get modifiable buffer from locked view");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
    if( i >= _height )
        throw std::logic_error("Out of bounds of buffer");
#endif
    return &_buffer[i];
}

template<typename Scalar>
inline const Scalar*
psp::Vector<Scalar>::LockedBuffer( int i ) const
{
#ifndef RELEASE
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
    if( i >= _height )
        throw std::logic_error("Out of bounds of buffer");
#endif
    if( _lockedView )
        return &_lockedBuffer[i];
    else
        return &_buffer[i];
}

template<typename Scalar>
inline void
psp::Vector<Scalar>::View( Vector<Scalar>& x )
{
    _viewing = true;
    _lockedView = false;
    _buffer = x.Buffer();
    _height = x.Height();
}

template<typename Scalar>
inline void
psp::Vector<Scalar>::View( Vector<Scalar>& x, int i, int height )
{
#ifndef RELEASE
    if( x.Height() < i+height )
        throw std::logic_error("Vector view goes out of bounds");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
#endif
    _viewing = true;
    _lockedView = false;
    _buffer = x.Buffer( i );
    _height = height;
}

template<typename Scalar>
inline void
psp::Vector<Scalar>::LockedView( const Vector<Scalar>& x )
{
    _viewing = true;
    _lockedView = true;
    _lockedBuffer = x.Buffer();
    _height = x.Height();
}

template<typename Scalar>
inline void
psp::Vector<Scalar>::LockedView( const Vector<Scalar>& x, int i, int height )
{
#ifndef RELEASE
    if( x.Height() < i+height )
        throw std::logic_error("Vector view goes out of bounds");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
#endif
    _viewing = true;
    _lockedView = true;
    _lockedBuffer = x.LockedBuffer( i );
    _height = height;
}

#endif // PSP_VECTOR_HPP
