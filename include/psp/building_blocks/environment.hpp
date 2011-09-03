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
#ifndef PSP_ENVIRONMENT_HPP
#define PSP_ENVIRONMENT_HPP 1

#include "mpi.h"
#include <algorithm>
#include <climits>
#include <complex>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <utility>
#include <vector>

#include "psp/config.h"

namespace psp {

typedef unsigned char byte;

template<typename Real>
Real
Abs( Real alpha )
{ return std::abs(alpha); }

template<typename Real>
Real
Abs( std::complex<Real> alpha )
{ return std::abs(alpha); }

template<typename Real>
inline Real
Conj( Real alpha )
{ return alpha; }

template<typename Real>
inline std::complex<Real>
Conj( std::complex<Real> alpha )
{ return std::conj( alpha ); }

// For reading and writing to a buffer
template<typename T>
inline void Write( byte*& head, const T& t )
{
    *((T*)head) = t;
    head += sizeof(T);
}

template<typename T>
inline void Write( byte** head, const T& t )
{
    *((T*)*head) = t;
    *head += sizeof(T);
}

template<typename T>
inline void Write( byte*& head, const T* buffer, int n )
{
    std::memcpy( head, buffer, n*sizeof(T) );
    head += n*sizeof(T);
}

template<typename T>
inline void Write( byte** head, const T* buffer, int n )
{
    std::memcpy( *head, buffer, n*sizeof(T) );
    *head += n*sizeof(T);
}

template<typename T>
inline T Read( const byte*& head )
{
    T retval = *((const T*)head);
    head += sizeof(T);
    return retval;
}

template<typename T>
inline T Read( const byte** head )
{
    T retval = *((const T*)*head);
    *head += sizeof(T);
    return retval;
}

template<typename T>
inline void Read( T* writeHead, const byte*& readHead, int n )
{
    std::memcpy( writeHead, readHead, n*sizeof(T) );
    readHead += n*sizeof(T);
}

template<typename T>
inline void Read( T* writeHead, const byte** readHead, int n )
{
    std::memcpy( writeHead, *readHead, n*sizeof(T) );
    *readHead += n*sizeof(T);
}

// For extracting the underlying real datatype, 
// e.g., typename RealBase<Scalar>::type a = 3.0;
template<typename Real>
struct RealBase
{ typedef Real type; };

template<typename Real>
struct RealBase<std::complex<Real> >
{ typedef Real type; };

// Create a wrappers around real and std::complex<real> types so that they
// can be conveniently printed in a more Matlab-compatible format.
//
// All printing of scalars should now be performed in the fashion:
//     std::cout << WrapScalar(alpha);
// where 'alpha' can be real or complex.

template<typename Real>
class ScalarWrapper
{
    const Real _value;
public:
    ScalarWrapper( const Real alpha ) : _value(alpha) { }

    friend std::ostream& operator<<
    ( std::ostream& out, const ScalarWrapper<Real> alpha )
    {
        out << alpha._value;
        return out;
    }
};

template<typename Real>
class ScalarWrapper<std::complex<Real> >
{
    const std::complex<Real> _value;
public:
    ScalarWrapper( const std::complex<Real> alpha ) : _value(alpha) { }

    friend std::ostream& operator<<
    ( std::ostream& os, const ScalarWrapper<std::complex<Real> > alpha )
    {
        os << std::real(alpha._value) << "+" << std::imag(alpha._value) << "i";
        return os;
    }
};

template<typename Real>
inline const ScalarWrapper<Real>
WrapScalar( const Real alpha )
{ return ScalarWrapper<Real>( alpha ); }

template<typename Real>
const ScalarWrapper<std::complex<Real> >
WrapScalar( const std::complex<Real> alpha )
{ return ScalarWrapper<std::complex<Real> >( alpha ); }

// These should typically only be used when not in RELEASE mode
void PushCallStack( const std::string s );
void PopCallStack();
void DumpCallStack();

inline unsigned Log2( unsigned N )
{
#ifndef RELEASE
    PushCallStack("Log2");
    if( N == 0 )
        throw std::logic_error("Cannot take integer log2 of 0");
    PopCallStack();
#endif
    int result = 0;
    if( N >= (1<<16)) { N >>= 16; result += 16; }
    if( N >= (1<< 8)) { N >>=  8; result +=  8; }
    if( N >= (1<< 4)) { N >>=  4; result +=  4; }
    if( N >= (1<< 2)) { N >>=  2; result +=  2; }
    if( N >= (1<< 1)) { N >>=  1; result +=  1; }
    return result;
}

inline void AddToMap
( std::map<int,int>& map, int key, int value )
{
    if( value == 0 )
        return;

    std::map<int,int>::iterator it;
    it = map.find( key );
    if( it == map.end() )
        map[key] = value;
    else
        it->second += value;
}

} // namespace psp

#endif // PSP_ENVIRONMENT_HPP
