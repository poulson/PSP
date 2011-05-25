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
#include <complex>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "psp/config.h"

namespace psp {

typedef unsigned char byte;

template<typename Real>
Real
Abs( Real alpha )
{ return alpha; }

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
struct RealBase< std::complex<Real> >
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
    const Real& _value;
public:
    ScalarWrapper( const Real& alpha ) : _value(alpha) { }

    friend std::ostream& operator<<
    ( std::ostream& out, const ScalarWrapper<Real>& alpha )
    {
        out << alpha._value;
        return out;
    }
};

template<typename Real>
class ScalarWrapper< std::complex<Real> >
{
    const std::complex<Real>& _value;
public:
    ScalarWrapper( const std::complex<Real>& alpha ) : _value(alpha) { }

    friend std::ostream& operator<<
    ( std::ostream& out, const ScalarWrapper< std::complex<Real> >& alpha )
    {
        out << std::real(alpha._value) << "+" << std::imag(alpha._value) << "i";
        return out;
    }
};

template<typename Scalar>
ScalarWrapper<Scalar>
WrapScalar( Scalar alpha )
{ return ScalarWrapper<Scalar>( alpha ); }

// These should typically only be used when not in RELEASE mode
void PushCallStack( const std::string& s );
void PopCallStack();
void DumpCallStack();

} // namespace psp

#endif // PSP_ENVIRONMENT_HPP
