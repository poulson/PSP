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
#ifndef PSP_HPP
#define PSP_HPP 1

#include "mpi.h"
#include <complex>
#include <iostream>

#include "psp/config.h"

namespace psp {

template<typename Real>
inline Real
Conj( Real alpha )
{ return alpha; }

template<typename Real>
inline std::complex<Real>
Conj( std::complex<Real> alpha )
{ return std::conj( alpha ); }

// Create a wrappers around real and std::complex<real> types so that they
// can be conveniently printed in a more Matlab-compatible format.
//
// All printing of scalars should now be performed in the fashion:
//     std::cout << WrapScalar(alpha);
// where 'alpha' can be real or complex.

template<typename Real>
class ScalarWrapper
{
    Real _value;
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
    std::complex<Real> _value;
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

} // namespace psp

#include "psp/blas.hpp"
#include "psp/lapack.hpp"
#include "psp/vector.hpp"
#include "psp/dense_matrix.hpp"
#include "psp/low_rank_matrix.hpp"
#include "psp/sparse_matrix.hpp"
#include "psp/abstract_hmatrix.hpp"
#include "psp/hmatrix_tools.hpp"
#include "psp/quasi2d_hmatrix.hpp"

#endif // PSP_HPP
