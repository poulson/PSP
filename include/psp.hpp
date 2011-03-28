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

#include "psp/config.h"

namespace psp {

template<typename Real>
inline Real
Conj( const Real alpha )
{ return alpha; }

template<typename Real>
inline std::complex<Real>
Conj( const std::complex<Real> alpha )
{ return std::conj( alpha ); }

} // namespace psp

#include "psp/blas.hpp"
#include "psp/lapack.hpp"
#include "psp/flat_matrices.hpp"
#include "psp/hmatrix_tools.hpp"
#include "psp/hmatrix_quasi2d.hpp"

#endif // PSP_HPP
