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

#include "clique.hpp"

namespace psp {

// Pull in some of Elemental's imported libraries
namespace blas = elem::blas;
namespace lapack = elem::lapack;
namespace mpi = elem::mpi;

// Pull in a number of useful enums from Elemental
using namespace elem::unit_or_non_unit_wrapper;
using namespace elem::distribution_wrapper;
using namespace elem::orientation_wrapper;
using namespace elem::upper_or_lower_wrapper;
using namespace elem::left_or_right_wrapper;
using namespace elem::vertical_or_horizontal_wrapper;
using namespace elem::forward_or_backward_wrapper;
using namespace elem::conjugation_wrapper;

// Pull in a few classes from Elemental
using elem::Complex;
using elem::Matrix;
using elem::Grid;
using elem::DistMatrix;

// Pull in a few indexing functions
using elem::LocalLength;
using elem::Shift;

// Pull in a few scalar math functions
using elem::Abs;
using elem::Conj;
using elem::Exp;

} // namespace psp

#include "psp/config.h"
#include "psp/finite_diff_control.hpp"
#include "psp/grid_data.hpp"
#include "psp/dist_helmholtz.hpp"

#endif // PSP_HPP
