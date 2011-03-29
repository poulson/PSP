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
#ifndef PSP_FACTOR_MATRIX_HPP
#define PSP_FACTOR_MATRIX_HPP 1

namespace psp {

// A basic low-rank matrix representation that is used for the blocks with
// sufficiently separated sources and targets
template<typename Scalar>
struct FactorMatrix
{
    int m; // height of matrix
    int n; // width of matrix
    int r; // rank of matrix
    // A = U V^H
    std::vector<Scalar> U; // buffer for m x r left set of vectors
    std::vector<Scalar> V; // buffer for n x r right set of vectors
};

} // namespace psp

#endif // PSP_FACTOR_MATRIX_HPP
