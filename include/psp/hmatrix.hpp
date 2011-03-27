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
#ifndef PSP_HMATRIX_HPP
#define PSP_HMATRIX_HPP 1

#include "psp/blas.hpp"
#include "psp/lapack.hpp"

namespace psp {

// For now will require that it be an H-matrix for a quasi-2d domain.
class HMatrix
{
    // A basic low-rank matrix respresentation
    struct FactorMatrix
    {
        int m; // height of matrix
        int n; // width of matrix
        int r; // rank of matrix
        // A = U V^H
        std::vector<PetscScalar> U; // buffer for m x r left set of vectors
        std::vector<PetscScalar> V; // buffer for n x r right set of vectors
    };

    //------------------------------------------------------------------------//
    // Building blocks for H-algebra                                          //
    //------------------------------------------------------------------------//

    // Ensure that the matrix A has a rank of at most 'maxRank'
    static void Compress( int maxRank, FactorMatrix& A );

    // Generalized add of two factor matrices, C := alpha A + beta B
    static void MatrixAdd
    ( PetscScalar alpha, const FactorMatrix& A,
      PetscScalar beta,  const FactorMatrix& B,
                               FactorMatrix& C );

    // Generalized add of two factor matrices, C := alpha A + beta B, 
    // where C is then forced to be of rank at most 'maxRank'
    static void MatrixAddRounded
    ( int maxRank,
      PetscScalar alpha, const FactorMatrix& A,
      PetscScalar beta,  const FactorMatrix& B,
                               FactorMatrix& C );

    // C := alpha A B
    static void MatrixMultiply
    ( PetscScalar alpha, const FactorMatrix& A, const FactorMatrix& B,
                               FactorMatrix& C );

    // y := alpha A x + beta y
    static void MatrixVector
    ( PetscScalar alpha, const FactorMatrix& A, 
                         const std::vector<PetscScalar>& x,
      PetscScalar beta,        std::vector<PetscScalar>& y );

    // y := alpha A x
    static void MatrixVector
    ( PetscScalar alpha, const FactorMatrix& A, 
                         const std::vector<PetscScalar>& x,
                               std::vector<PetscScalar>& y );

    //------------------------------------------------------------------------//
    // For mapping between different orderings                                //
    //------------------------------------------------------------------------//
    static void BuildNaturalToHierarchicalMap
    ( std::vector<int>& naturalToHierarchicalMap, int numLevels,
      int xSize, int ySize, int zSize );

    static void InvertMap
    (       std::vector<int>& invertedMap,
      const std::vector<int>& originalMap );

public:

    // For storing a single entry of a sparse matrix
    struct Entry
    {
        int i;
        int j;
        PetscScalar value;
    };

    // A basic sparse matrix representation
    struct SparseMatrix
    {
        int m; // height of matrix
        int n; // width of matrix
        std::vector<Entry> entries;
    };

    // TODO: Application and construction/destruction routines

};

} // namespace psp

#endif // PSP_HMATRIX_HPP
