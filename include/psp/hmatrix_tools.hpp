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

#include <complex>
#include <cstring>
#include <vector>

#include "psp/blas.hpp"
#include "psp/lapack.hpp"

namespace psp {
namespace hmatrix_tools {

// A basic dense matrix representation that is used for storing blocks 
// whose sources and targets are too close to represent as low rank
template<typename Scalar>
struct DenseMatrix
{
    bool symmetric;
    int m; // height of matrix
    int n; // width of matrix
    int ldim; // leading dimension of matrix
    std::vector<Scalar> buffer; // column-major buffer
};

// A simple Compressed Sparse Row (CSR) data structure
template<typename Scalar>
struct SparseMatrix
{
    bool symmetric;
    int m; // height of matrix
    int n; // width of matrix
    std::vector<Scalar> nonzeros;
    std::vector<int> columnIndices;
    std::vector<int> rowOffsets;
    // TODO: Routines for outputting in Matlab and PETSc formats?
};

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

//----------------------------------------------------------------------------//
// Building blocks for H-algebra                                              //
//----------------------------------------------------------------------------//

// Ensure that the matrix A has a rank of at most 'maxRank'
template<typename Real>
void Compress( int maxRank, FactorMatrix<Real>& A );

template<typename Real>
void Compress( int maxRank, FactorMatrix< std::complex<Real> >& A );

// Convert a subset of a sparse matrix to a dense matrix
template<typename Scalar>
void ConvertSubmatrix
( DenseMatrix<Scalar>& D, const SparseMatrix<Scalar>& S,
  int iStart, int iEnd, int jStart, int jEnd );

// Convert a subset of a sparse matrix to a factor matrix
template<typename Scalar>
void ConvertSubmatrix
( FactorMatrix<Scalar>& F, const SparseMatrix<Scalar>& S,
  int iStart, int iEnd, int jStart, int jEnd );

// Generalized add of two factor matrices, C := alpha A + beta B
template<typename Scalar>
void MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar>& A,
  Scalar beta,  const FactorMatrix<Scalar>& B,
                      FactorMatrix<Scalar>& C );

// Generalized add of two factor matrices, C := alpha A + beta B, 
// where C is then forced to be of rank at most 'maxRank'
template<typename Real>
void MatrixAddRounded
( int maxRank,
  Real alpha, const FactorMatrix<Real>& A,
  Real beta,  const FactorMatrix<Real>& B,
                    FactorMatrix<Real>& C );

template<typename Real>
void MatrixAddRounded
( int maxRank,
  std::complex<Real> alpha, const FactorMatrix< std::complex<Real> >& A,
  std::complex<Real> beta,  const FactorMatrix< std::complex<Real> >& B,
                                  FactorMatrix< std::complex<Real> >& C );

// C := alpha A B
template<typename Scalar>
void MatrixMultiply
( Scalar alpha, const FactorMatrix<Scalar>& A, 
                const FactorMatrix<Scalar>& B,
                      FactorMatrix<Scalar>& C );

// Dense y := alpha A x + beta y
template<typename Scalar>
void MatrixVector
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const std::vector<Scalar>& x,
  Scalar beta,        std::vector<Scalar>& y );

// Dense y := alpha A x
template<typename Scalar>
void MatrixVector
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const std::vector<Scalar>& x,
                      std::vector<Scalar>& y );

// Low-rank y := alpha A x + beta y
template<typename Scalar>
void MatrixVector
( Scalar alpha, const FactorMatrix<Scalar>& A, 
                const std::vector<Scalar>& x,
  Scalar beta,        std::vector<Scalar>& y );

// Low-rank y := alpha A x
template<typename Scalar>
void MatrixVector
( Scalar alpha, const FactorMatrix<Scalar>& A, 
                const std::vector<Scalar>& x,
                      std::vector<Scalar>& y );

//----------------------------------------------------------------------------//
// For mapping between different orderings                                    //
//----------------------------------------------------------------------------//
void BuildNaturalToHierarchicalQuasi2dMap
( std::vector<int>& map, int numLevels, int xSize, int ySize, int zSize );

void InvertMap
(       std::vector<int>& invertedMap,
  const std::vector<int>& originalMap );

} // namespace hmatrix_tools
} // namespace psp

#endif // PSP_HMATRIX_TOOLS_HPP
