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
#ifndef PSP_HMATRIX_TOOLS_HPP
#define PSP_HMATRIX_TOOLS_HPP 1

#include <complex>
#include <cstring>
#include <vector>

#include "psp/blas.hpp"
#include "psp/lapack.hpp"
#include "psp/flat_matrices.hpp"

namespace psp {
namespace hmatrix_tools {

//----------------------------------------------------------------------------//
// Building blocks for H-algebra.                                             //
//                                                                            //
// Routines are put here when they are needed for H-algebra but do not        //
// actually require a hierarchical data structure. This is meant to maximize  //
// the reusability of this code.                                              //
//----------------------------------------------------------------------------//

/*\
|*| Ensure that the factor matrix has a rank of at most 'maxRank'
\*/
template<typename Real>
void Compress( int maxRank, FactorMatrix<Real>& F );
template<typename Real>
void Compress( int maxRank, FactorMatrix< std::complex<Real> >& F );

/*\ 
|*| Convert a subset of a sparse matrix to dense/factor form
\*/
template<typename Scalar>
void ConvertSubmatrix
( DenseMatrix<Scalar>& D, const SparseMatrix<Scalar>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template<typename Scalar>
void ConvertSubmatrix
( FactorMatrix<Scalar>& F, const SparseMatrix<Scalar>& S,
  int iStart, int iEnd, int jStart, int jEnd );

/*\
|*| Generalized addition of two dense/factor matrices, C := alpha A + beta B
\*/
// D := alpha D + beta D
template<typename Scalar>
void MatrixAdd
( Scalar alpha, const DenseMatrix<Scalar>& A,
  Scalar beta,  const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// F := alpha F + beta F
template<typename Scalar>
void MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar>& A,
  Scalar beta,  const FactorMatrix<Scalar>& B,
                      FactorMatrix<Scalar>& C );
// D := alpha F + beta D
template<typename Scalar>
void MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar>& A,
  Scalar beta,  const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D + beta F
template<typename Scalar>
void MatrixAdd
( Scalar alpha, const DenseMatrix<Scalar>& A,
  Scalar beta,  const FactorMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F + beta F
template<typename Scalar>
void MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar>& A,
  Scalar beta,  const FactorMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );

/*\
|*| Generalized update of two dense/factor matrices, B := alpha A + beta B
\*/
// D := alpha D + beta D
template<typename Scalar>
void MatrixUpdate
( Scalar alpha, const DenseMatrix<Scalar>& A,
  Scalar beta,        DenseMatrix<Scalar>& B );
// F := alpha F + beta F
template<typename Scalar>
void MatrixUpdate
( Scalar alpha, const FactorMatrix<Scalar>& A,
  Scalar beta,        FactorMatrix<Scalar>& B );
// D := alpha F + beta D
template<typename Scalar>
void MatrixUpdate
( Scalar alpha, const FactorMatrix<Scalar>& A,
  Scalar beta,        DenseMatrix<Scalar>& B );

/*\
|*| Generalized add of two factor matrices, C := alpha A + beta B, 
|*| where C is then forced to be of rank at most 'maxRank'
\*/
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

/*\
|*| Generalized update of a factor matrix, B := alpha A + beta B, 
|*| where B is then forced to be of rank at most 'maxRank'
\*/
template<typename Real>
void MatrixUpdateRounded
( int maxRank,
  Real alpha, const FactorMatrix<Real>& A,
  Real beta,        FactorMatrix<Real>& B );
template<typename Real>
void MatrixUpdateRounded
( int maxRank,
  std::complex<Real> alpha, const FactorMatrix< std::complex<Real> >& A,
  std::complex<Real> beta,        FactorMatrix< std::complex<Real> >& B );

/*\
|*| Matrix Multiply, C := alpha A B
\*/
// D := alpha D D
template<typename Scalar>
void MatrixMultiply
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// F := alpha F F
template<typename Scalar>
void MatrixMultiply
( Scalar alpha, const FactorMatrix<Scalar>& A, 
                const FactorMatrix<Scalar>& B,
                      FactorMatrix<Scalar>& C );
// F := alpha D F
template<typename Scalar>
void MatrixMultiply
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar>& B,
                      FactorMatrix<Scalar>& C );
// F := alpha F D
template<typename Scalar>
void MatrixMultiply
( Scalar alpha, const FactorMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B,
                      FactorMatrix<Scalar>& C );

/*\
|*| Matrix-Vector multiply, y := alpha A x + beta y
\*/
// y := alpha D x + beta y
template<typename Scalar>
void MatrixVector
( Scalar alpha, const DenseMatrix<Scalar>& D, 
                const std::vector<Scalar>& x,
  Scalar beta,        std::vector<Scalar>& y );
// y := alpha F x + beta y
template<typename Scalar>
void MatrixVector
( Scalar alpha, const FactorMatrix<Scalar>& F, 
                const std::vector<Scalar>& x,
  Scalar beta,        std::vector<Scalar>& y );

/*\
|*| Matrix-Vector multiply, y := alpha A x 
\*/
// y := alpha D x
template<typename Scalar>
void MatrixVector
( Scalar alpha, const DenseMatrix<Scalar>& D, 
                const std::vector<Scalar>& x,
                      std::vector<Scalar>& y );
// y := alpha F x
template<typename Scalar>
void MatrixVector
( Scalar alpha, const FactorMatrix<Scalar>& F, 
                const std::vector<Scalar>& x,
                      std::vector<Scalar>& y );

/*\
|*| Dense inversion, D := inv(D)
\*/
template<typename Scalar>
void Invert( DenseMatrix<Scalar>& D );

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
