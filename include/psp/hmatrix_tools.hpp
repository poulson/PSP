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
#include <cstdlib> // for integer abs
#include <cstring> // for std::memset and std::memcpy
#include <vector> 

#include "psp/blas.hpp"
#include "psp/lapack.hpp"
#include "psp/vector.hpp"
#include "psp/dense_matrix.hpp"
#include "psp/low_rank_matrix.hpp"
#include "psp/sparse_matrix.hpp"
#include "psp/abstract_hmatrix.hpp"

namespace psp {
namespace hmatrix_tools {

template<typename Scalar>
void Print( const std::string& tag, const DenseMatrix<Scalar>& D );
template<typename Scalar,bool Conjugated>
void Print( const std::string& tag, const LowRankMatrix<Scalar,Conjugated>& F );
template<typename Scalar>
void Print( const std::string& tag, const SparseMatrix<Scalar>& S );

//----------------------------------------------------------------------------//
// Building blocks for H-algebra.                                             //
//                                                                            //
// Routines are put here when they are needed for H-algebra but do not        //
// actually require a hierarchical data structure. This is meant to maximize  //
// the reusability of this code.                                              //
//----------------------------------------------------------------------------//

/*\
|*| Ensure that the low-rank matrix has a rank of at most 'maxRank'
\*/
template<typename Real,bool Conjugated>
void Compress
( int maxRank, 
  DenseMatrix<Real>& D, 
  LowRankMatrix<Real,Conjugated>& F );
template<typename Real,bool Conjugated>
void Compress
( int maxRank, 
  DenseMatrix< std::complex<Real> >& D, 
  LowRankMatrix<std::complex<Real>,Conjugated>& F );
template<typename Real,bool Conjugated>
void Compress( int maxRank, LowRankMatrix<Real,Conjugated>& F );
template<typename Real,bool Conjugated>
void Compress( int maxRank, LowRankMatrix<std::complex<Real>,Conjugated>& F );

/*\
|*| Convert a subset of a sparse matrix to dense/low-rank form
\*/
template<typename Scalar>
void ConvertSubmatrix
( DenseMatrix<Scalar>& D, const SparseMatrix<Scalar>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template<typename Scalar,bool Conjugated>
void ConvertSubmatrix
( LowRankMatrix<Scalar,Conjugated>& F, const SparseMatrix<Scalar>& S,
  int iStart, int iEnd, int jStart, int jEnd );

/*\
|*| Generalized addition of two dense/low-rank matrices, C := alpha A + beta B
\*/
// D := alpha D + beta D
template<typename Scalar>
void MatrixAdd
( Scalar alpha, const DenseMatrix<Scalar>& A,
  Scalar beta,  const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// F := alpha F + beta F
template<typename Scalar,bool Conjugated>
void MatrixAdd
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A,
  Scalar beta,  const LowRankMatrix<Scalar,Conjugated>& B,
                      LowRankMatrix<Scalar,Conjugated>& C );
// D := alpha F + beta D
template<typename Scalar,bool Conjugated>
void MatrixAdd
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A,
  Scalar beta,  const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D + beta F
template<typename Scalar,bool Conjugated>
void MatrixAdd
( Scalar alpha, const DenseMatrix<Scalar>& A,
  Scalar beta,  const LowRankMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F + beta F
template<typename Scalar,bool Conjugated>
void MatrixAdd
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A,
  Scalar beta,  const LowRankMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C );

/*\
|*| Generalized update of two dense/low-rank matrices, B := alpha A + beta B
\*/
// D := alpha D + beta D
template<typename Scalar>
void MatrixUpdate
( Scalar alpha, const DenseMatrix<Scalar>& A,
  Scalar beta,        DenseMatrix<Scalar>& B );
// F := alpha F + beta F
template<typename Scalar,bool Conjugated>
void MatrixUpdate
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A,
  Scalar beta,        LowRankMatrix<Scalar,Conjugated>& B );
// D := alpha F + beta D
template<typename Scalar,bool Conjugated>
void MatrixUpdate
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A,
  Scalar beta,        DenseMatrix<Scalar>& B );

/*\
|*| Generalized add of two low-rank matrices, C := alpha A + beta B, 
|*| where C is then forced to be of rank at most 'maxRank'
\*/
template<typename Real,bool Conjugated>
void MatrixAddRounded
( int maxRank,
  Real alpha, const LowRankMatrix<Real,Conjugated>& A,
  Real beta,  const LowRankMatrix<Real,Conjugated>& B,
                    LowRankMatrix<Real,Conjugated>& C );
template<typename Real,bool Conjugated>
void MatrixAddRounded
( int maxRank,
  std::complex<Real> alpha, 
  const LowRankMatrix<std::complex<Real>,Conjugated>& A,
  std::complex<Real> beta,  
  const LowRankMatrix<std::complex<Real>,Conjugated>& B,
        LowRankMatrix<std::complex<Real>,Conjugated>& C );

/*\
|*| Generalized update of a low-rank matrix, B := alpha A + beta B, 
|*| where B is then forced to be of rank at most 'maxRank'
\*/
template<typename Real,bool Conjugated>
void MatrixUpdateRounded
( int maxRank,
  Real alpha, const LowRankMatrix<Real,Conjugated>& A,
  Real beta,        LowRankMatrix<Real,Conjugated>& B );
template<typename Real,bool Conjugated>
void MatrixUpdateRounded
( int maxRank,
  std::complex<Real> alpha, 
  const LowRankMatrix<std::complex<Real>,Conjugated>& A,
  std::complex<Real> beta, 
        LowRankMatrix<std::complex<Real>,Conjugated>& B
);

/*\
|*| Matrix Matrix multiply, C := alpha A B
|*|
|*| When the resulting matrix is dense, an update form is also provided, i.e.,
|*| C := alpha A B + beta C
|*|
|*| A routine for forming a low-rank matrix from the product of two black-box 
|*| matrix and matrix-transpose vector multiplication routines is also provided.
\*/
// D := alpha D D
template<typename Scalar>
void MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D D + beta D
template<typename Scalar>
void MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha D F
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D F + beta D
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha F D
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F D + beta D
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha F F
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A,
                const LowRankMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F F + beta D
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A,
                const LowRankMatrix<Scalar,Conjugated>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// F := alpha F F
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B,
                      LowRankMatrix<Scalar,Conjugated>& C );
// F := alpha D F
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B,
                      LowRankMatrix<Scalar,Conjugated>& C );
// F := alpha F D
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
                      LowRankMatrix<Scalar,Conjugated>& C );
// F := alpha D D
template<typename Real,bool Conjugated>
void MatrixMatrix
( int maxRank, Real alpha, 
  const DenseMatrix<Real>& A,
  const DenseMatrix<Real>& B,
        LowRankMatrix<Real,Conjugated>& C );
// F := alpha D D
template<typename Real,bool Conjugated>
void MatrixMatrix
( int maxRank, std::complex<Real> alpha, 
  const DenseMatrix< std::complex<Real> >& A,
  const DenseMatrix< std::complex<Real> >& B,
        LowRankMatrix<std::complex<Real>,Conjugated>& C );
// F := alpha D D + beta F
template<typename Real,bool Conjugated>
void MatrixMatrix
( int maxRank, Real alpha, 
  const DenseMatrix<Real>& A,
  const DenseMatrix<Real>& B,
  Real beta,
        LowRankMatrix<Real,Conjugated>& C );
// F := alpha D D + beta F
template<typename Real,bool Conjugated>
void MatrixMatrix
( int maxRank, std::complex<Real> alpha, 
  const DenseMatrix< std::complex<Real> >& A,
  const DenseMatrix< std::complex<Real> >& B,
  std::complex<Real> beta,
        LowRankMatrix<std::complex<Real>,Conjugated>& C );
// F := alpha H H,
template<typename Real,bool Conjugated>
void MatrixMatrix
( int oversampling,
  Real alpha, 
  const AbstractHMatrix<Real>& A,
  const AbstractHMatrix<Real>& B,
        LowRankMatrix<Real,Conjugated>& F );
template<typename Real,bool Conjugated>
void MatrixMatrix
( int oversampling,
  std::complex<Real> alpha, 
  const AbstractHMatrix< std::complex<Real> >& A,
  const AbstractHMatrix< std::complex<Real> >& B,
        LowRankMatrix<std::complex<Real>,Conjugated>& F );

/*\
|*| Matrix Transpose Matrix Multiply, C := alpha A^T B
|*|
|*| When the resulting matrix is dense, an update form is also provided, i.e.,
|*| C := alpha A^T B + beta C
|*|
|*| A routine for forming a low-rank matrix from the product of two black-box 
|*| matrix and matrix-transpose vector multiplication routines is also provided.
\*/
// D := alpha D^T D
template<typename Scalar>
void MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D^T D + beta D
template<typename Scalar>
void MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha D^T F
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D^T F + beta D
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha F^T D
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F^T D + beta D
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha F^T F
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A,
                const LowRankMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F^T F + beta D
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A,
                const LowRankMatrix<Scalar,Conjugated>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// F := alpha F^T F
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B,
                      LowRankMatrix<Scalar,Conjugated>& C );
// F := alpha D^T F
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B,
                      LowRankMatrix<Scalar,Conjugated>& C );
// F := alpha F^T D
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
                      LowRankMatrix<Scalar,Conjugated>& C );
// F := alpha D^T D
template<typename Real,bool Conjugated>
void MatrixTransposeMatrix
( int maxRank, Real alpha, 
  const DenseMatrix<Real>& A,
  const DenseMatrix<Real>& B,
        LowRankMatrix<Real,Conjugated>& C );
// F := alpha D^T D
template<typename Real,bool Conjugated>
void MatrixTransposeMatrix
( int maxRank, std::complex<Real> alpha, 
  const DenseMatrix< std::complex<Real> >& A,
  const DenseMatrix< std::complex<Real> >& B,
        LowRankMatrix<std::complex<Real>,Conjugated>& C );
// F := alpha D^T D + beta F
template<typename Real,bool Conjugated>
void MatrixTransposeMatrix
( int maxRank, Real alpha, 
  const DenseMatrix<Real>& A,
  const DenseMatrix<Real>& B,
  Real beta,
        LowRankMatrix<Real,Conjugated>& C );
// F := alpha D^T D + beta F
template<typename Real,bool Conjugated>
void MatrixTransposeMatrix
( int maxRank, std::complex<Real> alpha, 
  const DenseMatrix< std::complex<Real> >& A,
  const DenseMatrix< std::complex<Real> >& B,
  std::complex<Real> beta,
        LowRankMatrix<std::complex<Real>,Conjugated>& C );
// F := alpha H^T H
template<typename Real,bool Conjugated>
void MatrixTransposeMatrix
( int oversampling,
  Real alpha, 
  const AbstractHMatrix<Real>& A,
  const AbstractHMatrix<Real>& B,
        LowRankMatrix<Real,Conjugated>& F );
template<typename Real,bool Conjugated>
void MatrixTransposeMatrix
( int oversampling,
  std::complex<Real> alpha, 
  const AbstractHMatrix< std::complex<Real> >& A,
  const AbstractHMatrix< std::complex<Real> >& B,
        LowRankMatrix<std::complex<Real>,Conjugated>& F );

/*\
|*| Matrix Hermitian Transpose Matrix Multiply, C := alpha A^H B
|*|
|*| When the resulting matrix is dense, an update form is also provided, i.e.,
|*| C := alpha A^H B + beta C
|*|
|*| A routine for forming a low-rank matrix from the product of two black-box 
|*| matrix and matrix-transpose vector multiplication routines is also provided.
\*/
// D := alpha D^H D
template<typename Scalar>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D^H D + beta D
template<typename Scalar>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha D^H F
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D^H F + beta D
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha F^H D
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F^H D + beta D
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha F^H F
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A,
                const LowRankMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F^H F + beta D
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A,
                const LowRankMatrix<Scalar,Conjugated>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// F := alpha F^H F
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B,
                      LowRankMatrix<Scalar,Conjugated>& C );
// F := alpha D^H F
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B,
                      LowRankMatrix<Scalar,Conjugated>& C );
// F := alpha F^H D
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
                      LowRankMatrix<Scalar,Conjugated>& C );
// F := alpha D^H D
template<typename Real,bool Conjugated>
void MatrixHermitianTransposeMatrix
( int maxRank, Real alpha, 
  const DenseMatrix<Real>& A,
  const DenseMatrix<Real>& B,
        LowRankMatrix<Real,Conjugated>& C );
// F := alpha D^H D
template<typename Real,bool Conjugated>
void MatrixHermitianTransposeMatrix
( int maxRank, std::complex<Real> alpha, 
  const DenseMatrix< std::complex<Real> >& A,
  const DenseMatrix< std::complex<Real> >& B,
        LowRankMatrix<std::complex<Real>,Conjugated>& C );
// F := alpha D^H D + beta F
template<typename Real,bool Conjugated>
void MatrixHermitianTransposeMatrix
( int maxRank, Real alpha, 
  const DenseMatrix<Real>& A,
  const DenseMatrix<Real>& B,
  Real beta,
        LowRankMatrix<Real,Conjugated>& C );
// F := alpha D^H D + beta F
template<typename Real,bool Conjugated>
void MatrixHermitianTransposeMatrix
( int maxRank, std::complex<Real> alpha, 
  const DenseMatrix< std::complex<Real> >& A,
  const DenseMatrix< std::complex<Real> >& B,
  std::complex<Real> beta,
        LowRankMatrix<std::complex<Real>,Conjugated>& C );
// F := alpha H^H H
template<typename Real,bool Conjugated>
void MatrixHermitianTransposeMatrix
( int oversampling,
  Real alpha, 
  const AbstractHMatrix<Real>& A,
  const AbstractHMatrix<Real>& B,
        LowRankMatrix<Real,Conjugated>& F );
template<typename Real,bool Conjugated>
void MatrixHermitianTransposeMatrix
( int oversampling,
  std::complex<Real> alpha, 
  const AbstractHMatrix< std::complex<Real> >& A,
  const AbstractHMatrix< std::complex<Real> >& B,
        LowRankMatrix<std::complex<Real>,Conjugated>& F );

/*\
|*| Matrix-Vector multiply, y := alpha A x + beta y
\*/
// y := alpha D x + beta y
template<typename Scalar>
void MatrixVector
( Scalar alpha, const DenseMatrix<Scalar>& D, 
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y );
// y := alpha F x + beta y
template<typename Scalar,bool Conjugated>
void MatrixVector
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& F, 
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y );

/*\
|*| Matrix-Vector multiply, y := alpha A x 
\*/
// y := alpha D x
template<typename Scalar>
void MatrixVector
( Scalar alpha, const DenseMatrix<Scalar>& D, 
                const Vector<Scalar>& x,
                      Vector<Scalar>& y );
// y := alpha F x
template<typename Scalar,bool Conjugated>
void MatrixVector
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& F, 
                const Vector<Scalar>& x,
                      Vector<Scalar>& y );

/*\
|*| Matrix-Transpose-Vector multiply, y := alpha A^T x + beta y
\*/
// y := alpha D^T x + beta y
template<typename Scalar>
void MatrixTransposeVector
( Scalar alpha, const DenseMatrix<Scalar>& D, 
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y );
// y := alpha F^T x + beta y
template<typename Scalar,bool Conjugated>
void MatrixTransposeVector
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& F, 
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y );

/*\
|*| Matrix-Transpose-Vector multiply, y := alpha A^T x 
\*/
// y := alpha D^T x
template<typename Scalar>
void MatrixTransposeVector
( Scalar alpha, const DenseMatrix<Scalar>& D, 
                const Vector<Scalar>& x,
                      Vector<Scalar>& y );
// y := alpha F^T x
template<typename Scalar,bool Conjugated>
void MatrixTransposeVector
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& F, 
                const Vector<Scalar>& x,
                      Vector<Scalar>& y );

/*\
|*| Matrix-Hermitian-Transpose-Vector multiply, y := alpha A^H x + beta y
\*/
// y := alpha D^H x + beta y
template<typename Scalar>
void MatrixHermitianTransposeVector
( Scalar alpha, const DenseMatrix<Scalar>& D, 
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y );
// y := alpha F^H x + beta y
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeVector
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& F, 
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y );

/*\
|*| Matrix-Hermitian-Transpose-Vector multiply, y := alpha A^H x 
\*/
// y := alpha D^H x
template<typename Scalar>
void MatrixHermitianTransposeVector
( Scalar alpha, const DenseMatrix<Scalar>& D, 
                const Vector<Scalar>& x,
                      Vector<Scalar>& y );
// y := alpha F^H x
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeVector
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& F, 
                const Vector<Scalar>& x,
                      Vector<Scalar>& y );

/*\
|*| Dense inversion, D := inv(D)
\*/
template<typename Scalar>
void Invert( DenseMatrix<Scalar>& D );

/*\
|*| Scale a vector or matrix
\*/
template<typename Scalar>
void Scale( Scalar alpha, Vector<Scalar>& x );

template<typename Scalar>
void Scale( Scalar alpha, DenseMatrix<Scalar>& D );

template<typename Scalar,bool Conjugated>
void Scale( Scalar alpha, LowRankMatrix<Scalar,Conjugated>& F );

/*\
|*| Copy a vector or matrix
\*/
template<typename Scalar>
void Copy( const Vector<Scalar>& x, Vector<Scalar>& y );
template<typename Scalar>
void Copy( const std::vector<Scalar>& x, std::vector<Scalar>& y );
template<typename Scalar>
void Copy( const Vector<Scalar>& x, std::vector<Scalar>& y );
template<typename Scalar>
void Copy( const std::vector<Scalar>& x, Vector<Scalar>& y );
template<typename Scalar>
void Copy( const DenseMatrix<Scalar>& A, DenseMatrix<Scalar>& B );
template<typename Scalar,bool Conjugated>
void Copy
( const LowRankMatrix<Scalar,Conjugated>& A, 
        LowRankMatrix<Scalar,Conjugated>& B );

/*\
|*| Conjugate a vector or matrix
\*/
template<typename Real>
void Conjugate( Vector<Real>& x );
template<typename Real>
void Conjugate( Vector< std::complex<Real> >& x );

template<typename Real>
void Conjugate
( const Vector<Real>& x,
        Vector<Real>& y );
template<typename Real>
void Conjugate
( const Vector< std::complex<Real> >& x, 
        Vector< std::complex<Real> >& y );

template<typename Real>
void Conjugate( std::vector<Real>& x );
template<typename Real>
void Conjugate( std::vector< std::complex<Real> >& x );

template<typename Real>
void Conjugate
( const std::vector<Real>& x,
        std::vector<Real>& y );
template<typename Real>
void Conjugate
( const std::vector< std::complex<Real> >& x,
        std::vector< std::complex<Real> >& y );

template<typename Real>
void Conjugate
( const Vector<Real>& x,
        std::vector<Real>& y );
template<typename Real>
void Conjugate
( const Vector< std::complex<Real> >& x,
        std::vector< std::complex<Real> >& y );

template<typename Real>
void Conjugate
( const std::vector<Real>& x,
        Vector<Real>& y );
template<typename Real>
void Conjugate
( const std::vector< std::complex<Real> >& x,
        Vector< std::complex<Real> >& y );

template<typename Real>
void Conjugate( DenseMatrix<Real>& D );
template<typename Real>
void Conjugate( DenseMatrix< std::complex<Real> >& D );

template<typename Real>
void Conjugate
( const DenseMatrix<Real>& D1, 
        DenseMatrix<Real>& D2 );
template<typename Real>
void Conjugate
( const DenseMatrix< std::complex<Real> >& D1,
        DenseMatrix< std::complex<Real> >& D2 );

template<typename Real,bool Conjugated>
void Conjugate( LowRankMatrix<Real,Conjugated>& F );
template<typename Real,bool Conjugated>
void Conjugate( LowRankMatrix<std::complex<Real>,Conjugated>& F );

template<typename Real,bool Conjugated>
void Conjugate
( const LowRankMatrix<Real,Conjugated>& F1,
        LowRankMatrix<Real,Conjugated>& F2 );
template<typename Real,bool Conjugated>
void Conjugate
( const LowRankMatrix<std::complex<Real>,Conjugated>& F1,
        LowRankMatrix<std::complex<Real>,Conjugated>& F2 );

/*\
|*| For generating Gaussian random variables/vectors
\*/
template<typename Real>
void Uniform( Real& U );
template<typename Real>
void BoxMuller( Real& X, Real& Y );
template<typename Real>
void GaussianRandomVariable( Real& X );
template<typename Real>
void GaussianRandomVariable( std::complex<Real>& X );
template<typename Real>
void GaussianRandomVector( Vector<Real>& x );
template<typename Real>
void GaussianRandomVector( Vector< std::complex<Real> >& x );
template<typename Real>
void GaussianRandomVectors( DenseMatrix<Real>& A );
template<typename Real>
void GaussianRandomVectors( DenseMatrix< std::complex<Real> >& A );

} // namespace hmatrix_tools
} // namespace psp

//----------------------------------------------------------------------------//
// Header implementations                                                     //
//----------------------------------------------------------------------------//

/*\
|*| Copy a vector or matrix
\*/
template<typename Scalar>
void Copy( const psp::Vector<Scalar>& x, psp::Vector<Scalar>& y )
{
    y.Resize( x.Size() );
    std::memcpy( y.Buffer(), x.LockedBuffer(), x.Size()*sizeof(Scalar) );
}

template<typename Scalar>
void Copy( const std::vector<Scalar>& x, std::vector<Scalar>& y )
{
    y.resize( x.size() );
    std::memcpy( &y[0], &x[0], x.size()*sizeof(Scalar) );
}

template<typename Scalar>
void Copy( const psp::Vector<Scalar>& x, std::vector<Scalar>& y )
{
    y.resize( x.Size() );
    std::memcpy( &y[0], x.LockedBuffer(), x.Size()*sizeof(Scalar) );
}

template<typename Scalar>
void Copy( const std::vector<Scalar>& x, psp::Vector<Scalar>& y )
{
    y.Resize( x.size() );
    std::memcpy( y.Buffer(), &x[0], x.size()*sizeof(Scalar) );
}

template<typename Scalar>
void Copy( const psp::DenseMatrix<Scalar>& A, psp::DenseMatrix<Scalar>& B )
{
    const int m = A.Height();
    const int n = A.Width();
    B.SetType( A.Type() ); B.Resize( m, n );
    if( A.Symmetric() )
    {
        for( int j=0; j<n; ++j )
        {
            std::memcpy
            ( B.Buffer(j,j), A.LockedBuffer(j,j), (m-j)*sizeof(Scalar) );
        }
    }
    else
    {
        for( int j=0; j<n; ++j )
        {
            std::memcpy
            ( B.Buffer(0,j), A.LockedBuffer(0,j), m*sizeof(Scalar) );
        }
    }
}

template<typename Scalar,bool Conjugated>
void Copy
( const psp::LowRankMatrix<Scalar,Conjugated>& A, 
        psp::LowRankMatrix<Scalar,Conjugated>& B )
{
    Copy( A.U, B.U );
    Copy( A.V, B.V );
}

/*\
|*| Conjugate a vector or matrix
\*/

template<typename Real> 
void psp::hmatrix_tools::Conjugate
( psp::Vector<Real>& x ) 
{ }

template<typename Real>
void psp::hmatrix_tools::Conjugate
( psp::Vector< std::complex<Real> >& x )
{
    const int n = x.Size();
    std::complex<Real>* xBuffer = x.Buffer();
    for( int i=0; i<n; ++i ) 
        xBuffer[i] = Conj( xBuffer[i] );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const psp::Vector<Real>& x,
        psp::Vector<Real>& y )
{ 
    y.Resize( x.Size() );
    std::memcpy( y.Buffer(), x.LockedBuffer(), x.Size()*sizeof(Real) );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const psp::Vector< std::complex<Real> >& x, 
        psp::Vector< std::complex<Real> >& y )
{
    const int n = x.Size();
    y.Resize( n );
    const std::complex<Real>* RESTRICT xBuffer = x.LockedBuffer();
    std::complex<Real>* RESTRICT yBuffer = y.Buffer();
    for( int i=0; i<n; ++i )
        yBuffer[i] = Conj( xBuffer[i] );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( std::vector<Real>& x )
{ }

template<typename Real>
void psp::hmatrix_tools::Conjugate
( std::vector< std::complex<Real> >& x )
{
    const int n = x.size();
    std::complex<Real>* xBuffer = &x[0];
    for( int i=0; i<n; ++i )
        xBuffer[i] = Conj( xBuffer[i] );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const std::vector<Real>& x,
        std::vector<Real>& y )
{
    y.resize( x.size() );
    std::memcpy( &y[0], &x[0], x.size()*sizeof(Real) );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const std::vector< std::complex<Real> >& x,
        std::vector< std::complex<Real> >& y )
{
    const int n = x.size();
    y.resize( n );
    const std::complex<Real>* RESTRICT xBuffer = &x[0];
    std::complex<Real>* RESTRICT yBuffer = &y[0];
    for( int i=0; i<n; ++i )
        yBuffer[i] = Conj( xBuffer[i] );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const psp::Vector<Real>& x,
        std::vector<Real>& y )
{
    y.resize( x.Size() );
    std::memcpy( &y[0], x.Buffer(), x.Size()*sizeof(Real) );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const psp::Vector< std::complex<Real> >& x,
        std::vector< std::complex<Real> >& y )
{
    const int n = x.Size();
    y.resize( n );
    const std::complex<Real>* RESTRICT xBuffer = x.LockedBuffer();
    std::complex<Real>* RESTRICT yBuffer = &y[0];
    for( int i=0; i<n; ++i )
        yBuffer[i] = Conj( xBuffer[i] );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const std::vector<Real>& x,
        psp::Vector<Real>& y )
{
    y.Resize( x.size() );
    std::memcpy( y.Buffer(), &x[0], x.size()*sizeof(Real) );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const std::vector< std::complex<Real> >& x,
        psp::Vector< std::complex<Real> >& y )
{
    const int n = x.size();
    y.Resize( n );
    const std::complex<Real>* xBuffer = &x[0];
    std::complex<Real>* yBuffer = y.Buffer();
    for( int i=0; i<n; ++i )
        yBuffer[i] = Conj( xBuffer[i] );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( psp::DenseMatrix<Real>& D )
{ }

template<typename Real>
void psp::hmatrix_tools::Conjugate
( psp::DenseMatrix< std::complex<Real> >& D )
{
    const int m = D.Height();
    const int n = D.Width();
    for( int j=0; j<n; ++j )
    {
        std::complex<Real>* DCol = D.Buffer(0,j);
        for( int i=0; i<m; ++i )
            DCol[i] = Conj( DCol[i] );
    }
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const psp::DenseMatrix<Real>& D1, 
        psp::DenseMatrix<Real>& D2 )
{
    const int m = D1.Height();
    const int n = D1.Width();
    D2.SetType( D1.Type() ); 
    D2.Resize( m, n );
    if( D1.Symmetric() )
    {
        for( int j=0; j<n; ++j )
        {
            std::memcpy
            ( D2.Buffer(j,j), D1.LockedBuffer(j,j), (m-j)*sizeof(Real) );
        }
    }
    else
    {
        for( int j=0; j<n; ++j )
        {
            std::memcpy
            ( D2.Buffer(0,j), D1.LockedBuffer(0,j), m*sizeof(Real) );
        }
    }
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const psp::DenseMatrix< std::complex<Real> >& D1,
        psp::DenseMatrix< std::complex<Real> >& D2 )
{
    const int m = D1.Height();
    const int n = D1.Width();
    D2.SetType( D1.Type() );
    D2.Resize( m, n );
    if( D1.Symmetric() )
    {
        for( int j=0; j<n; ++j )
        {
            const std::complex<Real>* RESTRICT D1Col = D1.LockedBuffer(0,j);
            std::complex<Real>* RESTRICT D2Col = D2.Buffer(0,j);
            for( int i=j; i<m; ++i )
                D2Col[i] = Conj( D1Col[i] );
        }
    }
    else
    {
        for( int j=0; j<n; ++j )
        {
            const std::complex<Real>* RESTRICT D1Col = D1.LockedBuffer(0,j);
            std::complex<Real>* RESTRICT D2Col = D2.Buffer(0,j);
            for( int i=0; i<m; ++i )
                D2Col[i] = Conj( D1Col[i] );
        }
    }
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::Conjugate
( psp::LowRankMatrix<Real,Conjugated>& F )
{ }

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::Conjugate
( psp::LowRankMatrix<std::complex<Real>,Conjugated>& F )
{
    Conjugate( F.U );
    Conjugate( F.V );
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::Conjugate
( const psp::LowRankMatrix<Real,Conjugated>& F1,
        psp::LowRankMatrix<Real,Conjugated>& F2 )
{
    const int m = F1.Height();
    const int n = F1.Width();
    const int r = F1.Rank();
    F2.U.SetType( GENERAL ); F2.U.Resize( m, r );
    F2.V.SetType( GENERAL ); F2.V.Resize( n, r );
    Copy( F1.U, F2.U );
    Copy( F1.V, F2.V );
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::Conjugate
( const psp::LowRankMatrix<std::complex<Real>,Conjugated>& F1,
        psp::LowRankMatrix<std::complex<Real>,Conjugated>& F2 )
{
    const int m = F1.Height();
    const int n = F1.Width();
    const int r = F1.Rank();
    F2.U.SetType( GENERAL ); F2.U.Resize( m, r );
    F2.V.SetType( GENERAL ); F2.V.Resize( n, r );
    Conjugate( F1.U, F2.U );
    Conjugate( F1.V, F2.V );
}

/*\
|*| For scaling vectors and matrices
\*/

template<typename Scalar>
void psp::hmatrix_tools::Scale
( Scalar alpha, psp::Vector<Scalar>& x )
{
    if( alpha == (Scalar)0 )
        std::memset( x.Buffer(), 0, x.Size()*sizeof(Scalar) );
    else
        blas::Scal( x.Size(), alpha, x.Buffer(), 1 );
}

template<typename Scalar>
void psp::hmatrix_tools::Scale
( Scalar alpha, psp::DenseMatrix<Scalar>& D )
{
    const int m = D.Height();
    const int n = D.Width();

    if( alpha == (Scalar)1 )
        return;

    if( D.Symmetric() )
    {
        if( alpha == (Scalar)0 )
        {
            for( int j=0; j<n; ++j )
                std::memset( D.Buffer(j,j), 0, (m-j)*sizeof(Scalar) );
        }
        else
        {
            for( int j=0; j<n; ++j )
                blas::Scal( m-j, alpha, D.Buffer(j,j), 1 );
        }
    }
    else
    {
        if( alpha == (Scalar)0 )
        {
            for( int j=0; j<n; ++j )
                std::memset( D.Buffer(0,j), 0, m*sizeof(Scalar) );
        }
        else
        {
            for( int j=0; j<n; ++j )
                blas::Scal( m, alpha, D.Buffer(0,j), 1 );
        }
    }
}

template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::Scale
( Scalar alpha, psp::LowRankMatrix<Scalar,Conjugated>& F )
{
    if( alpha == (Scalar)0 )
    {
        F.U.Resize( F.Height(), 0 );
        F.V.Resize( F.Width(),  0 );
    }
    else
    {
        Scale( alpha, F.U );
    }
}

/*\
|*| For forming low-rank approximations to the product of H-matrices
\*/

// F := alpha H H,
template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( int oversampling, 
  Real alpha, 
  const psp::AbstractHMatrix<Real>& A, 
  const psp::AbstractHMatrix<Real>& B,
        psp::LowRankMatrix<Real,Conjugated>& F )
{
    const int maxRankA = std::min( A.Height(), A.Width() );
    const int maxRankB = std::min( B.Height(), B.Width() );
    const int maxRankAB = std::min( maxRankA, maxRankB );
    const int r = std::min( std::min(A.MaxRank(),B.MaxRank()), maxRankAB );

    // Generate a few more than r Gaussian random vectors
    DenseMatrix<Real> Omega( B.Width(), r+oversampling );
    GaussianRandomVectors( Omega );

    // Compute the action of (alpha A B) on Omega (into Y)
    DenseMatrix<Real> X;
    B.MapMatrix( alpha, Omega, X );
    DenseMatrix<Real> Y;
    A.MapMatrix( 1, X, Y );

    // Create a work vector that is sufficiently large for all operations
    const int lworkPivotedQR = lapack::PivotedQRWorkSize( r+oversampling );
    const int lworkSVD = lapack::SVDWorkSize( B.Width(), r+oversampling );
    const int lwork = std::max( lworkPivotedQR, lworkSVD );
    std::vector<Real> work( lwork );

    // Replace Y with an orthogonal matrix which spans its range
    {
        const int m = Y.Height();
        const int n = Y.Width();
        const int minDim = std::min( m, n );

        // Perform a pivoted QR decomposition on Y = (alpha A B) Omega
        std::vector<int> jpvt( n );
        std::vector<Real> tau( minDim );
        lapack::PivotedQR
        ( m, n, Y.Buffer(), Y.LDim(), &jpvt[0], &tau[0], &work[0], lwork );
        
        // Form the Q from the QR decomposition
        lapack::FormQ
        ( m, n, minDim, Y.Buffer(), Y.LDim(), &tau[0], &work[0], lwork );
    }

    // Compute (Q^T (alpha A B))^T = alpha B^T A^T Q into F.V
    A.TransposeMapMatrix( alpha, Y, X );
    B.TransposeMapMatrix( 1, X, F.V );

    // Compute the economic SVD of F.V = (Q^T (alpha A B))^T = U Sigma V^T,
    // overwriting F.V with U, and X with V^T. Then truncate the SVD to rank 
    // r and form V^T := Sigma V^T.
    {
        const int m = F.V.Height();
        const int n = F.V.Width();
        std::vector<Real> s( std::min(m,n) );
        X.Resize( std::min(m,n), n );
        lapack::SVD
        ( 'O', 'S', m, n, F.V.Buffer(), F.V.LDim(), &s[0], 0, 0, 
          X.Buffer(), X.LDim(), &work[0], lwork );

        // Truncate the SVD in-place
        F.V.Resize( m, r );
        s.resize( r );
        X.Resize( r, n );

        // V^T := Sigma V^T
        for( int i=0; i<r; ++i )
        {
            const Real sigma = s[i];
            Real* VTRow = X.Buffer(i,0);
            const int VTLDim = X.LDim();
            for( int j=0; j<n; ++j )
                VTRow[j*VTLDim] *= sigma;
        }
    }

    // F.U := Q (VT)^T = Q V
    F.U.Resize( Y.Height(), r );
    blas::Gemm
    ( 'N', 'T', Y.Height(), r, Y.Width(), 
      1, Y.LockedBuffer(), Y.LDim(), X.LockedBuffer(), X.LDim(), 
      0, F.U.Buffer(), F.U.LDim() );
}

// F := alpha H H,
template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( int oversampling, 
  std::complex<Real> alpha, 
  const psp::AbstractHMatrix< std::complex<Real> >& A, 
  const psp::AbstractHMatrix< std::complex<Real> >& B,
        psp::LowRankMatrix< std::complex<Real>,Conjugated>& F )
{
    typedef std::complex<Real> Scalar;

    const int maxRankA = std::min( A.Height(), A.Width() );
    const int maxRankB = std::min( B.Height(), B.Width() );
    const int maxRankAB = std::min( maxRankA, maxRankB );
    const int r = std::min( std::min(A.MaxRank(),B.MaxRank()), maxRankAB );

    // Generate a few more than r Gaussian random vectors
    DenseMatrix<Scalar> Omega( B.Width(), r+oversampling );
    GaussianRandomVectors( Omega );

    // Compute the action of (alpha A B) on Omega (into Y)
    DenseMatrix<Scalar> X;
    B.MapMatrix( alpha, Omega, X );
    DenseMatrix<Scalar> Y;
    A.MapMatrix( 1, X, Y );

    // Create work vectors that are sufficiently large for all operations
    const int lworkPivotedQR = lapack::PivotedQRWorkSize( r+oversampling );
    const int lrworkPivotedQR = lapack::PivotedQRRealWorkSize( r+oversampling );
    const int lworkSVD = lapack::SVDWorkSize( B.Width(), r+oversampling );
    const int lrworkSVD = lapack::SVDRealWorkSize( B.Width(), r+oversampling );
    const int lwork = std::max( lworkPivotedQR, lworkSVD );
    std::vector<Scalar> work( lwork );
    std::vector<Real> rwork( std::max(lrworkPivotedQR,lrworkSVD) );
    
    // Replace Y with an orthogonal matrix which spans its range
    {
        const int m = Y.Height();
        const int n = Y.Width();
        const int minDim = std::min( m, n );

        // Perform a pivoted QR decomposition on Y = (alpha A B) Omega
        std::vector<int> jpvt( n );
        std::vector<Scalar> tau( minDim );
        lapack::PivotedQR
        ( m, n, Y.Buffer(), Y.LDim(), &jpvt[0], &tau[0], 
          &work[0], lwork, &rwork[0] );

        // Form the Q from the QR decomposition
        lapack::FormQ
        ( m, n, minDim, Y.Buffer(), Y.LDim(), &tau[0], &work[0], lwork );
    }

    // Compute (Q^H (alpha AB))^H = conj(alpha) B^H A^H Q into F.V
    A.HermitianTransposeMapMatrix( Conj(alpha), Y, X );
    B.HermitianTransposeMapMatrix( 1, X, F.V );
        
    // Compute the economic SVD of F.V = (Q^H (alpha AB))^H = U Sigma V^H,
    // overwriting F.V with U, and X with V^H. Then truncate the SVD to rank 
    // r and form V^H := Sigma V^H.
    {
        const int m = F.V.Height();
        const int n = F.V.Width();
        std::vector<Real> s( std::min(m,n) );
        X.Resize( std::min(m,n), n );
        lapack::SVD
        ( 'O', 'S', m, n, F.V.Buffer(), F.V.LDim(), &s[0], 0, 0,
          X.Buffer(), X.LDim(), &work[0], lwork, &rwork[0] );

        // Truncate the SVD in-place
        F.V.Resize( m, r );
        s.resize( r );
        X.Resize( r, n );

        // V^H := Sigma V^H
        for( int i=0; i<r; ++i )
        {
            const Real sigma = s[i];
            Scalar* VHRow = X.Buffer(i,0);
            const int VHLDim = X.LDim();
            for( int j=0; j<n; ++j )
                VHRow[j*VHLDim] *= sigma;
        }
    }

    // F.U := Q (VH)^H = Q V
    blas::Gemm
    ( 'N', 'C', Y.Height(), r, Y.Width(), 
      1, Y.LockedBuffer(), Y.LDim(), X.LockedBuffer(), X.LDim(), 
      0, F.U.Buffer(), F.U.LDim() );
}

// F := alpha H^T H,
template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixTransposeMatrix
( int oversampling,
  Real alpha, 
  const psp::AbstractHMatrix<Real>& A,
  const psp::AbstractHMatrix<Real>& B,
        psp::LowRankMatrix<Real,Conjugated>& F )
{
    const int maxRankA = std::min( A.Height(), A.Width() );
    const int maxRankB = std::min( B.Height(), B.Width() );
    const int maxRankAB = std::min( maxRankA, maxRankB );
    const int r = std::min( std::min(A.MaxRank(),B.MaxRank()), maxRankAB );

    // Generate a few more than r Gaussian random vectors
    DenseMatrix<Real> Omega( B.Width(), r+oversampling );
    GaussianRandomVectors( Omega );

    // Compute the action of (alpha A^T B) on Omega (into Y)
    DenseMatrix<Real> X;
    B.MapMatrix( alpha, Omega, X );
    DenseMatrix<Real> Y;
    A.TransposeMapMatrix( 1, X, Y );

    // Create a work vector that is sufficiently large for all operations
    const int lworkPivotedQR = lapack::PivotedQRWorkSize( r+oversampling );
    const int lworkSVD = lapack::SVDWorkSize( B.Width(), r+oversampling );
    const int lwork = std::max( lworkPivotedQR, lworkSVD );
    std::vector<Real> work( lwork );

    // Replace Y with an orthogonal matrix which spans its range
    {
        const int m = Y.Height();
        const int n = Y.Width();
        const int minDim = std::min( m, n );

        // Perform a pivoted QR decomposition on Y = (alpha A^T B) Omega
        std::vector<int> jpvt( n );
        std::vector<Real> tau( minDim );
        lapack::PivotedQR
        ( m, n, Y.Buffer(), Y.LDim(), &jpvt[0], &tau[0], &work[0], lwork );

        // Form the Q from the QR decomposition
        lapack::FormQ
        ( m, n, minDim, Y.Buffer(), Y.LDim(), &tau[0], &work[0], lwork );
    }

    // Compute (Q^T (alpha A^T B))^T = alpha B^T A Q into F.V
    A.MapMatrix( alpha, Y, X );
    B.TransposeMapMatrix( 1, X, F.V );

    // Compute the economic SVD of F.V = (Q^T (alpha A^T B))^T = U Sigma V^T,
    // overwriting F.V with U, and X with V^T. Then truncate the SVD to rank 
    // r and form V^T := Sigma V^T.
    {
        const int m = F.V.Height();
        const int n = F.V.Width();
        std::vector<Real> s( std::min(m,n) );
        X.Resize( std::min(m,n), n );
        lapack::SVD
        ( 'O', 'S', m, n, F.V.Buffer(), F.V.LDim(), &s[0], 0, 0,
          X.Buffer(), X.LDim(), &work[0], lwork );

        // Truncate the SVD in-place
        F.V.Resize( m, r );
        s.resize( r );
        X.Resize( r, n );

        // V^T := Sigma V^T
        for( int i=0; i<r; ++i )
        {
            const Real sigma = s[i];
            Real* VTRow = X.Buffer(i,0);
            const int VTLDim = X.LDim();
            for( int j=0; j<n; ++j )
                VTRow[j*VTLDim] *= sigma;
        }
    }

    // F.U := Q (VT)^T = Q V
    F.U.Resize( Y.Height(), r );
    blas::Gemm
    ( 'N', 'T', Y.Height(), r, Y.Width(),
      1, Y.LockedBuffer(), Y.LDim(), X.LockedBuffer(), X.LDim(),
      0, F.U.Buffer(), F.U.LDim() );
}

// F := alpha H^T H,
template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixTransposeMatrix
( int oversampling,
  std::complex<Real> alpha, 
  const psp::AbstractHMatrix< std::complex<Real> >& A,
  const psp::AbstractHMatrix< std::complex<Real> >& B,
        psp::LowRankMatrix<std::complex<Real>,Conjugated>& F )
{
    typedef std::complex<Real> Scalar;

    const int maxRankA = std::min( A.Height(), A.Width() );
    const int maxRankB = std::min( B.Height(), B.Width() );
    const int maxRankAB = std::min( maxRankA, maxRankB );
    const int r = std::min( std::min(A.MaxRank(),B.MaxRank()), maxRankAB );

    // Generate a few more than r Gaussian random vectors
    DenseMatrix<Scalar> Omega( B.Width(), r+oversampling );
    GaussianRandomVectors( Omega );

    // Compute the action of (alpha A^T B) on Omega (into Y)
    DenseMatrix<Scalar> X;
    B.MapMatrix( alpha, Omega, X );
    DenseMatrix<Scalar> Y;
    A.TransposeMapMatrix( 1, X, Y );

    // Create work vectors that are sufficiently large for all operations
    const int lworkPivotedQR = lapack::PivotedQRWorkSize( r+oversampling );
    const int lrworkPivotedQR = lapack::PivotedQRRealWorkSize( r+oversampling );
    const int lworkSVD = lapack::SVDWorkSize( B.Width(), r+oversampling );
    const int lrworkSVD = lapack::SVDRealWorkSize( B.Width(), r+oversampling );
    const int lwork = std::max( lworkPivotedQR, lworkSVD );
    std::vector<Scalar> work( lwork );
    std::vector<Real> rwork( std::max(lrworkPivotedQR,lrworkSVD) );
    
    // Replace Y with an orthogonal matrix which spans its range
    {
        const int m = Y.Height();
        const int n = Y.Width();
        const int minDim = std::min( m, n );

        // Perform a pivoted QR decomposition on Y = (alpha A^T B) Omega
        std::vector<int> jpvt( n );
        std::vector<Scalar> tau( minDim );
        lapack::PivotedQR
        ( m, n, Y.Buffer(), Y.LDim(), &jpvt[0], &tau[0], 
          &work[0], lwork, &rwork[0] );

        // Form the Q from the QR decomposition
        lapack::FormQ
        ( m, n, minDim, Y.Buffer(), Y.LDim(), &tau[0], &work[0], lwork );
    }

    // Compute (Q^H (alpha A^T B))^H = conj(alpha) B^H conj(A) Q
    //                               = conj(alpha B^T A conj(Q))
    // into F.V.
    Conjugate( Y );
    A.MapMatrix( alpha, Y, X );
    B.TransposeMapMatrix( 1, X, F.V );
    Conjugate( F.V );
    Conjugate( Y );
        
    // Compute the economic SVD of F.V = (Q^H (alpha A^T B))^H = U Sigma V^H,
    // overwriting F.V with U, and X with V^H. Then truncate the SVD to rank 
    // r and form V^H := Sigma V^H.
    {
        const int m = F.V.Height();
        const int n = F.V.Width();
        std::vector<Real> s( std::min(m,n) );
        X.Resize( std::min(m,n), n );
        lapack::SVD
        ( 'O', 'S', m, n, F.V.Buffer(), F.V.LDim(), &s[0], 0, 0,
          X.Buffer(), X.LDim(), &work[0], lwork, &rwork[0] );

        // Truncate the SVD in-place
        F.V.Resize( m, r );
        s.resize( r );
        X.Resize( r, n );

        // V^H := Sigma V^H
        for( int i=0; i<r; ++i )
        {
            const Real sigma = s[i];
            Scalar* VHRow = X.Buffer(i,0);
            const int VHLDim = X.LDim();
            for( int j=0; j<n; ++j )
                VHRow[j*VHLDim] *= sigma;
        }
    }

    // F.U := Q (VH)^H = Q V
    F.U.Resize( Y.Height(), r );
    blas::Gemm
    ( 'N', 'C', Y.Height(), r, Y.Width(), 
      1, Y.LockedBuffer(), Y.LDim(), X.LockedBuffer(), X.LDim(), 
      0, F.U.Buffer(), F.U.LDim() );
}

// F := alpha H^H H,
template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, int oversampling, 
  Real alpha, 
  const psp::AbstractHMatrix<Real>& A,
  const psp::AbstractHMatrix<Real>& B,
        psp::LowRankMatrix<Real,Conjugated>& F )
{
    MatrixTransposeMatrix( maxRank, oversampling, alpha, A, B, F );
}

// F := alpha H^H H
template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int oversampling, 
  std::complex<Real> alpha, 
  const psp::AbstractHMatrix< std::complex<Real> >& A,
  const psp::AbstractHMatrix< std::complex<Real> >& B,
        psp::LowRankMatrix<std::complex<Real>,Conjugated>& F )
{
    typedef std::complex<Real> Scalar;

    const int maxRankA = std::min( A.Height(), A.Width() );
    const int maxRankB = std::min( B.Height(), B.Width() );
    const int maxRankAB = std::min( maxRankA, maxRankB );
    const int r = std::min( std::min(A.MaxRank(),B.MaxRank()), maxRankAB );

    // Generate a few more than r Gaussian random vectors
    DenseMatrix<Scalar> Omega( B.Width(), r+oversampling );
    GaussianRandomVectors( Omega );

    // Compute the action of (alpha A^H B) on Omega (into Y)
    DenseMatrix<Scalar> X;
    B.MapMatrix( alpha, Omega, X );
    DenseMatrix<Scalar> Y;
    A.HermitianTransposeMapMatrix( 1, X, Y );

    // Create work vectors that are sufficiently large for all operations
    const int lworkPivotedQR = lapack::PivotedQRWorkSize( r+oversampling );
    const int lrworkPivotedQR = lapack::PivotedQRRealWorkSize( r+oversampling );
    const int lworkSVD = lapack::SVDWorkSize( B.Width(), r+oversampling );
    const int lrworkSVD = lapack::SVDRealWorkSize( B.Width(), r+oversampling );
    const int lwork = std::max( lworkPivotedQR, lworkSVD );
    std::vector<Scalar> work( lwork );
    std::vector<Real> rwork( std::max(lrworkPivotedQR,lrworkSVD) );
    
    // Replace Y with an orthogonal matrix which spans its range
    {
        const int m = Y.Height();
        const int n = Y.Width();
        const int minDim = std::min( m, n );

        // Perform a pivoted QR decomposition on Y = (alpha A^H B) Omega
        std::vector<int> jpvt( n );
        std::vector<Scalar> tau( minDim );
        lapack::PivotedQR
        ( m, n, Y.Buffer(), Y.LDim(), &jpvt[0], &tau[0], 
          &work[0], lwork, &rwork[0] );

        // Form the Q from the QR decomposition
        lapack::FormQ
        ( m, n, minDim, Y.Buffer(), Y.LDim(), &tau[0], &work[0], lwork );
    }

    // Compute (Q^H (alpha A^H B))^H = conj(alpha) B^H A Q into F.V.
    A.MapMatrix( Conj(alpha), Y, X );
    B.HermitianTransposeMapMatrix( 1, X, F.V );
        
    // Compute the economic SVD of F.V = (Q^H (alpha A^H B))^H = U Sigma V^H,
    // overwriting F.V with U, and X with V^H. Then truncate the SVD to rank 
    // r and form V^H := Sigma V^H.
    {
        const int m = F.V.Height();
        const int n = F.V.Width();
        std::vector<Real> s( std::min(m,n) );
        X.Resize( std::min(m,n), n );
        lapack::SVD
        ( 'O', 'S', m, n, F.V.Buffer(), F.V.LDim(), &s[0], 0, 0,
          X.Buffer(), X.LDim(), &work[0], lwork, &rwork[0] );

        // Truncate the SVD in-place
        F.V.Resize( m, r );
        s.resize( r );
        X.Resize( r, n );

        // V^H := Sigma V^H
        for( int i=0; i<r; ++i )
        {
            const Real sigma = s[i];
            Scalar* VHRow = X.Buffer(i,0);
            const int VHLDim = X.LDim();
            for( int j=0; j<n; ++j )
                VHRow[j*VHLDim] *= sigma;
        }
    }

    // F.U := Q (VH)^H = Q V
    F.U.Resize( Y.Height(), r );
    blas::Gemm
    ( 'N', 'C', Y.Height(), r, Y.Width(), 
      1, Y.LockedBuffer(), Y.LDim(), X.LockedBuffer(), X.LDim(), 
      0, F.U.Buffer(), F.U.LDim() );
}

/*\
|*| For generating Gaussian random variables/vectors
\*/

// Return a uniform sample from [0,1]
template<typename Real>
inline void
psp::hmatrix_tools::Uniform
( Real& U )
{
    U = static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX);
}

template<typename Real>
inline void
psp::hmatrix_tools::BoxMuller
( Real& X, Real& Y )
{
    Real U, V;
    Uniform( U );
    Uniform( V );
    const Real A = sqrt(-2*log(U));
    const Real c = cos(2*M_PI*V);
    const Real s = sin(2*M_PI*V);
    X = A*c;
    Y = A*s;
}

template<typename Real>
inline void
psp::hmatrix_tools::GaussianRandomVariable
( Real& X )
{
    // Use half of Box-Muller
    Real U, V;
    Uniform( U );
    Uniform( V );
    X = sqrt(-2*log(U)) * cos(2*M_PI*V);
}

template<typename Real>
inline void
psp::hmatrix_tools::GaussianRandomVariable
( std::complex<Real>& X )
{
    Real Y, Z;
    BoxMuller( Y, Z );
    X = std::complex<Real>( Y, Z );
}

template<typename Real>
void
psp::hmatrix_tools::GaussianRandomVector
( psp::Vector<Real>& x )
{
    // Use BoxMuller for every pair of entries
    const int n = x.size();
    const int numPairs = (n+1)/2;
    Real* buffer = x.Buffer();
    for( int i=0; i<numPairs-1; ++i )
    {
        Real X, Y;
        BoxMuller( X, Y );
        buffer[2*i] = X;
        buffer[2*i+1] = Y;
    }
    if( n & 1 )
        GaussianRandomVariable( buffer[n-1] );
    else
        BoxMuller( buffer[n-2], buffer[n-1] );
}

template<typename Real>
void
psp::hmatrix_tools::GaussianRandomVector
( psp::Vector< std::complex<Real> >& x )
{
    const int n = x.size();
    std::complex<Real>* buffer = x.Buffer();
    for( int i=0; i<n; ++i )
        GaussianRandomVariable( buffer[i] );
}

template<typename Real>
void
psp::hmatrix_tools::GaussianRandomVectors
( psp::DenseMatrix<Real>& A )
{
    // Use BoxMuller for every pair of entries in each column
    A.SetType( GENERAL );
    const int m = A.Height();
    const int n = A.Width();
    const int numPairs = (m+1)/2;
    for( int j=0; j<n; ++j )
    {
        Real* ACol = A.Buffer(0,j);
        for( int i=0; i<numPairs-1; ++i )
        {
            Real X, Y;
            BoxMuller( X, Y );
            ACol[2*i] = X;
            ACol[2*i+1] = Y;
        }
        if( m & 1 )
            GaussianRandomVariable( ACol[n-1] );
        else
            BoxMuller( ACol[n-2], ACol[n-1] );
    }
}

template<typename Real>
void
psp::hmatrix_tools::GaussianRandomVectors
( psp::DenseMatrix< std::complex<Real> >& A )
{
    A.SetType( GENERAL );
    const int m = A.Height();
    const int n = A.Width();
    for( int j=0; j<n; ++j )
    {
        std::complex<Real>* ACol = A.Buffer(0,j);
        for( int i=0; i<m; ++i )
            GaussianRandomVariable( ACol[i] );
    }
}

#endif // PSP_HMATRIX_TOOLS_HPP
