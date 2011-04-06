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
#include "psp/factor_matrix.hpp"
#include "psp/sparse_matrix.hpp"
#include "psp/abstract_hmatrix.hpp"

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
template<typename Real,bool Conjugated>
void Compress( int maxRank, FactorMatrix<Real,Conjugated>& F );
template<typename Real,bool Conjugated>
void Compress( int maxRank, FactorMatrix<std::complex<Real>,Conjugated>& F );

/*\
|*| Convert a subset of a sparse matrix to dense/factor form
\*/
template<typename Scalar>
void ConvertSubmatrix
( DenseMatrix<Scalar>& D, const SparseMatrix<Scalar>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template<typename Scalar,bool Conjugated>
void ConvertSubmatrix
( FactorMatrix<Scalar,Conjugated>& F, const SparseMatrix<Scalar>& S,
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
template<typename Scalar,bool Conjugated>
void MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A,
  Scalar beta,  const FactorMatrix<Scalar,Conjugated>& B,
                      FactorMatrix<Scalar,Conjugated>& C );
// D := alpha F + beta D
template<typename Scalar,bool Conjugated>
void MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A,
  Scalar beta,  const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D + beta F
template<typename Scalar,bool Conjugated>
void MatrixAdd
( Scalar alpha, const DenseMatrix<Scalar>& A,
  Scalar beta,  const FactorMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F + beta F
template<typename Scalar,bool Conjugated>
void MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A,
  Scalar beta,  const FactorMatrix<Scalar,Conjugated>& B,
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
template<typename Scalar,bool Conjugated>
void MatrixUpdate
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A,
  Scalar beta,        FactorMatrix<Scalar,Conjugated>& B );
// D := alpha F + beta D
template<typename Scalar,bool Conjugated>
void MatrixUpdate
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A,
  Scalar beta,        DenseMatrix<Scalar>& B );

/*\
|*| Generalized add of two factor matrices, C := alpha A + beta B, 
|*| where C is then forced to be of rank at most 'maxRank'
\*/
template<typename Real,bool Conjugated>
void MatrixAddRounded
( int maxRank,
  Real alpha, const FactorMatrix<Real,Conjugated>& A,
  Real beta,  const FactorMatrix<Real,Conjugated>& B,
                    FactorMatrix<Real,Conjugated>& C );
template<typename Real,bool Conjugated>
void MatrixAddRounded
( int maxRank,
  std::complex<Real> alpha, const FactorMatrix<std::complex<Real>,Conjugated>& A,
  std::complex<Real> beta,  const FactorMatrix<std::complex<Real>,Conjugated>& B,
                                  FactorMatrix<std::complex<Real>,Conjugated>& C 
);

/*\
|*| Generalized update of a factor matrix, B := alpha A + beta B, 
|*| where B is then forced to be of rank at most 'maxRank'
\*/
template<typename Real,bool Conjugated>
void MatrixUpdateRounded
( int maxRank,
  Real alpha, const FactorMatrix<Real,Conjugated>& A,
  Real beta,        FactorMatrix<Real,Conjugated>& B );
template<typename Real,bool Conjugated>
void MatrixUpdateRounded
( int maxRank,
  std::complex<Real> alpha, const FactorMatrix<std::complex<Real>,Conjugated>& A,
  std::complex<Real> beta,        FactorMatrix<std::complex<Real>,Conjugated>& B
);

/*\
|*| Matrix Matrix multiply, C := alpha A B
|*|
|*| When the resulting matrix is dense, an update form is also provided, i.e.,
|*| C := alpha A B + beta C
|*|
|*| A routine for forming a factor matrix from the product of two black-box 
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
// F := alpha F F
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const FactorMatrix<Scalar,Conjugated>& B,
                      FactorMatrix<Scalar,Conjugated>& C );
// D := alpha D F
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D F + beta D
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha F D
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F D + beta D
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// F := alpha D F
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B,
                      FactorMatrix<Scalar,Conjugated>& C );
// F := alpha F D
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
                      FactorMatrix<Scalar,Conjugated>& C );
// F := alpha H H,
template<typename Scalar,bool Conjugated>
void MatrixMatrix
( int maxRank, int oversampling,
  Scalar alpha, const AbstractHMatrix<Scalar>& A,
                const AbstractHMatrix<Scalar>& B,
                      FactorMatrix<Scalar,Conjugated>& F );

/*\
|*| Matrix Transpose Matrix Multiply, C := alpha A^T B
|*|
|*| When the resulting matrix is dense, an update form is also provided, i.e.,
|*| C := alpha A^T B + beta C
|*|
|*| A routine for forming a factor matrix from the product of two black-box 
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
// F := alpha F^T F
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const FactorMatrix<Scalar,Conjugated>& B,
                      FactorMatrix<Scalar,Conjugated>& C );
// D := alpha D^T F
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D^T F + beta D
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha F^T D
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F^T D + beta D
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// F := alpha D^T F
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B,
                      FactorMatrix<Scalar,Conjugated>& C );
// F := alpha F^T D
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
                      FactorMatrix<Scalar,Conjugated>& C );
// F := alpha H^T H
template<typename Scalar,bool Conjugated>
void MatrixTransposeMatrix
( int maxRank, int oversampling,
  Scalar alpha, const AbstractHMatrix<Scalar>& A,
                const AbstractHMatrix<Scalar>& B,
                      FactorMatrix<Scalar,Conjugated>& F );

/*\
|*| Matrix Hermitian Transpose Matrix Multiply, C := alpha A^H B
|*|
|*| When the resulting matrix is dense, an update form is also provided, i.e.,
|*| C := alpha A^H B + beta C
|*|
|*| A routine for forming a factor matrix from the product of two black-box 
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
// F := alpha F^H F
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const FactorMatrix<Scalar,Conjugated>& B,
                      FactorMatrix<Scalar,Conjugated>& C );
// D := alpha D^H F
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D^H F + beta D
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha F^H D
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F^H D + beta D
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// F := alpha D^H F
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B,
                      FactorMatrix<Scalar,Conjugated>& C );
// F := alpha F^H D
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B,
                      FactorMatrix<Scalar,Conjugated>& C );
// F := alpha H^H H
template<typename Scalar,bool Conjugated>
void MatrixHermitianTransposeMatrix
( int maxRank, int oversampling,
  Scalar alpha, const AbstractHMatrix<Scalar>& A,
                const AbstractHMatrix<Scalar>& B,
                      FactorMatrix<Scalar,Conjugated>& F );

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
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& F, 
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
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& F, 
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
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& F, 
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
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& F, 
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
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& F, 
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
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& F, 
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
void Scale( Scalar alpha, FactorMatrix<Scalar,Conjugated>& F );

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
void Conjugate( FactorMatrix<Real,Conjugated>& F );
template<typename Real,bool Conjugated>
void Conjugate( FactorMatrix<std::complex<Real>,Conjugated>& F );

template<typename Real,bool Conjugated>
void Conjugate
( const FactorMatrix<Real,Conjugated>& F1,
        FactorMatrix<Real,Conjugated>& F2 );
template<typename Real,bool Conjugated>
void Conjugate
( const FactorMatrix<std::complex<Real>,Conjugated>& F1,
        FactorMatrix<std::complex<Real>,Conjugated>& F2 );

/*\
|*| For mapping between different orderings 
\*/
void BuildNaturalToHierarchicalQuasi2dMap
( std::vector<int>& map, int numLevels, int xSize, int ySize, int zSize );

void InvertMap
(       std::vector<int>& invertedMap,
  const std::vector<int>& originalMap );

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
|*| Conjugate a vector or matrix
\*/

template<typename Real> 
void psp::hmatrix_tools::Conjugate
( Vector<Real>& x ) 
{ }

template<typename Real>
void psp::hmatrix_tools::Conjugate
( Vector< std::complex<Real> >& x )
{
    const int n = x.Size();
    std::complex<Real>* xBuffer = x.Buffer();
    for( int i=0; i<n; ++i ) 
        xBuffer[i] = Conj( xBuffer[i] );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const Vector<Real>& x,
        Vector<Real>& y )
{ 
    y.Resize( x.Size() );
    std::memcpy( y.Buffer(), x.LockedBuffer(), x.Size()*sizeof(Real) );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const Vector< std::complex<Real> >& x, 
        Vector< std::complex<Real> >& y )
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
( const Vector<Real>& x,
        std::vector<Real>& y )
{
    y.resize( x.Size() );
    std::memcpy( &y[0], x.Buffer(), x.Size()*sizeof(Real) );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const Vector< std::complex<Real> >& x,
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
        Vector<Real>& y )
{
    y.Resize( x.size() );
    std::memcpy( y.Buffer(), &x[0], x.size()*sizeof(Real) );
}

template<typename Real>
void psp::hmatrix_tools::Conjugate
( const std::vector< std::complex<Real> >& x,
        Vector< std::complex<Real> >& y )
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
( DenseMatrix<Real>& D )
{ }

template<typename Real>
void psp::hmatrix_tools::Conjugate
( DenseMatrix< std::complex<Real> >& D )
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
( const DenseMatrix<Real>& D1, 
        DenseMatrix<Real>& D2 )
{
    const int m = D1.Height();
    const int n = D1.Width();
    D2.Resize( m, n );
    D2.SetType( D1.Type() ); 
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
( const DenseMatrix< std::complex<Real> >& D1,
        DenseMatrix< std::complex<Real> >& D2 )
{
    const int m = D1.Height();
    const int n = D1.Width();
    D2.Resize( m, n );
    D2.SetType( D1.Type() );
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
( FactorMatrix<Real,Conjugated>& F )
{ }

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::Conjugate
( FactorMatrix<std::complex<Real>,Conjugated>& F )
{
    const int m = F.m;
    const int n = F.n;
    const int r = F.r;
    std::complex<Real>* FU = &F.U[0];
    for( int i=0; i<m*r; ++i )
        FU[i] = Conj( FU[i] );
    std::complex<Real>* FV = &F.V[0];
    for( int i=0; i<n*r; ++i )
        FV[i] = Conj( FV[i] );
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::Conjugate
( const FactorMatrix<Real,Conjugated>& F1,
        FactorMatrix<Real,Conjugated>& F2 )
{
    const int m = F1.m;
    const int n = F1.n;
    const int r = F1.r;
    F2.m = m;
    F2.n = n;
    F2.r = r;
    F2.U.resize( m*r );
    F2.V.resize( n*r );
    std::memcpy( &F2.U[0], &F1.U[0], m*r*sizeof(Real) );
    std::memcpy( &F2.V[0], &F1.V[0], n*r*sizeof(Real) );
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::Conjugate
( const FactorMatrix<std::complex<Real>,Conjugated>& F1,
        FactorMatrix<std::complex<Real>,Conjugated>& F2 )
{
    const int m = F1.m;
    const int n = F1.n;
    const int r = F1.r;
    F2.m = m;
    F2.n = n;
    F2.r = r;
    F2.U.resize( m*r );
    F2.V.resize( n*r );
    const std::complex<Real>* RESTRICT F1U = &F1.U[0];
    std::complex<Real>* RESTRICT F2U = &F2.U[0];
    for( int i=0; i<m*r; ++i )
        F2U[i] = Conj( F1U[i] );
    const std::complex<Real>* RESTRICT F1V = &F1.V[0];
    std::complex<Real>* RESTRICT F2V = &F2.V[0];
    for( int i=0; i<n*r; ++i )
        F2V[i] = Conj( F1V[i] );
}

/*\
|*| For scaling vectors and matrices
\*/

template<typename Scalar>
void psp::hmatrix_tools::Scale
( Scalar alpha, Vector<Scalar>& x )
{
    if( alpha == (Scalar)0 )
        std::memset( x.Buffer(), 0, x.Size()*sizeof(Scalar) );
    else
        blas::Scal( x.Size(), alpha, x.Buffer(), 1 );
}

template<typename Scalar>
void psp::hmatrix_tools::Scale
( Scalar alpha, DenseMatrix<Scalar>& D )
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

template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::Scale
( Scalar alpha, FactorMatrix<Scalar,Conjugate>& F )
{
    if( alpha == (Scalar)0 )
    {
        F.r = 0;
        F.U.resize( 0 );
        F.V.resize( 0 );
    }
    else
    {
        blas::Scal( F.m*F.r, alpha, &F.U[0], 1 );
    }
}

/*\
|*| For forming low-rank approximations to the product of H-matrices
\*/

// F := alpha H H,
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( int maxRank, int oversampling, 
  Scalar alpha, const AbstractHMatrix<Scalar>& A, 
                const AbstractHMatrix<Scalar>& B,
                      FactorMatrix<Scalar,Conjugated>& F )
{
    // TODO
}

// F := alpha H^T H,
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixTransposeMatrix
( int maxRank, int oversampling,
  Scalar alpha, const AbstractHMatrix<Scalar>& A,
                const AbstractHMatrix<Scalar>& B,
                      FactorMatrix<Scalar,Conjugated>& F )
{
    // TODO
}

// F := alpha H^H H,
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, int oversampling, 
  Scalar alpha, const AbstractHMatrix<Scalar>& A,
                const AbstractHMatrix<Scalar>& B,
                      FactorMatrix<Scalar,Conjugated>& F )
{
    // TODO
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
( Vector<Real>& x )
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
( Vector< std::complex<Real> >& x )
{
    const int n = x.size();
    std::complex<Real>* buffer = x.Buffer();
    for( int i=0; i<n; ++i )
        GaussianRandomVariable( buffer[i] );
}

template<typename Real>
void
psp::hmatrix_tools::GaussianRandomVectors
( DenseMatrix<Real>& A )
{
    // Use BoxMuller for every pair of entries in each column
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
( DenseMatrix< std::complex<Real> >& A )
{
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
