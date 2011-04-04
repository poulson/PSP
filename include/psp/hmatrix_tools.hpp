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
template<typename Real,bool Conjugate>
void Compress( int maxRank, FactorMatrix<Real,Conjugate>& F );
template<typename Real,bool Conjugate>
void Compress( int maxRank, FactorMatrix<std::complex<Real>,Conjugate>& F );

/*\
|*| Convert a subset of a sparse matrix to dense/factor form
\*/
template<typename Scalar>
void ConvertSubmatrix
( DenseMatrix<Scalar>& D, const SparseMatrix<Scalar>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template<typename Scalar,bool Conjugate>
void ConvertSubmatrix
( FactorMatrix<Scalar,Conjugate>& F, const SparseMatrix<Scalar>& S,
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
template<typename Scalar,bool Conjugate>
void MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A,
  Scalar beta,  const FactorMatrix<Scalar,Conjugate>& B,
                      FactorMatrix<Scalar,Conjugate>& C );
// D := alpha F + beta D
template<typename Scalar,bool Conjugate>
void MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A,
  Scalar beta,  const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D + beta F
template<typename Scalar,bool Conjugate>
void MatrixAdd
( Scalar alpha, const DenseMatrix<Scalar>& A,
  Scalar beta,  const FactorMatrix<Scalar,Conjugate>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F + beta F
template<typename Scalar,bool Conjugate>
void MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A,
  Scalar beta,  const FactorMatrix<Scalar,Conjugate>& B,
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
template<typename Scalar,bool Conjugate>
void MatrixUpdate
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A,
  Scalar beta,        FactorMatrix<Scalar,Conjugate>& B );
// D := alpha F + beta D
template<typename Scalar,bool Conjugate>
void MatrixUpdate
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A,
  Scalar beta,        DenseMatrix<Scalar>& B );

/*\
|*| Generalized add of two factor matrices, C := alpha A + beta B, 
|*| where C is then forced to be of rank at most 'maxRank'
\*/
template<typename Real,bool Conjugate>
void MatrixAddRounded
( int maxRank,
  Real alpha, const FactorMatrix<Real,Conjugate>& A,
  Real beta,  const FactorMatrix<Real,Conjugate>& B,
                    FactorMatrix<Real,Conjugate>& C );
template<typename Real,bool Conjugate>
void MatrixAddRounded
( int maxRank,
  std::complex<Real> alpha, const FactorMatrix<std::complex<Real>,Conjugate>& A,
  std::complex<Real> beta,  const FactorMatrix<std::complex<Real>,Conjugate>& B,
                                  FactorMatrix<std::complex<Real>,Conjugate>& C 
);

/*\
|*| Generalized update of a factor matrix, B := alpha A + beta B, 
|*| where B is then forced to be of rank at most 'maxRank'
\*/
template<typename Real,bool Conjugate>
void MatrixUpdateRounded
( int maxRank,
  Real alpha, const FactorMatrix<Real,Conjugate>& A,
  Real beta,        FactorMatrix<Real,Conjugate>& B );
template<typename Real,bool Conjugate>
void MatrixUpdateRounded
( int maxRank,
  std::complex<Real> alpha, const FactorMatrix<std::complex<Real>,Conjugate>& A,
  std::complex<Real> beta,        FactorMatrix<std::complex<Real>,Conjugate>& B
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
template<typename Scalar,bool Conjugate>
void MatrixMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const FactorMatrix<Scalar,Conjugate>& B,
                      FactorMatrix<Scalar,Conjugate>& C );
// D := alpha D F
template<typename Scalar,bool Conjugate>
void MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D F + beta D
template<typename Scalar,bool Conjugate>
void MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha F D
template<typename Scalar,bool Conjugate>
void MatrixMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F D + beta D
template<typename Scalar,bool Conjugate>
void MatrixMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// F := alpha D F
template<typename Scalar,bool Conjugate>
void MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B,
                      FactorMatrix<Scalar,Conjugate>& C );
// F := alpha F D
template<typename Scalar,bool Conjugate>
void MatrixMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B,
                      FactorMatrix<Scalar,Conjugate>& C );
// F := alpha H H,
// where A( const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Y ) 
// forms y := A X, 
//       AH( const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Y ) 
// forms y := A^H X,
// etc.
template<typename Scalar,bool Conjugate,class A,class AT,class B,class BT>
void MatrixMatrix
( Scalar alpha, FactorMatrix<Scalar,Conjugate>& F );

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
template<typename Scalar,bool Conjugate>
void MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const FactorMatrix<Scalar,Conjugate>& B,
                      FactorMatrix<Scalar,Conjugate>& C );
// D := alpha D^T F
template<typename Scalar,bool Conjugate>
void MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D^T F + beta D
template<typename Scalar,bool Conjugate>
void MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha F^T D
template<typename Scalar,bool Conjugate>
void MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F^T D + beta D
template<typename Scalar,bool Conjugate>
void MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// F := alpha D^T F
template<typename Scalar,bool Conjugate>
void MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B,
                      FactorMatrix<Scalar,Conjugate>& C );
// F := alpha F^T D
template<typename Scalar,bool Conjugate>
void MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B,
                      FactorMatrix<Scalar,Conjugate>& C );
// F := alpha H^T H,
// where A( const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Y ) 
// forms y := A X, 
//       AH( const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Y ) 
// forms y := A^H X,
// etc.
template<typename Scalar,bool Conjugate,class A,class AT,class B,class BT>
void MatrixTransposeMatrix
( Scalar alpha, FactorMatrix<Scalar,Conjugate>& F );

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
template<typename Scalar,bool Conjugate>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const FactorMatrix<Scalar,Conjugate>& B,
                      FactorMatrix<Scalar,Conjugate>& C );
// D := alpha D^H F
template<typename Scalar,bool Conjugate>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha D^H F + beta D
template<typename Scalar,bool Conjugate>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// D := alpha F^H D
template<typename Scalar,bool Conjugate>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C );
// D := alpha F^H D + beta D
template<typename Scalar,bool Conjugate>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B,
  Scalar beta,        DenseMatrix<Scalar>& C );
// F := alpha D^H F
template<typename Scalar,bool Conjugate>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B,
                      FactorMatrix<Scalar,Conjugate>& C );
// F := alpha F^H D
template<typename Scalar,bool Conjugate>
void MatrixHermitianTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B,
                      FactorMatrix<Scalar,Conjugate>& C );
// F := alpha H^H H,
// where A( const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Y ) 
// forms y := A X, 
//       AH( const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Y ) 
// forms y := A^H X,
// etc.
template<typename Scalar,bool Conjugate,class A,class AT,class B,class BT>
void MatrixHermitianTransposeMatrix
( Scalar alpha, FactorMatrix<Scalar,Conjugate>& F );

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
template<typename Scalar,bool Conjugate>
void MatrixVector
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& F, 
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
template<typename Scalar,bool Conjugate>
void MatrixVector
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& F, 
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
template<typename Scalar,bool Conjugate>
void MatrixTransposeVector
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& F, 
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
template<typename Scalar,bool Conjugate>
void MatrixTransposeVector
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& F, 
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
template<typename Scalar,bool Conjugate>
void MatrixHermitianTransposeVector
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& F, 
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
template<typename Scalar,bool Conjugate>
void MatrixHermitianTransposeVector
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& F, 
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

template<typename Scalar,bool Conjugate>
void Scale( Scalar alpha, FactorMatrix<Scalar,Conjugate>& F );

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
GaussianRandomVariable( Real& X );
template<typename Real>
GaussianRandomVariable( std::complex<Real>& X );
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
// Abstract implementations                                                   //
//----------------------------------------------------------------------------//

// F := alpha H H,
// where A( const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Y ) 
// forms y := A X, 
//       AH( const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Y ) 
// forms y := A^H X,
// etc.
template<typename Scalar,bool Conjugate,class A,class AH,class B,class BH>
void MatrixMatrix
( Scalar alpha, FactorMatrix<Scalar,Conjugate>& F )
{
    // TODO
}

// F := alpha H^T H,
// where A( const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Y ) 
// forms y := A X, 
//       AH( const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Y ) 
// forms y := A^H X,
// etc.
template<typename Scalar,bool Conjugate,class A,class AH,class B,class BH>
void MatrixTransposeMatrix
( Scalar alpha, FactorMatrix<Scalar,Conjugate>& F )
{
    // TODO
}

// F := alpha H^H H,
// where A( const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Y ) 
// forms y := A X, 
//       AH( const DenseMatrix<Scalar>& X, DenseMatrix<Scalar>& Y ) 
// forms y := A^H X,
// etc.
template<typename Scalar,bool Conjugate,class A,class AH,class B,class BH>
void MatrixHermitianTransposeMatrix
( Scalar alpha, FactorMatrix<Scalar,Conjugate>& F )
{
    // TODO
}

// Return a uniform sample from [0,1]
template<typename Real>
inline void
Uniform( Real& U )
{
    U = static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX);
}

template<typename Real>
inline void
BoxMuller( Real& X, Real& Y )
{
    Real U, V;
    Uniform( U );
    Uniform( V );
    const Real a = sqrt(-2*log(U));
    const Real c = cos(2*M_PI*V);
    const Real s = sin(2*M_PI*V);
    X = A*c;
    Y = A*s;
}

template<typename Real>
inline void
GaussianRandomVariable( Real& X )
{
    // Use half of Box-Muller
    Real U;
    Uniform( U );
    X = sqrt(-2*log(U)) * cos(2*M_PI*V);
}

template<typename Real>
inline void
GaussianRandomVariable( std::complex<Real>& X )
{
    Real Y, Z;
    BoxMuller( Y, Z );
    X = std::complex<Real>( Y, Z );
}

template<typename Real>
void
GaussianRandomVector( Vector<Real>& x )
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
GaussianRandomVector( Vector< std::complex<Real> >& x )
{
    const int n = x.size();
    std::complex<Real>* buffer = x.Buffer();
    for( int i=0; i<n; ++i )
        GaussianRandomVariable( buffer[i] );
}

template<typename Real>
void
GaussianRandomVectors( DenseMatrix<Real>& A )
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
GaussianRandomVectors( DenseMatrix< std::complex<Real> >& A )
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
