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
#include "psp.hpp"

// Dense y := alpha A x + beta y
template<typename Scalar>
void psp::hmatrix_tools::MatrixVector
( Scalar alpha, const DenseMatrix<Scalar>& A,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
    if( A.Symmetric() )
    {
        blas::Symv
        ( 'L', A.Height(), 
          alpha, A.LockedBuffer(), A.LDim(), 
                 x.LockedBuffer(), 1, 
          beta,  y.Buffer(),       1 );
    }
    else
    {
        blas::Gemv
        ( 'N', A.Height(), A.Width(), 
          alpha, A.LockedBuffer(), A.LDim(), 
                 x.LockedBuffer(), 1, 
          beta,  y.Buffer(),       1 );
    }
}

// Dense y := alpha A x
template<typename Scalar>
void psp::hmatrix_tools::MatrixVector
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
    y.Resize( x.Size() );
    if( A.Symmetric() )
    {
        blas::Symv
        ( 'L', A.Height(), 
          alpha, A.LockedBuffer(), A.LDim(), 
                 x.LockedBuffer(), 1, 
          0,     y.Buffer(),       1 );
    }
    else
    {
        blas::Gemv
        ( 'N', A.Height(), A.Width(), 
          alpha, A.LockedBuffer(), A.LDim(), 
                 x.LockedBuffer(), 1, 
          0,     y.Buffer(),       1 );
    }
}

// Low-rank y := alpha A x + beta y
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixVector
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
    const int r = A.r;
    std::vector<Scalar> t(r);

    // Form t := alpha (A.V)^[T,H] x
    const char option = ( Conjugate ? 'C' : 'T' );
    blas::Gemv
    ( option, A.n, A.r, alpha, &A.V[0], A.n, x.LockedBuffer(), 1, 0, &t[0], 1 );

    // Form y := (A.U) t + beta y
    blas::Gemv
    ( 'N', A.m, A.r, 1, &A.U[0], A.m, &t[0], 1, beta, y.Buffer(), 1 );
}

// Low-rank y := alpha A x
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixVector
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
    const int r = A.r;
    std::vector<Scalar> t(r);

    // Form t := alpha (A.V)^[T,H] x
    const char option = ( Conjugate ? 'C' : 'T' );
    blas::Gemv
    ( option, A.n, A.r, alpha, &A.V[0], A.n, x.LockedBuffer(), 1, 0, &t[0], 1 );

    // Form y := (A.U) t
    y.Resize( x.Size() );
    blas::Gemv
    ( 'N', A.m, A.r, 1, &A.U[0], A.m, &t[0], 1, 0, y.Buffer(), 1 );
}

template void psp::hmatrix_tools::MatrixVector
( float alpha, const DenseMatrix<float>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void psp::hmatrix_tools::MatrixVector
( double alpha, const DenseMatrix<double>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const Vector< std::complex<float> >& x,
  std::complex<float> beta,        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const Vector< std::complex<double> >& x,
  std::complex<double> beta,        Vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixVector
( float alpha, const DenseMatrix<float>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void psp::hmatrix_tools::MatrixVector
( double alpha, const DenseMatrix<double>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const Vector< std::complex<float> >& x,
                                   Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const Vector< std::complex<double> >& x,
                                    Vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixVector
( float alpha, const FactorMatrix<float,false>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void psp::hmatrix_tools::MatrixVector
( float alpha, const FactorMatrix<float,true>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void psp::hmatrix_tools::MatrixVector
( double alpha, const FactorMatrix<double,false>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void psp::hmatrix_tools::MatrixVector
( double alpha, const FactorMatrix<double,true>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const Vector< std::complex<float> >& x,
  std::complex<float> beta,        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const Vector< std::complex<float> >& x,
  std::complex<float> beta,        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const Vector< std::complex<double> >& x,
  std::complex<double> beta,        Vector< std::complex<double> >& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const Vector< std::complex<double> >& x,
  std::complex<double> beta,        Vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixVector
( float alpha, const FactorMatrix<float,false>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void psp::hmatrix_tools::MatrixVector
( float alpha, const FactorMatrix<float,true>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void psp::hmatrix_tools::MatrixVector
( double alpha, const FactorMatrix<double,false>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void psp::hmatrix_tools::MatrixVector
( double alpha, const FactorMatrix<double,true>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const Vector< std::complex<float> >& x,
                                   Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const Vector< std::complex<float> >& x,
                                   Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const Vector< std::complex<double> >& x,
                                    Vector< std::complex<double> >& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const Vector< std::complex<double> >& x,
                                    Vector< std::complex<double> >& y );

