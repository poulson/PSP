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

// Dense y := alpha A^T x + beta y
template<typename Scalar>
void psp::hmatrix_tools::MatrixTransposeVector
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
        ( 'T', A.Height(), A.Width(), 
          alpha, A.LockedBuffer(), A.LDim(), 
                 x.LockedBuffer(), 1, 
          beta,  y.Buffer(),       1 );
    }
}

// Dense y := alpha A^T x
template<typename Scalar>
void psp::hmatrix_tools::MatrixTransposeVector
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
        ( 'T', A.Height(), A.Width(), 
          alpha, A.LockedBuffer(), A.LDim(), 
                 x.LockedBuffer(), 1, 
          0,     y.Buffer(),       1 );
    }
}

// Low-rank y := alpha A^T x + beta y
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixTransposeVector
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
    const int r = A.r;
    std::vector<Scalar> t(r);

    // Form t := alpha (A.U)^T x
    blas::Gemv
    ( 'T', A.m, A.r, alpha, &A.U[0], A.m, x.LockedBuffer(), 1, 0, &t[0], 1 );

    if( Conjugated )
    {
        Conjugate( t );
        Conjugate( y );
        blas::Gemv
        ( 'N', A.n, A.r, 1, &A.V[0], A.n, &t[0], 1, Conj(beta), y.Buffer(), 1 );
        Conjugate( y );
    }
    else
    {
        // Form y := (A.V) t + beta y
        blas::Gemv
        ( 'N', A.n, A.r, 1, &A.V[0], A.n, &t[0], 1, beta, y.Buffer(), 1 );
    }
}

// Low-rank y := alpha A^T x
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixTransposeVector
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
    const int r = A.r;
    std::vector<Scalar> t(r);

    // Form t := alpha (A.U)^T x
    blas::Gemv
    ( 'T', A.m, A.r, alpha, &A.U[0], A.m, x.LockedBuffer(), 1, 0, &t[0], 1 );

    y.Resize( x.Size() );
    if( Conjugated )
    {
        Conjugate( t );
        blas::Gemv
        ( 'N', A.n, A.r, 1, &A.V[0], A.n, &t[0], 1, 0, y.Buffer(), 1 );
        Conjugate( y );
    }
    else
    {
        // Form y := (A.V) t
        blas::Gemv
        ( 'N', A.n, A.r, 1, &A.V[0], A.n, &t[0], 1, 0, y.Buffer(), 1 );
    }
}

template void psp::hmatrix_tools::MatrixTransposeVector
( float alpha, const DenseMatrix<float>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( double alpha, const DenseMatrix<double>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const Vector< std::complex<float> >& x,
  std::complex<float> beta,        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const Vector< std::complex<double> >& x,
  std::complex<double> beta,        Vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixTransposeVector
( float alpha, const DenseMatrix<float>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( double alpha, const DenseMatrix<double>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const Vector< std::complex<float> >& x,
                                   Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const Vector< std::complex<double> >& x,
                                    Vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixTransposeVector
( float alpha, const FactorMatrix<float,false>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( float alpha, const FactorMatrix<float,true>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( double alpha, const FactorMatrix<double,false>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( double alpha, const FactorMatrix<double,true>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const Vector< std::complex<float> >& x,
  std::complex<float> beta,        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const Vector< std::complex<float> >& x,
  std::complex<float> beta,        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const Vector< std::complex<double> >& x,
  std::complex<double> beta,        Vector< std::complex<double> >& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const Vector< std::complex<double> >& x,
  std::complex<double> beta,        Vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixTransposeVector
( float alpha, const FactorMatrix<float,false>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( float alpha, const FactorMatrix<float,true>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( double alpha, const FactorMatrix<double,false>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( double alpha, const FactorMatrix<double,true>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const Vector< std::complex<float> >& x,
                                   Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const Vector< std::complex<float> >& x,
                                   Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const Vector< std::complex<double> >& x,
                                    Vector< std::complex<double> >& y );
template void psp::hmatrix_tools::MatrixTransposeVector
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const Vector< std::complex<double> >& x,
                                    Vector< std::complex<double> >& y );

