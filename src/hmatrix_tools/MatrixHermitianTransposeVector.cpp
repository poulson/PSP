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

// Dense y := alpha A^H x + beta y
template<typename Scalar>
void psp::hmatrix_tools::MatrixHermitianTransposeVector
( Scalar alpha, const DenseMatrix<Scalar>& A,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
    if( A.Symmetric() )
    {
        // Form conj(alpha x) in a buffer
        const int n = x.Size();
        std::vector<Scalar> xConj(n);
        Scalar* xConjBuffer = &xConj[0];
        const Scalar* xBuffer = x.LockedBuffer();
        for( int i=0; i<n; ++i )
            xConjBuffer[i] = Conj( alpha*xBuffer[i] );

        // Form y := conj(beta y)
        const int m = y.Size();
        Scalar* yBuffer = y.Buffer();
        for( int i=0; i<m; ++i )
            yBuffer[i] = Conj( beta*yBuffer[i] );

        // Form y := A x + y
        blas::Symv
        ( 'L', A.Height(), 
          1, A.LockedBuffer(), A.LDim(), 
             xConjBuffer,      1, 
          1, y.Buffer(),       1 );

        // Form y := conj(y)
        for( int i=0; i<m; ++i )
            yBuffer[i] = Conj( yBuffer[i] );
    }
    else
    {
        blas::Gemv
        ( 'C', A.Height(), A.Width(), 
          alpha, A.LockedBuffer(), A.LDim(), 
                 x.LockedBuffer(), 1, 
          beta,  y.Buffer(),       1 );
    }
}

// Dense y := alpha A^H x
template<typename Scalar>
void psp::hmatrix_tools::MatrixHermitianTransposeVector
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
    y.Resize( x.Size() );
    if( A.Symmetric() )
    {
        // Form conj(alpha x) in a buffer
        const int n = x.Size();
        std::vector<Scalar> xConj(n);
        Scalar* xConjBuffer = &xConj[0];
        const Scalar* xBuffer = x.LockedBuffer();
        for( int i=0; i<n; ++i )
            xConjBuffer[i] = Conj( alpha*xBuffer[i] );

        // Form y := A x
        blas::Symv
        ( 'L', A.Height(), 
          1, A.LockedBuffer(), A.LDim(), 
             xConjBuffer,      1, 
          0, y.Buffer(),       1 );

        // Conjugate y
        const int m = y.Size();
        Scalar* yBuffer = y.Buffer();
        for( int i=0; i<m; ++i )
            yBuffer[i] = Conj( yBuffer[i] );
    }
    else
    {
        blas::Gemv
        ( 'C', A.Height(), A.Width(), 
          alpha, A.LockedBuffer(), A.LDim(), 
                 x.LockedBuffer(), 1, 
          0,     y.Buffer(),       1 );
    }
}

// Low-rank y := alpha A^H x + beta y
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixHermitianTransposeVector
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
    const int r = A.r;
    std::vector<Scalar> t(r);

    // Form t := alpha (A.U)^H x
    blas::Gemv
    ( 'C', A.m, A.r, alpha, &A.U[0], A.m, x.LockedBuffer(), 1, 0, &t[0], 1 );

    if( Conjugate )
    {
        // Form y := (A.V) t + beta y
        blas::Gemv
        ( 'N', A.n, A.r, 1, &A.V[0], A.n, &t[0], 1, beta, y.Buffer(), 1 );
    }
    else
    {
        // t := conj(t)
        for( int i=0; i<r; ++i )
            t[i] = Conj( t[i] );

        // y := conj(beta y)
        const int n = A.n;
        Scalar* yBuffer = y.Buffer();
        for( int i=0; i<n; ++i )
            yBuffer[i] = Conj( beta*yBuffer[i] );

        // Form y := A.V t + y
        blas::Gemv
        ( 'N', A.n, A.r, 1, &A.V[0], A.n, &t[0], 1, 1, y.Buffer(), 1 );

        // y := conj(y)
        for( int i=0; i<n; ++i )
            yBuffer[i] = Conj( yBuffer[i] ); 
    }
}

// Low-rank y := alpha A^H x
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixHermitianTransposeVector
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
    const int r = A.r;
    std::vector<Scalar> t(r);

    // Form t := alpha (A.U)^H x
    blas::Gemv
    ( 'C', A.m, A.r, alpha, &A.U[0], A.m, x.LockedBuffer(), 1, 0, &t[0], 1 );

    y.Resize( x.Size() );
    if( Conjugate )
    {
        // Form y := (A.V) t
        blas::Gemv
        ( 'N', A.n, A.r, 1, &A.V[0], A.n, &t[0], 1, 0, y.Buffer(), 1 );
    }
    else
    {
        // t := conj(t)
        for( int i=0; i<r; ++i )
            t[i] = Conj( t[i] );

        // Form y := A.V t
        blas::Gemv
        ( 'N', A.n, A.r, 1, &A.V[0], A.n, &t[0], 1, 0, y.Buffer(), 1 );

        // y := conj(y)
        const int n = A.n;
        Scalar* yBuffer = y.Buffer();
        for( int i=0; i<n; ++i )
            yBuffer[i] = Conj( yBuffer[i] );
    }
}

template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( float alpha, const DenseMatrix<float>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( double alpha, const DenseMatrix<double>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const Vector< std::complex<float> >& x,
  std::complex<float> beta,        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const Vector< std::complex<double> >& x,
  std::complex<double> beta,        Vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( float alpha, const DenseMatrix<float>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( double alpha, const DenseMatrix<double>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const Vector< std::complex<float> >& x,
                                   Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const Vector< std::complex<double> >& x,
                                    Vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( float alpha, const FactorMatrix<float,false>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( float alpha, const FactorMatrix<float,true>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( double alpha, const FactorMatrix<double,false>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( double alpha, const FactorMatrix<double,true>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const Vector< std::complex<float> >& x,
  std::complex<float> beta,        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const Vector< std::complex<float> >& x,
  std::complex<float> beta,        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const Vector< std::complex<double> >& x,
  std::complex<double> beta,        Vector< std::complex<double> >& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const Vector< std::complex<double> >& x,
  std::complex<double> beta,        Vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( float alpha, const FactorMatrix<float,false>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( float alpha, const FactorMatrix<float,true>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( double alpha, const FactorMatrix<double,false>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( double alpha, const FactorMatrix<double,true>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const Vector< std::complex<float> >& x,
                                   Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const Vector< std::complex<float> >& x,
                                   Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const Vector< std::complex<double> >& x,
                                    Vector< std::complex<double> >& y );
template void psp::hmatrix_tools::MatrixHermitianTransposeVector
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const Vector< std::complex<double> >& x,
                                    Vector< std::complex<double> >& y );
