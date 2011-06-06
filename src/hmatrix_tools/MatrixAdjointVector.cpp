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
void psp::hmatrix_tools::MatrixAdjointVector
( Scalar alpha, const DenseMatrix<Scalar>& A,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixAdjointVector (y := D^H x + y)");
#endif
    if( A.Symmetric() )
    {
        Vector<Scalar> xConj;
        Conjugate( x, xConj );
        Conjugate( y );
        blas::Symv
        ( 'L', A.Height(), 
          Conj(alpha), A.LockedBuffer(), A.LDim(), 
                       xConj.Buffer(),   1, 
          Conj(beta),  y.Buffer(),       1 );
        Conjugate( y );
    }
    else
    {
        blas::Gemv
        ( 'C', A.Height(), A.Width(), 
          alpha, A.LockedBuffer(), A.LDim(), 
                 x.LockedBuffer(), 1, 
          beta,  y.Buffer(),       1 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Dense y := alpha A^H x
template<typename Scalar>
void psp::hmatrix_tools::MatrixAdjointVector
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixAdjointVector (y := D^H x)");
#endif
    y.Resize( A.Width() );
    if( A.Symmetric() )
    {
        Vector<Scalar> xConj;
        Conjugate( x, xConj );
        blas::Symv
        ( 'L', A.Height(), 
          Conj(alpha), A.LockedBuffer(), A.LDim(), 
                       xConj.Buffer(),   1, 
          0,           y.Buffer(),       1 );
        Conjugate( y );
    }
    else
    {
        blas::Gemv
        ( 'C', A.Height(), A.Width(), 
          alpha, A.LockedBuffer(), A.LDim(), 
                 x.LockedBuffer(), 1, 
          0,     y.Buffer(),       1 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Low-rank y := alpha A^H x + beta y
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixAdjointVector
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixAdjointVector (y := F x + y)");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int r = A.Rank();

    // Form t := alpha (A.U)^H x
    Vector<Scalar> t( r );
    blas::Gemv
    ( 'C', m, r, 
      alpha, A.U.LockedBuffer(), A.U.LDim(), 
             x.LockedBuffer(),   1, 
      0,     t.Buffer(),         1 );

    if( Conjugated )
    {
        // Form y := (A.V) t + beta y
        blas::Gemv
        ( 'N', n, r, 
          1,    A.V.LockedBuffer(), A.V.LDim(), 
                t.LockedBuffer(),   1, 
          beta, y.Buffer(),         1 );
    }
    else
    {
        Conjugate( t );
        Conjugate( y );
        blas::Gemv
        ( 'N', n, r, 
          1,          A.V.LockedBuffer(), A.V.LDim(), 
                      t.LockedBuffer(),   1, 
          Conj(beta), y.Buffer(),         1 );
        Conjugate( y );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Low-rank y := alpha A^H x
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixAdjointVector
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixAdjointVector (y := F x)");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int r = A.Rank();

    // Form t := alpha (A.U)^H x
    Vector<Scalar> t( r );
    blas::Gemv
    ( 'C', m, r, 
      alpha, A.U.LockedBuffer(), A.U.LDim(), 
             x.LockedBuffer(),   1, 
      0,     t.Buffer(),         1 );

    y.Resize( n );
    if( Conjugated )
    {
        // Form y := (A.V) t
        blas::Gemv
        ( 'N', n, r, 
          1, A.V.LockedBuffer(), A.V.LDim(), 
             t.LockedBuffer(),   1, 
          0, y.Buffer(),         1 );
    }
    else
    {
        Conjugate( t );
        blas::Gemv
        ( 'N', n, r, 
          1, A.V.LockedBuffer(), A.V.LDim(), 
             t.LockedBuffer(),   1, 
          0, y.Buffer(),         1 );
        Conjugate( y );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template void psp::hmatrix_tools::MatrixAdjointVector
( float alpha, const DenseMatrix<float>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( double alpha, const DenseMatrix<double>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const Vector< std::complex<float> >& x,
  std::complex<float> beta,        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const Vector< std::complex<double> >& x,
  std::complex<double> beta,        Vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixAdjointVector
( float alpha, const DenseMatrix<float>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( double alpha, const DenseMatrix<double>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const Vector< std::complex<float> >& x,
                                   Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const Vector< std::complex<double> >& x,
                                    Vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixAdjointVector
( float alpha, const LowRankMatrix<float,false>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( float alpha, const LowRankMatrix<float,true>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( double alpha, const LowRankMatrix<double,false>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( double alpha, const LowRankMatrix<double,true>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,false>& A,
  const Vector< std::complex<float> >& x,
  std::complex<float> beta, 
        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,true>& A,
  const Vector< std::complex<float> >& x,
  std::complex<float> beta, 
        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,false>& A,
  const Vector< std::complex<double> >& x,
  std::complex<double> beta,
        Vector< std::complex<double> >& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,true>& A,
  const Vector< std::complex<double> >& x,
  std::complex<double> beta, 
        Vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixAdjointVector
( float alpha, const LowRankMatrix<float,false>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( float alpha, const LowRankMatrix<float,true>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( double alpha, const LowRankMatrix<double,false>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( double alpha, const LowRankMatrix<double,true>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,false>& A,
  const Vector< std::complex<float> >& x,
        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,true>& A,
  const Vector< std::complex<float> >& x,
        Vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,false>& A,
  const Vector< std::complex<double> >& x,
        Vector< std::complex<double> >& y );
template void psp::hmatrix_tools::MatrixAdjointVector
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,true>& A,
  const Vector< std::complex<double> >& x,
        Vector< std::complex<double> >& y );
