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

// Dense C := alpha A B
template<typename Scalar>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.Resize( A.Height(), B.Width() );
    C.SetType( GENERAL );
    MatrixMatrix( alpha, A, B, (Scalar)0, C );
}

// Dense C := alpha A B + beta C
template<typename Scalar>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.Width() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( A.Symmetric() && B.Symmetric() )
        throw std::logic_error("Product of symmetric matrices not supported.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', C.Height(), C.Width(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(), 
          beta, C.Buffer(), C.LDim() );
    }
    else if( B.Symmetric() )
    {
        blas::Symm
        ( 'R', 'L', C.Height(), C.Width(),
          alpha, B.LockedBuffer(), B.LDim(), A.LockedBuffer(), A.LDim(),
          beta, C.Buffer(), C.LDim() );
    }
    else
    {
        blas::Gemm
        ( 'N', 'N', C.Height(), C.Width(), A.Width(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
          beta, C.Buffer(), C.LDim() );
    }
}

// Low-rank C := alpha A B
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const FactorMatrix<Scalar,Conjugate>& B, 
                      FactorMatrix<Scalar,Conjugate>& C )
{
#ifndef RELEASE
    if( A.n != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    C.m = A.m;
    C.n = B.n;
    if( A.r <= B.r )
    {
        C.r = A.r;
        C.U.resize( C.m*C.r );
        C.V.resize( C.n*C.r );

        if( Conjugate )
        {
            // C.U C.V^H := alpha (A.U A.V^H) (B.U B.V^H)
            //            = A.U (alpha A.V^H B.U B.V^H)
            //            = A.U (conj(alpha) B.V (B.U^H A.V))^H
            //
            // C.U := A.U
            // W := B.U^H A.V
            // C.V := conj(alpha) B.V W
            std::memcpy( &C.U[0], &A.U[0], A.m*A.r*sizeof(Scalar) );
            std::vector<Scalar> W( B.r*A.r );
            blas::Gemm
            ( 'C', 'N', B.r, A.r, B.m,
              1, &B.U[0], B.m, &A.V[0], A.n, 0, &W[0], B.r );
            blas::Gemm
            ( 'N', 'N', B.n, A.r, B.r,
              Conj(alpha), &B.V[0], B.n, &W[0], B.r, 0, &C.V[0], C.n );
        }
        else
        {
            // C.U C.V^T := alpha (A.U A.V^T) (B.U B.V^T)
            //            = A.U (alpha A.V^T B.U B.V^T)
            //            = A.U (alpha B.V (B.U^T A.V))^T
            //
            // C.U := A.U
            // W := B.U^T A.V
            // C.V := alpha B.V W
            std::memcpy( &C.U[0], &A.U[0], A.m*A.r*sizeof(Scalar) );
            std::vector<Scalar> W( B.r*A.r );
            blas::Gemm
            ( 'T', 'N', B.r, A.r, B.m,
              1, &B.U[0], B.m, &A.V[0], A.n, 0, &W[0], B.r );
            blas::Gemm
            ( 'N', 'N', B.n, A.r, B.r,
              alpha, &B.V[0], B.n, &W[0], B.r, 0, &C.V[0], C.n );
        }
    }
    else // B.r < A.r
    {
        C.r = B.r;
        C.U.resize( C.m*C.r );
        C.V.resize( C.n*C.r );

        if( Conjugate )
        {
            // C.U C.V^H := alpha (A.U A.V^H) (B.U B.V^H)
            //            = (alpha A.U (A.V^H B.U)) B.V^H
            //
            // W := A.V^H B.U
            // C.U := alpha A.U W
            // C.V := B.V
            std::vector<Scalar> W( A.r*B.r );
            blas::Gemm
            ( 'C', 'N', A.r, B.r, A.n,
              1, &A.V[0], A.n, &B.U[0], B.m, 0, &W[0], A.r );
            blas::Gemm
            ( 'N', 'N', A.m, B.r, A.r,
              alpha, &A.U[0], A.m, &W[0], A.r, 0, &C.U[0], C.m );
            std::memcpy( &C.V[0], &B.V[0], B.n*B.r*sizeof(Scalar) );
        }
        else
        {
            // C.U C.V^T := alpha (A.U A.V^T) (B.U B.V^T)
            //            = (alpha A.U (A.V^T B.U)) B.V^T
            //
            // W := A.V^T B.U
            // C.U := alpha A.U W
            // C.V := B.V
            std::vector<Scalar> W( A.r*B.r );
            blas::Gemm
            ( 'T', 'N', A.r, B.r, A.n,
              1, &A.V[0], A.n, &B.U[0], B.m, 0, &W[0], A.r );
            blas::Gemm
            ( 'N', 'N', A.m, B.r, A.r,
              alpha, &A.U[0], A.m, &W[0], A.r, 0, &C.U[0], C.m );
            std::memcpy( &C.V[0], &B.V[0], B.n*B.r*sizeof(Scalar) );
        }
    }
}

// Form a factor matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B, 
                      FactorMatrix<Scalar,Conjugate>& C )
{
#ifndef RELEASE
    if( A.Width() != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    C.m = A.Height();
    C.n = B.n;
    C.r = B.r;
    C.U.resize( C.m*C.r );
    C.V.resize( C.n*C.r );

    // Form C.U := A B.U
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', C.m, C.r, 
          alpha, A.LockedBuffer(), A.LDim(), &B.U[0], B.m, 0, &C.U[0], C.m );
    }
    else
    {
        blas::Gemm
        ( 'N', 'N', C.m, C.r, A.Width(),
          alpha, A.LockedBuffer(), A.LDim(), &B.U[0], B.m, 0, &C.U[0], C.m );
    }

    // Form C.V := B.V
    std::memcpy( &C.V[0], &B.V[0], B.n*B.r*sizeof(Scalar) );
}

// Form a factor matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B, 
                      FactorMatrix<Scalar,Conjugate>& C )
{
#ifndef RELEASE
    if( A.n != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    C.m = A.m;
    C.n = B.Width();
    C.r = A.r;
    C.U.resize( C.m*C.r );
    C.V.resize( C.n*C.r );

    if( Conjugate )
    {
        if( B.Symmetric() )
        {
            // C.U C.V^H := alpha (A.U A.V^H) B
            //            = A.U (alpha A.V^H B)
            //            = A.U (conj(alpha) B^H A.V)^H
            //            = A.U (conj(alpha B conj(A.V)))^H
            //
            // C.U := A.U
            // AVConj := conj(A.V)
            // C.V := alpha B AVConj
            // C.V := conj(C.V)
            std::memcpy( &C.U[0], &A.U[0], A.m*A.r*sizeof(Scalar) );
            std::vector<Scalar> AVConj( A.n*A.r );
            {
                const int An = A.n;
                const int Ar = A.r;
                const Scalar* RESTRICT AV = &A.V[0];
                Scalar* RESTRICT AVConjBuffer = &AVConj[0];
                for( int i=0; i<An*Ar; ++i )
                    AVConjBuffer[i] = Conj( AV[i] );
            }
            blas::Symm
            ( 'L', 'L', A.n, A.r,
              alpha, B.LockedBuffer(), B.LDim(), &AVConj[0], A.n, 
              0, &C.V[0], C.n );
            {
                const int Cn = C.n; 
                const int Cr = C.r;
                Scalar* CV = &C.V[0];
                for( int i=0; i<Cn*Cr; ++i )
                    CV[i] = Conj( CV[i] );
            }
        }
        else
        {
            // C.U C.V^H := alpha (A.U A.V^H) B
            //            = A.U (conj(alpha) B^H A.V)^H
            //
            // C.U := A.U
            // C.V := conj(alpha) B^H A.V
            std::memcpy( &C.U[0], &A.U[0], A.m*A.r*sizeof(Scalar) );
            blas::Gemm
            ( 'C', 'N', C.n, C.r, A.n,
              Conj(alpha), B.LockedBuffer(), B.LDim(), &A.V[0], A.n,
              0, &C.V[0], C.n );
        }
    }
    else
    {
        if( B.Symmetric() )
        {
            // C.U C.V^T := alpha (A.U A.V^T) B
            //            = A.U (alpha A.V^T B)
            //            = A.U (alpha B A.V)^T
            //
            // C.U := A.U
            // C.V := alpha B A.V
            std::memcpy( &C.U[0], &A.U[0], A.m*A.r*sizeof(Scalar) );
            blas::Symm
            ( 'L', 'L', A.n, A.r,
              alpha, B.LockedBuffer(), B.LDim(), &A.V[0], A.n, 
              0, &C.V[0], C.n );
        }
        else
        {
            // C.U C.V^T := alpha (A.U A.V^T) B
            //            = A.U (alpha B^T A.V)^T
            //
            // C.U := A.U
            // C.V := alpha B^T A.V
            std::memcpy( &C.U[0], &A.U[0], A.m*A.r*sizeof(Scalar) );
            blas::Gemm
            ( 'T', 'N', B.Width(), A.r, A.n,
              alpha, B.LockedBuffer(), B.LDim(), &A.V[0], A.n,
              0, &C.V[0], C.n );
        }
    }
}

// Form a dense matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.Resize( A.Height(), B.n );
    C.SetType( GENERAL );
    MatrixMatrix( alpha, A, B, (Scalar)0, C );
}

// Form a dense matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.Width() != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    // W := A B.U
    std::vector<Scalar> W( A.Height()*B.r );
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', C.Height(), B.r,
          1, A.LockedBuffer(), A.LDim(), &B.U[0], B.m, 0, &W[0], A.Height() );
    }
    else
    {
        blas::Gemm
        ( 'N', 'N', C.Height(), B.r, A.Width(),
          1, A.LockedBuffer(), A.LDim(), &B.U[0], B.m, 0, &W[0], A.Height() );
    }
    // C := alpha W B.V^[T,H] + beta C
    const char option = ( Conjugate ? 'C' : 'T' );
    blas::Gemm
    ( 'N', option, C.Height(), C.Width(), B.r,
      alpha, &W[0], A.Height(), &B.V[0], B.n, beta, C.Buffer(), C.LDim() );
}

// Form a dense matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.Resize( A.m, B.Width() );
    C.SetType( GENERAL );
    MatrixMatrix( alpha, A, B, (Scalar)0, C );
}

// Form a dense matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.n != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    if( Conjugate )
    {
        if( B.Symmetric() )
        {
            // C := alpha (A.U A.V^H) B + beta C
            //    = alpha A.U (A.V^H B) + beta C
            //    = alpha A.U (B conj(A.V))^T + beta C
            //
            // AVConj := conj(A.V)
            // W := B AVConj
            // C := alpha A.U W^T + beta C
            std::vector<Scalar> AVConj( A.n*A.r );
            {
                const int An = A.n;
                const int Ar = A.r;
                const Scalar* RESTRICT AV = &A.V[0];
                Scalar* RESTRICT AVConjBuffer = &AVConj[0];
                for( int i=0; i<An*Ar; ++i )
                    AVConjBuffer[i] = Conj( AV[i] );
            }
            std::vector<Scalar> W( A.n*A.r );
            blas::Symm
            ( 'L', 'L', A.n, A.r,
              1, B.LockedBuffer(), B.LDim(), &AVConj[0], A.n, 0, &W[0], A.n );
            blas::Gemm
            ( 'N', 'T', A.m, A.n, A.r,
              alpha, &A.U[0], A.m, &W[0], A.n, beta, C.Buffer(), C.LDim() );
        }
        else
        {
            // C := alpha (A.U A.V^H) B + beta C
            //    = alpha A.U (A.V^H B) + beta C
            //
            // W := A.V^H B
            // C := alpha A.U W + beta C
            std::vector<Scalar> W( A.r*B.Width() );
            blas::Gemm
            ( 'C', 'N', A.r, B.Width(), A.n,
              1, &A.V[0], A.n, B.LockedBuffer(), B.LDim(), 0, &W[0], A.r );
            blas::Gemm
            ( 'N', 'N', A.m, B.Width(), A.r,
              alpha, &A.U[0], A.m, &W[0], A.r, beta, C.Buffer(), C.LDim() );
        }
    }
    else
    {
        if( B.Symmetric() )
        {
            // C := alpha (A.U A.V^T) B + beta C
            //    = alpha A.U (A.V^T B) + beta C
            //    = alpha A.U (B A.V)^T + beta C
            //
            // W := B A.V
            // C := alpha A.U W^T + beta C
            std::vector<Scalar> W( A.n*A.r );
            blas::Symm
            ( 'L', 'L', A.n, A.r,
              1, B.LockedBuffer(), B.LDim(), &A.V[0], A.n, 0, &W[0], A.n );
            blas::Gemm
            ( 'N', 'T', A.m, A.n, A.r,
              alpha, &A.U[0], A.m, &W[0], A.n, beta, C.Buffer(), C.LDim() );
        }
        else
        {
            // C := alpha (A.U A.V^T) B + beta C
            //    = alpha A.U (A.V^T B) + beta C
            //
            // W := A.V^T B
            // C := alpha A.U W + beta C
            std::vector<Scalar> W( A.r*B.Width() );
            blas::Gemm
            ( 'T', 'N', A.r, B.Width(), A.n,
              1, &A.V[0], A.n, B.LockedBuffer(), B.LDim(), 0, &W[0], A.r );
            blas::Gemm
            ( 'N', 'N', A.m, B.Width(), A.r,
              alpha, &A.U[0], A.m, &W[0], A.r, beta, C.Buffer(), C.LDim() );
        }
    }
}

// Dense C := alpha A B
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A,
               const DenseMatrix<float>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Dense C := alpha A B + beta C
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A,
               const DenseMatrix<float>& B,
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );

// Low-rank C := alpha A B
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const FactorMatrix<float,false>& A,
               const FactorMatrix<float,false>& B,
                     FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const FactorMatrix<float,true>& A,
               const FactorMatrix<float,true>& B,
                     FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const FactorMatrix<double,false>& A,
                const FactorMatrix<double,false>& B,
                      FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const FactorMatrix<double,true>& A,
                const FactorMatrix<double,true>& B,
                      FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const FactorMatrix<std::complex<float>,false>& B,
                                   FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const FactorMatrix<std::complex<float>,true>& B,
                                   FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const FactorMatrix<std::complex<double>,false>& B,
                                    FactorMatrix<std::complex<double>,false>& C
);
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const FactorMatrix<std::complex<double>,true>& B,
                                    FactorMatrix<std::complex<double>,true>& C
);

// Form a dense matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,false>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,true>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,false>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,true>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,false>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,true>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,false>& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,true>& B,
                                    DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,false>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,true>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,false>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,true>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,false>& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,true>& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,false>& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,true>& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const FactorMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const FactorMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const FactorMatrix<double,false>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const FactorMatrix<double,true>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const FactorMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const FactorMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const FactorMatrix<double,false>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const FactorMatrix<double,true>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );

// Form a factor matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,false>& B, 
                     FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,true>& B, 
                     FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,false>& B,
                      FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,true>& B,
                      FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,false>& B,
                                   FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,true>& B,
                                   FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,false>& B,
                                    FactorMatrix<std::complex<double>,false>& C
);
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,true>& B,
                                    FactorMatrix<std::complex<double>,true>& C
);

// Form a factor matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const FactorMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
                     FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const FactorMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
                     FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const FactorMatrix<double,false>& A,
                const DenseMatrix<double>& B,
                      FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const FactorMatrix<double,true>& A,
                const DenseMatrix<double>& B,
                      FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    FactorMatrix<std::complex<double>,false>& C 
);
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    FactorMatrix<std::complex<double>,true>& C 
);
