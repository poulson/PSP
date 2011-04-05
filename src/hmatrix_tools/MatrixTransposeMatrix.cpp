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

// Dense C := alpha A^T B
template<typename Scalar>
void psp::hmatrix_tools::MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.Resize( A.Width(), B.Width() );
    C.SetType( GENERAL );
    MatrixTransposeMatrix( alpha, A, B, (Scalar)0, C );
}

// Dense C := alpha A^T B + beta C
template<typename Scalar>
void psp::hmatrix_tools::MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( B.Symmetric() )
        throw std::logic_error("BLAS does not support symm times trans");
#endif
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', C.Height(), C.Width(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(), 
          beta, C.Buffer(), C.LDim() );
    }
    else
    {
        blas::Gemm
        ( 'T', 'N', C.Height(), C.Width(), A.Height(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
          beta, C.Buffer(), C.LDim() );
    }
}

// Low-rank C := alpha A^T B
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const FactorMatrix<Scalar,Conjugate>& B, 
                      FactorMatrix<Scalar,Conjugate>& C )
{
#ifndef RELEASE
    if( A.m != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    C.m = A.n;
    C.n = B.n;
    if( A.r <= B.r )
    {
        C.r = A.r;
        C.U.resize( C.m*C.r );
        C.V.resize( C.n*C.r );

        if( Conjugate )
        {
            // C.U C.V^H := alpha (A.U A.V^H)^T (B.U B.V^H)
            //            = alpha conj(A.V) (A.U^T B.U B.V^H)
            //            = conj(A.V) (conj(alpha) B.V B.U^H conj(A.U))^H
            //            = conj(A.V) (conj(alpha) B.V (A.U^T B.U)^H)^H
            //
            // C.U := conj(A.V)
            // W := A.U^T B.U
            // C.V := conj(alpha) B.V W^H
            {
                const int An = A.n;
                const int Ar = A.r;
                const Scalar* RESTRICT AV = &A.V[0];
                Scalar* RESTRICT CU = &C.U[0];
                for( int i=0; i<An*Ar; ++i )
                    CU[i] = Conj( AV[i] );
            }
            std::vector<Scalar> W( A.r*B.r );
            blas::Gemm
            ( 'T', 'N', A.r, B.r, A.m,
              1, &A.U[0], A.m, &B.U[0], B.m, 0, &W[0], A.r );
            blas::Gemm
            ( 'N', 'C', B.n, A.r, B.r,
              Conj(alpha), &B.V[0], B.n, &W[0], A.r, 0, &C.V[0], C.n );
        }
        else
        {
            // C.U C.V^T := alpha (A.U A.V^T)^T (B.U B.V^T)
            //            = alpha A.V A.U^T B.U B.V^T
            //            = A.V (alpha A.U^T B.U B.V^T)
            //            = A.V (alpha B.V (B.U^T A.U))^T
            //
            // C.U := A.V
            // W := B.U^T A.U
            // C.V := alpha B.V W
            std::memcpy( &C.U[0], &A.V[0], A.n*A.r*sizeof(Scalar) );
            std::vector<Scalar> W( B.r*A.r );
            blas::Gemm
            ( 'T', 'N', B.r, A.r, B.m,
              1, &B.U[0], B.m, &A.U[0], A.m, 0, &W[0], B.r );
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
            // C.U C.V^H := alpha (A.U A.V^H)^T (B.U B.V^H)
            //            = alpha conj(A.V) A.U^T B.U B.V^H
            //            = (alpha conj(A.V) (A.U^T B.U)) B.V^H
            //
            // W := A.U^T B.U
            // AVConj := conj(A.V)
            // C.U := alpha AVConj W
            // C.V := B.V
            std::vector<Scalar> W( A.r*B.r );
            blas::Gemm
            ( 'T', 'N', A.r, B.r, A.m,
              1, &A.U[0], A.m, &B.U[0], B.m, 0, &W[0], A.r );
            std::vector<Scalar> AVConj( A.n*A.r );
            {
                const int An = A.n;
                const int Ar = A.r;
                const Scalar* RESTRICT AV = &A.V[0];
                Scalar* RESTRICT AVConjBuffer = &AVConj[0];
                for( int i=0; i<An*Ar; ++i )
                    AVConjBuffer[i] = Conj( AV[i] );
            }
            blas::Gemm
            ( 'N', 'N', A.n, B.r, A.r,
              alpha, &AVConj[0], A.n, &W[0], A.r, 0, &C.U[0], C.m );
            std::memcpy( &C.V[0], &B.V[0], B.n*B.r*sizeof(Scalar) );
        }
        else
        {
            // C.U C.V^T := alpha (A.U A.V^T)^T (B.U B.V^T)
            //            = alpha A.V A.U^T B.U B.V^T
            //            = (alpha A.V (A.U^T B.U)) B.V^T
            //
            // W := A.U^T B.U
            // C.U := alpha A.V W
            // C.V := B.V
            std::vector<Scalar> W( A.r*B.r );
            blas::Gemm
            ( 'T', 'N', A.r, B.r, A.m,
              1, &A.U[0], A.m, &B.U[0], B.m, 0, &W[0], A.r );
            blas::Gemm
            ( 'N', 'N', A.n, B.r, A.r,
              alpha, &A.V[0], A.n, &W[0], A.r, 0, &C.U[0], C.m );
            std::memcpy( &C.V[0], &B.V[0], B.n*B.r*sizeof(Scalar) );
        }
    }
}

// Form a factor matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B, 
                      FactorMatrix<Scalar,Conjugate>& C )
{
#ifndef RELEASE
    if( A.Height() != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    C.m = A.Width();
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
        ( 'T', 'N', C.m, C.r, A.Height(),
          alpha, A.LockedBuffer(), A.LDim(), &B.U[0], B.m, 0, &C.U[0], C.m );
    }

    // Form C.V := B.V
    std::memcpy( &C.V[0], &B.V[0], B.n*B.r*sizeof(Scalar) );
}

// Form a factor matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B, 
                      FactorMatrix<Scalar,Conjugate>& C )
{
#ifndef RELEASE
    if( A.m != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    C.m = A.n;
    C.n = B.Width();
    C.r = A.r;
    C.U.resize( C.m*C.r );
    C.V.resize( C.n*C.r );

    if( Conjugate )
    {
        if( B.Symmetric() )
        {
            // C.U C.V^H := alpha (A.U A.V^H)^T B
            //            = alpha conj(A.V) A.U^T B
            //            = conj(A.V) (alpha A.U^T B)
            //            = conj(A.V) (conj(alpha) B^H conj(A.U))^H
            //            = conj(A.V) (conj(alpha) conj(B) conj(A.U))^H
            //            = conj(A.V) (conj(alpha B A.U))^H
            //
            // C.U := conj(A.V)
            // C.V := alpha B A.U
            // C.V := conj(C.V)
            {
                const int An = A.n;
                const int Ar = A.r;
                const Scalar* RESTRICT AV = &A.V[0];
                Scalar* RESTRICT CU = &C.U[0];
                for( int i=0; i<An*Ar; ++i )
                    CU[i] = Conj( AV[i] );
            }
            blas::Symm
            ( 'L', 'L', A.m, A.r,
              alpha, B.LockedBuffer(), B.LDim(), &A.U[0], A.m, 
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
            // C.U C.V^H := alpha (A.U A.V^H)^T B
            //            = alpha conj(A.V) A.U^T B
            //            = conj(A.V) (alpha A.U^T B)
            //            = conj(A.V) (conj(alpha) B^H conj(A.U))^H
            //            = conj(A.V) (conj(alpha B^T A.U))^H
            //
            // C.U := conj(A.V)
            // C.V := alpha B^T A.U
            // C.V := conj(C.V)
            {
                const int An = A.n;
                const int Ar = A.r;
                const Scalar* RESTRICT AV = &A.V[0];
                Scalar* RESTRICT CU = &C.U[0];
                for( int i=0; i<An*Ar; ++i )
                    CU[i] = Conj( AV[i] );
            }
            blas::Gemm
            ( 'T', 'N', B.Width(), A.r, A.m,
              alpha, B.LockedBuffer(), B.LDim(), &A.U[0], A.m, 
              0, &C.V[0], C.n );
            {
                const int Cn = C.n;
                const int Cr = C.r;
                Scalar* CV = &C.V[0];
                for( int i=0; i<Cn*Cr; ++i )
                    CV[i] = Conj( CV[i] );
            }
        }
    }
    else
    {
        if( B.Symmetric() )
        {
            // C.U C.V^T := alpha (A.U A.V^T)^T B
            //            = alpha A.V A.U^T B
            //            = A.V (alpha A.U^T B)
            //            = A.V (alpha B A.U)^T
            //
            // C.U := A.V
            // C.V := alpha B A.U
            std::memcpy( &C.U[0], &A.V[0], A.n*A.r*sizeof(Scalar) );
            blas::Symm
            ( 'L', 'L', A.m, A.r,
              alpha, B.LockedBuffer(), B.LDim(), &A.U[0], A.m, 
              0, &C.V[0], C.n );
        }
        else
        {
            // C.U C.V^T := alpha (A.U A.V^T)^T B
            //            = alpha A.V A.U^T B
            //            = A.V (alpha B^T A.U)^T
            //
            // C.U := A.V
            // C.V := alpha B^T A.U
            std::memcpy( &C.U[0], &A.V[0], A.n*A.r*sizeof(Scalar) );
            blas::Gemm
            ( 'T', 'N', B.Width(), A.r, A.m,
              alpha, B.LockedBuffer(), B.LDim(), &A.U[0], A.m, 
              0, &C.V[0], C.n );
        }
    }
}

// Form a dense matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.Resize( A.Width(), B.n );
    C.SetType( GENERAL );
    MatrixTransposeMatrix( alpha, A, B, (Scalar)0, C );
}

// Form a dense matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.Height() != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Symmetric() )
        throw std::logic_error("Update is probably not symmetric.");
#endif
    // W := A^T B.U
    std::vector<Scalar> W( A.Width()*B.r );
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', C.Height(), B.r,
          1, A.LockedBuffer(), A.LDim(), &B.U[0], B.m, 0, &W[0], A.Width() );
    }
    else
    {
        blas::Gemm
        ( 'T', 'N', C.Height(), B.r, A.Height(),
          1, A.LockedBuffer(), A.LDim(), &B.U[0], B.m, 0, &W[0], A.Width() );
    }
    // C := alpha W B.V^[T,H] + beta C
    const char option = ( Conjugate ? 'C' : 'T' );
    blas::Gemm
    ( 'N', option, C.Height(), C.Width(), B.r,
      alpha, &W[0], A.Width(), &B.V[0], B.n, beta, C.Buffer(), C.LDim() );
}

// Form a dense matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.Resize( A.n, B.Width() );
    C.SetType( GENERAL );
    MatrixTransposeMatrix( alpha, A, B, (Scalar)0, C );
}

// Form a dense matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.m != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Symmetric() )
        throw std::logic_error("Update is probably not symmetric.");
#endif
    if( Conjugate )
    {
        if( B.Symmetric() )
        {
            // C := alpha (A.U A.V^H)^T B + beta C
            //    = alpha conj(A.V) A.U^T B + beta C
            //    = alpha conj(A.V) (B A.U)^T + beta C
            //
            // W := B A.U
            // AVConj := conj(A.V)
            // C := alpha AVConj W^T + beta C
            std::vector<Scalar> W( A.m*A.r );
            blas::Symm
            ( 'L', 'L', A.m, A.r,
              1, B.LockedBuffer(), B.LDim(), &A.U[0], A.m, 0, &W[0], A.m );
            std::vector<Scalar> AVConj( A.n*A.r );
            {
                const int An = A.n;
                const int Ar = A.r;
                const Scalar* RESTRICT AV = &A.V[0];
                Scalar* RESTRICT AVConjBuffer = &AVConj[0];
                for( int i=0; i<An*Ar; ++i )
                    AVConjBuffer[i] = Conj( AV[i] );
            }
            blas::Gemm
            ( 'N', 'T', A.n, A.m, A.r,
              alpha, &AVConj[0], A.n, &W[0], A.m, beta, C.Buffer(), C.LDim() );
        }
        else
        {
            // C := alpha (A.U A.V^H)^T B + beta C
            //    = alpha conj(A.V) A.U^T B + beta C
            //    = alpha conj(A.V) (A.U^T B) + beta C
            //
            // W := A.U^T B
            // AVConj := conj(A.V)
            // C := alpha AVConj W + beta C
            std::vector<Scalar> W( A.r*B.Width() );
            blas::Gemm
            ( 'T', 'N', A.r, B.Width(), A.m,
              1, &A.U[0], A.m, B.LockedBuffer(), B.LDim(), 0, &W[0], A.r );
            std::vector<Scalar> AVConj( A.n*A.r );
            {
                const int An = A.n;
                const int Ar = A.r;
                const Scalar* RESTRICT AV = &A.V[0];
                Scalar* RESTRICT AVConjBuffer = &AVConj[0];
                for( int i=0; i<An*Ar; ++i )
                    AVConjBuffer[i] = Conj( AV[i] );
            }
            blas::Gemm
            ( 'N', 'N', A.n, A.m, A.r,
              alpha, &AVConj[0], A.n, &W[0], A.r, beta, C.Buffer(), C.LDim() );
        }
    }
    else
    {
        if( B.Symmetric() )
        {
            // C := alpha (A.U A.V^T)^T B + beta C
            //    = alpha A.V A.U^T B + beta C
            //    = alpha A.V (B A.U)^T + beta C
            //
            // W := B A.U
            // C := alpha A.V W^T + beta C
            std::vector<Scalar> W( A.m*A.r );
            blas::Symm
            ( 'L', 'L', A.m, A.r,
              1, B.LockedBuffer(), B.LDim(), &A.U[0], A.m, 0, &W[0], A.m );
            blas::Gemm
            ( 'N', 'N', A.n, A.m, A.r,
              alpha, &A.V[0], A.n, &W[0], A.m, beta, C.Buffer(), C.LDim() );
        }
        else
        {
            // C := alpha (A.U A.V^T)^T B + beta C
            //    = alpha A.V (A.U^T B) + beta C
            //
            // W := A.U^T B
            // C := alpha A.V W + beta C
            std::vector<Scalar> W( A.r*B.Width() );
            blas::Gemm
            ( 'T', 'N', A.r, B.Width(), A.m,
              1, &A.U[0], A.m, B.LockedBuffer(), B.LDim(), 0, &W[0], A.r );
            blas::Gemm
            ( 'N', 'N', A.n, B.Width(), A.r,
              alpha, &A.V[0], A.n, &W[0], A.r, beta, C.Buffer(), C.LDim() );
        }
    }
}

// Dense C := alpha A^T B
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const DenseMatrix<float>& A,
               const DenseMatrix<float>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Dense C := alpha A^T B + beta C
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const DenseMatrix<float>& A,
               const DenseMatrix<float>& B,
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );

// Low-rank C := alpha A^T B
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const FactorMatrix<float,false>& A,
               const FactorMatrix<float,false>& B,
                     FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const FactorMatrix<float,true>& A,
               const FactorMatrix<float,true>& B,
                     FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const FactorMatrix<double,false>& A,
                const FactorMatrix<double,false>& B,
                      FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const FactorMatrix<double,true>& A,
                const FactorMatrix<double,true>& B,
                      FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const FactorMatrix<std::complex<float>,false>& B,
                                   FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const FactorMatrix<std::complex<float>,true>& B,
                                   FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const FactorMatrix<std::complex<double>,false>& B,
                                    FactorMatrix<std::complex<double>,false>& C
);
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const FactorMatrix<std::complex<double>,true>& B,
                                    FactorMatrix<std::complex<double>,true>& C
);

// Form a dense matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,false>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,true>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,false>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,true>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,false>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,true>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,false>& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,true>& B,
                                    DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,false>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,true>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,false>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,true>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,false>& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,true>& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,false>& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,true>& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const FactorMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const FactorMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const FactorMatrix<double,false>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const FactorMatrix<double,true>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const FactorMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const FactorMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const FactorMatrix<double,false>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const FactorMatrix<double,true>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );

// Form a factor matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,false>& B, 
                     FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,true>& B, 
                     FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,false>& B,
                      FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,true>& B,
                      FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,false>& B,
                                   FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,true>& B,
                                   FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,false>& B,
                                    FactorMatrix<std::complex<double>,false>& C
);
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,true>& B,
                                    FactorMatrix<std::complex<double>,true>& C
);

// Form a factor matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const FactorMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
                     FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( float alpha, const FactorMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
                     FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const FactorMatrix<double,false>& A,
                const DenseMatrix<double>& B,
                      FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( double alpha, const FactorMatrix<double,true>& A,
                const DenseMatrix<double>& B,
                      FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    FactorMatrix<std::complex<double>,false>& C 
);
template void psp::hmatrix_tools::MatrixTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    FactorMatrix<std::complex<double>,true>& C 
);
