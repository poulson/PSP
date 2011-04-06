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

// Dense C := alpha A^H B
template<typename Scalar>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.Resize( A.Width(), B.Width() );
    C.SetType( GENERAL );
    MatrixHermitianTransposeMatrix( alpha, A, B, (Scalar)0, C );
}

// Dense C := alpha A^H B + beta C
template<typename Scalar>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( B.Symmetric() )
        throw std::logic_error("BLAS does not support symm times trans.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    if( A.Symmetric() )
    {
        const int m = A.Height();
        const int n = B.Width();
        if( m <= 2*n )
        {
            // C := alpha A^H B + beta C
            //    = alpha conj(A) B + beta C
            //
            // AConj := conj(A)
            // C := alpha AConj B + beta C

            DenseMatrix<Scalar> AConj(m,m); 
            Conjugate( A, AConj );
            blas::Symm
            ( 'L', 'L', C.Height(), C.Width(),
              alpha, AConj.LockedBuffer(), AConj.LDim(), 
                     B.LockedBuffer(), B.LDim(),
              beta,  C.Buffer(), C.LDim() );
        }
        else
        {
            // C := alpha A^H B + beta C
            //    = alpha conj(A) B + beta C
            //    = conj(conj(alpha) A conj(B) + conj(beta) conj(C))
            //
            // BConj := conj(B)
            // CConj := conj(C)
            // C := conj(alpha) A BConj + conj(beta) CConj
            // C := conj(C)

            DenseMatrix<Scalar> BConj(m,n);
            Conjugate( B, BConj );
            Conjugate( C );
            blas::Symm
            ( 'L', 'L', C.Height(), C.Width(),
              Conj(alpha), A.LockedBuffer(), A.LDim(), 
                           BConj.LockedBuffer(), BConj.LDim(),
              Conj(beta),  C.Buffer(), C.LDim() );
            Conjugate( C );
        }
    }
    else
    {
        blas::Gemm
        ( 'C', 'N', C.Height(), C.Width(), A.Height(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
          beta, C.Buffer(), C.LDim() );
    }
}

// TODO: version of above routine that allows for temporary in-place conj of B

// Low-rank C := alpha A^H B
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const FactorMatrix<Scalar,Conjugated>& B, 
                      FactorMatrix<Scalar,Conjugated>& C )
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

        if( Conjugated )
        {
            // C.U C.V^H = alpha (A.U A.V^H)^H (B.U B.V^H) 
            //           = alpha A.V A.U^H B.U B.V^H 
            //           = A.V (conj(alpha) B.V (B.U^H A.U))^H 
            //           = A.V (conj(alpha) B.V W)^H
            //
            // C.U := A.V
            // W := B.U^H A.U
            // C.V := conj(alpha) B.V W
            std::memcpy( &C.U[0], &A.V[0], A.m*A.r*sizeof(Scalar) );
            std::vector<Scalar> W(B.r*A.r);
            blas::Gemm
            ( 'C', 'N', B.r, A.r, B.m, 
              1, &B.U[0], B.m, &A.U[0], A.m, 0, &W[0], B.r );
            blas::Gemm
            ( 'N', 'N', C.n, A.r, B.r,
              Conj(alpha), &B.V[0], B.n, &W[0], B.r, 0, &C.V[0], C.n );
        }
        else
        {
            // C.U C.V^T = alpha (A.U A.V^T)^H (B.U B.V^T)
            //           = alpha conj(A.V) A.U^H B.U B.V^T
            //           = conj(A.V) (alpha B.V (B.U^T conj(A.U)))^T
            //           = conj(A.V) (alpha B.V (A.U^H B.U)^T)^T
            //           = conj(A.V) (alpha B.V W^T)^T
            //
            // C.U := conj(A.V)
            // W := A.U^H B.U
            // C.V := alpha B.V W^T
            Conjugate( A.V, C.U );
            std::vector<Scalar> W( A.r*B.r );
            blas::Gemm
            ( 'C', 'N', A.r, B.r, A.m,
              1, &A.U[0], A.m, &B.U[0], B.m, 0, &W[0], A.r );
            blas::Gemm
            ( 'N', 'T', B.n, A.r, B.r,
              alpha, &B.V[0], B.n, &W[0], A.r, 0, &C.V[0], C.n );
        }
    }
    else // B.r < A.r
    {
        C.r = B.r;
        C.U.resize( C.m*C.r );
        C.V.resize( C.n*C.r );

        if( Conjugated )
        {
            // C.U C.V^H := alpha (A.U A.V^H)^H (B.U B.V^H)
            //            = alpha A.V A.U^H B.U B.V^H
            //            = (alpha A.V (A.U^H B.U)) B.V^H
            //            = (alpha A.V W) B.V^H
            //
            // W := A.U^H B.U
            // C.U := alpha A.V W
            // C.V := B.V
            std::vector<Scalar> W( A.r*B.r );
            blas::Gemm
            ( 'C', 'N', A.r, B.r, A.m,
              1, &A.U[0], A.m, &B.U[0], B.m, 0, &W[0], A.r );
            blas::Gemm
            ( 'N', 'N', A.n, A.r, B.r,
              alpha, &A.V[0], A.n, &W[0], A.r, 0, &C.U[0], C.m );
            std::memcpy( &C.V[0], &B.V[0], B.n*B.r*sizeof(Scalar) );
        }
        else
        {
            // C.U C.V^T := alpha (A.U A.V^T)^H (B.U B.V^T)
            //            = alpha conj(A.V) A.U^H B.U B.V^T
            //            = (alpha conj(A.V) A.U^H B.U) B.V^T
            //            = conj(conj(alpha) A.V A.U^T conj(B.U)) B.V^T
            //            = conj(conj(alpha) A.V (B.U^H A.U)^T) B.V^T
            //            = conj(conj(alpha) A.V W^T) B.V^T
            //
            // W := B.U^H A.U
            // C.U := conj(alpha) A.V W^T
            // C.U := conj(C.U)
            // C.V := B.V
            std::vector<Scalar> W( B.r*A.r );
            blas::Gemm
            ( 'C', 'N', B.r, A.r, B.m,
              1, &B.U[0], B.m, &A.U[0], A.m, 0, &W[0], B.r );
            blas::Gemm
            ( 'N', 'T', A.n, B.r, A.r,
              Conj(alpha), &A.V[0], A.n, &W[0], B.r, 0, &C.U[0], C.m );
            Conjugate( C.U );
            std::memcpy( &C.V[0], &B.V[0], B.n*B.r*sizeof(Scalar) );
        }
    }
}

// Form a factor matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B, 
                      FactorMatrix<Scalar,Conjugated>& C )
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

    if( A.Symmetric() )
    {
        // C.U C.V^[T,H] := alpha A^H B.U B.V^[T,H]
        //                = (alpha conj(A) B.U) B.V^[T,H]
        //                = conj(conj(alpha) A conj(B.U)) B.V^[T,H]
        //
        // BUConj := conj(B.U)
        // C.U := conj(alpha) A BUConj
        // C.U := conj(C.U)
        // C.V := B.V
        std::vector<Scalar> BUConj( B.m*B.r );
        Conjugate( B.U, BUConj );
        blas::Symm
        ( 'L', 'L', B.m, B.r,
          Conj(alpha), A.LockedBuffer(), A.LDim(), &BUConj[0], B.m,
          0, &C.U[0], C.m );
        Conjugate( C.U );
        std::memcpy( &C.V[0], &B.V[0], B.n*B.r*sizeof(Scalar) );
    }
    else
    {
        // C.U C.V^[T,H] := alpha A^H B.U B.V^[T,H]
        //                = (alpha A^H B.U) B.V^[T,H]
        blas::Gemm
        ( 'C', 'N', C.m, C.r, A.Height(),
          alpha, A.LockedBuffer(), A.LDim(), &B.U[0], B.m, 0, &C.U[0], C.m );
        std::memcpy( &C.V[0], &B.V[0], B.n*B.r*sizeof(Scalar) );
    }
}

// Form a factor matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B, 
                      FactorMatrix<Scalar,Conjugated>& C )
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

    if( Conjugated )
    {
        if( B.Symmetric() )
        {
            // C.U C.V^H := alpha (A.U A.V^H)^H B
            //            = alpha A.V A.U^H B
            //            = A.V (alpha A.U^H B)
            //            = A.V (conj(alpha) B^H A.U)^H
            //            = A.V (conj(alpha B conj(A.U)))^H
            //
            // C.U := A.V
            // AUConj := conj(A.U)
            // C.V := alpha B AUConj
            // C.V := conj(C.V)
            std::memcpy( &C.U[0], &A.V[0], A.n*A.r*sizeof(Scalar) );
            std::vector<Scalar> AUConj( A.m*A.r );
            Conjugate( A.U, AUConj );
            blas::Symm
            ( 'L', 'L', C.n, C.r,
              alpha, B.LockedBuffer(), B.LDim(), &AUConj[0], A.m, 
              0, &C.V[0], C.n );
            Conjugate( C.V );
        }
        else
        {
            // C.U C.V^H := alpha (A.U A.V^H)^H B
            //            = alpha A.V A.U^H B
            //            = A.V (conj(alpha) B^H A.U)^H
            //
            // C.U := A.V
            // C.V := conj(alpha) B^H A.U
            std::memcpy( &C.U[0], &A.V[0], A.n*A.r*sizeof(Scalar) );
            blas::Gemm
            ( 'C', 'N', C.n, C.r, A.m,
              alpha, B.LockedBuffer(), B.LDim(), &A.U[0], A.m, 
              0, &C.V[0], C.n );
        }
    }
    else
    {
        // C.U C.V^T := alpha (A.U A.V^T)^H B
        //            = alpha conj(A.V) A.U^H B
        //            = conj(A.V) (alpha B^T conj(A.U))^T
        //
        // C.U := conj(A.V)
        // AUConj := conj(A.U)
        // C.V := alpha B^T AUConj
        Conjugate( A.V, C.U );
        std::vector<Scalar> AUConj( A.m*A.r );
        Conjugate( A.U, AUConj );
        if( B.Symmetric() )
        {
            blas::Symm
            ( 'L', 'L', A.m, A.r, 
              alpha, B.LockedBuffer(), B.LDim(), &AUConj[0], A.m, 
              0, &C.V[0], C.n );
        }
        else
        {
            blas::Gemm
            ( 'T', 'N', C.n, C.r, A.m,
              alpha, B.LockedBuffer(), B.LDim(), &AUConj[0], A.m,
              0, &C.V[0], C.n );
        }
    }
}

// Form a dense matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.Resize( A.Width(), B.n );
    C.SetType( GENERAL );
    MatrixHermitianTransposeMatrix( alpha, A, B, (Scalar)0, C );
}

// Form a dense matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.Height() != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    if( A.Symmetric() )
    {
        // C := alpha A^H (B.U B.V^[T,H]) + beta C
        //    = alpha (conj(A) B.U) B.V^[T,H] + beta C
        //    = alpha conj(A conj(B.U)) B.V^[T,H] + beta C
        //
        // BUConj := conj(B.U)
        // W := A BUConj
        // W := conj(W)
        // C := alpha W B.V^[T,H] + beta C
        std::vector<Scalar> BUConj( B.m*B.r );
        Conjugate( B.U, BUConj );
        std::vector<Scalar> W( B.m*B.r );
        blas::Symm
        ( 'L', 'L', B.m, B.r,
          1, A.LockedBuffer(), A.LDim(), &BUConj[0], B.m, 0, &W[0], B.m );
        Conjugate( W );
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, C.Height(), C.Width(), B.r,
          alpha, &W[0], B.m, &B.V[0], B.n, beta, C.Buffer(), C.LDim() );
    }
    else
    {
        // C := alpha (A^H B.U) B.V^[T,H] + beta C
        //
        // W := A^H B.U
        // C := alpha W B.V^[T,H] + beta C
        std::vector<Scalar> W( A.Width()*B.r );
        blas::Gemm
        ( 'C', 'N', A.Width(), B.r, B.m,
          1, A.LockedBuffer(), A.LDim(), &B.U[0], B.m, 0, &W[0], A.Width() );
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, C.Height(), C.Width(), B.r,
          alpha, &W[0], A.Width(), &B.V[0], B.n, beta, C.Buffer(), C.LDim() );
    }
}

// Form a dense matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.Resize( A.n, B.Width() );
    C.SetType( GENERAL );
    MatrixHermitianTransposeMatrix( alpha, A, B, (Scalar)0, C );
}

// Form a dense matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.m != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    if( Conjugated )
    {
        if( B.Symmetric() )
        {
            // C := alpha (A.U A.V^H)^H B + beta C
            //    = alpha A.V (A.U^H B) + beta C
            //    = alpha A.V (B^T conj(A.U))^T + beta C
            //    = alpha A.V (B conj(A.U))^T + beta C
            //
            // AUConj := conj(A.U)
            // W := B AUConj
            // C := alpha A.V W^T + beta C
            std::vector<Scalar> AUConj( A.m*A.r );
            Conjugate( A.U, AUConj );
            std::vector<Scalar> W( A.m*A.r );
            blas::Symm
            ( 'L', 'L', A.m, A.r,
              1, B.LockedBuffer(), B.LDim(), &AUConj[0], A.m, 0, &W[0], A.m );
            blas::Gemm
            ( 'N', 'T', C.Height(), C.Width(), A.r,
              alpha, &A.V[0], A.n, &W[0], A.m, beta, C.Buffer(), C.LDim() );
        }
        else
        {
            // C := alpha (A.U A.V^H)^H B + beta C
            //    = alpha A.V (A.U^H B) + beta C
            //
            // W := A.U^H B
            // C := alpha A.V W + beta C
            std::vector<Scalar> W( A.r*B.Width() );
            blas::Gemm
            ( 'C', 'N', A.r, B.Width(), A.m,
              1, &A.U[0], A.m, B.LockedBuffer(), B.LDim(), 0, &W[0], A.r );
            blas::Gemm
            ( 'N', 'N', C.Height(), C.Width(), A.r,
              alpha, &A.V[0], A.n, &W[0], A.r, beta, C.Buffer(), C.LDim() );
        }
    }
    else
    {
        if( B.Symmetric() )
        {
            // C := alpha (A.U A.V^T)^H B + beta C
            //    = alpha conj(A.V) (A.U^H B) + beta C
            //    = alpha conj(A.V) (B conj(A.U))^T + beta C
            //
            // AUConj := conj(A.U)
            // W := B AUConj
            // AVConj := conj(A.V)
            // C := alpha AVConj W^T + beta C
            std::vector<Scalar> AUConj( A.m*A.r );
            Conjugate( A.U, AUConj );
            std::vector<Scalar> W( A.m*A.r );
            blas::Symm
            ( 'L', 'L', A.m, A.r,
              1, B.LockedBuffer(), B.LDim(), &AUConj[0], A.m, 0, &W[0], A.m );
            std::vector<Scalar> AVConj( A.n*A.r );
            Conjugate( A.V, AVConj );
            blas::Gemm
            ( 'N', 'T', A.n, A.m, A.r,
              alpha, &AVConj[0], A.n, &W[0], A.m, beta, C.Buffer(), C.LDim() );
        }
        else
        {
            // C := alpha (A.U A.V^T)^H B + beta C
            //    = alpha conj(A.V) (A.U^H B) + beta C
            //
            // W := A.U^H B
            // AVConj := conj(A.V)
            // C := alpha AVConj W + beta C
            std::vector<Scalar> W( A.r*B.Width() );
            blas::Gemm
            ( 'C', 'N', A.r, B.Width(), A.m,
              1, &A.U[0], A.m, B.LockedBuffer(), B.LDim(), 0, &W[0], A.r );
            std::vector<Scalar> AVConj( A.n*A.r );
            Conjugate( A.V, AVConj );
            blas::Gemm
            ( 'N', 'N', C.Height(), C.Width(), A.r,
              alpha, &AVConj[0], A.n, &W[0], A.r, beta, C.Buffer(), C.LDim() );
        }
    }
}

// Dense C := alpha A^H B
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const DenseMatrix<float>& A,
               const DenseMatrix<float>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Dense C := alpha A^H B + beta C
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const DenseMatrix<float>& A,
               const DenseMatrix<float>& B,
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );

// Low-rank C := alpha A^H B
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const FactorMatrix<float,false>& A,
               const FactorMatrix<float,false>& B,
                     FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const FactorMatrix<float,true>& A,
               const FactorMatrix<float,true>& B,
                     FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const FactorMatrix<double,false>& A,
                const FactorMatrix<double,false>& B,
                      FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const FactorMatrix<double,true>& A,
                const FactorMatrix<double,true>& B,
                      FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const FactorMatrix<std::complex<float>,false>& B,
                                   FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const FactorMatrix<std::complex<float>,true>& B,
                                   FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const FactorMatrix<std::complex<double>,false>& B,
                                    FactorMatrix<std::complex<double>,false>& C
);
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const FactorMatrix<std::complex<double>,true>& B,
                                    FactorMatrix<std::complex<double>,true>& C
);

// Form a dense matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,false>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,true>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,false>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,true>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,false>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,true>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,false>& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,true>& B,
                                    DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,false>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,true>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,false>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,true>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,false>& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,true>& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,false>& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,true>& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const FactorMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const FactorMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const FactorMatrix<double,false>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const FactorMatrix<double,true>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const FactorMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const FactorMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const FactorMatrix<double,false>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const FactorMatrix<double,true>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );

// Form a factor matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,false>& B, 
                     FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,true>& B, 
                     FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,false>& B,
                      FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,true>& B,
                      FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,false>& B,
                                   FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,true>& B,
                                   FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,false>& B,
                                    FactorMatrix<std::complex<double>,false>& C
);
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,true>& B,
                                    FactorMatrix<std::complex<double>,true>& C
);

// Form a factor matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const FactorMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
                     FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( float alpha, const FactorMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
                     FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const FactorMatrix<double,false>& A,
                const DenseMatrix<double>& B,
                      FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( double alpha, const FactorMatrix<double,true>& A,
                const DenseMatrix<double>& B,
                      FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    FactorMatrix<std::complex<double>,false>& C 
);
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    FactorMatrix<std::complex<double>,true>& C 
);
