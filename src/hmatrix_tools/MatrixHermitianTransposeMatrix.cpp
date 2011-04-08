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
    C.SetType( GENERAL );
    C.Resize( A.Width(), B.Width() );
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
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Width();
    const int n = B.Width();
    const int Ar = A.Rank();
    const int Br = B.Rank();
    if( Ar <= Br )
    {
        const int r = Ar;
        C.U.SetType( GENERAL ); C.U.Resize( m, r ); 
        C.V.SetType( GENERAL ); C.V.Resize( n, r ); 

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
            Copy( A.V, C.U );
            DenseMatrix<Scalar> W( Br, Ar );
            blas::Gemm
            ( 'C', 'N', Br, Ar, B.Height(), 
              1, B.U.LockedBuffer(), B.U.LDim(), 
                 A.U.LockedBuffer(), A.U.LDim(), 
              0, W.Buffer(),         W.LDim() );
            blas::Gemm
            ( 'N', 'N', n, Ar, Br,
              Conj(alpha), B.V.LockedBuffer(), B.V.LDim(), 
                           W.LockedBuffer(),   W.LDim(), 
              0,           C.V.Buffer(),       C.V.LDim() );
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
            DenseMatrix<Scalar> W( Ar, Br );
            blas::Gemm
            ( 'C', 'N', Ar, Br, A.Height(),
              1, A.U.LockedBuffer(), A.U.LDim(), 
                 B.U.LockedBuffer(), B.U.LDim(), 
              0, W.Buffer(),         W.LDim() );
            blas::Gemm
            ( 'N', 'T', n, Ar, Br,
              alpha, B.V.LockedBuffer(), B.V.LDim(), 
                     W.LockedBuffer(),   W.LDim(), 
              0,     C.V.Buffer(),       C.V.LDim() );
        }
    }
    else // B.r < A.r
    {
        const int r = Br;
        C.U.SetType( GENERAL ); C.U.Resize( m, r ); 
        C.V.SetType( GENERAL ); C.V.Resize( n, r ); 

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
            DenseMatrix<Scalar> W( Ar, Br );
            blas::Gemm
            ( 'C', 'N', Ar, Br, A.Height(),
              1, A.U.LockedBuffer(), A.U.LDim(),
                 B.U.LockedBuffer(), B.U.LDim(),
              0, W.Buffer(),         W.LDim() );
            blas::Gemm
            ( 'N', 'N', A.Width(), Ar, Br,
              alpha, A.V.LockedBuffer(), A.V.LDim(), 
                     W.LockedBuffer(),   W.LDim(), 
              0,     C.U.Buffer(),       C.U.LDim() );
            Copy( B.V, C.V );
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
            DenseMatrix<Scalar> W( Br, Ar );
            blas::Gemm
            ( 'C', 'N', Br, Ar, B.Height(),
              1, B.U.LockedBuffer(), B.U.LDim(), 
                 A.U.LockedBuffer(), A.U.LDim(), 
              0, W.Buffer(),         W.LDim() );
            blas::Gemm
            ( 'N', 'T', A.Width(), Br, Ar,
              Conj(alpha), A.V.LockedBuffer(), A.V.LDim(), 
                           W.LockedBuffer(),   W.LDim(), 
              0,           C.U.Buffer(),       C.U.LDim() );
            Conjugate( C.U );
            Copy( B.V, C.V );
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
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Width();
    const int n = B.Width();
    const int r = B.Rank();

    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

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
        DenseMatrix<Scalar> BUConj;
        Conjugate( B.U, BUConj );
        blas::Symm
        ( 'L', 'L', B.Height(), r,
          Conj(alpha), A.LockedBuffer(),      A.LDim(), 
                       BUConj.LockedBuffer(), BUConj.LDim(),
          0,           C.U.Buffer(),          C.U.LDim() );
        Conjugate( C.U );
        Copy( B.V, C.V );
    }
    else
    {
        // C.U C.V^[T,H] := alpha A^H B.U B.V^[T,H]
        //                = (alpha A^H B.U) B.V^[T,H]
        blas::Gemm
        ( 'C', 'N', m, r, A.Height(),
          alpha, A.LockedBuffer(),   A.LDim(), 
                 B.U.LockedBuffer(), B.U.LDim(), 
          0,     C.U.Buffer(),       C.U.LDim() );
        Copy( B.V, C.V );
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
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Width();
    const int n = B.Width();
    const int r = A.Rank();

    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

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
            Copy( A.V, C.U );
            DenseMatrix<Scalar> AUConj;
            Conjugate( A.U, AUConj );
            blas::Symm
            ( 'L', 'L', n, r,
              alpha, B.LockedBuffer(),      B.LDim(), 
                     AUConj.LockedBuffer(), AUConj.LDim(), 
              0,     C.V.Buffer(),          C.V.LDim() );
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
            Copy( A.V, C.U );
            blas::Gemm
            ( 'C', 'N', n, r, A.Height(),
              alpha, B.LockedBuffer(),   B.LDim(), 
                     A.U.LockedBuffer(), A.U.LDim(), 
              0,     C.V.Buffer(),       C.V.LDim() );
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
        DenseMatrix<Scalar> AUConj;
        Conjugate( A.U, AUConj );
        if( B.Symmetric() )
        {
            blas::Symm
            ( 'L', 'L', A.Height(), r, 
              alpha, B.LockedBuffer(),      B.LDim(), 
                     AUConj.LockedBuffer(), AUConj.LDim(), 
              0,     C.V.Buffer(),          C.V.LDim() );
        }
        else
        {
            blas::Gemm
            ( 'T', 'N', n, r, A.Height(),
              alpha, B.LockedBuffer(),      B.LDim(), 
                     AUConj.LockedBuffer(), AUConj.LDim(),
              0,     C.V.Buffer(),          C.V.LDim() );
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
    C.SetType( GENERAL );
    C.Resize( A.Width(), B.Width() );
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
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    const int m = C.Height();
    const int n = C.Width();
    const int r = B.Rank();

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
        DenseMatrix<Scalar> BUConj;
        Conjugate( B.U, BUConj );
        DenseMatrix<Scalar> W( B.Height(), r );
        blas::Symm
        ( 'L', 'L', B.Height(), r,
          1, A.LockedBuffer(),      A.LDim(), 
             BUConj.LockedBuffer(), BUConj.LDim(), 
          0, W.Buffer(),            W.LDim() );
        Conjugate( W );
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, m, n, r,
          alpha, W.LockedBuffer(),   W.LDim(), 
                 B.V.LockedBuffer(), B.V.LDim(), 
          beta,  C.Buffer(),         C.LDim() );
    }
    else
    {
        // C := alpha (A^H B.U) B.V^[T,H] + beta C
        //
        // W := A^H B.U
        // C := alpha W B.V^[T,H] + beta C
        DenseMatrix<Scalar> W( m, r );
        blas::Gemm
        ( 'C', 'N', m, r, B.Height(),
          1, A.LockedBuffer(),   A.LDim(), 
             B.U.LockedBuffer(), B.U.LDim(), 
          0, W.Buffer(),         W.LDim() );
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, m, n, r,
          alpha, W.LockedBuffer(),   W.LDim(), 
                 B.V.LockedBuffer(), B.V.LDim(), 
          beta,  C.Buffer(),         C.LDim() );
    }
}

// Form a dense matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.SetType( GENERAL );
    C.Resize( A.Width(), B.Width() );
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
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    const int m = A.Width();
    const int n = B.Width();
    const int r = A.Rank();

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
            DenseMatrix<Scalar> AUConj;
            Conjugate( A.U, AUConj );
            DenseMatrix<Scalar> W( A.Height(), r );
            blas::Symm
            ( 'L', 'L', A.Height(), r,
              1, B.LockedBuffer(),      B.LDim(), 
                 AUConj.LockedBuffer(), AUConj.LDim(), 
              0, W.Buffer(),            W.LDim() );
            blas::Gemm
            ( 'N', 'T', m, n, r,
              alpha, A.V.LockedBuffer(), A.V.LDim(), 
                     W.LockedBuffer(),   W.LDim(), 
              beta,  C.Buffer(),         C.LDim() );
        }
        else
        {
            // C := alpha (A.U A.V^H)^H B + beta C
            //    = alpha A.V (A.U^H B) + beta C
            //
            // W := A.U^H B
            // C := alpha A.V W + beta C
            DenseMatrix<Scalar> W( r, n );
            blas::Gemm
            ( 'C', 'N', r, n, A.Height(),
              1, A.U.LockedBuffer(), A.U.LDim(), 
                 B.LockedBuffer(),   B.LDim(), 
              0, W.Buffer(),         W.LDim() );
            blas::Gemm
            ( 'N', 'N', m, n, r,
              alpha, A.V.LockedBuffer(), A.V.LDim(), 
                     W.LockedBuffer(),   W.LDim(), 
              beta,  C.Buffer(),         C.LDim() );
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
            DenseMatrix<Scalar> AUConj;
            Conjugate( A.U, AUConj );
            DenseMatrix<Scalar> W( A.Height(), r );
            blas::Symm
            ( 'L', 'L', A.Height(), r,
              1, B.LockedBuffer(),      B.LDim(), 
                 AUConj.LockedBuffer(), AUConj.LDim(), 
              0, W.Buffer(),            W.LDim() );
            DenseMatrix<Scalar> AVConj;
            Conjugate( A.V, AVConj );
            blas::Gemm
            ( 'N', 'T', m, A.Height(), r,
              alpha, AVConj.LockedBuffer(), AVConj.LDim(), 
                     W.LockedBuffer(),      W.LDim(), 
              beta, C.Buffer(),             C.LDim() );
        }
        else
        {
            // C := alpha (A.U A.V^T)^H B + beta C
            //    = alpha conj(A.V) (A.U^H B) + beta C
            //
            // W := A.U^H B
            // AVConj := conj(A.V)
            // C := alpha AVConj W + beta C
            DenseMatrix<Scalar> W( r, n );
            blas::Gemm
            ( 'C', 'N', r, n, A.Height(),
              1, A.U.LockedBuffer(), A.U.LDim(),
                 B.LockedBuffer(),   B.LDim(), 
              0, W.Buffer(),         W.LDim() );
            DenseMatrix<Scalar> AVConj;
            Conjugate( A.V, AVConj );
            blas::Gemm
            ( 'N', 'N', m, n, r,
              alpha, AVConj.LockedBuffer(), AVConj.LDim(), 
                     W.LockedBuffer(),      W.LDim(), 
              beta,  C.Buffer(),            C.LDim() );
        }
    }
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, Real alpha,
  const DenseMatrix<Real>& A,
  const DenseMatrix<Real>& B,
        FactorMatrix<Real,Conjugated>& C )
{
    MatrixTransposeMatrix( maxRank, alpha, A, B, C );
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, std::complex<Real> alpha,
  const DenseMatrix< std::complex<Real> >& A,
  const DenseMatrix< std::complex<Real> >& B,
        FactorMatrix<std::complex<Real>,Conjugated>& C )
{
    typedef std::complex<Real> Scalar;

    const int m = A.Width();
    const int n = B.Width();
    const int minDim = std::min( m, n );
    const int r = std::min( minDim, maxRank );

    // C.U := alpha A^H B
    MatrixHermitianTransposeMatrix( alpha, A, B, C.U );

    // Get the economic SVD of C.U, C.U = U Sigma V^H, overwriting C.U with U.
    Vector<Real> s( minDim );
    DenseMatrix<Scalar> VH( minDim, n );
    const int lwork = lapack::SVDWorkSize( m, n );
    std::vector<Scalar> work( lwork );
    std::vector<Real> rwork( 5*minDim );
    lapack::SVD
    ( 'O', 'S', m, n, C.U.Buffer(), C.U.LDim(),
      s.Buffer(), 0, 0, VH.Buffer(), VH.LDim(),
      &work[0], lwork, &rwork[0] );

    // Truncate the SVD in-place
    C.U.Resize( m, r );
    s.Resize( r );
    VH.Resize( r, n );

    C.V.SetType( GENERAL ); C.V.Resize( n, r );
    if( Conjugated )
    {
        // Put (Sigma V^H)^H = V Sigma into C.V
        for( int j=0; j<r; ++j )
        {
            const Real sigma = s.Get(j);
            Scalar* RESTRICT VCol = C.V.Buffer(0,j);
            const Scalar* RESTRICT VHRow = VH.LockedBuffer(j,0);
            const int VHLDim = VH.LDim();
            for( int i=0; i<n; ++i )
                VCol[i] = sigma*VHRow[i*VHLDim];
        }
    }
    else
    {
        // Put (Sigma V^H)^T = conj(V) Sigma into C.V
        for( int j=0; j<r; ++j )
        {
            const Real sigma = s.Get(j);
            Scalar* RESTRICT VCol = C.V.Buffer(0,j);
            const Scalar* RESTRICT VHRow = VH.LockedBuffer(j,0);
            const int VHLDim = VH.LDim();
            for( int i=0; i<n; ++i )
                VCol[i] = sigma*Conj(VHRow[i*VHLDim]);
        }
    }
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, Real alpha,
  const DenseMatrix<Real>& A,
  const DenseMatrix<Real>& B,
  Real beta,
  FactorMatrix<Real,Conjugated>& C )
{
    MatrixTransposeMatrix( maxRank, alpha, A, B, beta, C );
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, std::complex<Real> alpha,
  const DenseMatrix< std::complex<Real> >& A,
  const DenseMatrix< std::complex<Real> >& B,
  std::complex<Real> beta,
        FactorMatrix< std::complex<Real>,Conjugated>& C )
{
    typedef std::complex<Real> Scalar;

    // D := alpha A^H B + beta C
    DenseMatrix<Scalar> D;
    MatrixHermitianTransposeMatrix( alpha, A, B, D );
    MatrixUpdate( beta, C, (Scalar)1, D );

    // Force D into a factor matrix of rank 'maxRank'
    Compress( maxRank, D, C );
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

// Generate a factor matrix from the product of two dense matrices
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, float alpha,
  const DenseMatrix<float>& A,
  const DenseMatrix<float>& B,
        FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, float alpha,
  const DenseMatrix<float>& A,
  const DenseMatrix<float>& B,
        FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, double alpha,
  const DenseMatrix<double>& A,
  const DenseMatrix<double>& B,
        FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, double alpha,
  const DenseMatrix<double>& A,
  const DenseMatrix<double>& B,
        FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, std::complex<float> alpha,
  const DenseMatrix< std::complex<float> >& A,
  const DenseMatrix< std::complex<float> >& B,
        FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, std::complex<float> alpha,
  const DenseMatrix< std::complex<float> >& A,
  const DenseMatrix< std::complex<float> >& B,
        FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, std::complex<double> alpha,
  const DenseMatrix< std::complex<double> >& A,
  const DenseMatrix< std::complex<double> >& B,
        FactorMatrix<std::complex<double>,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, std::complex<double> alpha,
  const DenseMatrix< std::complex<double> >& A,
  const DenseMatrix< std::complex<double> >& B,
        FactorMatrix<std::complex<double>,true>& C );

// Update a factor matrix from the product of two dense matrices
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, float alpha,
  const DenseMatrix<float>& A,
  const DenseMatrix<float>& B,
  float beta,
        FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, float alpha,
  const DenseMatrix<float>& A,
  const DenseMatrix<float>& B,
  float beta,
        FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, double alpha,
  const DenseMatrix<double>& A,
  const DenseMatrix<double>& B,
  double beta,
        FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, double alpha,
  const DenseMatrix<double>& A,
  const DenseMatrix<double>& B,
  double beta,
        FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, std::complex<float> alpha,
  const DenseMatrix< std::complex<float> >& A,
  const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,
        FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, std::complex<float> alpha,
  const DenseMatrix< std::complex<float> >& A,
  const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,
        FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, std::complex<double> alpha,
  const DenseMatrix< std::complex<double> >& A,
  const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,
        FactorMatrix<std::complex<double>,false>& C );
template void psp::hmatrix_tools::MatrixHermitianTransposeMatrix
( int maxRank, std::complex<double> alpha,
  const DenseMatrix< std::complex<double> >& A,
  const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,
        FactorMatrix<std::complex<double>,true>& C );
