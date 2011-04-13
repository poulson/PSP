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
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (D := D D)");
#endif
    C.SetType( GENERAL );
    C.Resize( A.Height(), B.Width() );
    MatrixMatrix( alpha, A, B, (Scalar)0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

// Dense C := alpha A B + beta C
template<typename Scalar>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (D := D D + D)");
    if( A.Width() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Height() != A.Height() || C.Width() != B.Width() )
        throw std::logic_error("C does not conform with AB");
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
#ifndef RELEASE
    PopCallStack();
#endif
}

// Form a dense matrix from a dense matrix times a low-rank matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B, 
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (D := D F)");
#endif
    C.SetType( GENERAL );
    C.Resize( A.Height(), B.Width() );
    MatrixMatrix( alpha, A, B, (Scalar)0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

// Form a dense matrix from a dense matrix times a low-rank matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (D := D F + D)");
    if( A.Width() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Height() != A.Height() || C.Width() != B.Width() )
        throw std::logic_error("C does not conform with AB");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    // W := A B.U
    DenseMatrix<Scalar> W( A.Height(), B.Rank() );
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', A.Height(), B.Rank(),
          1, A.LockedBuffer(),   A.LDim(), 
             B.U.LockedBuffer(), B.U.LDim(), 
          0, W.Buffer(),         W.LDim() );
    }
    else
    {
        blas::Gemm
        ( 'N', 'N', A.Height(), B.Rank(), A.Width(),
          1, A.LockedBuffer(),   A.LDim(), 
             B.U.LockedBuffer(), B.U.LDim(), 
          0, W.Buffer(),         W.LDim() );
    }
    // C := alpha W B.V^[T,H] + beta C
    const char option = ( Conjugated ? 'C' : 'T' );
    blas::Gemm
    ( 'N', option, C.Height(), C.Width(), B.Rank(),
      alpha, W.LockedBuffer(),   W.LDim(), 
             B.V.LockedBuffer(), B.V.LDim(), 
      beta,  C.Buffer(),         C.LDim() );
#ifndef RELEASE
    PopCallStack();
#endif
}

// Form a dense matrix from a low-rank matrix times a dense matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (D := F D)");
#endif
    C.SetType( GENERAL );
    C.Resize( A.Height(), B.Width() );
    MatrixMatrix( alpha, A, B, (Scalar)0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

// Form a dense matrix from a low-rank matrix times a dense matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (D := F D + D)");
    if( A.Width() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Height() != A.Height() || C.Width() != B.Width() )
        throw std::logic_error("C does not conform with AB");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    if( Conjugated )
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
            DenseMatrix<Scalar> AVConj;
            Conjugate( A.V, AVConj );
            DenseMatrix<Scalar> W( A.Width(), A.Rank() );
            blas::Symm
            ( 'L', 'L', A.Width(), A.Rank(),
              1, B.LockedBuffer(),      B.LDim(), 
                 AVConj.LockedBuffer(), AVConj.LDim(), 
              0, W.Buffer(),            W.LDim() );
            blas::Gemm
            ( 'N', 'T', A.Height(), A.Width(), A.Rank(),
              alpha, A.U.LockedBuffer(), A.U.LDim(), 
                     W.LockedBuffer(),   W.LDim(), 
              beta,  C.Buffer(),         C.LDim() );
        }
        else
        {
            // C := alpha (A.U A.V^H) B + beta C
            //    = alpha A.U (A.V^H B) + beta C
            //
            // W := A.V^H B
            // C := alpha A.U W + beta C
            DenseMatrix<Scalar> W( A.Rank(), B.Width() );
            blas::Gemm
            ( 'C', 'N', A.Rank(), B.Width(), A.Width(),
              1, A.V.LockedBuffer(), A.V.LDim(), 
                 B.LockedBuffer(),   B.LDim(), 
              0, W.Buffer(),         W.LDim() );
            blas::Gemm
            ( 'N', 'N', A.Height(), B.Width(), A.Rank(),
              alpha, A.U.LockedBuffer(), A.U.LDim(), 
                     W.LockedBuffer(),   W.LDim(), 
              beta,  C.Buffer(),         C.LDim() );
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
            DenseMatrix<Scalar> W( A.Width(), A.Rank() );
            blas::Symm
            ( 'L', 'L', A.Width(), A.Rank(),
              1, B.LockedBuffer(),   B.LDim(), 
                 A.V.LockedBuffer(), A.V.LDim(), 
              0, W.Buffer(),         W.LDim() );
            blas::Gemm
            ( 'N', 'T', A.Height(), A.Width(), A.Rank(),
              alpha, A.U.LockedBuffer(), A.U.LDim(), 
                     W.LockedBuffer(),   W.LDim(), 
              beta,  C.Buffer(),         C.LDim() );
        }
        else
        {
            // C := alpha (A.U A.V^T) B + beta C
            //    = alpha A.U (A.V^T B) + beta C
            //
            // W := A.V^T B
            // C := alpha A.U W + beta C
            DenseMatrix<Scalar> W( A.Rank(), B.Width() );
            blas::Gemm
            ( 'T', 'N', A.Rank(), B.Width(), A.Width(),
              1, A.V.LockedBuffer(), A.V.LDim(), 
                 B.LockedBuffer(),   B.LDim(), 
              0, W.Buffer(),         A.Rank() );
            blas::Gemm
            ( 'N', 'N', A.Height(), B.Width(), A.Rank(),
              alpha, A.U.LockedBuffer(), A.U.LDim(), 
                     W.LockedBuffer(),   W.LDim(), 
              beta,  C.Buffer(),         C.LDim() );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Form a dense matrix from the product of two low-rank matrices.
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A,
                const LowRankMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (D := F F)");
#endif
    C.SetType( GENERAL ); C.Resize( A.Height(), B.Width() );
    MatrixMatrix( alpha, A, B, (Scalar)0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

// Update a dense matrix from the product of two low-rank matrices.
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A,
                const LowRankMatrix<Scalar,Conjugated>& B,
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (D := F F + D)");
    if( A.Width() != B.Height() )
        throw std::logic_error("A and B do not conform in MatrixMatrix");
    if( C.Height() != A.Height() || C.Width() != B.Width() )
        throw std::logic_error("C does not conform with AB");
#endif
    const char option = ( Conjugated ? 'C' : 'T' );
    DenseMatrix<Scalar> W( A.Rank(), B.Rank() );
    blas::Gemm
    ( option, 'N', A.Rank(), B.Rank(), A.Width(),
      1, A.V.LockedBuffer(), A.V.LDim(), B.U.LockedBuffer(), B.U.LDim(),
      0, W.Buffer(), W.LDim() );
    DenseMatrix<Scalar> X( A.Height(), B.Rank() );
    blas::Gemm
    ( 'N', 'N', A.Height(), B.Rank(), A.Rank(),
      1, A.U.LockedBuffer(), A.U.LDim(), W.LockedBuffer(), W.LDim(),
      0, X.Buffer(), X.LDim() );
    blas::Gemm
    ( 'N', option, C.Height(), C.Width(), B.Rank(),
      alpha, X.LockedBuffer(), X.LDim(), B.V.LockedBuffer(), B.V.LDim(),
      beta,  C.Buffer(), C.LDim() );
#ifndef RELEASE
    PopCallStack();
#endif
}

// Low-rank C := alpha A B
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B, 
                      LowRankMatrix<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (F := F F)");
    if( A.Width() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Height();
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
            // C.U C.V^H := alpha (A.U A.V^H) (B.U B.V^H)
            //            = A.U (alpha A.V^H B.U B.V^H)
            //            = A.U (conj(alpha) B.V (B.U^H A.V))^H
            //
            // C.U := A.U
            // W := B.U^H A.V
            // C.V := conj(alpha) B.V W
            Copy( A.U, C.U );
            DenseMatrix<Scalar> W( Br, Ar );
            blas::Gemm
            ( 'C', 'N', Br, Ar, B.Height(),
              1, B.U.LockedBuffer(), B.U.LDim(), 
                 A.V.LockedBuffer(), A.V.LDim(), 
              0, W.Buffer(),         W.LDim() );
            blas::Gemm
            ( 'N', 'N', n, Ar, Br,
              Conj(alpha), B.V.LockedBuffer(), B.V.LDim(), 
                           W.LockedBuffer(),   W.LDim(), 
              0,           C.V.Buffer(),       C.V.LDim() );
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
            Copy( A.U, C.U );
            DenseMatrix<Scalar> W( Br, Ar );
            blas::Gemm
            ( 'T', 'N', Br, Ar, B.Height(),
              1, B.U.LockedBuffer(), B.U.LDim(),
                 A.V.LockedBuffer(), A.V.LDim(),
              0, W.Buffer(),         W.LDim() );
            blas::Gemm
            ( 'N', 'N', n, Ar, Br,
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
            // C.U C.V^H := alpha (A.U A.V^H) (B.U B.V^H)
            //            = (alpha A.U (A.V^H B.U)) B.V^H
            //
            // W := A.V^H B.U
            // C.U := alpha A.U W
            // C.V := B.V
            DenseMatrix<Scalar> W( Ar, Br );
            blas::Gemm
            ( 'C', 'N', Ar, Br, A.Width(),
              1, A.V.LockedBuffer(), A.V.LDim(), 
                 B.U.LockedBuffer(), B.U.LDim(), 
              0, W.Buffer(),         W.LDim() );
            blas::Gemm
            ( 'N', 'N', m, Br, Ar,
              alpha, A.U.LockedBuffer(), A.U.LDim(), 
                     W.LockedBuffer(),   W.LDim(), 
              0,     C.U.Buffer(),       C.U.LDim() );
            Copy( B.V, C.V );
        }
        else
        {
            // C.U C.V^T := alpha (A.U A.V^T) (B.U B.V^T)
            //            = (alpha A.U (A.V^T B.U)) B.V^T
            //
            // W := A.V^T B.U
            // C.U := alpha A.U W
            // C.V := B.V
            DenseMatrix<Scalar> W( Ar, Br );
            blas::Gemm
            ( 'T', 'N', Ar, Br, A.Width(),
              1, A.V.LockedBuffer(), A.V.LDim(), 
                 B.U.LockedBuffer(), B.U.LDim(), 
              0, W.Buffer(),         W.LDim() );
            blas::Gemm
            ( 'N', 'N', m, Br, Ar,
              alpha, A.U.LockedBuffer(), A.U.LDim(), 
                     W.LockedBuffer(),   W.LDim(), 
              0,     C.U.Buffer(),       C.U.LDim() );
            Copy( B.V, C.V );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Form a low-rank matrix from a dense matrix times a low-rank matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const LowRankMatrix<Scalar,Conjugated>& B, 
                      LowRankMatrix<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (F := D F)");
    if( A.Width() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Height();
    const int n = B.Width();
    const int r = B.Rank();

    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

    // Form C.U := A B.U
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', m, r, 
          alpha, A.LockedBuffer(),   A.LDim(), 
                 B.U.LockedBuffer(), B.U.LDim(), 
          0,     C.U.Buffer(),       C.U.LDim() );
    }
    else
    {
        blas::Gemm
        ( 'N', 'N', m, r, A.Width(),
          alpha, A.LockedBuffer(),   A.LDim(), 
                 B.U.LockedBuffer(), B.U.LDim(), 
          0,     C.U.Buffer(),       C.U.LDim() );
    }

    // Form C.V := B.V
    Copy( B.V, C.V );
#ifndef RELEASE
    PopCallStack();
#endif
}

// Form a low-rank matrix from a low-rank matrix times a dense matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B, 
                      LowRankMatrix<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (F := F D)");
    if( A.Width() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Height();
    const int n = B.Width();
    const int r = A.Rank();

    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

    if( Conjugated )
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
            Copy( A.U, C.U );
            DenseMatrix<Scalar> AVConj;
            Conjugate( A.V, AVConj );
            blas::Symm
            ( 'L', 'L', A.Width(), r,
              alpha, B.LockedBuffer(),      B.LDim(), 
                     AVConj.LockedBuffer(), AVConj.LDim(), 
              0,     C.V.Buffer(),          C.V.LDim() );
            Conjugate( C.V );
        }
        else
        {
            // C.U C.V^H := alpha (A.U A.V^H) B
            //            = A.U (conj(alpha) B^H A.V)^H
            //
            // C.U := A.U
            // C.V := conj(alpha) B^H A.V
            Copy( A.U, C.U );
            blas::Gemm
            ( 'C', 'N', n, r, A.Width(),
              Conj(alpha), B.LockedBuffer(),   B.LDim(), 
                           A.V.LockedBuffer(), A.V.LDim(),
              0,           C.V.Buffer(),       C.V.LDim() );
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
            Copy( A.U, C.U );
            blas::Symm
            ( 'L', 'L', A.Width(), r,
              alpha, B.LockedBuffer(),   B.LDim(), 
                     A.V.LockedBuffer(), A.V.LDim(), 
              0,     C.V.Buffer(),       C.V.LDim() );
        }
        else
        {
            // C.U C.V^T := alpha (A.U A.V^T) B
            //            = A.U (alpha B^T A.V)^T
            //
            // C.U := A.U
            // C.V := alpha B^T A.V
            Copy( A.U, C.U );
            blas::Gemm
            ( 'T', 'N', n, r, A.Width(),
              alpha, B.LockedBuffer(),   B.LDim(), 
                     A.V.LockedBuffer(), A.V.LDim(),
              0,     C.V.Buffer(),       C.V.LDim() );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( int maxRank, Real alpha, 
  const DenseMatrix<Real>& A,
  const DenseMatrix<Real>& B,
        LowRankMatrix<Real,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (F := D D)");
#endif
    const int m = A.Height();
    const int n = B.Width();
    const int minDim = std::min( m, n );
    const int r = std::min( minDim, maxRank );

    // C.U := alpha A B
    MatrixMatrix( alpha, A, B, C.U );

    // Get the economic SVD of C.U, C.U = U Sigma V^T, overwriting C.U with U.
    Vector<Real> s( minDim );
    DenseMatrix<Real> VT( minDim, n );
    const int lwork = lapack::SVDWorkSize( m, n );
    std::vector<Real> work( lwork );
    lapack::SVD
    ( 'O', 'S', m, n, C.U.Buffer(), C.U.LDim(), 
      s.Buffer(), 0, 0, VT.Buffer(), VT.LDim(),
      &work[0], lwork );

    // Truncate the SVD in-place
    C.U.Resize( m, r );
    s.Resize( r );
    VT.Resize( r, n );

    // Put (Sigma V^T)^T = V Sigma into C.V
    C.V.SetType( GENERAL ); C.V.Resize( n, r );
    for( int j=0; j<r; ++j )
    {
        const Real sigma = s.Get(j);
        Real* RESTRICT VCol = C.V.Buffer(0,j);
        const Real* RESTRICT VTRow = VT.LockedBuffer(j,0);
        const int VTLDim = VT.LDim();
        for( int i=0; i<n; ++i )
            VCol[i] = sigma*VTRow[i*VTLDim];
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( int maxRank, std::complex<Real> alpha, 
  const DenseMatrix< std::complex<Real> >& A,
  const DenseMatrix< std::complex<Real> >& B,
        LowRankMatrix< std::complex<Real>,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (F := D D)");
#endif
    typedef std::complex<Real> Scalar;

    const int m = A.Height();
    const int n = B.Width();
    const int minDim = std::min( m, n );
    const int r = std::min( minDim, maxRank );

    // C.U := alpha A B
    MatrixMatrix( alpha, A, B, C.U );

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
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( int maxRank, Real alpha, 
  const DenseMatrix<Real>& A,
  const DenseMatrix<Real>& B,
  Real beta,
  LowRankMatrix<Real,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (F := D D + F)");
    if( A.Width() != B.Height() )
        throw std::logic_error("A and B not conformal in MatrixMatrix");
    if( C.Height() != A.Height() || C.Width() != A.Width() )
        throw std::logic_error("C not conformal with AB");
#endif
    // D := alpha A B + beta C
    DenseMatrix<Real> D;
    MatrixMatrix( alpha, A, B, D );
    MatrixUpdate( beta, C, (Real)1, D );

    // Truncate D down to a low-rank matrix of rank 'maxRank'
    Compress( maxRank, D, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Real,bool Conjugated>
void psp::hmatrix_tools::MatrixMatrix
( int maxRank, std::complex<Real> alpha, 
  const DenseMatrix< std::complex<Real> >& A,
  const DenseMatrix< std::complex<Real> >& B,
  std::complex<Real> beta,
        LowRankMatrix< std::complex<Real>,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixMatrix (F := D D + F)");
    if( A.Width() != B.Height() )
        throw std::logic_error("A and B not conformal in MatrixMatrix");
    if( C.Height() != A.Height() || C.Width() != A.Width() )
        throw std::logic_error("C not conformal with AB");
#endif
    typedef std::complex<Real> Scalar;

    // D := alpha A B + beta C
    DenseMatrix<Scalar> D;
    MatrixMatrix( alpha, A, B, D );
    MatrixUpdate( beta, C, (Scalar)1, D );

    // Truncate D down to a low-rank matrix of rank 'maxRank'
    Compress( maxRank, D, C );
#ifndef RELEASE
    PopCallStack();
#endif
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

// Form a dense matrix from a dense matrix times a low-rank matrix
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A, 
               const LowRankMatrix<float,false>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A, 
               const LowRankMatrix<float,true>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const LowRankMatrix<double,false>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const LowRankMatrix<double,true>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, 
  const DenseMatrix< std::complex<float> >& A,
  const LowRankMatrix<std::complex<float>,false>& B,
        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, 
  const DenseMatrix< std::complex<float> >& A,
  const LowRankMatrix<std::complex<float>,true>& B,
        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const DenseMatrix< std::complex<double> >& A,
  const LowRankMatrix<std::complex<double>,false>& B,
        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const DenseMatrix< std::complex<double> >& A,
  const LowRankMatrix<std::complex<double>,true>& B,
        DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a dense matrix times a low-rank matrix
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A, 
               const LowRankMatrix<float,false>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A, 
               const LowRankMatrix<float,true>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const LowRankMatrix<double,false>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const LowRankMatrix<double,true>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, 
  const DenseMatrix< std::complex<float> >& A,
  const LowRankMatrix<std::complex<float>,false>& B,
  std::complex<float> beta, 
        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, 
  const DenseMatrix< std::complex<float> >& A,
  const LowRankMatrix<std::complex<float>,true>& B,
  std::complex<float> beta, 
        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const DenseMatrix< std::complex<double> >& A,
  const LowRankMatrix<std::complex<double>,false>& B,
  std::complex<double> beta,        
        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const DenseMatrix< std::complex<double> >& A,
  const LowRankMatrix<std::complex<double>,true>& B,
  std::complex<double> beta,
        DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a low-rank matrix times a dense matrix
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const LowRankMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const LowRankMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const LowRankMatrix<double,false>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const LowRankMatrix<double,true>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,false>& A,
  const DenseMatrix< std::complex<float> >& B,
        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,true>& A,
  const DenseMatrix< std::complex<float> >& B,
        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,false>& A,
  const DenseMatrix< std::complex<double> >& B,
        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,true>& A,
  const DenseMatrix< std::complex<double> >& B,
        DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a low-rank matrix times a dense matrix
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const LowRankMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const LowRankMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const LowRankMatrix<double,false>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const LowRankMatrix<double,true>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const LowRankMatrix<std::complex<float>,false>& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, const LowRankMatrix<std::complex<float>,true>& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,false>& A,
  const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        
        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,true>& A,
  const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        
        DenseMatrix< std::complex<double> >& C );

template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const LowRankMatrix<float,false>& A,
               const LowRankMatrix<float,false>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const LowRankMatrix<float,true>& A,
               const LowRankMatrix<float,true>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const LowRankMatrix<double,false>& A,
                const LowRankMatrix<double,false>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const LowRankMatrix<double,true>& A,
                const LowRankMatrix<double,true>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha,
  const LowRankMatrix<std::complex<float>,false>& A,
  const LowRankMatrix<std::complex<float>,false>& B,
        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha,
  const LowRankMatrix<std::complex<float>,true>& A,
  const LowRankMatrix<std::complex<float>,true>& B,
        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha,
  const LowRankMatrix<std::complex<double>,false>& A,
  const LowRankMatrix<std::complex<double>,false>& B,
        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha,
  const LowRankMatrix<std::complex<double>,true>& A,
  const LowRankMatrix<std::complex<double>,true>& B,
        DenseMatrix< std::complex<double> >& C );

template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const LowRankMatrix<float,false>& A,
               const LowRankMatrix<float,false>& B,
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const LowRankMatrix<float,true>& A,
               const LowRankMatrix<float,true>& B,
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const LowRankMatrix<double,false>& A,
                const LowRankMatrix<double,false>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const LowRankMatrix<double,true>& A,
                const LowRankMatrix<double,true>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha,
  const LowRankMatrix<std::complex<float>,false>& A,
  const LowRankMatrix<std::complex<float>,false>& B,
  std::complex<float> beta,
        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha,
  const LowRankMatrix<std::complex<float>,true>& A,
  const LowRankMatrix<std::complex<float>,true>& B,
  std::complex<float> beta,
        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha,
  const LowRankMatrix<std::complex<double>,false>& A,
  const LowRankMatrix<std::complex<double>,false>& B,
  std::complex<double> beta,
        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha,
  const LowRankMatrix<std::complex<double>,true>& A,
  const LowRankMatrix<std::complex<double>,true>& B,
  std::complex<double> beta,
        DenseMatrix< std::complex<double> >& C );

// Low-rank C := alpha A B
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const LowRankMatrix<float,false>& A,
               const LowRankMatrix<float,false>& B,
                     LowRankMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const LowRankMatrix<float,true>& A,
               const LowRankMatrix<float,true>& B,
                     LowRankMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const LowRankMatrix<double,false>& A,
                const LowRankMatrix<double,false>& B,
                      LowRankMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const LowRankMatrix<double,true>& A,
                const LowRankMatrix<double,true>& B,
                      LowRankMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,false>& A,
  const LowRankMatrix<std::complex<float>,false>& B,
        LowRankMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,true>& A,
  const LowRankMatrix<std::complex<float>,true>& B,
        LowRankMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,false>& A,
  const LowRankMatrix<std::complex<double>,false>& B,
        LowRankMatrix<std::complex<double>,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,true>& A,
  const LowRankMatrix<std::complex<double>,true>& B,
        LowRankMatrix<std::complex<double>,true>& C );

// Form a low-rank matrix from a dense matrix times a low-rank matrix
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A, 
               const LowRankMatrix<float,false>& B, 
                     LowRankMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const DenseMatrix<float>& A, 
               const LowRankMatrix<float,true>& B, 
                     LowRankMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const LowRankMatrix<double,false>& B,
                      LowRankMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const DenseMatrix<double>& A,
                const LowRankMatrix<double,true>& B,
                      LowRankMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, 
  const DenseMatrix< std::complex<float> >& A,
  const LowRankMatrix<std::complex<float>,false>& B,
        LowRankMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, 
  const DenseMatrix< std::complex<float> >& A,
  const LowRankMatrix<std::complex<float>,true>& B,
        LowRankMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const DenseMatrix< std::complex<double> >& A,
  const LowRankMatrix<std::complex<double>,false>& B,
        LowRankMatrix<std::complex<double>,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const DenseMatrix< std::complex<double> >& A,
  const LowRankMatrix<std::complex<double>,true>& B,
        LowRankMatrix<std::complex<double>,true>& C );

// Form a low-rank matrix from a low-rank matrix times a dense matrix
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const LowRankMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
                     LowRankMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( float alpha, const LowRankMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
                     LowRankMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const LowRankMatrix<double,false>& A,
                const DenseMatrix<double>& B,
                      LowRankMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( double alpha, const LowRankMatrix<double,true>& A,
                const DenseMatrix<double>& B,
                      LowRankMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,false>& A,
  const DenseMatrix< std::complex<float> >& B,
        LowRankMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,true>& A,
  const DenseMatrix< std::complex<float> >& B,
        LowRankMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,false>& A,
  const DenseMatrix< std::complex<double> >& B,
        LowRankMatrix<std::complex<double>,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,true>& A,
  const DenseMatrix< std::complex<double> >& B,
        LowRankMatrix<std::complex<double>,true>& C );

// Generate a low-rank matrix from the product of two dense matrices
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, float alpha, 
  const DenseMatrix<float>& A,
  const DenseMatrix<float>& B,
        LowRankMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, float alpha, 
  const DenseMatrix<float>& A,
  const DenseMatrix<float>& B,
        LowRankMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, double alpha,
  const DenseMatrix<double>& A,
  const DenseMatrix<double>& B,
        LowRankMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, double alpha,
  const DenseMatrix<double>& A,
  const DenseMatrix<double>& B,
        LowRankMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, std::complex<float> alpha,
  const DenseMatrix< std::complex<float> >& A,
  const DenseMatrix< std::complex<float> >& B,
        LowRankMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, std::complex<float> alpha,
  const DenseMatrix< std::complex<float> >& A,
  const DenseMatrix< std::complex<float> >& B,
        LowRankMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, std::complex<double> alpha,
  const DenseMatrix< std::complex<double> >& A,
  const DenseMatrix< std::complex<double> >& B,
        LowRankMatrix<std::complex<double>,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, std::complex<double> alpha,
  const DenseMatrix< std::complex<double> >& A,
  const DenseMatrix< std::complex<double> >& B,
        LowRankMatrix<std::complex<double>,true>& C );

// Update a low-rank matrix from the product of two dense matrices
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, float alpha, 
  const DenseMatrix<float>& A,
  const DenseMatrix<float>& B,
  float beta,
        LowRankMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, float alpha, 
  const DenseMatrix<float>& A,
  const DenseMatrix<float>& B,
  float beta,
        LowRankMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, double alpha,
  const DenseMatrix<double>& A,
  const DenseMatrix<double>& B,
  double beta,
        LowRankMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, double alpha,
  const DenseMatrix<double>& A,
  const DenseMatrix<double>& B,
  double beta,
        LowRankMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, std::complex<float> alpha,
  const DenseMatrix< std::complex<float> >& A,
  const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,
        LowRankMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, std::complex<float> alpha,
  const DenseMatrix< std::complex<float> >& A,
  const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,
        LowRankMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, std::complex<double> alpha,
  const DenseMatrix< std::complex<double> >& A,
  const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,
        LowRankMatrix<std::complex<double>,false>& C );
template void psp::hmatrix_tools::MatrixMatrix
( int maxRank, std::complex<double> alpha,
  const DenseMatrix< std::complex<double> >& A,
  const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,
        LowRankMatrix<std::complex<double>,true>& C );
