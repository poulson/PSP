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
    C.SetType( GENERAL );
    C.Resize( A.Width(), B.Width() );
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
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixTransposeMatrix
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
            // C.U C.V^H := alpha (A.U A.V^H)^T (B.U B.V^H)
            //            = alpha conj(A.V) (A.U^T B.U B.V^H)
            //            = conj(A.V) (conj(alpha) B.V B.U^H conj(A.U))^H
            //            = conj(A.V) (conj(alpha) B.V (A.U^T B.U)^H)^H
            //
            // C.U := conj(A.V)
            // W := A.U^T B.U
            // C.V := conj(alpha) B.V W^H
            Conjugate( A.V, C.U );
            DenseMatrix<Scalar> W( Ar, Br );
            blas::Gemm
            ( 'T', 'N', Ar, Br, A.Height(),
              1, A.U.LockedBuffer(), A.U.LDim(), 
                 B.U.LockedBuffer(), B.U.LDim(), 
              0, W.Buffer(),         W.LDim() );
            blas::Gemm
            ( 'N', 'C', n, Ar, Br,
              Conj(alpha), B.V.LockedBuffer(), B.V.LDim(), 
                           W.LockedBuffer(),   W.LDim(), 
              0,           C.V.Buffer(),       C.V.LDim() );
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
            Copy( A.V, C.U );
            DenseMatrix<Scalar> W( Br, Ar );
            blas::Gemm
            ( 'T', 'N', Br, Ar, B.Height(),
              1, B.U.LockedBuffer(), B.U.LDim(), 
                 A.U.LockedBuffer(), A.U.LDim(), 
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
            // C.U C.V^H := alpha (A.U A.V^H)^T (B.U B.V^H)
            //            = alpha conj(A.V) A.U^T B.U B.V^H
            //            = (alpha conj(A.V) (A.U^T B.U)) B.V^H
            //
            // W := A.U^T B.U
            // AVConj := conj(A.V)
            // C.U := alpha AVConj W
            // C.V := B.V
            DenseMatrix<Scalar> W( Ar, Br );
            blas::Gemm
            ( 'T', 'N', Ar, Br, A.Height(),
              1, A.U.LockedBuffer(), A.U.LDim(), 
                 B.U.LockedBuffer(), B.U.LDim(), 
              0, W.Buffer(),         W.LDim() );
            DenseMatrix<Scalar> AVConj;
            Conjugate( A.V, AVConj );
            blas::Gemm
            ( 'N', 'N', m, Br, Ar,
              alpha, AVConj.LockedBuffer(), AVConj.LDim(), 
                     W.LockedBuffer(),      W.LDim(), 
              0,     C.U.Buffer(),          C.U.LDim() );
            Copy( B.V, C.V );
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
            DenseMatrix<Scalar> W( Ar, Br );
            blas::Gemm
            ( 'T', 'N', Ar, Br, A.Height(),
              1, A.U.LockedBuffer(), A.U.LDim(), 
                 B.U.LockedBuffer(), B.U.LDim(), 
              0, W.Buffer(),         W.LDim() );
            blas::Gemm
            ( 'N', 'N', m, Br, Ar,
              alpha, A.V.LockedBuffer(), A.V.LDim(), 
                     W.LockedBuffer(),   W.LDim(), 
              0,     C.U.Buffer(),       C.U.LDim() );
            Copy( B.V, C.V );
        }
    }
}

// Form a factor matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixTransposeMatrix
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
        ( 'T', 'N', m, r, A.Height(),
          alpha, A.LockedBuffer(),   A.LDim(), 
                 B.U.LockedBuffer(), B.U.LDim(), 
          0,     C.U.Buffer(),       C.U.LDim() );
    }

    // Form C.V := B.V
    Copy( B.V, C.V );
}

// Form a factor matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixTransposeMatrix
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
            Conjugate( A.V, C.U );
            blas::Symm
            ( 'L', 'L', A.Height(), r,
              alpha, B.LockedBuffer(),   B.LDim(), 
                     A.U.LockedBuffer(), A.U.LDim(), 
              0,     C.V.Buffer(),       C.V.LDim() );
            Conjugate( C.V );
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
            Conjugate( A.V, C.U );
            blas::Gemm
            ( 'T', 'N', n, r, A.Height(),
              alpha, B.LockedBuffer(),   B.LDim(), 
                     A.U.LockedBuffer(), A.U.LDim(), 
              0,     C.V.Buffer(),       C.V.LDim() );
            Conjugate( C.V );
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
            Copy( A.V, C.U );
            blas::Symm
            ( 'L', 'L', A.Height(), r,
              alpha, B.LockedBuffer(),   B.LDim(), 
                     A.U.LockedBuffer(), A.U.LDim(), 
              0,     C.V.Buffer(),       C.V.LDim() );
        }
        else
        {
            // C.U C.V^T := alpha (A.U A.V^T)^T B
            //            = alpha A.V A.U^T B
            //            = A.V (alpha B^T A.U)^T
            //
            // C.U := A.V
            // C.V := alpha B^T A.U
            Copy( A.V, C.U );
            blas::Gemm
            ( 'T', 'N', n, r, A.Height(),
              alpha, B.LockedBuffer(),   B.LDim(), 
                     A.U.LockedBuffer(), A.U.LDim(), 
              0,     C.V.Buffer(),       C.V.LDim() );
        }
    }
}

// Form a dense matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.SetType( GENERAL );
    C.Resize( A.Width(), B.Width() );
    MatrixTransposeMatrix( alpha, A, B, (Scalar)0, C );
}

// Form a dense matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixTransposeMatrix
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugated>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Symmetric() )
        throw std::logic_error("Update is probably not symmetric.");
#endif
    // W := A^T B.U
    DenseMatrix<Scalar> W( A.Width(), B.Rank() );
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', A.Width(), B.Rank(),
          1, A.LockedBuffer(), A.LDim(), B.U.LockedBuffer(), B.U.LDim(), 
          0, W.Buffer(), W.LDim() );
    }
    else
    {
        blas::Gemm
        ( 'T', 'N', A.Width(), B.Rank(), A.Height(),
          1, A.LockedBuffer(), A.LDim(), B.U.LockedBuffer(), B.U.LDim(), 
          0, W.Buffer(), W.LDim() );
    }
    // C := alpha W B.V^[T,H] + beta C
    const char option = ( Conjugated ? 'C' : 'T' );
    blas::Gemm
    ( 'N', option, C.Height(), C.Width(), B.Rank(),
      alpha, W.LockedBuffer(), W.LDim(), B.V.LockedBuffer(), B.V.LDim(), 
      beta,  C.Buffer(), C.LDim() );
}

// Form a dense matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.SetType( GENERAL );
    C.Resize( A.Width(), B.Width() );
    MatrixTransposeMatrix( alpha, A, B, (Scalar)0, C );
}

// Form a dense matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixTransposeMatrix
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
                const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Symmetric() )
        throw std::logic_error("Update is probably not symmetric.");
#endif
    const int m = A.Width();
    const int n = B.Width();
    const int r = A.Rank();

    if( Conjugated )
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
            DenseMatrix<Scalar> W( A.Height(), r );
            blas::Symm
            ( 'L', 'L', A.Height(), r,
              1, B.LockedBuffer(), B.LDim(), A.U.LockedBuffer(), A.U.LDim(), 
              0, W.Buffer(), W.LDim() );
            DenseMatrix<Scalar> AVConj;
            Conjugate( A.V, AVConj );
            blas::Gemm
            ( 'N', 'T', m, A.Height(), r,
              alpha, AVConj.LockedBuffer(), AVConj.LDim(), 
                     W.LockedBuffer(),      W.LDim(), 
              beta,  C.Buffer(),            C.LDim() );
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
            DenseMatrix<Scalar> W( r, n );
            blas::Gemm
            ( 'T', 'N', r, n, A.Height(),
              1, A.U.LockedBuffer(), A.U.LDim(), B.LockedBuffer(), B.LDim(), 
              0, W.Buffer(), W.LDim() );
            DenseMatrix<Scalar> AVConj;
            Conjugate( A.V, AVConj );
            blas::Gemm
            ( 'N', 'N', m, A.Height(), r,
              alpha, AVConj.LockedBuffer(), AVConj.LDim(), 
                     W.LockedBuffer(),      W.LDim(), 
              beta,  C.Buffer(),            C.LDim() );
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
            DenseMatrix<Scalar> W( A.Height(), r );
            blas::Symm
            ( 'L', 'L', A.Height(), r,
              1, B.LockedBuffer(), B.LDim(), A.U.LockedBuffer(), A.U.LDim(), 
              0, W.Buffer(), W.LDim() );
            blas::Gemm
            ( 'N', 'N', m, A.Height(), r,
              alpha, A.V.LockedBuffer(), A.V.LDim(), W.LockedBuffer(), W.LDim(),
              beta,  C.Buffer(), C.LDim() );
        }
        else
        {
            // C := alpha (A.U A.V^T)^T B + beta C
            //    = alpha A.V (A.U^T B) + beta C
            //
            // W := A.U^T B
            // C := alpha A.V W + beta C
            DenseMatrix<Scalar> W( r, n );
            blas::Gemm
            ( 'T', 'N', r, n, A.Height(),
              1, A.U.LockedBuffer(), A.U.LDim(), B.LockedBuffer(), B.LDim(), 
              0, W.Buffer(), W.LDim() );
            blas::Gemm
            ( 'N', 'N', m, n, r,
              alpha, A.V.LockedBuffer(), A.V.LDim(), W.LockedBuffer(), W.LDim(),
              beta,  C.Buffer(), C.LDim() );
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
