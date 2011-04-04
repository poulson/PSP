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
    if( A.r < B.r )
    {
        // C := conj(A.V) (conj(alpha) B.V (A.U^T B.U)^H)^H
        //                  or
        // C := A.V       (alpha       B.V (A.U^T B.U)^T))^T
        C.r = A.r;

        // C.U := conj(A.V)
        //       or
        // C.U := A.V
        C.U.resize( C.m*C.r );
        if( Conjugate )
        {
            const int m = C.m;
            const int r = C.r;
            for( int j=0; j<r; ++j )
            {
                const Scalar* RESTRICT AV = &A.V[j*A.n];
                Scalar* RESTRICT CU = &C.U[j*A.n];
                for( int i=0; i<m; ++i )
                    CU[i] = Conj( AV[i] );
            }
        }
        else
        {
            std::memcpy( &C.U[0], &A.V[0], C.m*C.r*sizeof(Scalar) );
        }

        // C.V := conj(alpha) B.V (A.U^T B.U)^H
        //                  or
        // C.V := alpha       B.V (A.U^T B.U)^T
        C.V.resize( C.n*C.r );
        std::vector<Scalar> W( A.r*B.r );
        // W := A.U^T B.U
        blas::Gemm
        ( 'T', 'N', A.r, B.r, A.m,
          1, &A.U[0], A.m, &B.U[0], B.m, 0, &W[0], A.r );
        // C.V := conj(alpha) B.V W = conj(alpha) B.V (A.U^T B.U)^H
        //                  or
        // C.V := alpha       B.V W = alpha       B.V (A.U^T B.U)^T
        const char option = ( Conjugate ? 'C' : 'T' );
        const Scalar scale = ( Conjugate ? Conj(alpha) : alpha );
        blas::Gemm
        ( 'N', option, C.n, C.r, B.r, 
          scale, &B.V[0], B.n, &W[0], B.r, 0, &C.V[0], C.n );
    }
    else
    {
        // C := (alpha conj(A.V) (A.U^T B.U)) B.V^H
        //                     or
        // C := (alpha A.V       (A.U^T B.U)) B.V^T
        C.r = B.r;

        // C.U := alpha conj(A.V) (A.U^T B.U)
        //                  or
        // C.U := alpha A.V       (A.U^T B.U)
        C.U.resize( C.m*C.r );
        std::vector<Scalar> W( A.r*B.r );
        // W := A.U^T B.U
        blas::Gemm
        ( 'T', 'N', A.r, B.r, A.m,
          1, &A.U[0], A.m, &B.U[0], B.m, 0, &W[0], A.r );
        // C.U := alpha conj(A.V) W = alpha conj(A.V) (A.U^T B.U)
        //                      or
        // C.U := alpha A.V       W = alpha A.V       (A.U^T B.U)
        if( Conjugate )
        {
            const int size = A.n*A.r;
            std::vector<Scalar> AVConj( size );
            const Scalar* RESTRICT AV = &A.V[0];
            for( int i=0; i<size; ++i )
                AVConj[i] = Conj( AV[i] );

            blas::Gemm
            ( 'N', 'N', C.m, C.r, A.r,
              alpha, &AVConj[0], A.n, &W[0], A.r, 0, &C.U[0], C.m );
        }
        else
        {
            blas::Gemm
            ( 'N', 'N', C.m, C.r, A.r,
              alpha, &A.V[0], A.n, &W[0], A.r, 0, &C.U[0], C.m );
        }

        // C.V := B.V
        C.V.resize( C.n*C.r );
        std::memcpy( &C.V[0], &B.V[0], C.n*C.r*sizeof(Scalar) );
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
        // Form C.U := conj(A.V)
        const int r = A.r;
        const int n = A.n;
        const Scalar* RESTRICT AV = &A.V[0];
        Scalar* RESTRICT AVConj = &C.U[0];
        for( int j=0; j<r; ++j )
            for( int i=0; i<n; ++i )
                AVConj[i] = Conj( AV[i] );
    }
    else
    {
        // Form C.U := A.V
        std::memcpy( &C.U[0], &A.V[0], A.n*A.r*sizeof(Scalar) );
    }

    // Form C.V := conj( alpha B^T A.U )
    //                or
    //      C.V :=       alpha B^T A.U
    if( B.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', C.n, C.r, 
          alpha, B.LockedBuffer(), B.LDim(), &A.U[0], A.m, 
          0, &C.V[0], C.n );
        if( Conjugate )
        {
            Scalar* CV = &C.V[0];
            const int size = C.n*C.r;
            for( int i=0; i<size; ++i )
                CV[i] = Conj( CV[i] );
        }
    }
    else
    {
        blas::Gemm
        ( 'T', 'N', C.n, C.r, B.Height(),
          alpha, B.LockedBuffer(), B.LDim(), &A.U[0], A.m, 0, &C.V[0], C.n );
        if( Conjugate )
        {
            Scalar* CV = &C.V[0];
            const int size = C.n*C.r;
            for( int i=0; i<size; ++i )
                CV[i] = Conj( CV[i] );
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
    // C := alpha W B.V^[T,H] + beta C = alpha (A^T B.U) B.V^[T,H] + beta C
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
#endif
    if( B.Symmetric() )
    {
        // W := B A.U 
        std::vector<Scalar> W( B.Height()*A.r );
        blas::Symm
        ( 'L', 'L', B.Height(), A.r,
          1, B.LockedBuffer(), B.LDim(), &A.U[0], A.m, 
          0, &W[0], B.Height() );

        if( Conjugate )
        {
            // Form AVConj := conj(A.V)
            const int size = A.n*A.r;
            const Scalar* RESTRICT AV = &A.V[0];
            std::vector<Scalar> AVConj( size );
            for( int i=0; i<size; ++i )
                AVConj[i] = Conj( AV[i] );

            // C := alpha conj(A.V) W^T + beta C 
            //    = alpha conj(A.V) (A.U^T B) + beta C
            blas::Gemm
            ( 'N', 'T', C.Height(), C.Width(), A.r,
              alpha, &AVConj[0], A.n, &W[0], B.Height(), 
              beta, C.Buffer(), C.LDim() );
        }
        else
        {
            // C := alpha A.V W^T + beta C = alpha A.V (A.U^T B) + beta C
            blas::Gemm
            ( 'N', 'T', C.Height(), C.Width(), A.r,
              alpha, &A.V[0], A.n, &W[0], B.Height(), 
              beta, C.Buffer(), C.LDim() );
        }
    }
    else
    {
        // W := A.U^T B
        std::vector<Scalar> W( A.r*B.Width() );
        blas::Gemm
        ( 'T', 'N', A.r, B.Width(), A.m,
          1, &A.U[0], A.m, B.LockedBuffer(), B.LDim(), 0, &W[0], A.r );

        if( Conjugate )
        {
            // Form AVConj := conj(A.V)
            const int size = A.n*A.r;
            const Scalar* RESTRICT AV = &A.V[0];
            std::vector<Scalar> AVConj( size );
            for( int i=0; i<size; ++i )
                AVConj[i] = Conj( AV[i] );

            // C := alpha conj(A.V) W + beta C 
            //    = alpha conj(A.V) (A.U^T B) + beta C
            blas::Gemm
            ( 'N', 'N', C.Height(), C.Width(), A.r,
              alpha, &AVConj[0], A.n, &W[0], A.r, 
              beta, C.Buffer(), C.LDim() );
        }
        else
        {
            // C := alpha A.V W + beta C = alpha A.V (A.U^T B) + beta C
            blas::Gemm
            ( 'N', 'N', C.Height(), C.Width(), A.r,
              alpha, &A.V[0], A.n, &W[0], A.r,
              beta, C.Buffer(), C.LDim() );
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
