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
void psp::hmatrix_tools::MatrixMultiply
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.Resize( A.Height(), B.Width() );
    C.SetType( GENERAL );
    MatrixMultiply( alpha, A, B, (Scalar)0, C );
}

// Dense C := alpha A B + beta C
template<typename Scalar>
void psp::hmatrix_tools::MatrixMultiply
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.Width() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( A.Symmetric() && B.Symmetric() )
        throw std::logic_error("Product of symmetric matrices not supported.");
#endif
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', C.Height(), C.Width(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(), 
          0, C.Buffer(), C.LDim() );
    }
    else if( B.Symmetric() )
    {
        blas::Symm
        ( 'R', 'L', C.Height(), C.Width(),
          alpha, B.LockedBuffer(), B.LDim(), A.LockedBuffer(), A.LDim(),
          0, C.Buffer(), C.LDim() );
    }
    else
    {
        blas::Gemm
        ( 'N', 'N', C.Height(), C.Width(), A.Width(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
          0, C.Buffer(), C.LDim() );
    }
}

// Low-rank C := alpha A B
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixMultiply
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
    if( A.r < B.r )
    {
        // C := A.U (alpha (A.V^H B.U) B.V^H)
        //                  or
        // C := A.U (alpha (A.V^T B.U) B.V^T)
        C.r = A.r;

        // C.U := A.U
        C.U.resize( C.m*C.r );
        std::memcpy( &C.U[0], &A.U[0], C.m*C.r*sizeof(Scalar) );

        // C.V := conj(alpha) B.V (B.U^H A.V)
        //                  or
        // C.V := alpha       B.V (B.U^T A.V)
        //
        // We need a temporary buffer of size B.r*A.r; for now, we will allocate
        // it on the fly, but eventually we may want to have a permanent buffer
        // of a fixed size.
        C.V.resize( C.n*C.r );
        std::vector<Scalar> W( B.r*A.r );
        // W := B.U^[T,H] A.V
        const char option = ( Conjugate ? 'C' : 'T' );
        blas::Gemm
        ( option, 'N', B.r, A.r, B.m, 
          1, &B.U[0], B.m, &A.V[0], A.n, 0, &W[0], B.r );
        // C.V := conj(alpha) B.V W = conj(alpha) B.V (B.U^H A.V)
        //                  or
        // C.V := alpha       B.V W = alpha       B.V (B.U^T A.V)
        const Scalar scale = ( Conjugate ? Conj(alpha) : alpha );
        blas::Gemm
        ( 'N', 'N', C.n, C.r, B.r, 
          scale, &B.V[0], B.n, &W[0], B.r, 0, &C.V[0], C.n );
    }
    else
    {
        // C := (alpha A.U (A.V^[T,H] B.U)) B.V^[T,H]
        C.r = B.r;

        // C.U := alpha A.U (A.V^[T,H B.U)
        C.U.resize( C.m*C.r );
        std::vector<Scalar> W( A.r*B.r );
        // W := A.V^[T,H] B.U
        const char option = ( Conjugate ? 'C' : 'N' );
        blas::Gemm
        ( option, 'N', A.r, B.r, A.n,
          1, &A.V[0], A.n, &B.U[0], B.m, 0, &W[0], A.r );
        // C.U := alpha A.U W = alpha A.U (A.V^[T,H] B.U)
        blas::Gemm
        ( 'N', 'N', C.m, C.r, A.r,
          alpha, &A.U[0], A.m, &W[0], A.r, 0, &C.U[0], C.m );

        // C.V := B.V
        C.V.resize( C.n*C.r );
        std::memcpy( &C.V[0], &B.V[0], C.n*C.r*sizeof(Scalar) );
    }
}

// Form a factor matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixMultiply
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
void psp::hmatrix_tools::MatrixMultiply
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

    // Form C.U := A.U
    std::memcpy( &C.U[0], &A.U[0], A.m*A.r*sizeof(Scalar) );

    // Form C.V := conj(alpha) B^H A.V
    //            or
    //      C.V := alpha       B^T A.V
    if( B.Symmetric() )
    {
        if( Conjugate )
        {
            // Since B is symmetric, 
            //   conj(alpha) B^H V = conj(alpha) conj(B) V = 
            //   conj(alpha B conj(V))
            const int size = A.n*A.r;
            std::vector<Scalar> W( size );
            const Scalar* RESTRICT AV = &A.V[0];
            for( int i=0; i<size; ++i )
                W[i] = Conj( AV[i] );

            blas::Symm
            ( 'L', 'L', C.n, C.r, 
              alpha, B.LockedBuffer(), B.LDim(), &W[0], A.n, 
              0, &C.V[0], C.n );

            Scalar* CV = &C.V[0];
            for( int i=0; i<size; ++i )
                CV[i] = Conj( CV[i] );
        }
        else
        {
            blas::Symm
            ( 'L', 'L', C.n, C.r,
              alpha, B.LockedBuffer(), B.LDim(), &A.V[0], A.n, 
              0, &C.V[0], C.n );
        }
    }
    else
    {
        const char option = ( Conjugate ? 'C' : 'T' );
        const Scalar scale = ( Conjugate ? Conj(alpha) : alpha );
        blas::Gemm
        ( option, 'N', C.n, C.r, B.Height(),
          scale, B.LockedBuffer(), B.LDim(), &A.V[0], A.n, 0, &C.V[0], C.n );
    }
}

// Form a dense matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixMultiply
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.Resize( A.Height(), B.n );
    C.SetType( GENERAL );
    MatrixMultiply( alpha, A, B, (Scalar)0, C );
}

// Form a dense matrix from a dense matrix times a factor matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixMultiply
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const FactorMatrix<Scalar,Conjugate>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.Width() != B.m )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
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
    // C := alpha W B.V^[T,H] = alpha (A B.U) B.V^[T,H]
    const char option = ( Conjugate ? 'C' : 'T' );
    blas::Gemm
    ( 'N', option, C.Height(), C.Width(), B.r,
      alpha, &W[0], A.Height(), &B.V[0], B.n, 0, C.Buffer(), C.LDim() );
}

// Form a dense matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixMultiply
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
    C.Resize( A.m, B.Width() );
    C.SetType( GENERAL );
    MatrixMultiply( alpha, A, B, (Scalar)0, C );
}

// Form a dense matrix from a factor matrix times a dense matrix
template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::MatrixMultiply
( Scalar alpha, const FactorMatrix<Scalar,Conjugate>& A, 
                const DenseMatrix<Scalar>& B, 
  Scalar beta,        DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.n != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    if( B.Symmetric() )
    {
        if( Conjugate )
        {
            // Form AVConj := conj(A.V)
            const int size = A.n*A.r;
            const Scalar* RESTRICT AV = &A.V[0];
            std::vector<Scalar> AVConj( size );
            for( int i=0; i<size; ++i )
                AVConj[i] = Conj( AV[i] );
        
            // W := B conj(A.V) = (A.V^H B)^T
            std::vector<Scalar> W( B.Height()*A.r );
            blas::Symm
            ( 'L', 'L', B.Height(), A.r,
              1, B.LockedBuffer(), B.LDim(), &AVConj[0], A.n, 
              0, &W[0], B.Height() );

            // C := alpha A.U W^T =  alpha A.U (A.V^H B)
            blas::Gemm
            ( 'N', 'T', C.Height(), C.Width(), A.r,
              alpha, &A.U[0], A.m, &W[0], B.Height(), 
              0, C.Buffer(), C.LDim() );
        }
        else
        {
            // W := B A.V = (A.V^T B)^T
            std::vector<Scalar> W( B.Height()*A.r );
            blas::Symm
            ( 'L', 'L', B.Height(), A.r,
              1, B.LockedBuffer(), B.LDim(), &A.V[0], A.n, 
              0, &W[0], B.Height() );

            // C := alpha A.U W^T =  alpha A.U (A.V^T B)
            blas::Gemm
            ( 'N', 'T', C.Height(), C.Width(), A.r,
              alpha, &A.U[0], A.m, &W[0], B.Height(), 
              0, C.Buffer(), C.LDim() );
        }
    }
    else
    {
        std::vector<Scalar> W( A.r*B.Width() );
        // W := A.V^[T,H] B
        const char option = ( Conjugate ? 'C' : 'T' );
        blas::Gemm
        ( option, 'N', A.r, B.Width(), A.n,
          1, &A.V[0], A.n, B.LockedBuffer(), B.LDim(), 0, &W[0], A.r );
        // C := alpha A.U W = alpha A.U (A.V^[T,H] B)
        blas::Gemm
        ( 'N', 'N', C.Height(), C.Width(), A.r,
          alpha, &A.U[0], A.m, &W[0], A.r, 0, C.Buffer(), C.LDim() );
    }
}

// Dense C := alpha A B
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const DenseMatrix<float>& A,
               const DenseMatrix<float>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const DenseMatrix<double>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Dense C := alpha A B + beta C
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const DenseMatrix<float>& A,
               const DenseMatrix<float>& B,
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const DenseMatrix<double>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );

// Low-rank C := alpha A B
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const FactorMatrix<float,false>& A,
               const FactorMatrix<float,false>& B,
                     FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const FactorMatrix<float,true>& A,
               const FactorMatrix<float,true>& B,
                     FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const FactorMatrix<double,false>& A,
                const FactorMatrix<double,false>& B,
                      FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const FactorMatrix<double,true>& A,
                const FactorMatrix<double,true>& B,
                      FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const FactorMatrix<std::complex<float>,false>& B,
                                   FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const FactorMatrix<std::complex<float>,true>& B,
                                   FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const FactorMatrix<std::complex<double>,false>& B,
                                    FactorMatrix<std::complex<double>,false>& C
);
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const FactorMatrix<std::complex<double>,true>& B,
                                    FactorMatrix<std::complex<double>,true>& C
);

// Form a dense matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,false>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,true>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,false>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,true>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,false>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,true>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,false>& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,true>& B,
                                    DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,false>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,true>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,false>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,true>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,false>& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,true>& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,false>& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,true>& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const FactorMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const FactorMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const FactorMatrix<double,false>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const FactorMatrix<double,true>& A,
                const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Form a dense matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const FactorMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const FactorMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
  float beta,        DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const FactorMatrix<double,false>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const FactorMatrix<double,true>& A,
                const DenseMatrix<double>& B,
  double beta,        DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const DenseMatrix< std::complex<float> >& B,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const DenseMatrix< std::complex<double> >& B,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& C );

// Form a factor matrix from a dense matrix times a factor matrix
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,false>& B, 
                     FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const DenseMatrix<float>& A, 
               const FactorMatrix<float,true>& B, 
                     FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,false>& B,
                      FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const DenseMatrix<double>& A,
                const FactorMatrix<double,true>& B,
                      FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,false>& B,
                                   FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const FactorMatrix<std::complex<float>,true>& B,
                                   FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,false>& B,
                                    FactorMatrix<std::complex<double>,false>& C
);
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const FactorMatrix<std::complex<double>,true>& B,
                                    FactorMatrix<std::complex<double>,true>& C
);

// Form a factor matrix from a factor matrix times a dense matrix
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const FactorMatrix<float,false>& A, 
               const DenseMatrix<float>& B, 
                     FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixMultiply
( float alpha, const FactorMatrix<float,true>& A, 
               const DenseMatrix<float>& B, 
                     FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const FactorMatrix<double,false>& A,
                const DenseMatrix<double>& B,
                      FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixMultiply
( double alpha, const FactorMatrix<double,true>& A,
                const DenseMatrix<double>& B,
                      FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
                             const DenseMatrix< std::complex<float> >& B,
                                   FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    FactorMatrix<std::complex<double>,false>& C 
);
template void psp::hmatrix_tools::MatrixMultiply
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
                              const DenseMatrix< std::complex<double> >& B,
                                    FactorMatrix<std::complex<double>,true>& C 
);
