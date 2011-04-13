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

// Dense C := alpha A + beta B
template<typename Scalar>
void psp::hmatrix_tools::MatrixAdd
( Scalar alpha, const DenseMatrix<Scalar>& A, 
  Scalar beta,  const DenseMatrix<Scalar>& B, 
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixAdd (D := D + D)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to add nonconforming matrices.");
    // TODO: Allow for A and B to have different types
    if( A.Type() != B.Type() )
        throw std::logic_error("MatrixAdd with different types not written");
#endif
    const int m = A.Height();
    const int n = A.Width();

    C.SetType( A.Type() );
    C.Resize( m, n );

    if( C.Symmetric() )
    {
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
            const Scalar* RESTRICT BCol = B.LockedBuffer(0,j);
            Scalar* RESTRICT CCol = C.Buffer(0,j);
            for( int i=j; i<m; ++i )
                CCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
    else
    {
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
            const Scalar* RESTRICT BCol = B.LockedBuffer(0,j);
            Scalar* RESTRICT CCol = C.Buffer(0,j);
            for( int i=0; i<m; ++i )
                CCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Low-rank C := alpha A + beta B
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixAdd
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
  Scalar beta,  const LowRankMatrix<Scalar,Conjugated>& B, 
                      LowRankMatrix<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixAdd (F := F + F)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int Ar = A.Rank();
    const int Br = B.Rank();
    const int r = Ar + Br;
    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

    // C.U := [(alpha A.U), (beta B.U)]
    // Copy in (alpha A.U)
    for( int j=0; j<Ar; ++j )
    {
        Scalar* RESTRICT CUACol = C.U.Buffer(0,j);
        const Scalar* RESTRICT AUCol = A.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            CUACol[i] = alpha*AUCol[i];
    }
    // Copy in (beta B.U)
    for( int j=0; j<Br; ++j )
    {
        Scalar* RESTRICT CUBCol = C.U.Buffer(0,j+Ar);
        const Scalar* RESTRICT BUCol = B.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            CUBCol[i] = beta*BUCol[i];
    }

    // C.V := [A.V B.V]
    for( int j=0; j<Ar; ++j )
    {
        std::memcpy
        ( C.V.Buffer(0,j), A.V.LockedBuffer(0,j), n*sizeof(Scalar) );
    }
    for( int j=0; j<Br; ++j )
    {
        std::memcpy
        ( C.V.Buffer(0,j+Ar), B.V.LockedBuffer(0,j), n*sizeof(Scalar) );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Dense from sum of low-rank and dense:  C := alpha A + beta B
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixAdd
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
  Scalar beta,  const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixAdd (D := F + D)");
    if( A.Height() != B.Height() || A.Width() != B.Width()  )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int r = A.Rank();

    C.SetType( GENERAL );
    C.Resize( m, n );

    if( B.Symmetric() )
    {
        // Form the full C := beta B from the symmetric B
        
        // Form the lower triangle
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT BCol = B.LockedBuffer(0,j);
            Scalar* RESTRICT CCol = C.Buffer(0,j);
            for( int i=j; i<n; ++i )
                CCol[i] = beta*BCol[i];
        }

        // Transpose the strictly-lower triangle into the upper triangle
        const int ldc = C.LDim();
        for( int j=0; j<n-1; ++j )
        {
            const Scalar* CCol = C.LockedBuffer(0,j);
            Scalar* CRow = C.Buffer(j,0);
            for( int i=j+1; i<n; ++i )
                CRow[i*ldc] = CCol[i];
        }

        // C := alpha A + C = alpha A.U A.V^[T,H] + C
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, m, n, r, 
          alpha, A.U.LockedBuffer(), A.U.LDim(), 
                 A.V.LockedBuffer(), A.V.LDim(),
          1,     C.Buffer(),         C.LDim() );
    }
    else
    {
        // Form C := beta B
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT BCol = B.LockedBuffer(0,j);
            Scalar* RESTRICT CCol = C.Buffer(0,j);
            for( int i=0; i<m; ++i )
                CCol[i] = beta*BCol[i];
        }

        // C := alpha A + C = alpha A.U A.V^[T,H] + C
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, m, n, r, 
          alpha, A.U.LockedBuffer(), A.U.LDim(), 
                 A.V.LockedBuffer(), A.V.LDim(),
          1,     C.Buffer(),         C.LDim() );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Dense from sum of dense and low-rank:  C := alpha A + beta B
// The arguments are switched for generality, so just call the other version.
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixAdd
( Scalar alpha, const DenseMatrix<Scalar>& A, 
  Scalar beta,  const LowRankMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixAdd (D := D + F)");
#endif
    MatrixAdd( beta, B, alpha, A, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

// Dense as sum of two low-rank matrices
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixAdd
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
  Scalar beta,  const LowRankMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    PushCallStack("hmatrix_tools::MatrixAdd (D := F + F)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int r = A.Rank();

    C.SetType( GENERAL );
    C.Resize( m, n );

    // C := alpha A = alpha A.U A.V^[T,H] + C
    const char option = ( Conjugated ? 'C' : 'T' );
    blas::Gemm
    ( 'N', option, m, n, r, 
      alpha, A.U.LockedBuffer(), A.U.LDim(), 
             A.V.LockedBuffer(), A.V.LDim(),
      0,     C.Buffer(),         C.LDim() );
    // C := beta B + C = beta B.U B.V^[T,H] + C
    blas::Gemm
    ( 'N', option, m, n, r, 
      beta, B.U.LockedBuffer(), B.U.LDim(), 
            B.V.LockedBuffer(), B.V.LDim(),
      1,    C.Buffer(),         C.LDim() );
#ifndef RELEASE
    PopCallStack();
#endif
}

// Dense C := alpha A + beta B
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const DenseMatrix<float>& A,
  float beta,  const DenseMatrix<float>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const DenseMatrix<double>& A,
  double beta,  const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
  std::complex<float> beta,  const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
  std::complex<double> beta,  const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Low-rank C := alpha A + beta B
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const LowRankMatrix<float,false>& A,
  float beta,  const LowRankMatrix<float,false>& B,
                     LowRankMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const LowRankMatrix<float,true>& A,
  float beta,  const LowRankMatrix<float,true>& B,
                     LowRankMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const LowRankMatrix<double,false>& A,
  double beta,  const LowRankMatrix<double,false>& B,
                      LowRankMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const LowRankMatrix<double,true>& A,
  double beta,  const LowRankMatrix<double,true>& B,
                      LowRankMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const LowRankMatrix<std::complex<float>,false>& A,
  std::complex<float> beta,  const LowRankMatrix<std::complex<float>,false>& B,
                                   LowRankMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const LowRankMatrix<std::complex<float>,true>& A,
  std::complex<float> beta,  const LowRankMatrix<std::complex<float>,true>& B,
                                   LowRankMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const LowRankMatrix<std::complex<double>,false>& A,
  std::complex<double> beta,  const LowRankMatrix<std::complex<double>,false>& B,
                                    LowRankMatrix<std::complex<double>,false>& C
);
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const LowRankMatrix<std::complex<double>,true>& A,
  std::complex<double> beta,  const LowRankMatrix<std::complex<double>,true>& B,
                                    LowRankMatrix<std::complex<double>,true>& C
);


// Dense as sum of low-rank and dense, C := alpha A + beta B
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const LowRankMatrix<float,false>& A,
  float beta,  const DenseMatrix<float>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const LowRankMatrix<float,true>& A,
  float beta,  const DenseMatrix<float>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const LowRankMatrix<double,false>& A,
  double beta,  const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const LowRankMatrix<double,true>& A,
  double beta,  const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const LowRankMatrix<std::complex<float>,false>& A,
  std::complex<float> beta,  const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const LowRankMatrix<std::complex<float>,true>& A,
  std::complex<float> beta,  const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const LowRankMatrix<std::complex<double>,false>& A,
  std::complex<double> beta,  const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const LowRankMatrix<std::complex<double>,true>& A,
  std::complex<double> beta,  const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Dense as sum of dense and low-rank, C := alpha A + beta B
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const DenseMatrix<float>& A,
  float beta,  const LowRankMatrix<float,false>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const DenseMatrix<float>& A,
  float beta,  const LowRankMatrix<float,true>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const DenseMatrix<double>& A,
  double beta,  const LowRankMatrix<double,false>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const DenseMatrix<double>& A,
  double beta,  const LowRankMatrix<double,true>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
  std::complex<float> beta,  const LowRankMatrix<std::complex<float>,false>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
  std::complex<float> beta,  const LowRankMatrix<std::complex<float>,true>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
  std::complex<double> beta,  const LowRankMatrix<std::complex<double>,false>& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
  std::complex<double> beta,  const LowRankMatrix<std::complex<double>,true>& B,
                                    DenseMatrix< std::complex<double> >& C );

// Dense as sum of two low-rank matrices
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const LowRankMatrix<float,false>& A,
  float beta,  const LowRankMatrix<float,false>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const LowRankMatrix<float,true>& A,
  float beta,  const LowRankMatrix<float,true>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const LowRankMatrix<double,false>& A,
  double beta,  const LowRankMatrix<double,false>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const LowRankMatrix<double,true>& A,
  double beta,  const LowRankMatrix<double,true>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const LowRankMatrix<std::complex<float>,false>& A,
  std::complex<float> beta,  const LowRankMatrix<std::complex<float>,false>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const LowRankMatrix<std::complex<float>,true>& A,
  std::complex<float> beta,  const LowRankMatrix<std::complex<float>,true>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const LowRankMatrix<std::complex<double>,false>& A,
  std::complex<double> beta,  const LowRankMatrix<std::complex<double>,false>& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const LowRankMatrix<std::complex<double>,true>& A,
  std::complex<double> beta,  const LowRankMatrix<std::complex<double>,true>& B,
                                    DenseMatrix< std::complex<double> >& C );

