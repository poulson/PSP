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
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to add nonconforming matrices.");
    // TODO: Allow for A and B to have different types
    if( A.Type() != B.Type() )
        throw std::logic_error("MatrixAdd with different types not written");
#endif
    C.Resize( A.Height(), A.Width() );
    C.SetType( A.Type() );

    const int m = C.Height();
    const int n = C.Width();
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
}

// Low-rank C := alpha A + beta B
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
  Scalar beta,  const FactorMatrix<Scalar,Conjugated>& B, 
                      FactorMatrix<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    if( A.m != B.m || A.n != B.n )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    C.m = A.m;
    C.n = A.n;
    C.r = A.r + B.r;

    // C.U := [(alpha A.U), (beta B.U)]
    C.U.resize( C.m*C.r );
    // Copy in (alpha A.U)
    {
        const int r = A.r;
        const int m = A.m;
        Scalar* RESTRICT CU_A = &C.U[0];
        const Scalar* RESTRICT AU = &A.U[0]; 
        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                CU_A[i+j*m] = alpha*AU[i+j*m];
    }
    // Copy in (beta B.U)
    {
        const int r = B.r;
        const int m = A.m;
        Scalar* RESTRICT CU_B = &C.U[C.m*A.r];
        const Scalar* RESTRICT BU = &B.U[0];
        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                CU_B[i+j*m] = beta*BU[i+j*m];
    }

    // C.V := [A.V B.V]
    C.V.resize( C.n*C.r );
    std::memcpy( &C.V[0], &A.V[0], C.n*A.r*sizeof(Scalar) );
    std::memcpy( &C.V[C.n*A.r], &B.V[0], C.n*B.r*sizeof(Scalar) );
}

// Dense from sum of factor and dense:  C := alpha A + beta B
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
  Scalar beta,  const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.m != B.Height() || A.n != B.Width()  )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    C.Resize( A.m, A.n );
    C.SetType( GENERAL );

    const int m = A.m;
    const int n = A.n;

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
        ( 'N', option, A.m, A.n, A.r, 
          alpha, &A.U[0], A.m, &A.V[0], A.n,
          1, C.Buffer(), C.LDim() );
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
        ( 'N', option, A.m, A.n, A.r, 
          alpha, &A.U[0], A.m, &A.V[0], A.n,
          1, C.Buffer(), C.LDim() );
    }
}

// Dense from sum of dense and factor:  C := alpha A + beta B
// The arguments are switched for generality, so just call the other version.
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixAdd
( Scalar alpha, const DenseMatrix<Scalar>& A, 
  Scalar beta,  const FactorMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C )
{
    MatrixAdd( beta, B, alpha, A, C );
}

// Dense as sum of two factors
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar,Conjugated>& A, 
  Scalar beta,  const FactorMatrix<Scalar,Conjugated>& B,
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.m != B.m || A.n != B.n  )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    C.Resize( A.m, A.n );
    C.SetType( GENERAL );

    // C := alpha A = alpha A.U A.V^[T,H] + C
    const char option = ( Conjugated ? 'C' : 'T' );
    blas::Gemm
    ( 'N', option, A.m, A.n, A.r, 
      alpha, &A.U[0], A.m, &A.V[0], A.n,
      0, C.Buffer(), C.LDim() );
    // C := beta B + C = beta B.U B.V^[T,H] + C
    blas::Gemm
    ( 'N', option, B.m, B.n, B.r, 
      beta, &B.U[0], B.m, &B.V[0], B.n,
      1, C.Buffer(), C.LDim() );
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
( float alpha, const FactorMatrix<float,false>& A,
  float beta,  const FactorMatrix<float,false>& B,
                     FactorMatrix<float,false>& C );
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const FactorMatrix<float,true>& A,
  float beta,  const FactorMatrix<float,true>& B,
                     FactorMatrix<float,true>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const FactorMatrix<double,false>& A,
  double beta,  const FactorMatrix<double,false>& B,
                      FactorMatrix<double,false>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const FactorMatrix<double,true>& A,
  double beta,  const FactorMatrix<double,true>& B,
                      FactorMatrix<double,true>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
  std::complex<float> beta,  const FactorMatrix<std::complex<float>,false>& B,
                                   FactorMatrix<std::complex<float>,false>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
  std::complex<float> beta,  const FactorMatrix<std::complex<float>,true>& B,
                                   FactorMatrix<std::complex<float>,true>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
  std::complex<double> beta,  const FactorMatrix<std::complex<double>,false>& B,
                                    FactorMatrix<std::complex<double>,false>& C
);
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
  std::complex<double> beta,  const FactorMatrix<std::complex<double>,true>& B,
                                    FactorMatrix<std::complex<double>,true>& C
);


// Dense as sum of factor and dense, C := alpha A + beta B
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const FactorMatrix<float,false>& A,
  float beta,  const DenseMatrix<float>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const FactorMatrix<float,true>& A,
  float beta,  const DenseMatrix<float>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const FactorMatrix<double,false>& A,
  double beta,  const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const FactorMatrix<double,true>& A,
  double beta,  const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
  std::complex<float> beta,  const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
  std::complex<float> beta,  const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
  std::complex<double> beta,  const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
  std::complex<double> beta,  const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Dense as sum of dense and factor, C := alpha A + beta B
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const DenseMatrix<float>& A,
  float beta,  const FactorMatrix<float,false>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const DenseMatrix<float>& A,
  float beta,  const FactorMatrix<float,true>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const DenseMatrix<double>& A,
  double beta,  const FactorMatrix<double,false>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const DenseMatrix<double>& A,
  double beta,  const FactorMatrix<double,true>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
  std::complex<float> beta,  const FactorMatrix<std::complex<float>,false>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
  std::complex<float> beta,  const FactorMatrix<std::complex<float>,true>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
  std::complex<double> beta,  const FactorMatrix<std::complex<double>,false>& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
  std::complex<double> beta,  const FactorMatrix<std::complex<double>,true>& B,
                                    DenseMatrix< std::complex<double> >& C );

// Dense as sum of two factors
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const FactorMatrix<float,false>& A,
  float beta,  const FactorMatrix<float,false>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const FactorMatrix<float,true>& A,
  float beta,  const FactorMatrix<float,true>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const FactorMatrix<double,false>& A,
  double beta,  const FactorMatrix<double,false>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const FactorMatrix<double,true>& A,
  double beta,  const FactorMatrix<double,true>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,false>& A,
  std::complex<float> beta,  const FactorMatrix<std::complex<float>,false>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const FactorMatrix<std::complex<float>,true>& A,
  std::complex<float> beta,  const FactorMatrix<std::complex<float>,true>& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,false>& A,
  std::complex<double> beta,  const FactorMatrix<std::complex<double>,false>& B,
                                    DenseMatrix< std::complex<double> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const FactorMatrix<std::complex<double>,true>& A,
  std::complex<double> beta,  const FactorMatrix<std::complex<double>,true>& B,
                                    DenseMatrix< std::complex<double> >& C );

