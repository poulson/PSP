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
    if( A.m != B.m || A.n != B.n || A.symmetric != B.symmetric )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    C.m = A.m;
    C.n = A.n;
    C.ldim = C.m;
    C.symmetric = A.symmetric; 
    C.buffer.resize( C.ldim*C.n );

    const int m = C.m;
    const int n = C.n;
    const int lda = A.ldim;
    const int ldb = B.ldim;
    const int ldc = C.ldim;
    if( C.symmetric )
    {
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT ACol = &A.buffer[j*lda];
            const Scalar* RESTRICT BCol = &B.buffer[j*ldb];
            Scalar* RESTRICT CCol = &C.buffer[j*ldc];
            for( int i=j; i<m; ++i )
                CCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
    else
    {
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT ACol = &A.buffer[j*lda];
            const Scalar* RESTRICT BCol = &B.buffer[j*ldb];
            Scalar* RESTRICT CCol = &C.buffer[j*ldc];
            for( int i=0; i<m; ++i )
                CCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
}

// Low-rank C := alpha A + beta B
template<typename Scalar>
void psp::hmatrix_tools::MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar>& A, 
  Scalar beta,  const FactorMatrix<Scalar>& B, 
                      FactorMatrix<Scalar>& C )
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
template<typename Scalar>
void psp::hmatrix_tools::MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar>& A, 
  Scalar beta,  const DenseMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.m != B.m || A.n != B.n  )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    C.m = A.m;
    C.n = A.n;
    C.ldim = C.m;
    C.buffer.resize( C.ldim*C.n );
    C.symmetric = false;

    const int m = A.m;
    const int n = A.n;
    const int ldb = B.ldim;
    const int ldc = C.ldim;

    if( B.symmetric )
    {
        // Form the full C := beta B from the symmetric B
        
        // Form the lower triangle
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT BCol = &B.buffer[j*ldb];
            Scalar* RESTRICT CCol = &C.buffer[j*ldc];
            for( int i=j; i<n; ++i )
                CCol[i] = beta*BCol[i];
        }

        // Transpose the strictly-lower triangle into the upper triangle
        for( int j=0; j<n-1; ++j )
        {
            const Scalar* CCol = &C.buffer[j*ldc];
            Scalar* CRow = &C.buffer[j];
            for( int i=j+1; i<n; ++i )
                CRow[i*ldc] = CCol[i];
        }

        // C := alpha A + C
        blas::Gemm
        ( 'N', 'C', A.m, A.n, A.r, 
          alpha, &A.U[0], A.m, &A.V[0], A.n,
          1, &C.buffer[0], C.ldim );
    }
    else
    {
        // Form C := beta B
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT BCol = &B.buffer[j*ldb];
            Scalar* RESTRICT CCol = &C.buffer[j*ldc];
            for( int i=0; i<m; ++i )
                CCol[i] = beta*BCol[i];
        }

        // C := alpha A + C
        blas::Gemm
        ( 'N', 'C', A.m, A.n, A.r, 
          alpha, &A.U[0], A.m, &A.V[0], A.n,
          1, &C.buffer[0], C.ldim );
    }
}

// Dense from sum of dense and factor:  C := alpha A + beta B
// The arguments are switched for generality, so just call the other version.
template<typename Scalar>
void psp::hmatrix_tools::MatrixAdd
( Scalar alpha, const DenseMatrix<Scalar>& A, 
  Scalar beta,  const FactorMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C )
{
    MatrixAdd( beta, B, alpha, A, C );
}

// Dense as sum of two factors
template<typename Scalar>
void psp::hmatrix_tools::MatrixAdd
( Scalar alpha, const FactorMatrix<Scalar>& A, 
  Scalar beta,  const FactorMatrix<Scalar>& B,
                      DenseMatrix<Scalar>& C )
{
#ifndef RELEASE
    if( A.m != B.m || A.n != B.n  )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    C.m = A.m;
    C.n = A.n;
    C.ldim = C.m;
    C.buffer.resize( C.ldim*C.n );
    C.symmetric = false;

    const int m = A.m;
    const int n = A.n;
    const int ldc = C.ldim;

    // C := alpha A
    blas::Gemm
    ( 'N', 'C', A.m, A.n, A.r, 
      alpha, &A.U[0], A.m, &A.V[0], A.n,
      0, &C.buffer[0], C.ldim );
    // C := beta B + C
    blas::Gemm
    ( 'N', 'C', B.m, B.n, B.r, 
      beta, &B.U[0], B.m, &B.V[0], B.n,
      1, &C.buffer[0], C.ldim );
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
( float alpha, const FactorMatrix<float>& A,
  float beta,  const FactorMatrix<float>& B,
                     FactorMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const FactorMatrix<double>& A,
  double beta,  const FactorMatrix<double>& B,
                      FactorMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const FactorMatrix< std::complex<float> >& A,
  std::complex<float> beta,  const FactorMatrix< std::complex<float> >& B,
                                   FactorMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const FactorMatrix< std::complex<double> >& A,
  std::complex<double> beta,  const FactorMatrix< std::complex<double> >& B,
                                    FactorMatrix< std::complex<double> >& C );

// Dense as sum of factor and dense, C := alpha A + beta B
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const FactorMatrix<float>& A,
  float beta,  const DenseMatrix<float>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const FactorMatrix<double>& A,
  double beta,  const DenseMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const FactorMatrix< std::complex<float> >& A,
  std::complex<float> beta,  const DenseMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const FactorMatrix< std::complex<double> >& A,
  std::complex<double> beta,  const DenseMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Dense as sum of dense and factor, C := alpha A + beta B
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const DenseMatrix<float>& A,
  float beta,  const FactorMatrix<float>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const DenseMatrix<double>& A,
  double beta,  const FactorMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
  std::complex<float> beta,  const FactorMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
  std::complex<double> beta,  const FactorMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

// Dense as sum of two factors
template void psp::hmatrix_tools::MatrixAdd
( float alpha, const FactorMatrix<float>& A,
  float beta,  const FactorMatrix<float>& B,
                     DenseMatrix<float>& C );
template void psp::hmatrix_tools::MatrixAdd
( double alpha, const FactorMatrix<double>& A,
  double beta,  const FactorMatrix<double>& B,
                      DenseMatrix<double>& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<float> alpha, const FactorMatrix< std::complex<float> >& A,
  std::complex<float> beta,  const FactorMatrix< std::complex<float> >& B,
                                   DenseMatrix< std::complex<float> >& C );
template void psp::hmatrix_tools::MatrixAdd
( std::complex<double> alpha, const FactorMatrix< std::complex<double> >& A,
  std::complex<double> beta,  const FactorMatrix< std::complex<double> >& B,
                                    DenseMatrix< std::complex<double> >& C );

