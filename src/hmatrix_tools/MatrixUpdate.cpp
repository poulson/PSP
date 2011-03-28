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

// Dense B := alpha A + beta B
template<typename Scalar>
void psp::hmatrix_tools::MatrixUpdate
( Scalar alpha, const DenseMatrix<Scalar>& A, 
  Scalar beta,        DenseMatrix<Scalar>& B )
{
#ifndef RELEASE
    if( A.m != B.m || A.n != B.n || A.symmetric != B.symmetric )
        throw std::logic_error("Tried to update with nonconforming matrices.");
#endif
    const int m = A.m;
    const int n = A.n;
    const int lda = A.ldim;
    const int ldb = B.ldim;
    if( A.symmetric )
    {
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT ACol = &A.buffer[j*lda];
            Scalar* RESTRICT BCol = &B.buffer[j*ldb];
            for( int i=j; i<m; ++i )
                BCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
    else
    {
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT ACol = &A.buffer[j*lda];
            Scalar* RESTRICT BCol = &B.buffer[j*ldb];
            for( int i=0; i<m; ++i )
                BCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
}

// Low-rank B := alpha A + beta B
template<typename Scalar>
void psp::hmatrix_tools::MatrixUpdate
( Scalar alpha, const FactorMatrix<Scalar>& A, 
  Scalar beta,        FactorMatrix<Scalar>& B )
{
#ifndef RELEASE
    if( A.m != B.m || A.n != B.n )
        throw std::logic_error("Tried to update with nonconforming matrices.");
#endif
    const int newRank = A.r + B.r;

    // B.U := [(beta B.U), (alpha A.U)]
    B.U.resize( B.m*newRank );
    // Scale B.U by beta
    {
        const int r = B.r;
        const int m = B.m;
        Scalar* BU_B = &B.U[0];
        for( int i=0; i<r*m; ++i )
            BU_B[i] *= beta;
    }
    // Copy in (alpha A.U)
    {
        const int r = A.r;
        const int m = A.m;
        Scalar* RESTRICT BU_A = &B.U[B.r*m];
        const Scalar* RESTRICT AU = &A.U[0]; 
        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                BU_A[i+j*m] = alpha*AU[i+j*m];
    }

    // B.V := [B.V A.V]
    B.V.resize( B.n*newRank );
    std::memcpy( &B.V[B.n*B.r], &A.V[0], A.n*A.r*sizeof(Scalar) );

    // Mark the new rank
    B.r = newRank;
}

// Dense updated with factored, B := alpha A + beta B
template<typename Scalar>
void psp::hmatrix_tools::MatrixUpdate
( Scalar alpha, const FactorMatrix<Scalar>& A, 
  Scalar beta,        DenseMatrix<Scalar>& B )
{
#ifndef RELEASE
    if( A.m != B.m || A.n != B.n  )
        throw std::logic_error("Tried to update with nonconforming matrices.");
    if( B.symmetric )
        throw std::logic_error("Unsafe update of symmetric dense matrix.");
#endif
    const int m = A.m;
    const int n = A.n;
    const int lda = A.ldim;
    const int ldb = B.ldim;

    blas::Gemm
    ( 'N', 'C', A.m, A.n, A.r, 
      alpha, &A.U[0], A.m, &A.V[0], A.n, 
      beta, &B.buffer[0], B.ldim );
}

// Dense update B := alpha A + beta B
template void psp::hmatrix_tools::MatrixUpdate
( float alpha, const DenseMatrix<float>& A,
  float beta,        DenseMatrix<float>& B );
template void psp::hmatrix_tools::MatrixUpdate
( double alpha, const DenseMatrix<double>& A,
  double beta,        DenseMatrix<double>& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& B );

// Low-rank update B := alpha A + beta B
template void psp::hmatrix_tools::MatrixUpdate
( float alpha, const FactorMatrix<float>& A,
  float beta,        FactorMatrix<float>& B );
template void psp::hmatrix_tools::MatrixUpdate
( double alpha, const FactorMatrix<double>& A,
  double beta,        FactorMatrix<double>& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<float> alpha, const FactorMatrix< std::complex<float> >& A,
  std::complex<float> beta,        FactorMatrix< std::complex<float> >& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<double> alpha, const FactorMatrix< std::complex<double> >& A,
  std::complex<double> beta,        FactorMatrix< std::complex<double> >& B );

// Dense updated with factor, B := alpha A + beta B
template void psp::hmatrix_tools::MatrixUpdate
( float alpha, const FactorMatrix<float>& A,
  float beta,        DenseMatrix<float>& B );
template void psp::hmatrix_tools::MatrixUpdate
( double alpha, const FactorMatrix<double>& A,
  double beta,        DenseMatrix<double>& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<float> alpha, const FactorMatrix< std::complex<float> >& A,
  std::complex<float> beta,        DenseMatrix< std::complex<float> >& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<double> alpha, const FactorMatrix< std::complex<double> >& A,
  std::complex<double> beta,        DenseMatrix< std::complex<double> >& B );
