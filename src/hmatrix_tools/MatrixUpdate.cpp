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
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to update with nonconforming matrices.");
    // TODO: Allow for A to be symmetric when B is general
    if( A.Symmetric() && B.General() )
        throw std::logic_error("A-symmetric/B-general not yet implemented.");
    if( A.General() && B.Symmetric() )
        throw std::logic_error("Cannot update a symmetric matrix with a general one");
#endif
    const int m = A.Height();
    const int n = A.Width();
    if( A.Symmetric() )
    {
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
            Scalar* RESTRICT BCol = B.Buffer(0,j);
            for( int i=j; i<m; ++i )
                BCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
    else
    {
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
            Scalar* RESTRICT BCol = B.Buffer(0,j);
            for( int i=0; i<m; ++i )
                BCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
}

// Low-rank B := alpha A + beta B
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixUpdate
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
  Scalar beta,        LowRankMatrix<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to update with nonconforming matrices.");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int Ar = A.Rank();
    const int Br = B.Rank();
    const int newRank = Ar + Br;

    // B.U := [(beta B.U), (alpha A.U)]
    Scale( beta, B.U );
    B.U.Resize( B.Height(), newRank );
    // Copy in (alpha A.U)
    for( int j=0; j<Ar; ++j )
    {
        Scalar* RESTRICT BUACol = B.U.Buffer(0,j+Br);
        const Scalar* RESTRICT AUCol = A.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            BUACol[i] = alpha*AUCol[i];
    }

    // B.V := [B.V A.V]
    B.V.Resize( B.Width(), newRank );
    for( int j=0; j<Ar; ++j )
    {
        std::memcpy
        ( B.V.Buffer(0,j+Br), A.V.LockedBuffer(0,j), n*sizeof(Scalar) );
    }
}

// Dense updated with low-rank, B := alpha A + beta B
template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::MatrixUpdate
( Scalar alpha, const LowRankMatrix<Scalar,Conjugated>& A, 
  Scalar beta,        DenseMatrix<Scalar>& B )
{
#ifndef RELEASE
    if( A.Height() != B.Height() || A.Width() != B.Width()  )
        throw std::logic_error("Tried to update with nonconforming matrices.");
    if( B.Symmetric() )
        throw std::logic_error("Unsafe update of symmetric dense matrix.");
#endif
    const char option = ( Conjugated ? 'C' : 'T' );
    blas::Gemm
    ( 'N', option, A.Height(), A.Width(), A.Rank(), 
      alpha, A.U.LockedBuffer(), A.U.LDim(), A.V.LockedBuffer(), A.V.LDim(), 
      beta, B.Buffer(), B.LDim() );
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
( float alpha, const LowRankMatrix<float,false>& A,
  float beta,        LowRankMatrix<float,false>& B );
template void psp::hmatrix_tools::MatrixUpdate
( float alpha, const LowRankMatrix<float,true>& A,
  float beta,        LowRankMatrix<float,true>& B );
template void psp::hmatrix_tools::MatrixUpdate
( double alpha, const LowRankMatrix<double,false>& A,
  double beta,        LowRankMatrix<double,false>& B );
template void psp::hmatrix_tools::MatrixUpdate
( double alpha, const LowRankMatrix<double,true>& A,
  double beta,        LowRankMatrix<double,true>& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,false>& A,
  std::complex<float> beta,
        LowRankMatrix<std::complex<float>,false>& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,true>& A,
  std::complex<float> beta,
        LowRankMatrix<std::complex<float>,true>& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,false>& A,
  std::complex<double> beta, 
        LowRankMatrix<std::complex<double>,false>& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,true>& A,
  std::complex<double> beta, 
        LowRankMatrix<std::complex<double>,true>& B );


// Dense updated with low-rank, B := alpha A + beta B
template void psp::hmatrix_tools::MatrixUpdate
( float alpha, const LowRankMatrix<float,false>& A,
  float beta,        DenseMatrix<float>& B );
template void psp::hmatrix_tools::MatrixUpdate
( float alpha, const LowRankMatrix<float,true>& A,
  float beta,        DenseMatrix<float>& B );
template void psp::hmatrix_tools::MatrixUpdate
( double alpha, const LowRankMatrix<double,false>& A,
  double beta,        DenseMatrix<double>& B );
template void psp::hmatrix_tools::MatrixUpdate
( double alpha, const LowRankMatrix<double,true>& A,
  double beta,        DenseMatrix<double>& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,false>& A,
  std::complex<float> beta,
        DenseMatrix< std::complex<float> >& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<float> alpha, 
  const LowRankMatrix<std::complex<float>,true>& A,
  std::complex<float> beta,
        DenseMatrix< std::complex<float> >& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,false>& A,
  std::complex<double> beta,
        DenseMatrix< std::complex<double> >& B );
template void psp::hmatrix_tools::MatrixUpdate
( std::complex<double> alpha, 
  const LowRankMatrix<std::complex<double>,true>& A,
  std::complex<double> beta,
        DenseMatrix< std::complex<double> >& B );
