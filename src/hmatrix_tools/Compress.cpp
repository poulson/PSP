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

// A :~= A
//
// Approximate A with a given maximum rank.
template<typename Real,bool Conjugate>
void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<Real,Conjugate>& A )
{
    const int m = A.m;
    const int n = A.n;
    const int r = A.r;
    const int roundedRank = std::min( r, maxRank );
    if( roundedRank == r )
        return;

    // Grab enough workspace for our entire rounded addition
    const int leftPanelSize = m*r;
    const int rightPanelSize = n*r;
    const int blockSize = r*r;
    const int lworkSVD = 4*r*r;
    std::vector<Real> buffer
    ( 2*blockSize+std::max(lworkSVD,std::max(leftPanelSize,rightPanelSize)) );

#if defined(PIVOTED_QR)
    // TODO
    throw std::logic_error("Pivoted QR is not yet supported.");
#else
    // Perform an unpivoted QR decomposition on A.U
    std::vector<Real> tauU( std::min( m, r ) );
    lapack::QR( m, r, &A.U[0], m, &tauU[0], &buffer[0], leftPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Perform an unpivoted QR decomposition on A.V
    std::vector<Real> tauV( std::min( n, r ) );
    lapack::QR
    ( n, r, &A.V[0], n, &tauV[0], &buffer[0], rightPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    {
        Real* RESTRICT W = &buffer[0];
        const Real* RESTRICT R1 = &A.U[0];
        std::memset( W, 0, blockSize*sizeof(Real) );
        for( int j=0; j<r; ++j )
            for( int i=0; i<=j; ++i )
                W[i+j*r] = R1[i+j*m];
    }

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize): R1                                                     //
    //------------------------------------------------------------------------//

    // Update W := R1 R2^T. We are unfortunately performing 2x as many
    // flops as are required.
    blas::Trmm( 'R', 'U', 'T', 'N', r, r, 1, &A.V[0], n, &buffer[0], r );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize): R1 R2^T                                                //
    //------------------------------------------------------------------------//

    // Get the SVD of R1 R2^T, overwriting R1 R2^T with U
    std::vector<Real> s( r );
    lapack::SVD
    ( 'O', 'A', r, r, &buffer[0], r, &s[0], 0, 0, 
      &buffer[blockSize], r, &buffer[2*blockSize], lworkSVD );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize):           U of R1 R2^T                                 //
    //  [blockSize,2*blockSize): V^T of R1 R2^T                               //
    //------------------------------------------------------------------------//

    // Copy the result of the QR factorization of A.U into a temporary buffer
    std::memcpy( &buffer[2*blockSize], &A.U[0], m*r*sizeof(Real) );
    // Logically shrink A.U
    A.U.resize( m*roundedRank );
    // Zero the shrunk buffer
    std::memset( &A.U[0], 0, A.U.size()*sizeof(Real) );
    // Copy the scaled U from the SVD of R1 R2^T into the top of the matrix
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = s[j];
        const Real* RESTRICT UCol = &buffer[j*r];
        Real* RESTRICT UColScaled = &A.U[j*m];
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Hit the matrix from the left with Q1 from the QR decomp of the orig A.U
    lapack::ApplyQ
    ( 'L', 'T', m, roundedRank, r, &buffer[2*blockSize], m, &tauU[0], 
      &A.U[0], m, &buffer[0], blockSize );

    // Copy the result of the QR factorization of A.V into a temporary buffer
    std::memcpy( &buffer[2*blockSize], &A.V[0], n*r*sizeof(Real) );
    // Logically shrink A.V
    A.V.resize( n*roundedRank );
    // Zero the shrunk buffer
    std::memset( &A.V[0], 0, A.V.size()*sizeof(Real) );
    // Copy V=(V^T)^T from the SVD of R1 R2^T into the top of A.V
    for( int j=0; j<roundedRank; ++j )
    {
        const Real* RESTRICT VTRow = &buffer[blockSize+j];
        Real* RESTRICT VCol = &A.V[j*n];
        for( int i=0; i<r; ++i )
            VCol[i] = VTRow[i*r];
    }
    // Hit the matrix from the left with Q2 from the QR decomp of the orig A.V
    lapack::ApplyQ
    ( 'L', 'T', n, roundedRank, r, &buffer[2*blockSize], n, &tauV[0],
      &A.V[0], n, &buffer[0], blockSize );
#endif // PIVOTED_QR

    // Mark the matrix as having the new reduced rank
    A.r = roundedRank;
}

// A :~= A
//
// Approximate A with a given maximum rank. This implementation handles both 
// cases, A = U V^T (Conjugate=false), and A = U V^H (Conjugate=true)
template<typename Real,bool Conjugate>
void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<std::complex<Real>,Conjugate>& A )
{
    typedef std::complex<Real> Scalar;

    const int m = A.m;
    const int n = A.n;
    const int r = A.r;
    const int roundedRank = std::min( r, maxRank );
    if( roundedRank == r )
        return;

    // Grab enough workspace for our entire rounded addition
    const int leftPanelSize = m*r;
    const int rightPanelSize = n*r;
    const int blockSize = r*r;
    const int lworkSVD = 4*r*r;
    std::vector<Scalar> buffer
    ( 2*blockSize+std::max(lworkSVD,std::max(leftPanelSize,rightPanelSize)) );

#if defined(PIVOTED_QR)
    // TODO
    throw std::logic_error("Pivoted QR is not yet supported.");
#else
    // Perform an unpivoted QR decomposition on A.U
    std::vector<Scalar> tauU( std::min( m, r ) );
    lapack::QR( m, r, &A.U[0], m, &tauU[0], &buffer[0], leftPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Perform an unpivoted QR decomposition on A.V
    std::vector<Scalar> tauV( std::min( n, r ) );
    lapack::QR( n, r, &A.V[0], n, &tauV[0], &buffer[0], rightPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    {
        Scalar* RESTRICT W = &buffer[0];
        const Scalar* RESTRICT R1 = &A.U[0];
        std::memset( W, 0, blockSize*sizeof(Scalar) );
        for( int j=0; j<r; ++j )
            for( int i=0; i<=j; ++i )
                W[i+j*r] = R1[i+j*m];
    }

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize): R1                                                     //
    //------------------------------------------------------------------------//

    // Update W := R1 R2^[T,H].
    // We are unfortunately performing 2x as many flops as required.
    const char option = ( Conjugate ? 'C' : 'T' );
    blas::Trmm( 'R', 'U', option, 'N', r, r, 1, &A.V[0], n, &buffer[0], r );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize): R1 R2^[T,H]                                            //
    //------------------------------------------------------------------------//

    // Get the SVD of R1 R2^[T,H], overwriting it with U
    std::vector<Real> realBuffer( 6*r );
    lapack::SVD
    ( 'O', 'A', r, r, &buffer[0], r, &realBuffer[0], 0, 0, 
      &buffer[blockSize], r, &buffer[2*blockSize], lworkSVD, &realBuffer[r] );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize):           U of R1 R2^[T,H]                             //
    //  [blockSize,2*blockSize): V^H of R1 R2^[T,H]                           //
    //                                                                        //
    // realBuffer contains:                                                   //
    //   [0,r): singular values of R1 R2^[T,H]                                //
    //------------------------------------------------------------------------//

    // Copy the result of the QR factorization of A.U into a temporary buffer
    std::memcpy( &buffer[2*blockSize], &A.U[0], m*r*sizeof(Scalar) );
    // Logically shrink A.U
    A.U.resize( m*roundedRank );
    // Zero the shrunk buffer
    std::memset( &A.U[0], 0, A.U.size()*sizeof(Scalar) );
    // Copy the scaled U from the SVD of R1 R2^[T,H] into the top of the matrix
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = realBuffer[j];
        const Scalar* RESTRICT UCol = &buffer[j*r];
        Scalar* RESTRICT UColScaled = &A.U[j*m];
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Hit the matrix from the left with Q1 from the QR decomp of the orig A.U
    lapack::ApplyQ
    ( 'L', 'C', m, roundedRank, r, &buffer[2*blockSize], m, &tauU[0], 
      &A.U[0], m, &buffer[0], blockSize );

    // Copy the result of the QR factorization of A.V into a temporary buffer
    std::memcpy( &buffer[2*blockSize], &A.V[0], n*r*sizeof(Scalar) );
    // Logically shrink A.V
    A.V.resize( n*roundedRank );
    // Zero the shrunk buffer
    std::memset( &A.V[0], 0, A.V.size()*sizeof(Scalar) );
    if( Conjugate )
    {
        // Copy V=(V^H)^H from the SVD of R1 R2^H into the top of A.V
        for( int j=0; j<roundedRank; ++j )
        {
            const Scalar* RESTRICT VHRow = &buffer[blockSize+j];
            Scalar* RESTRICT VCol = &A.V[j*n];
            for( int i=0; i<r; ++i )
                VCol[i] = Conj( VHRow[i*r] );
        }
    }
    else
    {
        // Copy conj(V)=(V^H)^T from the SVD of R1 R2^T into the top of A.V
        for( int j=0; j<roundedRank; ++j )
        {
            const Scalar* RESTRICT VHRow = &buffer[blockSize+j];
            Scalar* RESTRICT VColConj = &A.V[j*n];
            for( int i=0; i<r; ++i )
                VColConj[i] = VHRow[i*r];
        }
    }
    // Hit the matrix from the left with Q2 from the QR decomp of the orig A.V
    lapack::ApplyQ
    ( 'L', 'C', n, roundedRank, r, &buffer[2*blockSize], n, &tauV[0],
      &A.V[0], n, &buffer[0], blockSize );
#endif // PIVOTED_QR

    // Mark the matrix as having the new reduced rank
    A.r = roundedRank;
}

template void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<float,false>& A );
template void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<float,true>& A );
template void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<double,false>& A );
template void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<double,true>& A );
template void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<std::complex<float>,false>& A );
template void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<std::complex<float>,true>& A );
template void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<std::complex<double>,false>& A );
template void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<std::complex<double>,true>& A );
