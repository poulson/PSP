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
template<typename Real>
void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<Real>& A )
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

    // Update W := R1 (R2^H). We are unfortunately performing 2x as many
    // flops as are required.
    blas::Trmm( 'R', 'U', 'T', 'N', r, r, 1, &A.V[0], n, &buffer[0], r );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize): R1 R2^H                                                //
    //------------------------------------------------------------------------//

    // Get the SVD of R1 R2^H, overwriting R1 R2^H with U
    std::vector<Real> s( r );
    lapack::SVD
    ( 'O', 'A', r, r, &buffer[0], r, &s[0], 0, 0, 
      &buffer[blockSize], r, &buffer[2*blockSize], lworkSVD );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize):           U of R1 R2^H                                 //
    //  [blockSize,2*blockSize): V^H of R1 R2^H                               //
    //------------------------------------------------------------------------//

    // Copy the result of the QR factorization of A.U into a temporary buffer
    std::memcpy( &buffer[2*blockSize], &A.U[0], m*r*sizeof(Real) );
    // Logically shrink A.U
    A.U.resize( m*roundedRank );
    // Zero the shrunk buffer
    std::memset( &A.U[0], 0, A.U.size()*sizeof(Real) );
    // Copy the scaled U from the SVD of R1 R2^H into the top of the matrix
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = s[j];
        const Real* RESTRICT UColOrig = &buffer[j*r];
        Real* RESTRICT UColScaled = &A.U[j*m];
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UColOrig[i];
    }
    // Hit the matrix from the left with Q1 from the QR decomp of the orig A.U
    lapack::ApplyQ
    ( 'L', 'C', m, roundedRank, r, &buffer[2*blockSize], m, &tauU[0], 
      &A.U[0], m, &buffer[0], blockSize );

    // Copy the result of the QR factorization of A.V into a temporary buffer
    std::memcpy( &buffer[2*blockSize], &A.V[0], n*r*sizeof(Real) );
    // Logically shrink A.V
    A.V.resize( n*roundedRank );
    // Zero the shrunk buffer
    std::memset( &A.V[0], 0, A.V.size()*sizeof(Real) );
    // Copy the conj-trans of V from the SVD of R1 R2^H into the top of A.V
    for( int j=0; j<roundedRank; ++j )
    {
        const Real* RESTRICT VRowOrig = &buffer[blockSize+j];
        Real* RESTRICT VColConj = &A.V[j*n];
        for( int i=0; i<r; ++i )
            VColConj[i] = VRowOrig[i*r];
    }
    // Hit the matrix from the left with Q2 from the QR decomp of the orig A.V
    lapack::ApplyQ
    ( 'L', 'C', n, roundedRank, r, &buffer[2*blockSize], n, &tauV[0],
      &A.V[0], n, &buffer[0], blockSize );
#endif // PIVOTED_QR

    // Mark the matrix as having the new reduced rank
    A.r = roundedRank;
}

// A :~= A
//
// Approximate A with a given maximum rank.
template<typename Real>
void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix< std::complex<Real> >& A )
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
    lapack::QR
    ( n, r, &A.V[0], n, &tauV[0], &buffer[0], rightPanelSize );

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

    // Update W := R1 (R2^H). We are unfortunately performing 2x as many
    // flops as are required.
    blas::Trmm( 'R', 'U', 'T', 'N', r, r, 1, &A.V[0], n, &buffer[0], r );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize): R1 R2^H                                                //
    //------------------------------------------------------------------------//

    // Get the SVD of R1 R2^H, overwriting R1 R2^H with U
    std::vector<Real> realBuffer( 6*r );
    lapack::SVD
    ( 'O', 'A', r, r, &buffer[0], r, &realBuffer[0], 0, 0, 
      &buffer[blockSize], r, &buffer[2*blockSize], lworkSVD, &realBuffer[r] );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize):           U of R1 R2^H                                 //
    //  [blockSize,2*blockSize): V^H of R1 R2^H                               //
    //                                                                        //
    // realBuffer contains:                                                   //
    //   [0,r): singular values of R1 R2^H                                    //
    //------------------------------------------------------------------------//

    // Copy the result of the QR factorization of A.U into a temporary buffer
    std::memcpy( &buffer[2*blockSize], &A.U[0], m*r*sizeof(Scalar) );
    // Logically shrink A.U
    A.U.resize( m*roundedRank );
    // Zero the shrunk buffer
    std::memset( &A.U[0], 0, A.U.size()*sizeof(Scalar) );
    // Copy the scaled U from the SVD of R1 R2^H into the top of the matrix
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = realBuffer[j];
        const Scalar* RESTRICT UColOrig = &buffer[j*r];
        Scalar* RESTRICT UColScaled = &A.U[j*m];
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UColOrig[i];
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
    // Copy the conj-trans of V from the SVD of R1 R2^H into the top of A.V
    for( int j=0; j<roundedRank; ++j )
    {
        const Scalar* RESTRICT VRowOrig = &buffer[blockSize+j];
        Scalar* RESTRICT VColConj = &A.V[j*n];
        for( int i=0; i<r; ++i )
            VColConj[i] = Conj( VRowOrig[i*r] );
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
( int maxRank, FactorMatrix<float>& A );
template void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<double>& A );
template void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix< std::complex<float> >& A );
template void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix< std::complex<double> >& A );
