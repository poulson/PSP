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
#include <cstring>

// A :~= A
//
// Approximate A with a given maximum rank.
void psp::hmat::Compress
( int maxRank, FactorMatrix& A )
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
    std::vector<PetscScalar> buffer
    ( 3*blockSize+std::max(lworkSVD,std::max(leftPanelSize,rightPanelSize)) );

#if defined(PIVOTED_QR)
    // TODO: 
#else
    // Perform an unpivoted QR decomposition on A.U
    std::vector<PetscScalar> tauU( std::min( m, r ) );
    lapack::QR( m, r, &A.U[0], m, &tauU[0], &buffer[0], leftPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Perform an unpivoted QR decomposition on A.V
    std::vector<PetscScalar> tauV( std::min( n, r ) );
    lapack::QR
    ( n, r, &A.V[0], n, &tauV[0], &buffer[0], rightPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    {
        PetscScalar* RESTRICT W = &buffer[0];
        const PetscScalar* RESTRICT R1 = &A.U[0];
        std::memset( W, 0, blockSize*sizeof(PetscScalar) );
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

    // Get the SVD of R1 R2^H
    std::vector<PetscReal> realBuffer( 6*r );
    lapack::SVD
    ( 'A', 'A', r, r, &buffer[0], r,
      &realBuffer[0], &buffer[blockSize], r, &buffer[2*blockSize], r, 
      &buffer[3*blockSize], lworkSVD, &realBuffer[r] );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize):             trash                                      //
    //  [blockSize,2*blockSize):   U of R1 R2^H                               //
    //  [2*blockSize,3*blockSize): V^H of R1 R2^H                             //
    //                                                                        //
    // realBuffer contains:                                                   //
    //   [0,r): singular values of R1 R2^H                                    //
    //------------------------------------------------------------------------//

    // Copy the result of the QR factorization of A.U into a temporary buffer
    std::memcpy( &buffer[3*blockSize], &A.U[0], m*r*sizeof(PetscScalar) );
    // Logically shrink A.U
    A.U.resize( m*roundedRank );
    // Zero the shrunk buffer
    std::memset( &A.U[0], 0, A.U.size()*sizeof(PetscScalar) );
    // Copy the scaled U from the SVD of R1 R2^H into the top of the matrix
    for( int j=0; j<roundedRank; ++j )
    {
        const PetscReal sigma = &realBuffer[j];
        const PetscScalar* RESTRICT UColOrig = &buffer[blocksize+j*r];
        PetscScalar* RESTRICT UColScaled = &A.U[j*m];
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UColOrig[i];
    }
    // Hit the matrix from the left with Q1 from the QR decomp of the orig A.U
    lapack::ApplyQ
    ( 'L', 'C', m, roundedRank, r, &buffer[3*blockSize], m, &tauU[0], 
      &A.U[0], m, &buffer[0], blockSize );

    // Copy the result of the QR factorization of A.V into a temporary buffer
    std::memcpy( &buffer[3*blockSize], &A.V[0], n*r*sizeof(PetscScalar) );
    // Logically shrink A.V
    A.V.resize( n*roundedRank );
    // Zero the shrunk buffer
    std::memset( &A.V[0], 0, A.V.size()*sizeof(PetscScalar) );
    // Copy the conj-trans of V from the SVD of R1 R2^H into the top of A.V
    for( int j=0; j<roundedRank; ++j )
    {
        const PetscScalar* RESTRICT VRowOrig = &buffer[2*blockSize+j];
        PetscScalar* RESTRICT VColConj = &A.V[j*n];
        for( int i=0; i<r; ++i )
            VColConj = std::conj( VRowOrig[i*r] );
    }
    // Hit the matrix from the left with Q2 from the QR decomp of the orig A.V
    lapack::ApplyQ
    ( 'L', 'C', n, roundedRank, r, &buffer[3*blockSize], n, &tauV[0],
      &A.V[0], n, &buffer[0], blockSize );
#endif // PIVOTED_QR

    // Mark the matrix as having the new reduced rank
    A.r = roundedRank;
}

