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

// C :~= alpha A + beta B
//
// TODO:
// We could make use of a pivoted QLP factorization [Stewart, 1999] in order to
// get a more accurate approximation to the truncated singular value 
// decomposition in O(mnk) work for an m x n matrix and k singular vectors.
//
// See Huckaby and Chan's 2004 paper:
// "Stewart's pivoted QLP decomposition for low-rank matrices".
//
// TODO: 
// Carefully consider whether or not we can avoid explicitly expanding our
// low-rank form before performing the QR/QLP decomposition. My suspicion is no,
// so perhaps we should instead aim for avoiding the buffer allocation cost by 
// leaving one around.
void psp::hmat::MatAddRounded
( const int maxRank,
  const PetscScalar alpha, 
  const FactorMatrix& A, 
  const PetscScalar beta,
  const FactorMatrix& B, 
        FactorMatrix& C )
{
    const int m = A.m;
    const int n = A.n;
    const int r = A.r + B.r;
    const int roundedRank = std::min( r, maxRank );

    // Grab enough workspace for our entire rounded addition
    const int leftPanelSize = m*r;
    const int rightPanelSize = n*r;
    const int blockSize = r*r;
    const int lworkSVD = 4*r*r;
    std::vector<PetscScalar> buffer
    ( leftPanelSize+rightPanelSize+
      std::max(std::max(leftPanelSize,rightPanelSize),3*blockSize+lworkSVD) );

    // Put [(alpha A.U), (beta B.U)] into the first 'leftPanelSize' entries of
    // our buffer and [A.V, B.V] into the next 'rightPanelSize' entries
    // Copy in (alpha A.U)
    {
        const int Ar = A.r;
        PetscScalar* RESTRICT packedAU = &buffer[0];
        const PetscScalar* RESTRICT AU = &A.U[0];
        for( PetscInt j=0; j<Ar; ++j )
            for( PetscInt i=0; i<m; ++i )
                packedAU[i+j*m] = alpha*AU[i+j*m];
    }
    // Copy in (beta B.U)
    {
        const int Br = B.r;
        PetscScalar* RESTRICT packedBU = &buffer[m*A.r];
        const PetscScalar* RESTRICT BU = &B.U[0];
        for( PetscInt j=0; j<Br; ++j )
            for( PetscInt i=0; i<m; ++i )
                packedBU[i+j*m] = beta*BU[i+j*m];
    }
    // Copy in A.V
    std::memcpy
    ( &buffer[leftPanelSize], &A.V[0], n*A.r*sizeof(PetscScalar) );
    // Copy in B.V
    std::memcpy
    ( &buffer[leftPanelSize+n*A.r], &B.V[0], n*B.r*sizeof(PetscScalar) );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize): [(alpha A.U), (beta B.U)]                          //
    //  [leftPanelSize,leftPanelSize+rightPanelSize): [A.V, B.V]              //
    //------------------------------------------------------------------------//
    const int offset = leftPanelSize + rightPanelSize;

#if defined(PIVOTED_QR)
    // Perform a pivoted QR decomposition on A.U

    // Perform a pivoted QR decomposition on A.V

    // TODO: 
    // Form the compressed sum using the truncated SVD of R1 R2^H.

#else
    // Perform an unpivoted QR decomposition on [(alpha A.U), (beta B.U)]
    std::vector<PetscScalar> tauU( std::min( m, r ) );
    lapack::QR( m, r, &buffer[0], m, &tauU[0], &buffer[offset], leftPanelSize );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):      qr([(alpha A.U), (beta B.U)])                 //
    //  [leftPanelSize,offset): [A.V, B.V]                                    //
    //------------------------------------------------------------------------//

    // Perform an unpivoted QR decomposition on [A.V, B.V]
    std::vector<PetscScalar> tauV( std::min( n, r ) );
    lapack::QR
    ( n, r, &buffer[leftPanelSize], n, &tauV[0], 
      &buffer[offset], rightPanelSize );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):      qr([(alpha A.U), (beta B.U)])                 //
    //  [leftPanelSize,offset): qr([A.V, B.V])                                //
    //------------------------------------------------------------------------//

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    {
        PetscScalar* RESTRICT W = &buffer[offset];
        const PetscScalar* RESTRICT R1 = &buffer[0];
        std::memset( W, 0, blockSize*sizeof(PetscScalar) );
        for( int j=0; j<r; ++j )
            for( int i=0; i<=j; ++i )
                W[i+j*r] = R1[i+j*m];
    }

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):         qr([(alpha A.U), (beta B.U)])              //
    //  [leftPanelSize,offset):    qr([A.V, B.V])                             //
    //  [offset,offset+blockSize): R1                                         //
    //------------------------------------------------------------------------//

    // Update W := R1 (R2^H). We are unfortunately performing 2x as many
    // flops as are required.
    blas::Trmm
    ( 'R', 'U', 'T', 'N', r, r, 
      1, &buffer[leftPanelSize], n, &buffer[offset], r );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):         qr([(alpha A.U), (beta B.U)])              //
    //  [leftPanelSize,offset):    qr([A.V, B.V])                             //
    //  [offset,offset+blockSize): R1 R2^H                                    //
    //------------------------------------------------------------------------//

    // Get the SVD of R1 R2^H
    std::vector<PetscReal> realBuffer( 6*r );
    lapack::SVD
    ( 'A', 'A', r, r, &buffer[offset], r, &realBuffer[0], 
      &buffer[offset+blockSize], r, &buffer[offset+2*blockSize], r, 
      &buffer[offset+3*blockSize], lworkSVD, &realBuffer[r] );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,leftPanelSize):                       qr([(alpha A.U),(beta B.U)]) //
    //  [leftPanelSize,offset):                  qr([A.V, B.V])               //
    //  [offset,offset+blockSize):               trash                        //
    //  [offset+blockSize,offset+2*blockSize):   U of R1 R2^H                 //
    //  [offset+2*blockSize,offset+3*blockSize): V^H of R1 R2^H               //
    //                                                                        //
    // realBuffer contains:                                                   //
    //   [0,r): singular values of R1 R2^H                                    //
    //   [r,6r): trash                                                        //
    //------------------------------------------------------------------------//

    // Set up space in C
    C.m = m;
    C.n = n;
    C.U.resize( m*roundedRank );
    C.V.resize( n*roundedRank );

    // Form the rounded C.U by first filling it with 
    //  | S*U |, and then hitting it from the left with Q1
    //  |  0  |
    std::memset( &C.U[0], 0, m*roundedRank*sizeof(PetscScalar) );
    for( int j=0; j<roundedRank; ++j )
    {
        const PetscReal sigma = &realBuffer[j];
        PetscScalar* UColOrig = &buffer[offset+blockSpace+j*r];
        PetscScalar* UColScaled = &C.U[j*roundedRank];
        for( int i=0; i<roundedRank; ++i )
            UColScaled[i] = sigma*UColOrig[i];
    }
    // Apply Q1 and use the trashed R1 R2^H space for our work buffer
    lapack::ApplyQ
    ( 'L', 'C', m, roundedRank, r, &buffer[0], C.m, &tauU[0], &C.U[0], C.m, 
      &buffer[offset], blockSize );

    // Form the rounded C.V by first filling it with 
    //  | conj(V^H) |, and then hitting it from the right with Q2
    //  |     0     |
    // HERE
#endif
}

