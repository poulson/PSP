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
template<typename Real,bool Conjugated>
void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<Real,Conjugated>& A )
{
    const int m = A.Height();
    const int n = A.Width();
    const int r = A.Rank();
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
    throw std::logic_error("Pivoted QR is not yet supported for this routine.");
#else
    // Perform an unpivoted QR decomposition on A.U
    std::vector<Real> tauU( std::min( m, r ) );
    lapack::QR
    ( m, r, A.U.Buffer(), A.U.LDim(), &tauU[0], &buffer[0], leftPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Perform an unpivoted QR decomposition on A.V
    std::vector<Real> tauV( std::min( n, r ) );
    lapack::QR
    ( n, r, A.V.Buffer(), A.V.LDim(), &tauV[0], &buffer[0], rightPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    {
        std::memset( &buffer[0], 0, blockSize*sizeof(Real) );
        for( int j=0; j<r; ++j )
            std::memcpy( &buffer[j*r], A.U.LockedBuffer(0,j), j*sizeof(Real) );
    }

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize): R1                                                     //
    //------------------------------------------------------------------------//

    // Update W := R1 R2^T. We are unfortunately performing 2x as many
    // flops as are required.
    blas::Trmm
    ( 'R', 'U', 'T', 'N', r, r, 
      1, A.V.LockedBuffer(), A.V.LDim(), &buffer[0], r );

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
    for( int j=0; j<r; ++j )
    {
        std::memcpy
        ( &buffer[2*blockSize+j*m], A.U.LockedBuffer(0,j), m*sizeof(Real) );
    }
    // Logically shrink A.U 
    A.U.Resize( m, roundedRank );
    // Zero the shrunk buffer
    Scale( (Real)0, A.U );
    // Copy the scaled U from the SVD of R1 R2^T into the top of the matrix
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = s[j];
        const Real* RESTRICT UCol = &buffer[j*r];
        Real* RESTRICT UColScaled = A.U.Buffer(0,j);
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Hit the matrix from the left with Q1 from the QR decomp of the orig A.U
    lapack::ApplyQ
    ( 'L', 'N', m, roundedRank, r, &buffer[2*blockSize], m, &tauU[0], 
      A.U.Buffer(), A.U.LDim(), &buffer[0], blockSize );

    // Copy the result of the QR factorization of A.V into a temporary buffer
    for( int j=0; j<r; ++j )
    {
        std::memcpy
        ( &buffer[2*blockSize+j*n], A.V.LockedBuffer(0,j), n*sizeof(Real) );
    }
    // Logically shrink A.V
    A.V.Resize( n, roundedRank );
    // Zero the shrunk buffer
    Scale( (Real)0, A.V );
    // Copy V=(V^T)^T from the SVD of R1 R2^T into the top of A.V
    for( int j=0; j<roundedRank; ++j )
    {
        const Real* RESTRICT VTRow = &buffer[blockSize+j];
        Real* RESTRICT VCol = A.V.Buffer(0,j);
        for( int i=0; i<r; ++i )
            VCol[i] = VTRow[i*r];
    }
    // Hit the matrix from the left with Q2 from the QR decomp of the orig A.V
    lapack::ApplyQ
    ( 'L', 'N', n, roundedRank, r, &buffer[2*blockSize], n, &tauV[0],
      A.V.Buffer(), A.V.LDim(), &buffer[0], blockSize );
#endif // PIVOTED_QR
}

// A :~= A
//
// Approximate A with a given maximum rank. This implementation handles both 
// cases, A = U V^T (Conjugated=false), and A = U V^H (Conjugated=true)
template<typename Real,bool Conjugated>
void psp::hmatrix_tools::Compress
( int maxRank, FactorMatrix<std::complex<Real>,Conjugated>& A )
{
    typedef std::complex<Real> Scalar;

    const int m = A.Height();
    const int n = A.Width();
    const int r = A.Rank();
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
    throw std::logic_error("Pivoted QR is not yet supported for this routine.");
#else
    // Perform an unpivoted QR decomposition on A.U
    std::vector<Scalar> tauU( std::min( m, r ) );
    lapack::QR
    ( m, r, A.U.Buffer(), A.U.LDim(), &tauU[0], &buffer[0], leftPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Perform an unpivoted QR decomposition on A.V
    std::vector<Scalar> tauV( std::min( n, r ) );
    lapack::QR
    ( n, r, A.V.Buffer(), A.V.LDim(), &tauV[0], &buffer[0], rightPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    {
        std::memset( &buffer[0], 0, blockSize*sizeof(Scalar) );
        for( int j=0; j<r; ++j )
            std::memcpy( &buffer[j*r], A.U.LockedBuffer(0,j), j*sizeof(Real) );
    }

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize): R1                                                     //
    //------------------------------------------------------------------------//

    // Update W := R1 R2^[T,H].
    // We are unfortunately performing 2x as many flops as required.
    const char option = ( Conjugated ? 'C' : 'T' );
    blas::Trmm
    ( 'R', 'U', option, 'N', r, r, 
      1, A.V.LockedBuffer(), A.V.LDim(), &buffer[0], r );

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
    for( int j=0; j<r; ++j )
    {
        std::memcpy
        ( &buffer[2*blockSize+j*m], A.U.LockedBuffer(0,j), m*sizeof(Real) );
    }
    // Logically shrink A.U
    A.U.Resize( m, roundedRank );
    // Zero the shrunk buffer
    Scale( (Scalar)0, A.U );
    // Copy the scaled U from the SVD of R1 R2^[T,H] into the top of the matrix
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = realBuffer[j];
        const Scalar* RESTRICT UCol = &buffer[j*r];
        Scalar* RESTRICT UColScaled = A.U.Buffer(0,j);
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Hit the matrix from the left with Q1 from the QR decomp of the orig A.U
    lapack::ApplyQ
    ( 'L', 'N', m, roundedRank, r, &buffer[2*blockSize], m, &tauU[0], 
      A.U.Buffer(), A.U.LDim(), &buffer[0], blockSize );

    // Copy the result of the QR factorization of A.V into a temporary buffer
    for( int j=0; j<r; ++j )
    {
        std::memcpy
        ( &buffer[2*blockSize+j*n], A.V.LockedBuffer(0,j), n*sizeof(Real) );
    }
    // Logically shrink A.V
    A.V.Resize( n, roundedRank );
    // Zero the shrunk buffer
    Scale( (Scalar)0, A.V );
    if( Conjugated )
    {
        // Copy V=(V^H)^H from the SVD of R1 R2^H into the top of A.V
        for( int j=0; j<roundedRank; ++j )
        {
            const Scalar* RESTRICT VHRow = &buffer[blockSize+j];
            Scalar* RESTRICT VCol = A.V.Buffer(0,j);
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
            Scalar* RESTRICT VColConj = A.V.Buffer(0,j);
            for( int i=0; i<r; ++i )
                VColConj[i] = VHRow[i*r];
        }
    }
    // Hit the matrix from the left with Q2 from the QR decomp of the orig A.V
    lapack::ApplyQ
    ( 'L', 'N', n, roundedRank, r, &buffer[2*blockSize], n, &tauV[0],
      A.V.Buffer(), A.V.LDim(), &buffer[0], blockSize );
#endif // PIVOTED_QR
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
