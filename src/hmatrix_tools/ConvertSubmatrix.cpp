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

template<typename Scalar>
void psp::hmatrix_tools::ConvertSubmatrix
( DenseMatrix<Scalar>& D, const SparseMatrix<Scalar>& S, 
  int iStart, int iEnd, int jStart, int jEnd )
{
    // Initialize the dense matrix to all zeros
    if( S.symmetric && iStart == jStart )
        D.SetType( SYMMETRIC );
    else 
        D.SetType( GENERAL );
    D.Resize( iEnd-iStart, jEnd-jStart );
    Scale( (Scalar)0, D );
#ifndef RELEASE
    if( D.Symmetric() && iEnd != jEnd )
        throw std::logic_error("Invalid submatrix of symmetric sparse matrix.");
#endif

    // Add in the nonzeros, one row at a time
    const int m = D.Height();
    const int ldim = D.LDim();
    Scalar* DBuffer = D.Buffer();
    for( int iOffset=0; iOffset<m; ++iOffset )
    {
        const int thisRowOffset = S.rowOffsets[iStart+iOffset];
        const int nextRowOffset = S.rowOffsets[iStart+iOffset+1];

        const int* thisSetOfColIndices = &S.columnIndices[thisRowOffset];
        for( int k=0; k<nextRowOffset-thisRowOffset; ++k )
        {
            const int thisColIndex = thisSetOfColIndices[k];
            if( thisColIndex < jStart )
                continue;
            else if( thisColIndex < jEnd )
            {
                const int jOffset = thisColIndex - jStart;
                DBuffer[iOffset+jOffset*ldim] = S.nonzeros[thisRowOffset+k];
            }
            else
                break;
        }
    }
}

template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::ConvertSubmatrix
( LowRankMatrix<Scalar,Conjugated>& F, const SparseMatrix<Scalar>& S,
  int iStart, int iEnd, int jStart, int jEnd )
{
    // Count the number of nonzeros in the submatrix
    const int nonzeroStart = S.rowOffsets[iStart];
    const int nonzeroEnd = S.rowOffsets[iEnd];
    const int numNonzeros = nonzeroEnd - nonzeroStart;

    // Initialize the low-rank matrix to all zeros
    const int m = iEnd - iStart;
    const int n = jEnd - jStart;
    const int r = numNonzeros;
    F.U.SetType( GENERAL ); F.U.Resize( m, r );
    F.V.SetType( GENERAL ); F.V.Resize( n, r );
    Scale( (Scalar)0, F.U );
    Scale( (Scalar)0, F.V );

    // Fill in the representation of each nonzero using the appropriate column
    // of identity in F.U and the appropriate scaled column of identity in 
    // F.V
    int rankCounter = 0;
    for( int iOffset=0; iOffset<m; ++iOffset )
    {
        const int thisRowOffset = S.rowOffsets[iStart+iOffset];
        const int nextRowOffset = S.rowOffsets[iStart+iOffset+1];

        const int* thisSetOfColIndices = &S.columnIndices[thisRowOffset];
        for( int k=0; k<nextRowOffset-thisRowOffset; ++k )
        {
            const int thisColIndex = thisSetOfColIndices[k];
            if( thisColIndex < jStart )
                continue;
            else if( thisColIndex < jEnd )
            {
                const int jOffset = thisColIndex - jStart;
                const Scalar value = S.nonzeros[thisRowOffset+k];
                F.U.Set(iOffset,rankCounter,1);
                if( Conjugated )
                    F.V.Set(jOffset,rankCounter,Conj(value));
                else
                    F.V.Set(jOffset,rankCounter,value);
                ++rankCounter;
            }
            else
                break;
        }
    }
}

template void psp::hmatrix_tools::ConvertSubmatrix
(       DenseMatrix<float>& D, 
  const SparseMatrix<float>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void psp::hmatrix_tools::ConvertSubmatrix
(       DenseMatrix<double>& D, 
  const SparseMatrix<double>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void psp::hmatrix_tools::ConvertSubmatrix
(       DenseMatrix< std::complex<float> >& D, 
  const SparseMatrix< std::complex<float> >& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void psp::hmatrix_tools::ConvertSubmatrix
(       DenseMatrix< std::complex<double> >& D,
  const SparseMatrix< std::complex<double> >& S,
  int iStart, int iEnd, int jStart, int jEnd );

template void psp::hmatrix_tools::ConvertSubmatrix
(       LowRankMatrix<float,false>& F,
  const SparseMatrix<float>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void psp::hmatrix_tools::ConvertSubmatrix
(       LowRankMatrix<float,true>& F,
  const SparseMatrix<float>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void psp::hmatrix_tools::ConvertSubmatrix
(       LowRankMatrix<double,false>& F,
  const SparseMatrix<double>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void psp::hmatrix_tools::ConvertSubmatrix
(       LowRankMatrix<double,true>& F,
  const SparseMatrix<double>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void psp::hmatrix_tools::ConvertSubmatrix
(       LowRankMatrix<std::complex<float>,false>& F,
  const SparseMatrix< std::complex<float> >& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void psp::hmatrix_tools::ConvertSubmatrix
(       LowRankMatrix<std::complex<float>,true>& F,
  const SparseMatrix< std::complex<float> >& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void psp::hmatrix_tools::ConvertSubmatrix
(       LowRankMatrix<std::complex<double>,false>& F,
  const SparseMatrix< std::complex<double> >& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void psp::hmatrix_tools::ConvertSubmatrix
(       LowRankMatrix<std::complex<double>,true>& F,
  const SparseMatrix< std::complex<double> >& S,
  int iStart, int iEnd, int jStart, int jEnd );
