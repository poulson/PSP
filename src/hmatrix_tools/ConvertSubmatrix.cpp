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
    std::memset( D.Buffer(), 0, D.LDim()*D.Width()*sizeof(Scalar) );
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

template<typename Scalar>
void psp::hmatrix_tools::ConvertSubmatrix
( FactorMatrix<Scalar>& F, const SparseMatrix<Scalar>& S,
  int iStart, int iEnd, int jStart, int jEnd )
{
    // Count the number of nonzeros in the submatrix
    const int nonzeroStart = S.rowOffsets[iStart];
    const int nonzeroEnd = S.rowOffsets[iEnd];
    const int numNonzeros = nonzeroEnd - nonzeroStart;

    // Initialize the factor matrix to all zeros
    F.m = iEnd - iStart;
    F.n = jEnd - jStart;
    F.r = numNonzeros;
    F.U.resize( F.m*F.r );
    F.V.resize( F.n*F.r );
    std::memset( &F.U[0], 0, F.m*F.r*sizeof(Scalar) );
    std::memset( &F.V[0], 0, F.n*F.r*sizeof(Scalar) );

    // Fill in the representation of each nonzero using the appropriate column
    // of identity in F.U and the appropriate scaled column of identity in 
    // F.V
    int rankCounter = 0;
    const int m = F.m;
    const int n = F.n;
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
                const Scalar conjValue = Conj( S.nonzeros[thisRowOffset+k] );
                F.U[iOffset+rankCounter*m] = 1;
                F.V[jOffset+rankCounter*n] = conjValue;
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
(       FactorMatrix<float>& F,
  const SparseMatrix<float>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void psp::hmatrix_tools::ConvertSubmatrix
(       FactorMatrix<double>& F,
  const SparseMatrix<double>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void psp::hmatrix_tools::ConvertSubmatrix
(       FactorMatrix< std::complex<float> >& F,
  const SparseMatrix< std::complex<float> >& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void psp::hmatrix_tools::ConvertSubmatrix
(       FactorMatrix< std::complex<double> >& F,
  const SparseMatrix< std::complex<double> >& S,
  int iStart, int iEnd, int jStart, int jEnd );
