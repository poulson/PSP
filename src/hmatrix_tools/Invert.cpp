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
void psp::hmatrix_tools::Invert
( DenseMatrix<Scalar>& D )
{
#ifndef RELEASE
    if( D.Height() != D.Width() )
        throw std::logic_error("Tried to invert a non-square dense matrix.");
#endif
    const int n = D.Height();
    const int lwork = std::min( 64*n, n*n );
    std::vector<int> ipiv( n );
    std::vector<Scalar> work( lwork );
    if( D.Symmetric() )
    {
        lapack::LDLT( 'L', n, D.Buffer(), D.LDim(), &ipiv[0], &work[0], lwork );
        lapack::InvertLDLT( 'L', n, D.Buffer(), D.LDim(), &ipiv[0], &work[0] );
    }
    else
    {
        lapack::LU( n, n, D.Buffer(), D.LDim(), &ipiv[0] );
        lapack::InvertLU( n, D.Buffer(), D.LDim(), &ipiv[0], &work[0], lwork );
    }
}

template void psp::hmatrix_tools::Invert
( DenseMatrix<float>& D );
template void psp::hmatrix_tools::Invert
( DenseMatrix<double>& D );
template void psp::hmatrix_tools::Invert
( DenseMatrix< std::complex<float> >& D );
template void psp::hmatrix_tools::Invert
( DenseMatrix< std::complex<double> >& D );
