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

namespace {

template<typename Real>
void PrintScalar( Real alpha )
{
    std::cout << alpha;
}

template<typename Real>
void PrintScalar( std::complex<Real> alpha )
{
    std::cout << std::real(alpha) << "+" << std::imag(alpha) << "i";
}

} // anonymous namespace

template<typename Scalar>
void psp::hmatrix_tools::Print
( const std::string& tag, const DenseMatrix<Scalar>& D )
{
    std::cout << tag << "\n";
    if( D.Symmetric() )
    {
        const int m = D.Height();
        for( int i=0; i<m; ++i )
        {
            for( int j=0; j<=i; ++j )
            {
                PrintScalar( D.Get(i,j) );
                std::cout << " ";
            }
            for( int j=i+1; j<m; ++j )
            {
                PrintScalar( D.Get(j,i) );
                std::cout << " ";
            }
            std::cout << "\n";
        }
    }
    else
    {
        const int m = D.Height(); 
        const int n = D.Width();
        for( int i=0; i<m; ++i )
        {
            for( int j=0; j<n; ++j )
            {
                PrintScalar( D.Get(i,j) );
                std::cout << " ";
            }
            std::cout << "\n";
        }
    }
    std::cout << std::endl;
}

template<typename Scalar,bool Conjugated>
void psp::hmatrix_tools::Print
( const std::string& tag, const LowRankMatrix<Scalar,Conjugated>& F )
{
    std::ostringstream streamU;
    streamU << tag << ".U";
    Print( streamU.str(), F.U );
    
    std::ostringstream streamV;
    streamV << tag << ".V";
    Print( streamV.str(), F.V );
}

template<typename Scalar>
void psp::hmatrix_tools::Print
( const std::string& tag, const SparseMatrix<Scalar>& S )
{
    if( S.symmetric )
        std::cout << tag << "(symmetric)\n";
    else
        std::cout << tag << "\n";

    const int m = S.height;
    for( int i=0; i<m; ++i )
    {
        const int numCols = S.rowOffsets[i+1]-S.rowOffsets[i];
        const int rowOffset = S.rowOffsets[i];
        for( int k=0; k<numCols; ++k )
        {
            const int j = S.columnIndices[rowOffset+k];
            const Scalar alpha = S.nonzeros[rowOffset+k];
            std::cout << "(" << i << "," << j << "): ";
            PrintScalar( alpha );
            std::cout << "\n";
        }
    }
    std::cout << std::endl;
}

template void psp::hmatrix_tools::Print
( const std::string& tag, const DenseMatrix<float>& D );
template void psp::hmatrix_tools::Print
( const std::string& tag, const DenseMatrix<double>& D );
template void psp::hmatrix_tools::Print
( const std::string& tag, const DenseMatrix< std::complex<float> >& D );
template void psp::hmatrix_tools::Print
( const std::string& tag, const DenseMatrix< std::complex<double> >& D );

template void psp::hmatrix_tools::Print
( const std::string& tag, const LowRankMatrix<float,false>& F );
template void psp::hmatrix_tools::Print
( const std::string& tag, const LowRankMatrix<float,true>& F );
template void psp::hmatrix_tools::Print
( const std::string& tag, const LowRankMatrix<double,false>& F );
template void psp::hmatrix_tools::Print
( const std::string& tag, const LowRankMatrix<double,true>& F );
template void psp::hmatrix_tools::Print
( const std::string& tag, const LowRankMatrix<std::complex<float>,false>& F );
template void psp::hmatrix_tools::Print
( const std::string& tag, const LowRankMatrix<std::complex<float>,true>& F );
template void psp::hmatrix_tools::Print
( const std::string& tag, const LowRankMatrix<std::complex<double>,false>& F );
template void psp::hmatrix_tools::Print
( const std::string& tag, const LowRankMatrix<std::complex<double>,true>& F );

template void psp::hmatrix_tools::Print
( const std::string& tag, const SparseMatrix<float>& S );
template void psp::hmatrix_tools::Print
( const std::string& tag, const SparseMatrix<double>& S );
template void psp::hmatrix_tools::Print
( const std::string& tag, const SparseMatrix< std::complex<float> >& S );
template void psp::hmatrix_tools::Print
( const std::string& tag, const SparseMatrix< std::complex<double> >& S );

