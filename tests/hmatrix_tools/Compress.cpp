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

int
main( int argc, char* argv[] )
{
    const int m = 8;
    const int n = 8;
    const int r = 3;

    std::cout << "----------------------------------------------------\n"
              << "Testing double-precision Compress                   \n"
              << "----------------------------------------------------" 
              << std::endl;
    {
        psp::DenseMatrix<double> D( m, n );

        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                D.Set( i, j, (double)i+j );
        psp::hmatrix_tools::Print( "D", D );

        psp::LowRankMatrix<double,false> F;
        psp::hmatrix_tools::Compress( r, D, F );

        psp::hmatrix_tools::Print( "F.U F.V^T ~= D", F );
    }

    std::cout << "----------------------------------------------------\n"
              << "Testing complex double-precision Compress           \n"
              << "----------------------------------------------------" 
              << std::endl;
    {
        psp::DenseMatrix< std::complex<double> > D( m, n );

        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                D.Set( i, j, std::complex<double>(i+j,i-j) );
        psp::hmatrix_tools::Print( "D", D );

        // F = F.U F.V^T
        psp::LowRankMatrix<std::complex<double>,false> FFalse;
        psp::DenseMatrix< std::complex<double> > DCopy;
        psp::hmatrix_tools::Copy( D, DCopy );
        psp::hmatrix_tools::Compress( r, DCopy, FFalse );

        // F = F.U F.V^H
        psp::LowRankMatrix<std::complex<double>,true> FTrue;
        psp::hmatrix_tools::Copy( D, DCopy );
        psp::hmatrix_tools::Compress( r, DCopy, FTrue );

        psp::hmatrix_tools::Print( "F.U F.V^T ~= D", FFalse );
        psp::hmatrix_tools::Print( "F.U F.V^H ~= D", FTrue );
    }

    return 0;
}
