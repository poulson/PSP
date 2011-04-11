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
              << "Testing double-precision dense MatrixAdd            \n"
              << "----------------------------------------------------" 
              << std::endl;
    {
        psp::DenseMatrix<double> A( m, n );
        psp::DenseMatrix<double> B( m, n );
        psp::DenseMatrix<double> C;

        // Set A to all 1's
        for( int j=0; j<n; ++j )
            for( int i=0; i<m; ++i )
                A.Set( i, j, 1.0 );

        // Set B to all 2's
        for( int j=0; j<n; ++j )
            for( int i=0; i<m; ++i )
                B.Set( i, j, 2.0 );

        psp::hmatrix_tools::Print( "A", A );
        psp::hmatrix_tools::Print( "B", B );

        // C := 3 A + 5 B
        psp::hmatrix_tools::MatrixAdd( 3.0, A, 5.0, B, C );

        psp::hmatrix_tools::Print( "C := 3A + 5B", C );
    }

    std::cout << "----------------------------------------------------\n"
              << "Testing complex single-precision dense MatrixAdd    \n"
              << "----------------------------------------------------" 
              << std::endl;
    {
        psp::DenseMatrix< std::complex<float> > A( m, n );
        psp::DenseMatrix< std::complex<float> > B( m, n );
        psp::DenseMatrix< std::complex<float> > C;

        // Set each entry of A to (1 + 2i)
        for( int j=0; j<n; ++j )
            for( int i=0; i<m; ++i )
                A.Set( i, j, std::complex<float>(1,2) );

        // Set each entry of B to (3 + 4i)
        for( int j=0; j<n; ++j )
            for( int i=0; i<m; ++i )
                B.Set( i, j, std::complex<float>(3,4) );

        psp::hmatrix_tools::Print( "A", A );
        psp::hmatrix_tools::Print( "B", B );

        // C := (5 + 6i)A + (7 + 8i)B
        psp::hmatrix_tools::MatrixAdd
        ( std::complex<float>(5,6), A, std::complex<float>(7,8), B, C );

        psp::hmatrix_tools::Print( "C := (5+6i)A + (7+8i)B", C );
    }

    std::cout << "----------------------------------------------------\n"
              << "Testing double-precision low-rank MatrixAdd         \n"
              << "----------------------------------------------------" 
              << std::endl;
    {
        psp::LowRankMatrix<double,false> A; 
        psp::LowRankMatrix<double,false> B;
        psp::LowRankMatrix<double,false> C;

        A.U.Resize( m, r );
        A.V.Resize( n, r );
        B.U.Resize( m, r );
        B.V.Resize( n, r );

        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                A.U.Set( i, j, (double)j );
        for( int j=0; j<r; ++j )
            for( int i=0; i<n; ++i )
                A.V.Set( i, j, (double)i+j );

        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                B.U.Set( i, j, (double)-j );
        for( int j=0; j<r; ++j )
            for( int i=0; i<n; ++i )
                B.V.Set( i, j, (double)-(i+j) );

        psp::hmatrix_tools::Print( "A", A );
        psp::hmatrix_tools::Print( "B", B );

        // C := 3A + 5B
        psp::hmatrix_tools::MatrixAdd( 3.0, A, 5.0, B, C );

        psp::hmatrix_tools::Print( "C := 3A + 5B", C );
    }

    return 0;
}
