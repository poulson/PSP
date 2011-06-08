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
    try
    {
        psp::Dense<double> D( m, n );

        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                D.Set( i, j, (double)i+j );
        D.Print( "D" );

        psp::LowRank<double,false> F;
        psp::hmat_tools::Compress( r, D, F );

        F.Print( "F.U F.V^T ~= D" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        psp::DumpCallStack();
#endif
    }

    std::cout << "----------------------------------------------------\n"
              << "Testing complex double-precision Compress           \n"
              << "----------------------------------------------------" 
              << std::endl;
    try
    {
        psp::Dense< std::complex<double> > D( m, n );

        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                D.Set( i, j, std::complex<double>(i+j,i-j) );
        D.Print( "D" );

        // F = F.U F.V^T
        psp::LowRank<std::complex<double>,false> FFalse;
        psp::Dense< std::complex<double> > DCopy;
        psp::hmat_tools::Copy( D, DCopy );
        psp::hmat_tools::Compress( r, DCopy, FFalse );
        FFalse.Print( "F.U F.V^T ~= D" );

        // F = F.U F.V^H
        psp::LowRank<std::complex<double>,true> FTrue;
        psp::hmat_tools::Copy( D, DCopy );
        psp::hmat_tools::Compress( r, DCopy, FTrue );
        FTrue.Print( "F.U F.V^H ~= D" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        psp::DumpCallStack();
#endif
    }

    return 0;
}
