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

void Usage()
{
    std::cout << "PackedQR <r> <print?>\n"
              << "r: size of the problem\n"
              << "print?: print out matrices?" << std::endl;
}

int
main( int argc, char* argv[] )
{
    if( argc < 3 )
    {
        Usage();
        return 0;
    }
    const int r = atoi( argv[1] );
    const bool print = atoi( argv[2] );

    std::cout << "----------------------------------------------------\n"
              << "Testing double-precision PackedQR                   \n"
              << "----------------------------------------------------" 
              << std::endl;
    try
    {
        // Fill a packed version of two concatenated upper triangular r x r
        // matrices with uniformly random samples from (0,1].
        const int packedSize = r*r + r;
        std::vector<double> packedA( packedSize );
        for( int j=0; j<packedSize; ++j )
            packedA[j] = psp::SerialUniform<double>();
        if( print )
            psp::hmat_tools::PrintPacked( r, &packedA[0], "packedA:" );

        // Allocate a workspace and perform the packed QR
        std::vector<double> tau(r), work(r);
        psp::hmat_tools::PackedQR( r, &packedA[0], &tau[0], &work[0] );
        if( print )
        {
            psp::hmat_tools::PrintPacked( r, &packedA[0], "packedQR:" );
            std::cout << "tau:\n";
            for( int j=0; j<r; ++j )
                std::cout << tau[j] << "\n";
            std::cout << std::endl;
        }

        // Copy the R into a zeroed 2r x r matrix
        psp::Dense<double> B( 2*r, r );
        psp::hmat_tools::Scale( 0.0, B );
        for( int j=0; j<r; ++j )
            for( int i=0; i<=j; ++i )
                B.Set( i, j, packedA[j*(j+1)+i] );
        B.Print( "R" );

        psp::hmat_tools::ApplyPackedQ( r, &packedA[0], &tau[0], B, &work[0] );
        B.Print( "QR ~= A" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        psp::DumpCallStack();
#endif
    }

    std::cout << "----------------------------------------------------\n"
              << "Testing complex double-precision PackedQR           \n"
              << "----------------------------------------------------" 
              << std::endl;
    try
    {
        // Fill a packed version of two concatenated upper triangular r x r
        // matrices with uniformly random samples from (0,1].
        const int packedSize = r*r + r;
        std::vector<std::complex<double> > packedA( packedSize );
        for( int j=0; j<packedSize; ++j )
            packedA[j] = std::complex<double>(psp::SerialUniform<double>(),
                                              psp::SerialUniform<double>());
        if( print )
            psp::hmat_tools::PrintPacked( r, &packedA[0], "packedA:" );

        // Allocate a workspace and perform the packed QR
        std::vector<std::complex<double> > tau(r), work(r);
        psp::hmat_tools::PackedQR( r, &packedA[0], &tau[0], &work[0] );
        if( print )
        {
            psp::hmat_tools::PrintPacked( r, &packedA[0], "packedQR:" );
            std::cout << "tau:\n";
            for( int j=0; j<r; ++j )
                std::cout << tau[j] << "\n";
            std::cout << std::endl;
        }

        // Copy the R into a zeroed 2r x r matrix
        psp::Dense<std::complex<double> > B( 2*r, r );
        psp::hmat_tools::Scale( std::complex<double>(0), B );
        for( int j=0; j<r; ++j )
            for( int i=0; i<=j; ++i )
                B.Set( i, j, packedA[j*(j+1)+i] );
        B.Print( "R" );

        psp::hmat_tools::ApplyPackedQ( r, &packedA[0], &tau[0], B, &work[0] );
        B.Print( "QR ~= A" );
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
