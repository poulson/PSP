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
    std::cout << "PackedQR <r> <s> <t> <print?>\n"
              << "r: width of triangular matrices\n"
              << "s: height of top matrix    (s <= r)\n"
              << "t: height of bottom matrix (t <= r)\n"
              << "print?: print out matrices?" << std::endl;
}

int
main( int argc, char* argv[] )
{
    if( argc < 5 )
    {
        Usage();
        return 0;
    }
    const int r = atoi( argv[1] );
    const int s = atoi( argv[2] );
    const int t = atoi( argv[3] );
    const bool print = atoi( argv[4] );

    if( s > r || t > r )
    {
        std::cout << "s and t cannot be greater than r" << std::endl;
        return 0;
    }

    std::cout << "----------------------------------------------------\n"
              << "Testing complex double-precision PackedQR           \n"
              << "----------------------------------------------------" 
              << std::endl;
    try
    {
        // Fill a packed version of two concatenated upper triangular s x r
        // and t x r matrices with Gaussian random variables.
        const int minDimT = std::min(s,r);
        const int minDimB = std::min(t,r);
        const int packedSize = 
            (minDimT*minDimT+minDimT)/2 + (r-minDimT)*minDimT + 
            (minDimB*minDimB+minDimB)/2 + (r-minDimB)*minDimB;

        // Form the random packed A
        std::vector<std::complex<double> > packedA( packedSize );
        for( int k=0; k<packedSize; ++k )
            psp::SerialGaussianRandomVariable( packedA[k] );

        // Expand the packed A
        psp::Dense<std::complex<double> > A( s+t, r );
        psp::hmat_tools::Scale( std::complex<double>(0), A );
        {
            int k=0;
            for( int j=0; j<r; ++j )
            {
                for( int i=0; i<std::min(j+1,s); ++i )
                    A.Set( i, j, packedA[k++] );
                for( int i=s; i<std::min(j+s+1,s+t); ++i )
                    A.Set( i, j, packedA[k++] );
            }
        }
        if( print )
        {
            psp::hmat_tools::PrintPacked( "packedA:", r, s, t, &packedA[0] );
            std::cout << std::endl;
        }

        // Allocate a workspace and perform the packed QR
        std::vector<std::complex<double> > tau( std::min(s+t,r) ), work(s+t);
        psp::hmat_tools::PackedQR( r, s, t, &packedA[0], &tau[0], &work[0] );
        if( print )
        {
            psp::hmat_tools::PrintPacked( "packedQR:", r, s, t, &packedA[0] );
            std::cout << "\ntau:\n";
            for( unsigned j=0; j<tau.size(); ++j )
                std::cout << 
                    psp::ScalarWrapper<std::complex<double> >(tau[j]) << "\n";
            std::cout << std::endl;
        }

        // Copy the R into a zeroed (s+t) x r matrix
        psp::Dense<std::complex<double> > B( s+t, r );
        psp::hmat_tools::Scale( std::complex<double>(0), B );
        int k=0;
        for( int j=0; j<r; ++j )
        {
            const int S = std::min(j+1,s);
            const int T = std::min(j+1,t);
            const int U = std::min(j+1,S+T);

            for( int i=0; i<U; ++i )
                B.Set( i, j, packedA[k++] );

            k += (S+T) - U;
        }
        if( print )
        {
            B.Print( "R" );
            std::cout << std::endl;
        }

        psp::hmat_tools::ApplyPackedQFromLeft
        ( r, s, t, &packedA[0], &tau[0], B, &work[0] );
        {
            double maxError = 0;
            for( int j=0; j<r; ++j )
                for( int i=0; i<s+t; ++i )
                    maxError = 
                        std::max(maxError,psp::Abs(B.Get(i,j)-A.Get(i,j)));
            std::cout << "||QR-A||_oo = " << maxError << std::endl;
        }
        if( print )
        {
            B.Print( "QR ~= A" );
            std::cout << std::endl;
        }

        // Create an (s+t) x (s+t) identity matrix and then apply Q' Q from 
        // the left
        B.Resize( s+t, s+t );
        work.resize( s+t );
        psp::hmat_tools::Scale( std::complex<double>(0), B );
        for( int j=0; j<s+t; ++j )
            B.Set( j, j, 1.0 );
        psp::hmat_tools::ApplyPackedQFromLeft
        ( r, s, t, &packedA[0], &tau[0], B, &work[0] );
        psp::hmat_tools::ApplyPackedQAdjointFromLeft
        ( r, s, t, &packedA[0], &tau[0], B, &work[0] );
        {
            double maxError = 0;
            for( int j=0; j<s+t; ++j )
            {
                for( int i=0; i<s+t; ++i )
                {
                    const std::complex<double> computed = B.Get(i,j);
                    if( i == j )
                        maxError = std::max(maxError,psp::Abs(computed-1.0));
                    else
                        maxError = std::max(maxError,psp::Abs(computed));
                }
            }
            std::cout << "||I - Q'QI||_oo = " << maxError << std::endl;
        }
        if( print )
        {
            B.Print( "Q' Q I" );
            std::cout << std::endl;
        }

        // Create an (s+t) x (s+t) identity matrix and then apply Q' Q from 
        // the right
        psp::hmat_tools::Scale( std::complex<double>(0), B );
        for( int j=0; j<s+t; ++j )
            B.Set( j, j, 1.0 );
        psp::hmat_tools::ApplyPackedQAdjointFromRight
        ( r, s, t, &packedA[0], &tau[0], B, &work[0] );
        psp::hmat_tools::ApplyPackedQFromRight
        ( r, s, t, &packedA[0], &tau[0], B, &work[0] );
        {
            double maxError = 0;
            for( int j=0; j<s+t; ++j )
            {
                for( int i=0; i<s+t; ++i )
                {
                    const std::complex<double> computed = B.Get(i,j);
                    if( i == j )
                        maxError = std::max(maxError,psp::Abs(computed-1.0));
                    else
                        maxError = std::max(maxError,psp::Abs(computed));
                }
            }
            std::cout << "||I - IQ'Q||_oo = " << maxError << std::endl;
        }
        if( print )
        {
            B.Print( "I Q' Q" );
            std::cout << std::endl;
        }
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
