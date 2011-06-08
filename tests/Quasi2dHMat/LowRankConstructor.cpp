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
    const int xSize = 4;
    const int ySize = 4;
    const int zSize = 2;
    const int m = xSize*ySize*zSize;
    const int n = xSize*ySize*zSize;
    const int r = 3;

    std::cout << "----------------------------------------------------\n"
              << "Converting double-precision low-rank to Quasi2dHMat \n"
              << "----------------------------------------------------" 
              << std::endl;
    try
    {
        psp::LowRank<double,false> F;
        F.U.Resize( m, r );
        F.V.Resize( n, r );
        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                F.U.Set( i, j, (double)i+j );
        for( int j=0; j<r; ++j )
            for( int i=0; i<n; ++i )
                F.V.Set( i, j, (double)i-j );
        F.Print( "F" );

        psp::Quasi2dHMat<double,false> 
            H( F, 2, r, false, xSize, ySize, zSize );

        psp::Vector<double> x( n );
        double* xBuffer = x.Buffer();
        for( int i=0; i<n; ++i )
            xBuffer[i] = 1.0;
        x.Print( "x" );

        psp::Vector<double> y;
        H.Multiply( 2.0, x, y );
        y.Print( "y := 2 H x ~= 2 F x" );
        H.TransposeMultiply( 2.0, x, y );
        y.Print( "y := 2 H^T x ~= 2 F^T x" );
        H.AdjointMultiply( 2.0, x, y );
        y.Print( "y := 2 H^H x ~= 2 F^H x" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        psp::DumpCallStack();
#endif
    }
    
    std::cout << "-----------------------------------------------\n"
              << "Converting double-complex sparse to Quasi2dHMat\n"
              << "------------------------------------------------" 
              << std::endl;
    try
    {
        psp::LowRank<std::complex<double>,false> F;
        F.U.Resize( m, r );
        F.V.Resize( n, r );
        for( int j=0; j<r; ++j )
            for( int i=0; i<m; ++i )
                F.U.Set( i, j, std::complex<double>(i,j) );
        for( int j=0; j<r; ++j )
            for( int i=0; i<n; ++i )
                F.V.Set( i, j, std::complex<double>(i+j,i-j) );
        F.Print( "F" );

        psp::Quasi2dHMat<std::complex<double>,false> 
            H( F, 2, r, false, xSize, ySize, zSize );

        psp::Vector< std::complex<double> > x( n );
        std::complex<double>* xBuffer = x.Buffer();
        for( int i=0; i<n; ++i )
            xBuffer[i] = std::complex<double>(1.0,3.0);
        x.Print( "x" );

        psp::Vector< std::complex<double> > y;
        H.Multiply( std::complex<double>(4.0,5.0), x, y );
        y.Print( "y := (4+5i)H x ~= (4+5i)F x" );
        H.TransposeMultiply( std::complex<double>(4.0,5.0), x, y );
        y.Print( "y := (4+5i)H^T x ~= (4+5i)F^T x" );
        H.AdjointMultiply( std::complex<double>(4.0,5.0), x, y );
        y.Print( "y := (4+5i)H^H x ~= (4+5i)F^H x" );
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
