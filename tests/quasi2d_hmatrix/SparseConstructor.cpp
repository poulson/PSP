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

    std::cout << "-----------------------------------------------------\n"
              << "Converting double-precision sparse to Quasi2dHMatrix \n"
              << "-----------------------------------------------------" 
              << std::endl;
    try
    {
        psp::SparseMatrix<double> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;

        for( int i=0; i<m; ++i )
        {
            S.rowOffsets.push_back( S.nonzeros.size() );

            if( i >= xSize )
            {
                S.nonzeros.push_back( S.nonzeros.size()+1 );
                S.columnIndices.push_back( i-xSize );
            }

            if( i >= 1 )
            {
                S.nonzeros.push_back( S.nonzeros.size()+1 );
                S.columnIndices.push_back( i-1 );
            }

            S.nonzeros.push_back( S.nonzeros.size()+1 );    
            S.columnIndices.push_back( i );

            if( i+1 < n )
            {
                S.nonzeros.push_back( S.nonzeros.size()+1 );
                S.columnIndices.push_back( i+1 );
            }

            if( i+xSize < n )
            {
                S.nonzeros.push_back( S.nonzeros.size()+1 );
                S.columnIndices.push_back( i+xSize );
            }
        }
        S.rowOffsets.push_back( S.nonzeros.size() );
        S.Print( "S" );

        psp::Quasi2dHMatrix<double,false> 
            H( S, 2, r, false, xSize, ySize, zSize );

        psp::Vector<double> x( n );
        double* xBuffer = x.Buffer();
        for( int i=0; i<n; ++i )
            xBuffer[i] = 1.0;
        x.Print( "x" );

        psp::Vector<double> y;
        H.MapVector( 2.0, x, y );
        y.Print( "y := 2 H x ~= 2 S x" );
        H.TransposeMapVector( 2.0, x, y );
        y.Print( "y := 2 H^T x ~= 2 S^T x" );
        H.HermitianTransposeMapVector( 2.0, x, y );
        y.Print( "y := 2 H^H x ~= 2 S^H x" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
    }
    
    std::cout << "-------------------------------------------------------------\n"
              << "Converting complex double-precision sparse to Quasi2dHMatrix \n"
              << "-------------------------------------------------------------" 
              << std::endl;
    try
    {
        psp::SparseMatrix< std::complex<double> > S;
        S.height = m;
        S.width = n;
        S.symmetric = false;

        for( int i=0; i<m; ++i )
        {
            S.rowOffsets.push_back( S.nonzeros.size() );

            if( i >= xSize )
            {
                S.nonzeros.push_back( std::complex<double>(i,S.nonzeros.size()+1) );
                S.columnIndices.push_back( i-xSize );
            }

            if( i >= 1 )
            {
                S.nonzeros.push_back( std::complex<double>(i,S.nonzeros.size()+1) );
                S.columnIndices.push_back( i-1 );
            }

            S.nonzeros.push_back( std::complex<double>(i,S.nonzeros.size()+1) );
            S.columnIndices.push_back( i );

            if( i+1 < n )
            {
                S.nonzeros.push_back( std::complex<double>(i,S.nonzeros.size()+1) );
                S.columnIndices.push_back( i+1 );
            }

            if( i+xSize < n )
            {
                S.nonzeros.push_back( std::complex<double>(i,S.nonzeros.size()+1) );
                S.columnIndices.push_back( i+xSize );
            }
        }
        S.rowOffsets.push_back( S.nonzeros.size() );
        S.Print( "S" );

        psp::Quasi2dHMatrix<std::complex<double>,false> 
            H( S, 2, r, false, xSize, ySize, zSize );

        psp::Vector< std::complex<double> > x( n );
        std::complex<double>* xBuffer = x.Buffer();
        for( int i=0; i<n; ++i )
            xBuffer[i] = std::complex<double>(1.0,3.0);
        x.Print( "x" );

        psp::Vector< std::complex<double> > y;
        H.MapVector( std::complex<double>(4.0,5.0), x, y );
        y.Print( "y := (4+5i)H x ~= (4+5i)S x" );
        H.TransposeMapVector( std::complex<double>(4.0,5.0), x, y );
        y.Print( "y := (4+5i)H^T x ~= (4+5i)S^T x" );
        H.HermitianTransposeMapVector( std::complex<double>(4.0,5.0), x, y );
        y.Print( "y := (4+5i)H^H x ~= (4+5i)S^H x" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
    }

    return 0;
}
