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
#include <iostream>

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
              << "Converting double-precision sparse to dense         \n"
              << "----------------------------------------------------" 
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

        psp::DenseMatrix<double> D;
        psp::hmatrix_tools::ConvertSubmatrix( D, S, 0, m, 0, n );
        D.Print( "D" );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
    }

    return 0;
}
