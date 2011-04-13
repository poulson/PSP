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
    std::cout << "SpeedTest <xSize> <ySize> <zSize> <numLevels> <r>" 
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    if( argc < 6 )
    {
        Usage();
        return 0;
    }
    const int xSize = atoi( argv[1] );
    const int ySize = atoi( argv[2] );
    const int zSize = atoi( argv[3] );
    const int numLevels = atoi( argv[4] );
    const int r = atoi( argv[5] );

    const int m = xSize*ySize*zSize;
    const int n = xSize*ySize*zSize;

    std::cout << "--------------------------------------------------\n"
              << "Testing double-precision sparse to Quasi2dHMatrix \n"
              << "--------------------------------------------------" 
              << std::endl;
    try
    {
        psp::SparseMatrix<double> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;

        std::vector<int> map;
        psp::Quasi2dHMatrix<double,false>::BuildNaturalToHierarchicalMap
        ( map, xSize, ySize, zSize, numLevels );

        std::vector<int> inverseMap( map.size() );
        for( int i=0; i<map.size(); ++i )
            inverseMap[map[i]] = i;

        std::cout << "Filling sparse matrix...";
        std::cout.flush();
        int value = 1;
        for( int i=0; i<m; ++i )
        {
            S.rowOffsets.push_back( S.nonzeros.size() );

            const int iNatural = inverseMap[i];

            if( iNatural >= xSize*ySize )
            {
                S.nonzeros.push_back( value++ );
                S.columnIndices.push_back( map[iNatural-xSize*ySize] );
            }

            if( (iNatural % (xSize*ySize)) >= xSize )
            {
                S.nonzeros.push_back( value++ );
                S.columnIndices.push_back( map[iNatural-xSize] );
            }

            if( (iNatural % xSize) >= 1 )
            {
                S.nonzeros.push_back( value++ );
                S.columnIndices.push_back( map[iNatural-1] );
            }

            S.nonzeros.push_back( value++ );    
            S.columnIndices.push_back( i );

            if( (iNatural % xSize) != xSize-1 )
            {
                S.nonzeros.push_back( value++ );
                S.columnIndices.push_back( map[iNatural+1] );
            }

            if( (iNatural % (xSize*ySize))/xSize != ySize-1 )
            {
                S.nonzeros.push_back( value++ );
                S.columnIndices.push_back( map[iNatural+xSize] );
            }

            if( iNatural < xSize*ySize*(zSize-1) )
            {
                S.nonzeros.push_back( value++ );
                S.columnIndices.push_back( map[iNatural+xSize*ySize] );
            }

            // Keep the entries from getting too big
            value = (value % 10000) + 1;
        }
        S.rowOffsets.push_back( S.nonzeros.size() );
        std::cout << "done." << std::endl;

        std::cout << "Constructing H-matrix...";
        std::cout.flush();
        psp::Quasi2dHMatrix<double,false> 
            H( S, numLevels, r, false, xSize, ySize, zSize );
        std::cout << "done" << std::endl;

        std::cout << "Inverting the H-matrix...";
        std::cout.flush();
        H.Invert();
        std::cout << "done" << std::endl;
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
