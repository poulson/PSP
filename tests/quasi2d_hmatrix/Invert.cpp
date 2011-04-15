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
    std::cout << "Invert <xSize> <ySize> <zSize> <numLevels> <r> <print?>" 
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    if( argc < 7 )
    {
        Usage();
        return 0;
    }
    const int xSize = atoi( argv[1] );
    const int ySize = atoi( argv[2] );
    const int zSize = atoi( argv[3] );
    const int numLevels = atoi( argv[4] );
    const int r = atoi( argv[5] );
    const bool print = atoi( argv[6] );

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

        std::vector<int> inverseMap( m );
        for( int i=0; i<m; ++i )
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
        if( print )
            S.Print( "S" );

        // Convert to H-matrix form
        std::cout << "Constructing H-matrix...";
        std::cout.flush();
        psp::Quasi2dHMatrix<double,false> 
            H( S, numLevels, r, true, xSize, ySize, zSize );
        std::cout << "done" << std::endl;
        if( print )
            H.Print( "H" );

        // Test against a vector of all 1's
        psp::Vector<double> x;
        x.Resize( m );
        double* xBuffer = x.Buffer();
        for( int i=0; i<m; ++i )
            xBuffer[i] = 1.0;
        if( print )
            x.Print( "x" );
        std::cout << "Multiplying H-matrix by a vector...";
        std::cout.flush();
        psp::Vector<double> y;
        H.MapVector( 1.0, x, y );
        std::cout << "done" << std::endl;
        if( print )
            y.Print( "y := H x ~= S x" );

        // Make a copy for inversion
        std::cout << "Making a copy of the H-matrix for inversion...";
        std::cout.flush();
        psp::Quasi2dHMatrix<double,false> invH;
        invH.CopyFrom( H );
        std::cout << "done" << std::endl;

        // Invert the copy
        std::cout << "Inverting the H-matrix...";
        std::cout.flush();
        invH.Invert();
        std::cout << "done" << std::endl;
        if( print )
            invH.Print( "inv(H)" );

        // Test for discrepancies in x and inv(H) H x
        std::cout << "Multiplying the inverse by a vector...";
        std::cout.flush();
        psp::Vector<double> z;
        invH.MapVector( 1.0, y, z );
        std::cout << "done" << std::endl;
        if( print )
        {
            y.Print( "y := H x ~= S x" );
            z.Print( "z := inv(H) H x ~= x" );
        }
        const double xNormL1 = m;
        const double xNormL2 = sqrt( m );
        const double xNormLInf = 1.0;
        double errorNormL1, errorNormL2, errorNormLInf;
        {
            errorNormL1 = 0;
            errorNormLInf = 0;

            double* zBuffer = z.Buffer();
            double errorNormL2Squared = 0;
            for( int i=0; i<m; ++i )
            {
                const double deviation = std::abs( zBuffer[i] - 1.0 );
                errorNormL1 += deviation;
                errorNormL2Squared += deviation*deviation;
                errorNormLInf = std::max( errorNormLInf, deviation );
            }
            errorNormL2 = std::sqrt( errorNormL2Squared );
        }
        std::cout << "||e||_1 =  " << errorNormL1 << "\n"
                  << "||e||_2 =  " << errorNormL2 << "\n"
                  << "||e||_oo = " << errorNormLInf << "\n"
                  << "\n"
                  << "||e||_1  / ||x||_1  = " << errorNormL1/xNormL1 << "\n"
                  << "||e||_2  / ||x||_2  = " << errorNormL2/xNormL2 << "\n"
                  << "||e||_oo / ||x||_oo = " << errorNormLInf/xNormLInf << "\n"
                  << std::endl;
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
