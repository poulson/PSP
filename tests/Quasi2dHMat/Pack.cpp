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
    std::cout << "Pack <xSize> <ySize> <zSize> <numLevels> "
                 "<strongly admissible?> <r> <print?>" << std::endl;
}

template<typename Real>
void
FormRow
( int x, int y, int z, int xSize, int ySize, int zSize, 
  std::vector< std::complex<Real> >& row, std::vector<int>& colIndices )
{
    typedef std::complex<Real> Scalar;
    const int rowIdx = x + xSize*y + xSize*ySize*z;

    row.resize( 0 );
    colIndices.resize( 0 );

    // Set up the diagonal entry
    colIndices.push_back( rowIdx );
    row.push_back( (Scalar)8 );

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        colIndices.push_back( (x-1) + xSize*y + xSize*ySize*z );
        row.push_back( (Scalar)-1 );
    }

    // Back connection to (x+1,y,z)
    if( x != xSize-1 )
    {
        colIndices.push_back( (x+1) + xSize*y + xSize*ySize*z );
        row.push_back( (Scalar)-1 );
    }

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        colIndices.push_back( x + xSize*(y-1) + xSize*ySize*z );
        row.push_back( (Scalar)-1 );
    }

    // Right connection to (x,y+1,z)
    if( y != ySize-1 )
    {
        colIndices.push_back( x + xSize*(y+1) + xSize*ySize*z );
        row.push_back( (Scalar)-1 );
    }

    // Top connection to (x,y,z-1)
    if( z != 0 )
    {
        colIndices.push_back( x + xSize*y + xSize*ySize*(z-1) );
        row.push_back( (Scalar)-1 );
    }

    // Bottom connection to (x,y,z+1)
    if( z != zSize-1 )
    {
        colIndices.push_back( x + xSize*y + xSize*ySize*(z+1) );
        row.push_back( (Scalar)-1 );
    }
}

int
main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    const int rank = psp::mpi::CommRank( MPI_COMM_WORLD );

    if( argc < 8 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const int xSize = atoi( argv[1] );
    const int ySize = atoi( argv[2] );
    const int zSize = atoi( argv[3] );
    const int numLevels = atoi( argv[4] );
    const bool stronglyAdmissible = atoi( argv[5] );
    const int r = atoi( argv[6] );
    const bool print = atoi( argv[7] );

    const int m = xSize*ySize*zSize;
    const int n = xSize*ySize*zSize;

    if( rank == 0 )
    {
        std::cout << "----------------------------------------------------\n"
                  << "Testing complex double Quasi2dHMat packing/unpacking\n"
                  << "----------------------------------------------------" 
                  << std::endl;
    }
    try
    {
        typedef std::complex<double> Scalar;
        typedef psp::Quasi2dHMat<Scalar,false> Quasi2d;

        psp::Sparse<Scalar> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;

        std::vector<int> map;
        Quasi2d::BuildNaturalToHierarchicalMap
        ( map, xSize, ySize, zSize, numLevels );

        std::vector<int> inverseMap( m );
        for( int i=0; i<m; ++i )
            inverseMap[map[i]] = i;

        if( rank == 0 )
        {
            std::cout << "Filling sparse matrix...";
            std::cout.flush();
        }
        double fillStartTime = psp::mpi::WallTime();
        std::vector<Scalar> row;
        std::vector<int> colIndices;
        for( int i=0; i<m; ++i )
        {
            S.rowOffsets.push_back( S.nonzeros.size() );
            const int iNatural = inverseMap[i];
            const int x = iNatural % xSize;
            const int y = (iNatural/xSize) % ySize;
            const int z = iNatural/(xSize*ySize);

            FormRow
            ( x, y, z, xSize, ySize, zSize, row, colIndices );

            for( unsigned j=0; j<row.size(); ++j )
            {
                S.nonzeros.push_back( row[j] );
                S.columnIndices.push_back( map[colIndices[j]] );
            }
        }
        S.rowOffsets.push_back( S.nonzeros.size() );
        double fillStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << fillStopTime-fillStartTime << " seconds." 
                      << std::endl;
            if( print )
                S.Print("S");
        }

        // Convert to H-matrix form
        if( rank == 0 )
        {
            std::cout << "Constructing H-matrix...";
            std::cout.flush();
        }
        double constructStartTime = psp::mpi::WallTime();
        Quasi2d H( S, numLevels, r, stronglyAdmissible, xSize, ySize, zSize );
        double constructStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << constructStopTime-constructStartTime 
                      << " seconds." << std::endl;
            if( print )
                H.Print("H");
        }

        // Test against a vector of all 1's
        psp::Vector<Scalar> x;
        x.Resize( m );
        Scalar* xBuffer = x.Buffer();
        for( int i=0; i<m; ++i )
            xBuffer[i] = 1.0;
        if( rank == 0 )
        {
            std::cout << "Multiplying H-matrix by a vector of all ones...";
            std::cout.flush();
        }
        psp::Vector<Scalar> y;
        double matVecStartTime = psp::mpi::WallTime();
        H.Multiply( 1.0, x, y );
        double matVecStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << matVecStopTime-matVecStartTime 
                      << " seconds." << std::endl;
            if( print )
                y.Print("y := H x ~= S x");
        }

        // Pack the H-matrix
        std::vector<psp::byte> packedHMat;
        if( rank == 0 )
        {
            std::cout << "Packing H-matrix...";
            std::cout.flush();
        }
        double packStartTime = psp::mpi::WallTime();
        H.Pack( packedHMat );
        double packStopTime = psp::mpi::WallTime();
        double sizeInMB = ((double)packedHMat.size())/(1024.*1024.);
        if( rank == 0 )
        {
            std::cout << "done: " << packStopTime-packStartTime << " seconds.\n"
                      << "Packed size: " << sizeInMB << " MB." << std::endl;
        }

        // Unpack the H-matrix
        if( rank == 0 )
        {
            std::cout << "Unpacking H-matrix...";
            std::cout.flush();
        }
        double unpackStartTime = psp::mpi::WallTime();
        Quasi2d HCopy( packedHMat );
        double unpackStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << unpackStopTime-unpackStartTime 
                      << " seconds." << std::endl;
            if( print )
                HCopy.Print("Unpacked copy of H-matrix");
        }

        // Check that the copied H-matrix has the same action on our vector of 
        // all 1's
        psp::Vector<Scalar> z;
        if( rank == 0 )
        {
            std::cout << "Multiplying H-matrix by a vector...";
            std::cout.flush();
        }
        matVecStartTime = psp::mpi::WallTime();
        HCopy.Multiply( 1.0, x, z );
        matVecStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << matVecStopTime-matVecStartTime 
                      << " seconds." << std::endl;
            if( print )
                z.Print("z := HCopy x ~= S x");
            for( int i=0; i<z.Height(); ++i )
            {
                if( z.Get(i) != y.Get(i) )
                {
                    std::ostringstream s;
                    s << "Action of copied H-matrix differed at index " << i;
                    throw std::logic_error( s.str().c_str() );
                }
            }
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Process " << rank << " caught message: " << e.what() 
                  << std::endl;
#ifndef RELEASE
        psp::DumpCallStack();
#endif
    }
    
    MPI_Finalize();
    return 0;
}
