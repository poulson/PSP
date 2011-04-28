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
#include <algorithm>

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
    const int p = psp::mpi::CommSize( MPI_COMM_WORLD );

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
        std::cout << "-------------------------------------------------------\n"
                  << "Testing complex double Quasi2dHMatrix packing/unpacking\n"
                  << "into DistQuasi2dHMatrix                                \n"
                  << "-------------------------------------------------------" 
                  << std::endl;
    }
    try
    {
        typedef std::complex<double> Scalar;
        typedef psp::Quasi2dHMatrix<Scalar,false> Quasi2d;
        typedef psp::DistQuasi2dHMatrix<Scalar,false> DistQuasi2d;

        psp::SparseMatrix<Scalar> S;
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
        double fillStartTime = MPI_Wtime();
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
        double fillStopTime = MPI_Wtime();
        if( rank == 0 )
        {
            std::cout << "done: " << fillStopTime-fillStartTime << " seconds." 
                      << std::endl;
        }

        // Convert to H-matrix form
        if( rank == 0 )
        {
            std::cout << "Constructing H-matrix...";
            std::cout.flush();
        }
        double constructStartTime = MPI_Wtime();
        Quasi2d H( S, numLevels, r, stronglyAdmissible, xSize, ySize, zSize );
        double constructStopTime = MPI_Wtime();
        if( rank == 0 )
        {
            std::cout << "done: " << constructStopTime-constructStartTime 
                      << " seconds." << std::endl;
            if( print )
                H.Print("H");
        }

        // Set up our subcommunicators and compute the packed sizes
        psp::Subcomms subcomms( MPI_COMM_WORLD );
        std::vector<std::size_t> packedSizes;
        DistQuasi2d::PackedSizes( packedSizes, H, subcomms ); 
        const std::size_t myMaxSize = 
            *(std::max_element( packedSizes.begin(), packedSizes.end() ));

        // Pack for a DistQuasi2dHMatrix
        if( rank == 0 )
        {
            std::cout << "Packing H-matrix for distribution...";
            std::cout.flush();
        }
        double packStartTime = MPI_Wtime();
        std::vector<psp::byte> sendBuffer( p*myMaxSize );
        std::vector<psp::byte*> packedPieces( p );
        for( int i=0; i<p; ++i )
            packedPieces[i] = &sendBuffer[i*myMaxSize];
        DistQuasi2d::Pack( packedPieces, H, subcomms );
        double packStopTime = MPI_Wtime();
        if( rank == 0 )
        {
            std::cout << "done: " << packStopTime-packStartTime << " seconds."
                      << std::endl;
        }

        // Compute the maximum package size
        int myIntMaxSize, intMaxSize;
        {
            myIntMaxSize = myMaxSize;
            psp::mpi::AllReduce
            ( &myIntMaxSize, &intMaxSize, 1, MPI_MAX, MPI_COMM_WORLD );
        }
        if( rank == 0 )
        {
            std::cout << "Maximum per-process message size: " 
                      << ((double)intMaxSize)/(1024.*1024.) << " MB." 
                      << std::endl;
        }
 
        // AllToAll
        if( rank == 0 )
        {
            std::cout << "AllToAll redistribution...";
            std::cout.flush();
        }
        double allToAllStartTime = MPI_Wtime();
        std::vector<psp::byte> recvBuffer( p*intMaxSize );
        psp::mpi::AllToAll
        ( &sendBuffer[0], myIntMaxSize, &recvBuffer[0], intMaxSize,
          MPI_COMM_WORLD );
        double allToAllStopTime = MPI_Wtime();
        MPI_Barrier( MPI_COMM_WORLD );
        if( rank == 0 )
        {
            std::cout << "done: " << allToAllStopTime-allToAllStartTime
                      << " seconds." << std::endl;
        }

        // Unpack our part of the matrix defined by process 0
        if( rank == 0 )
        {
            std::cout << "Unpacking...";
            std::cout.flush();
        }
        double unpackStartTime = MPI_Wtime();
        DistQuasi2d distH( &recvBuffer[0], subcomms );
        double unpackStopTime = MPI_Wtime();
        if( rank == 0 )
        {
            std::cout << "done: " << unpackStopTime-unpackStartTime
                      << " seconds." << std::endl;
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
