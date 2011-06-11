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
                 "<strongly admissible?> <r> <print?> <print structure?>" 
              << std::endl;
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

    if( argc < 9 )
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
    const bool printStructure = atoi( argv[8] );

    const int m = xSize*ySize*zSize;
    const int n = xSize*ySize*zSize;

    if( rank == 0 )
    {
        std::cout << "----------------------------------------------------\n"
                  << "Testing complex double Quasi2dHMat packing/unpacking\n"
                  << "into DistQuasi2dHMat                                \n"
                  << "----------------------------------------------------" 
                  << std::endl;
    }
    try
    {
        typedef std::complex<double> Scalar;
        typedef psp::Quasi2dHMat<Scalar,false> Quasi2d;
        typedef psp::DistQuasi2dHMat<Scalar,false> DistQuasi2d;

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
            std::cout << "Filling sparse matrices...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
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
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double fillStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << fillStopTime-fillStartTime << " seconds." 
                      << std::endl;
        }

        // Convert to H-matrix form
        if( rank == 0 )
        {
            std::cout << "Constructing H-matrices...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double constructStartTime = psp::mpi::WallTime();
        Quasi2d H( S, numLevels, r, stronglyAdmissible, xSize, ySize, zSize );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double constructStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << constructStopTime-constructStartTime 
                      << " seconds." << std::endl;
            if( print )
                H.Print("H");
            if( printStructure )
            {
                H.LatexWriteStructure("H_serial_structure");
                H.MScriptWriteStructure("H_serial_structure");
            }
        }

        // Invert the H-matrix
        if( rank == 0 )
        {
            std::cout << "Inverting H-matrices...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double invertStartTime = psp::mpi::WallTime();
        H.DirectInvert();
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double invertStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << invertStopTime-invertStartTime
                      << " seconds." << std::endl;
            if( print )
                H.Print("inv(H)");
            if( printStructure )
            {
                H.LatexWriteStructure("invH_serial_structure");
                H.MScriptWriteStructure("invH_serial_structure");
            }
        }

        // Store the result of a serial hmat-mat
        if( rank == 0 )
        {
            std::cout << "Y := H X...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double hmatMatStartTime = psp::mpi::WallTime();
        psp::Dense<Scalar> X( n, 30 );
        for( int j=0; j<X.Width(); ++j )
            for( int i=0; i<n; ++i )
                X.Set( i, j, i+j );
        psp::Dense<Scalar> Y;
        H.Multiply( (Scalar)1, X, Y );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double hmatMatStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << hmatMatStopTime-hmatMatStartTime 
                      << " seconds." << std::endl;
        }
        
        // Store the result of a serial hmat-trans-mat
        if( rank == 0 )
        {
            std::cout << "Z := H' X...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double hmatAdjointMatStartTime = psp::mpi::WallTime();
        psp::Dense<Scalar> Z;
        H.AdjointMultiply( (Scalar)1, X, Z );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double hmatAdjointMatStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " 
                      << hmatAdjointMatStopTime-hmatAdjointMatStartTime 
                      << " seconds." << std::endl;
        }

        // Set up our subcommunicators and compute the packed sizes
        DistQuasi2d::Teams teams( MPI_COMM_WORLD );
        std::vector<std::size_t> packedSizes;
        DistQuasi2d::PackedSizes( packedSizes, H, teams ); 
        const std::size_t myMaxSize = 
            *(std::max_element( packedSizes.begin(), packedSizes.end() ));

        // Pack for a DistQuasi2dHMat
        if( rank == 0 )
        {
            std::cout << "Packing H-matrix for distribution...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double packStartTime = psp::mpi::WallTime();
        std::vector<psp::byte> sendBuffer( p*myMaxSize );
        std::vector<psp::byte*> packedPieces( p );
        for( int i=0; i<p; ++i )
            packedPieces[i] = &sendBuffer[i*myMaxSize];
        DistQuasi2d::Pack( packedPieces, H, teams );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double packStopTime = psp::mpi::WallTime();
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
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double allToAllStartTime = psp::mpi::WallTime();
        std::vector<psp::byte> recvBuffer( p*intMaxSize );
        psp::mpi::AllToAll
        ( &sendBuffer[0], myIntMaxSize, &recvBuffer[0], intMaxSize,
          MPI_COMM_WORLD );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double allToAllStopTime = psp::mpi::WallTime();
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
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double unpackStartTime = psp::mpi::WallTime();
        DistQuasi2d distH( &recvBuffer[0], teams );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double unpackStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << unpackStopTime-unpackStartTime
                      << " seconds." << std::endl;
        }
        if( printStructure )
        {
            distH.LatexWriteLocalStructure("distH_structure");
            distH.MScriptWriteLocalStructure("distH_structure");
        }

        // Apply the distributed H-matrix
        if( rank == 0 )
        {
            std::cout << "Distributed Y := H X...";
            std::cout.flush();
        }
        psp::Dense<Scalar> XLocal;
        XLocal.LockedView
        ( X, distH.FirstLocalCol(), 0, distH.LocalWidth(), X.Width() );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double distHmatMatStartTime = psp::mpi::WallTime();
        psp::Dense<Scalar> YLocal;
        distH.Multiply( (Scalar)1, XLocal, YLocal );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double distHmatMatStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << distHmatMatStopTime-distHmatMatStartTime
                      << " seconds." << std::endl;
        }

        // Measure how close our result is to the serial results
        if( rank == 0 )
        {
            std::cout << "Comparing serial and distributed results...";
            std::cout.flush();
        }
        psp::Dense<Scalar> YLocalTruth;
        YLocalTruth.View
        ( Y, distH.FirstLocalRow(), 0, distH.LocalHeight(), X.Width() );
        for( int j=0; j<YLocal.Width(); ++j )
        {
            for( int i=0; i<YLocal.Height(); ++i )
            {
                if( std::abs(YLocalTruth.Get(i,j)-YLocal.Get(i,j)) > 1e-8 )
                {
                    std::ostringstream ss;
                    ss << "Answer differed at local index (" 
                        << i << "," << j << "), truth was "
                       << YLocalTruth.Get(i,j) << ", computed was "
                       << YLocal.Get(i,j) << std::endl;
                    throw std::logic_error( ss.str().c_str() );
                }
            }
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        if( rank == 0 )
            std::cout << "done" << std::endl;
        
        // Apply the adjoint of distributed H-matrix
        if( rank == 0 )
        {
            std::cout << "Distributed Z := H' X...";
            std::cout.flush();
        }
        XLocal.LockedView
        ( X, distH.FirstLocalRow(), 0, distH.LocalHeight(), X.Width() );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double distHmatAdjointMatStartTime = psp::mpi::WallTime();
        psp::Dense<Scalar> ZLocal;
        distH.AdjointMultiply( (Scalar)1, XLocal, ZLocal );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double distHmatAdjointMatStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " 
                      << distHmatAdjointMatStopTime-
                         distHmatAdjointMatStartTime
                      << " seconds." << std::endl;
        }

        // Measure how close our result is to the serial results
        if( rank == 0 )
        {
            std::cout << "Comparing serial and distributed results...";
            std::cout.flush();
        }
        psp::Dense<Scalar> ZLocalTruth;
        ZLocalTruth.View
        ( Z, distH.FirstLocalCol(), 0, distH.LocalWidth(), X.Width() );
        for( int j=0; j<ZLocal.Width(); ++j )
        {
            for( int i=0; i<ZLocal.Height(); ++i )
            {
                if( std::abs(ZLocalTruth.Get(i,j)-ZLocal.Get(i,j)) > 1e-8 )
                {
                    std::ostringstream ss;
                    ss << "Answer differed at local index (" 
                        << i << "," << j << "), truth was "
                       << ZLocalTruth.Get(i,j) << ", computed was "
                       << ZLocal.Get(i,j) << std::endl;
                    throw std::logic_error( ss.str().c_str() );
                }
            }
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        if( rank == 0 )
            std::cout << "done" << std::endl;
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

