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
    std::cout << "MultiplyRandom <xSize> <ySize> <zSize> <numLevels> "
                 "<strongly admissible?> <maxRank> <print structure?>"
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    const int commRank = psp::mpi::CommRank( MPI_COMM_WORLD );
    const int commSize = psp::mpi::CommSize( MPI_COMM_WORLD );

    psp::UInt64 seed;
    seed.d[0] = 17U;
    seed.d[1] = 21U;
    psp::SeedParallelLcg( commRank, commSize, seed );

    if( argc < 8 )
    {
        if( commRank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const int xSize = atoi( argv[1] );
    const int ySize = atoi( argv[2] );
    const int zSize = atoi( argv[3] );
    const int numLevels = atoi( argv[4] );
    const bool stronglyAdmissible = atoi( argv[5] );
    const int maxRank = atoi( argv[6] );
    const bool printStructure = atoi( argv[7] );

    if( commRank == 0 )
    {
        std::cout << "----------------------------------------------------\n"
                  << "Testing performance of H-matrix/H-matrix mult on    \n"
                  << "random matrices\n"
                  << "----------------------------------------------------" 
                  << std::endl;
    }
    try
    {
        typedef std::complex<double> Scalar;
        typedef psp::DistQuasi2dHMat<Scalar,false> DistQuasi2d;

        // Set up two random distributed H-matrices
        if( commRank == 0 )
        {
            std::cout << "Creating random distributed H-matrices for "
                      <<  "performance testing...";
            std::cout.flush();
        }
        const double createStartTime = psp::mpi::WallTime();
        DistQuasi2d::Teams teams( MPI_COMM_WORLD );
        DistQuasi2d A
        ( numLevels, maxRank, stronglyAdmissible, xSize, ySize, zSize, teams );
        DistQuasi2d B
        ( numLevels, maxRank, stronglyAdmissible, xSize, ySize, zSize, teams );
        A.SetToRandom();
        B.SetToRandom();
        const double createStopTime = psp::mpi::WallTime();
        if( commRank == 0 )
        {
            std::cout << "done: " << createStopTime-createStartTime
                      << " seconds." << std::endl;
        }

        if( printStructure )
        {
            if( commRank == 0 )
            {
                std::cout 
                    << "numLevels: " << numLevels << "\n"
                    << "maxRank:   " << maxRank << "\n"
                    << "stronglyAdmissible: " << stronglyAdmissible << "\n"
                    << "xSize:              " << xSize << "\n"
                    << "ySize:              " << ySize << "\n"
                    << "zSize:              " << zSize << std::endl;
            }
            A.LatexWriteLocalStructure("A_structure");
            A.MScriptWriteLocalStructure("A_structure");
        }

        // Attempt to multiply the two matrices
        if( commRank == 0 )
        {
            std::cout << "Multiplying distributed H-matrices...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double multStartTime = psp::mpi::WallTime();
        DistQuasi2d C( teams );
        A.Multiply( (Scalar)1, B, C );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double multStopTime = psp::mpi::WallTime();
        if( commRank == 0 )
        {
            std::cout << "done: " << multStopTime-multStartTime
                      << " seconds." << std::endl;
        }
        if( printStructure )
        {
            C.LatexWriteLocalStructure("C_ghosted_structure");
            C.MScriptWriteLocalStructure("C_ghosted_structure");
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Process " << commRank << " caught message: " << e.what() 
                  << std::endl;
#ifndef RELEASE
        psp::DumpCallStack();
#endif
    }
    
    MPI_Finalize();
    return 0;
}

