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
    std::cout << "Ghost <xSize> <ySize> <zSize> <numLevels> "
                 "<strongly admissible?> <maxRank> <print structure?>" 
              << std::endl;
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
    const int maxRank = atoi( argv[6] );
    const bool printStructure = atoi( argv[7] );

    if( rank == 0 )
    {
        std::cout << "----------------------------------------------------\n"
                  << "Testing formation of ghost nodes\n"
                  << "----------------------------------------------------" 
                  << std::endl;
    }
    try
    {
        typedef std::complex<double> Scalar;
        typedef psp::DistQuasi2dHMat<Scalar,false> DistQuasi2d;

        // Create a random distributed H-matrix
        DistQuasi2d::Teams teams( MPI_COMM_WORLD );
        DistQuasi2d H
        ( numLevels, maxRank, stronglyAdmissible, xSize, ySize, zSize, teams );
        H.SetToRandom();

        // Form the ghost nodes
        if( rank == 0 )
        {
            std::cout << "Forming ghost nodes...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double ghostStartTime = psp::mpi::WallTime();
        H.FormTargetGhostNodes();
        H.FormSourceGhostNodes();
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double ghostStopTime = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << ghostStopTime-ghostStartTime
                      << " seconds." << std::endl;
        }
        if( printStructure )
        {
            H.LatexWriteLocalStructure("H_ghosted_structure");
            H.MScriptWriteLocalStructure("H_ghosted_structure");
        }

        // Form the ghost nodes again
        if( rank == 0 )
        {
            std::cout << "Forming ghost nodes a second time...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double ghostStartTime2 = psp::mpi::WallTime();
        H.FormTargetGhostNodes();
        H.FormSourceGhostNodes();
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double ghostStopTime2 = psp::mpi::WallTime();
        if( rank == 0 )
        {
            std::cout << "done: " << ghostStopTime2-ghostStartTime2
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

