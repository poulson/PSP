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
    MPI_Init( &argc, &argv );
    int rank = psp::mpi::CommRank( MPI_COMM_WORLD );
    int commSize = psp::mpi::CommSize( MPI_COMM_WORLD );

    // Print the first 3*commSize entries of the serial RNG and the first 
    // 3 entries from each process from the parallel RNG.
    try
    {
        psp::UInt64 seed = {{ 17U, 0U }};
        if( rank == 0 )
        {
            psp::SeedSerialLcg( seed );
            std::cout << "Serial values:" << std::endl;
            for( int i=0; i<3*commSize; ++i )
            {
                psp::UInt64 state = psp::SerialLcg();
                std::cout << state[0] << " " << state[1] << "\n";
            }
            std::cout << std::endl;
        }

        std::vector<psp::UInt32> myValues( 6 );
        std::vector<psp::UInt32> values( 6*commSize );
        psp::SeedParallelLcg( rank, commSize, seed );
        for( int i=0; i<3; ++i )
        {
            psp::UInt64 state = psp::ParallelLcg();
            myValues[2*i] = state[0];
            myValues[2*i+1] = state[1];
        }
        psp::mpi::AllGather( &myValues[0], 6, &values[0], 6, MPI_COMM_WORLD );
        if( rank == 0 )
        {
            std::cout << "Parallel values:" << std::endl;
            for( int i=0; i<3; ++i )
            {
                for( int j=0; j<commSize; ++j )
                {
                    const int k = i+3*j;
                    std::cout << values[2*k] << " " << values[2*k+1] << "\n";
                }
            }
            std::cout << std::endl;
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        psp::DumpCallStack();
#endif
    }

    MPI_Finalize();
    return 0;
}
