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
    const int commSize = psp::mpi::CommSize( MPI_COMM_WORLD );
    const int commRank = psp::mpi::CommRank( MPI_COMM_WORLD );
    if( commSize != 1 )
    {
        if( commRank == 0 )
            std::cerr << "This test should be run with a single MPI process" 
                      << std::endl;
        MPI_Finalize();
        return 0;
    }

    try
    {
        std::cout << "Running timing experiment, please wait a few seconds..."
                  << std::endl;
        psp::Timer timer;
        timer.Start( 0 );
        sleep( 1 );
        timer.Start( 1 );
        sleep( 1 );
        timer.Start( 2 );
        sleep( 1 );
        timer.Start( 3 );
        sleep( 1 );
        timer.Stop( 3 );
        timer.Stop( 2 );
        timer.Stop( 1 );
        timer.Stop( 0 );

        std::cout << "Timer 0: " << timer.GetTime( 0 ) << " seconds\n"
                  << "Timer 1: " << timer.GetTime( 1 ) << " seconds\n"
                  << "Timer 2: " << timer.GetTime( 2 ) << " seconds\n"
                  << "Timer 3: " << timer.GetTime( 3 ) << " seconds\n"
                  << std::endl;

        std::cout << "Repeating experiment without resetting." << std::endl;

        timer.Start( 0 );
        sleep( 1 );
        timer.Start( 1 );
        sleep( 1 );
        timer.Start( 2 );
        sleep( 1 );
        timer.Start( 3 );
        sleep( 1 );
        timer.Stop( 3 );
        timer.Stop( 2 );
        timer.Stop( 1 );
        timer.Stop( 0 );

        std::cout << "Timer 0: " << timer.GetTime( 0 ) << " seconds\n"
                  << "Timer 1: " << timer.GetTime( 1 ) << " seconds\n"
                  << "Timer 2: " << timer.GetTime( 2 ) << " seconds\n"
                  << "Timer 3: " << timer.GetTime( 3 ) << " seconds\n"
                  << std::endl;
        
        std::cout << "Repeating experiment after clearing timer 0." 
                  << std::endl;
        timer.Clear( 0 );

        timer.Start( 0 );
        sleep( 1 );
        timer.Start( 1 );
        sleep( 1 );
        timer.Start( 2 );
        sleep( 1 );
        timer.Start( 3 );
        sleep( 1 );
        timer.Stop( 3 );
        timer.Stop( 2 );
        timer.Stop( 1 );
        timer.Stop( 0 );

        std::cout << "Timer 0: " << timer.GetTime( 0 ) << " seconds\n"
                  << "Timer 1: " << timer.GetTime( 1 ) << " seconds\n"
                  << "Timer 2: " << timer.GetTime( 2 ) << " seconds\n"
                  << "Timer 3: " << timer.GetTime( 3 ) << " seconds\n"
                  << std::endl;

        std::cout << "Repeating experiment after clearing all timers." 
                  << std::endl;
        timer.Clear();

        timer.Start( 0 );
        sleep( 1 );
        timer.Start( 1 );
        sleep( 1 );
        timer.Start( 2 );
        sleep( 1 );
        timer.Start( 3 );
        sleep( 1 );
        timer.Stop( 3 );
        timer.Stop( 2 );
        timer.Stop( 1 );
        timer.Stop( 0 );

        std::cout << "Timer 0: " << timer.GetTime( 0 ) << " seconds\n"
                  << "Timer 1: " << timer.GetTime( 1 ) << " seconds\n"
                  << "Timer 2: " << timer.GetTime( 2 ) << " seconds\n"
                  << "Timer 3: " << timer.GetTime( 3 ) << " seconds\n"
                  << std::endl;
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
