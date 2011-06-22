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
    try
    {
        psp::MemoryMap<int,psp::Dense<double> > memoryMap;

        memoryMap[17] = new psp::Dense<double>( 1, 1 );
        memoryMap[11] = new psp::Dense<double>( 2, 2 );
        memoryMap[23] = new psp::Dense<double>( 3, 3 );
        memoryMap[7]  = new psp::Dense<double>( 4, 4 );

        int numEntries = memoryMap.Size();
        std::cout << "size of memory map: " << numEntries << std::endl;
        memoryMap.ResetIterator();
        for( int entry=0; entry<numEntries; ++entry )
        {
            const psp::Dense<double>& D = *memoryMap.NextEntry(); 
            std::cout << "Index " << memoryMap.CurrentIndex() << ": " 
                      << D.Height() << " x " << D.Width() << "\n"; 
        }
        std::cout << std::endl;

        // Loop over again, and delete the third entry. At the end of each
        // loop, print the total size
        memoryMap.ResetIterator();
        for( int entry=0; entry<numEntries; ++entry )
        {
            const psp::Dense<double>& D = *memoryMap.NextEntry();
            if( entry == 2 )
            {
                memoryMap.EraseLastEntry();
                std::cout << "Erased third entry, new size is: " 
                          << memoryMap.Size() << "\n";
            }
            std::cout << "Index " << memoryMap.CurrentIndex() << ": "
                      << D.Height() << " x " << D.Width() << "\n";
        }
        std::cout << std::endl;
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
