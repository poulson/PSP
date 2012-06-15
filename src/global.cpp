/*
   Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
   of a sweeping preconditioner for 3d Helmholtz equations.

   Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and
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

namespace { 
bool pspInitializedClique; 
bool initializedPsp = false;
#ifndef RELEASE
std::stack<std::string> callStack;
#endif
}

namespace psp {

bool Initialized()
{ return ::initializedPsp; }

void Initialize( int& argc, char**& argv )
{
    // If PSP has already been initialized, this is a no-op
    if( ::initializedPsp )
        return;

    const bool mustInitClique = !cliq::Initialized();
    if( mustInitClique )
    {
        cliq::Initialize( argc, argv );
        ::pspInitializedClique = true;
    }
    else
    {
        ::pspInitializedClique = false;
    }
    ::initializedPsp = true;
}

void Finalize()
{
    // If PSP is not currently initialized, then this is a no-op
    if( !::initializedPsp )
        return;
    
    if( ::pspInitializedClique )
        cliq::Finalize();

    ::initializedPsp = false;
}

#ifndef RELEASE
void PushCallStack( std::string s )
{ ::callStack.push( s ); }

void PopCallStack()
{ ::callStack.pop(); }

void DumpCallStack()
{
    std::ostringstream msg;
    while( !::callStack.empty() )
    {
        msg << "[" << ::callStack.size() << "]: " << ::callStack.top() << "\n";
        ::callStack.pop();
    }
    std::cerr << msg.str() << std::endl;
}
#endif // ifndef RELEASE

} // namespace psp
