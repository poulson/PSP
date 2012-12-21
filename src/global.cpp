/*
   Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and 
   The University of Texas at Austin
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
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
