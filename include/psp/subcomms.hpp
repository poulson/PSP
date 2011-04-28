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
#ifndef PSP_COMM_TEAMS_HPP
#define PSP_COMM_TEAMS_HPP 1

#include "mpi.h"

namespace psp {

class Subcomms
{
private:
    std::vector<MPI_Comm> _subComms;
public:
    Subcomms( MPI_Comm comm );
    ~Subcomms();

    unsigned NumLevels() const;
    MPI_Comm Subcomm( unsigned level ) const;
};

} // namespace psp

//----------------------------------------------------------------------------//
// Inlined implementation                                                     //
//----------------------------------------------------------------------------//

namespace psp {

inline
Subcomms::Subcomms( MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("Subcomms::Subcomms");
#endif
    const int rank = mpi::CommRank( comm );
    const int p = mpi::CommSize( comm );
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");

    // Simple (yet slow) method for computing log2(p)
    unsigned numLevels = 1;
    unsigned teamSize = p;
    while( teamSize != 1 )
    {
        if( teamSize >= 4 )    
            teamSize >>= 2; 
        else // teamSize == 2
            teamSize = 1;
        ++numLevels;
    }

    _subComms.resize( numLevels );
    mpi::CommDup( comm, _subComms[0] );
    teamSize = p;
    for( unsigned i=1; i<numLevels; ++i )
    {
        if( teamSize >= 4 )
            teamSize >>= 2;
        else
            teamSize = 1;
        const int color = rank/teamSize;
        const int key = rank - color*teamSize;
        mpi::CommSplit( comm, color, key, _subComms[i] );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

inline
Subcomms::~Subcomms()
{
#ifndef RELEASE
    PushCallStack("Subcomms::~Subcomms");
#endif
    for( unsigned i=0; i<_subComms.size(); ++i )
        mpi::CommFree( _subComms[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline unsigned
Subcomms::NumLevels() const
{
    return _subComms.size();
}

// Return the single-process communicator when querying for levels deeper than
// the last.
inline MPI_Comm
Subcomms::Subcomm( unsigned level ) const
{
    return _subComms[std::min(level,(unsigned)_subComms.size()-1)];
}

} // namespace psp

#endif // PSP_COMM_TEAMS_HPP
