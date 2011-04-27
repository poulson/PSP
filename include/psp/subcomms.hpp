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
    unsigned log2p = 0;
    unsigned temp = p;
    while( temp >>= 1 )
        ++log2p;

    _subComms.resize( log2p+1 );
    mpi::CommDup( comm, _subComms[0] );
    for( unsigned i=1; i<=log2p; ++i )
    {
        const int color = rank/(p>>i);
        const int key = rank - color;
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

inline MPI_Comm
Subcomms::Subcomm( unsigned level ) const
{
    return _subComms[level];
}

} // namespace psp

#endif // PSP_COMM_TEAMS_HPP
