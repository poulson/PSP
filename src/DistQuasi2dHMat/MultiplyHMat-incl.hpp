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

#include "./MultiplyHMatFormGhostRanks-incl.hpp"
#include "./MultiplyHMatMain-incl.hpp"
#include "./MultiplyHMatFHH-incl.hpp"
#include "./MultiplyHMatUpdates-incl.hpp"

// C := alpha A B
template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::Multiply
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Multiply");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    A.MultiplyHMatFullAccumulate( alpha, B, C );
    //A.MultiplyHMatSingleLevelAccumulate( alpha, B, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatFullAccumulate
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatFullAccumulate");
    if( Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( _numLevels != B._numLevels )
        throw std::logic_error("H-matrices must have same number of levels");
    if( _zSize != B._zSize )
        throw std::logic_error("Mismatched z sizes");
    if( _level != B._level )
        throw std::logic_error("Mismatched levels");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    A.RequireRoot();
    A.PruneGhostNodes();
    B.PruneGhostNodes();
    C.Clear();

    A.FormTargetGhostNodes();
    B.FormSourceGhostNodes();
    A.MultiplyHMatFormGhostRanks( B );

    const int startLevel = 0;
    const int endLevel = A.NumLevels();
    A.MultiplyHMatMainPrecompute( alpha, B, C, startLevel, endLevel );
    A.MultiplyHMatMainSums( B, C, startLevel, endLevel );
    A.MultiplyHMatMainPassData( alpha, B, C, startLevel, endLevel );

    A.MultiplyHMatMainBroadcasts( B, C, startLevel, endLevel );
    A.MultiplyHMatMainPostcompute( alpha, B, C, startLevel, endLevel );
    A.MultiplyHMatFHHPrecompute( alpha, B, C, startLevel, endLevel );
    A.MultiplyHMatFHHSums( alpha, B, C, startLevel, endLevel );
    A.MultiplyHMatFHHPassData( alpha, B, C, startLevel, endLevel );
    A.MultiplyHMatFHHBroadcasts( alpha, B, C, startLevel, endLevel );
    A.MultiplyHMatFHHPostcompute( alpha, B, C, startLevel, endLevel );
    A.MultiplyHMatFHHFinalize( B, C, startLevel, endLevel );

    C.MultiplyHMatUpdates();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MultiplyHMatSingleLevelAccumulate
( Scalar alpha, DistQuasi2dHMat<Scalar,Conjugated>& B,
                DistQuasi2dHMat<Scalar,Conjugated>& C )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyHMatSingleLevelAccumulate");
    if( Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( _numLevels != B._numLevels )
        throw std::logic_error("H-matrices must have same number of levels");
    if( _zSize != B._zSize )
        throw std::logic_error("Mismatched z sizes");
    if( _level != B._level )
        throw std::logic_error("Mismatched levels");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;
    A.RequireRoot();
    A.PruneGhostNodes();
    B.PruneGhostNodes();
    C.Clear();

    A.FormTargetGhostNodes();
    B.FormSourceGhostNodes();
    A.MultiplyHMatFormGhostRanks( B );

    const int numLevels = A.NumLevels();
    for( int level=0; level<numLevels; ++level )
    {
        const int startLevel = level;
        const int endLevel = level+1;

        A.MultiplyHMatMainPrecompute( alpha, B, C, startLevel, endLevel );
        A.MultiplyHMatMainSums( B, C, startLevel, endLevel );
        A.MultiplyHMatMainPassData( alpha, B, C, startLevel, endLevel );

        A.MultiplyHMatMainBroadcasts( B, C, startLevel, endLevel );
        A.MultiplyHMatMainPostcompute( alpha, B, C, startLevel, endLevel );
        A.MultiplyHMatFHHPrecompute( alpha, B, C, startLevel, endLevel );
        A.MultiplyHMatFHHSums( alpha, B, C, startLevel, endLevel );
        A.MultiplyHMatFHHPassData( alpha, B, C, startLevel, endLevel );
        A.MultiplyHMatFHHBroadcasts( alpha, B, C, startLevel, endLevel );
        A.MultiplyHMatFHHPostcompute( alpha, B, C, startLevel, endLevel );
        A.MultiplyHMatFHHFinalize( B, C, startLevel, endLevel );

        C.MultiplyHMatUpdates();
    }
#ifndef RELEASE
    PopCallStack();
#endif
}


