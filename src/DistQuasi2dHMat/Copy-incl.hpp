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

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::CopyFrom
( const DistQuasi2dHMat<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::CopyFrom");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    A._numLevels = B._numLevels;
    A._maxRank = B._maxRank;
    A._targetOffset = B._targetOffset;
    A._sourceOffset = B._sourceOffset;
    A._stronglyAdmissible = B._stronglyAdmissible;

    A._xSizeTarget = B._xSizeTarget;
    A._ySizeTarget = B._ySizeTarget;
    A._xSizeSource = B._xSizeSource;
    A._ySizeSource = B._ySizeSource;
    A._zSize = B._zSize;

    A._xTarget = B._xTarget;
    A._yTarget = B._yTarget;
    A._xSource = B._xSource;
    A._ySource = B._ySource;

    A._teams = B._teams;
    A._level = B._level;
    A._inTargetTeam = B._inTargetTeam;
    A._inSourceTeam = B._inSourceTeam;
    A._targetRoot = B._targetRoot;
    A._sourceRoot = B._sourceRoot;

    A._block.Clear();
    A._block.type = B._block.type;

    switch( B._block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        A._block.data.N = A.NewNode();    
        Node& nodeA = *A._block.data.N;
        const Node& nodeB = *B._block.data.N;
        for( int j=0; j<16; ++j )
            nodeA.children[j] = new DistQuasi2dHMat<Scalar,Conjugated>;

        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeA.Child(t,s).CopyFrom( nodeB.Child(t,s) );
        break;
    }
    case DIST_LOW_RANK:
    {
        DistLowRank& DFA = *A._block.data.DF;
        const DistLowRank& DFB = *B._block.data.DF;

        DFA.rank = DFB.rank;
        if( B._inTargetTeam )
            hmat_tools::Copy( DFB.ULocal, DFA.ULocal );
        if( B._inSourceTeam )
            hmat_tools::Copy( DFB.VLocal, DFA.VLocal );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SFA = *A._block.data.SF;
        const SplitLowRank& SFB = *B._block.data.SF;

        SFA.rank = SFB.rank;
        hmat_tools::Copy( SFB.D, SFA.D );
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& FA = *A._block.data.F;
        const LowRank<Scalar,Conjugated>& FB = *B._block.data.F;

        hmat_tools::Copy( FB, FA );
        break;
    }
    case SPLIT_DENSE:
        if( B._inSourceTeam )
            hmat_tools::Copy( B._block.data.SD->D, A._block.data.SD->D );
        break;
    case DENSE:
        hmat_tools::Copy( *B._block.data.D, *A._block.data.D );
        break;
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

