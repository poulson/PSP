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
psp::DistQuasi2dHMat<Scalar,Conjugated>::SetToRandom()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::SetToRandom");
#endif
    const int maxRank = MaxRank();
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                _block.data.N->Child(t,s).SetToRandom();
        break;
    case DIST_LOW_RANK:
    {
        DistLowRank& DF = *_block.data.DF;
        const int localHeight = DF.ULocal.Height();
        const int localWidth = DF.VLocal.Height();

        DF.rank = maxRank;
        DF.ULocal.Resize( localHeight, maxRank );
        DF.VLocal.Resize( localWidth, maxRank );
        ParallelGaussianRandomVectors( DF.ULocal );
        ParallelGaussianRandomVectors( DF.VLocal );
        break;
    }
    case SPLIT_LOW_RANK:
    {
        SplitLowRank& SF = *_block.data.SF;
        const int length = SF.D.Height();

        SF.rank = maxRank;
        SF.D.Resize( length, maxRank );
        ParallelGaussianRandomVectors( SF.D );
        break;
    }
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;
        const int height = F.U.Height();
        const int width = F.V.Height();

        F.U.Resize( height, maxRank );
        F.V.Resize( width, maxRank );
        ParallelGaussianRandomVectors( F.U );
        ParallelGaussianRandomVectors( F.V );
        break;
    }
    case DIST_LOW_RANK_GHOST:
        _block.data.DFG->rank = maxRank;
       break;
    case SPLIT_LOW_RANK_GHOST:
        _block.data.SFG->rank = maxRank;
        break;
    case LOW_RANK_GHOST:
        _block.data.FG->rank = maxRank;
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
            ParallelGaussianRandomVectors( _block.data.SD->D );
        break;
    case DENSE:
        ParallelGaussianRandomVectors( *_block.data.D );
        break;
    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

