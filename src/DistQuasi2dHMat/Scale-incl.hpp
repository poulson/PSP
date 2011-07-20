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
psp::DistQuasi2dHMat<Scalar,Conjugated>::Scale( Scalar alpha )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Scale");
#endif
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                _block.data.N->Child(t,s).Scale( alpha );
        break;

    case DIST_LOW_RANK:
        if( alpha == (Scalar)0 )
        {
            _block.data.DF->rank = 0;
            if( _inTargetTeam )
            {
                Dense<Scalar>& ULocal = _block.data.DF->ULocal;
                ULocal.Resize( ULocal.Height(), 0, ULocal.Height() );
            }
            if( _inSourceTeam )
            {
                Dense<Scalar>& VLocal = _block.data.DF->VLocal;
                VLocal.Resize( VLocal.Height(), 0, VLocal.Height() );
            }
        }
        else if( _inTargetTeam )
            hmat_tools::Scale( alpha, _block.data.DF->ULocal );
        break;
    case SPLIT_LOW_RANK:
        if( alpha == (Scalar)0 )
        {
            _block.data.SF->rank = 0;
            Dense<Scalar>& D = _block.data.SF->D;
            D.Resize( D.Height(), 0, D.Height() );
        }
        else if( _inTargetTeam )
            hmat_tools::Scale( alpha, _block.data.SF->D );
        break;
    case LOW_RANK:
        hmat_tools::Scale( alpha, *_block.data.F );
        break;
    case DIST_LOW_RANK_GHOST:
        if( alpha == (Scalar)0 )
            _block.data.DFG->rank = 0;
    case SPLIT_LOW_RANK_GHOST:
        if( alpha == (Scalar)0 )
            _block.data.SFG->rank = 0;
    case LOW_RANK_GHOST:
        if( alpha == (Scalar)0 )
            _block.data.FG->rank = 0;

    case SPLIT_DENSE:
        if( _inSourceTeam )
            hmat_tools::Scale( alpha, _block.data.SD->D );
        break;
    case DENSE:
        hmat_tools::Scale( alpha, *_block.data.D );
        break;

    default:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

