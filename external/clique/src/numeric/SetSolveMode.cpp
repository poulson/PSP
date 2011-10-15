/*
   Clique: a scalable implementation of the multifrontal algorithm

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
#include "clique.hpp"
using namespace elemental;

// This routine could be modified later so that it uses much less memory
// by replacing the '=' redistributions with piece-by-piece redistributions.
template<typename F>
void clique::numeric::SetSolveMode( SymmFrontTree<F>& L, SolveMode mode )
{
#ifndef RELEASE
    PushCallStack("numeric::SetSolveMode");
#endif
    // Check if this call can be a no-op
    if( mode == L.dist.mode ) 
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    L.dist.mode = mode;
    const int numSupernodes = L.dist.fronts.size();    
    if( numSupernodes == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    DistSymmFront<F>& leafFront = L.dist.fronts[0];
    if( mode == FEW_RHS )
    {
        leafFront.front1dL.LockedView
        ( leafFront.front2dL.Height(), 
          leafFront.front2dL.Width(), 0,
          leafFront.front2dL.LockedLocalBuffer(), 
          leafFront.front2dL.LocalLDim(),
          leafFront.front2dL.Grid() );
        for( int s=1; s<numSupernodes; ++s )
        {
            DistSymmFront<F>& front = L.dist.fronts[s];
            front.front1dL.Empty();
            front.front1dL.SetGrid( front.front2dL.Grid() );
            front.front1dL = front.front2dL;
            front.front2dL.Empty();
        }
    }
    else
    {
        leafFront.front2dL.LockedView
        ( leafFront.front1dL.Height(), 
          leafFront.front1dL.Width(), 0, 0,
          leafFront.front1dL.LockedLocalBuffer(), 
          leafFront.front1dL.LocalLDim(),
          leafFront.front1dL.Grid() );
        for( int s=1; s<numSupernodes; ++s )
        {
            DistSymmFront<F>& front = L.dist.fronts[s];
            front.front2dL.Empty();
            front.front2dL.SetGrid( front.front1dL.Grid() );
            front.front2dL = front.front1dL;
            front.front1dL.Empty();
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template void clique::numeric::SetSolveMode
( SymmFrontTree<float>& L, SolveMode mode );

template void clique::numeric::SetSolveMode
( SymmFrontTree<double>& L, SolveMode mode );

template void clique::numeric::SetSolveMode
( SymmFrontTree<std::complex<float> >& L, SolveMode mode );

template void clique::numeric::SetSolveMode
( SymmFrontTree<std::complex<double> >& L, SolveMode mode );
