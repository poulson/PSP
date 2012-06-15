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
#ifndef PSP_DIST_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
#define PSP_DIST_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP 1

namespace psp {

template<typename F>
inline void
DistFrontCompressedBlockLowerForwardSolve
( const DistCompressedFront<F>& front, DistMatrix<F,VC,STAR>& X )
{
#ifndef RELEASE
    PushCallStack("DistFrontCompressedBlockLowerForwardSolve");
#endif
    const int numKeptModesA = front.AGreens.size();
    const int numKeptModesB = front.BGreens.size();

    const Grid& g = *front.grid;

    // XT := inv(ATL) XT
    //     = (C_A o G_A) XT
    // TODO    

    // XB := XB - LB XT
    //     = XB - (C_B o G_B) XT
    if( numKeptModesB != 0 )
    {
        // TODO 
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void
DistFrontCompressedBlockLowerBackwardSolve
( Orientation orientation, 
  const DistCompressedFront<F>& front, DistMatrix<F,VC,STAR>& X )
{
#ifndef RELEASE
    PushCallStack();
#endif
    const int numKeptModesA = front.AGreens.size();
    const int numKeptModesB = front.BGreens.size();
    if( numKeptModesB == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const Grid& g = *front.grid;

    // YT := LB^[T/H] XB
    //     = (C_B o G_B)^[T/H] XB
    //     = (C_B^[T/H] o G_B^[T/H]) XB
    // TODO

    // XT := XT - inv(ATL) YT
    //     = XT - (C_A o G_A) YT
    // TODO
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_DIST_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
