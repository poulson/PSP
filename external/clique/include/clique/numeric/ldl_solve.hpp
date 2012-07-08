/*
   Clique: a scalable implementation of the multifrontal algorithm

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
#ifndef CLIQUE_LDL_SOLVE_HPP
#define CLIQUE_LDL_SOLVE_HPP 1

namespace cliq {

template<typename F>
void LDLSolve
( Orientation orientation,
  const SymmInfo& info, const SymmFrontTree<F>& L, Matrix<F>& localX );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F>
inline void LDLSolve
( Orientation orientation,
  const SymmInfo& info, const SymmFrontTree<F>& L, Matrix<F>& localX )
{
#ifndef RELEASE
    PushCallStack("LDLSolve");
    if( orientation == NORMAL )
        throw std::logic_error("Invalid orientation for LDL");
#endif
    // Solve against unit diagonal L
    LowerSolve( NORMAL, UNIT, info, L, localX );

    // Solve against diagonal
    DiagonalSolve( info, L, localX );

    // Solve against the (conjugate-)transpose of the unit diagonal L
    LowerSolve( orientation, UNIT, info, L, localX );
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace cliq

#endif // CLIQUE_LDL_SOLVE_HPP
