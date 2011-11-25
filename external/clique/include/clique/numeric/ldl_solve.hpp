/*
   Clique: a scalable implementation of the multifrontal algorithm

   Copyright (C) 2010-2011 Jack Poulson <jack.poulson@gmail.com>
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
#ifndef CLIQUE_NUMERIC_LDL_SOLVE_HPP
#define CLIQUE_NUMERIC_LDL_SOLVE_HPP 1

namespace clique {
namespace numeric {

template<typename F>
void LDLSolve
( Orientation orientation,
  const symbolic::SymmFact& S,
  const numeric::SymmFrontTree<F>& L,
        Matrix<F>& localX, 
        bool checkIfSingular=true,
        bool singleL11AllGather=false );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F>
void LDLSolve
( Orientation orientation,
  const symbolic::SymmFact& S,
  const numeric::SymmFrontTree<F>& L,
        Matrix<F>& localX, 
        bool checkIfSingular,
        bool singleL11AllGather )
{
#ifndef RELEASE
    PushCallStack("numeric::LDLSolve");
    if( orientation == NORMAL )
        throw std::logic_error("Invalid orientation for LDL");
#endif
    // Solve against unit diagonal L
    clique::numeric::LowerSolve
    ( NORMAL, UNIT, S, L, localX, checkIfSingular, singleL11AllGather );

    // Solve against diagonal
    clique::numeric::DiagonalSolve( S, L, localX, checkIfSingular );

    // Solve against the (conjugate-)transpose of the unit diagonal L
    clique::numeric::LowerSolve
    ( orientation, UNIT, S, L, localX, checkIfSingular, singleL11AllGather );
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace numeric
} // namespace clique

#endif /* CLIQUE_NUMERIC_LDL_SOLVE_HPP */

