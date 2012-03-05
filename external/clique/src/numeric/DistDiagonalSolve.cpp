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
#include "clique.hpp"

namespace cliq {

template<typename F> // F representa a real or complex field
void numeric::DistDiagonalSolve
( const symbolic::SymmFact& S,
  const numeric::SymmFrontTree<F>& L,
        Matrix<F>& localX )
{
    using namespace symbolic;
#ifndef RELEASE
    PushCallStack("numeric::DistDiagonalSolve");
#endif
    const int numDistSupernodes = S.dist.supernodes.size();
    const int width = localX.Width();

    for( int s=1; s<numDistSupernodes; ++s )
    {
        const DistSymmFactSupernode& sn = S.dist.supernodes[s];
        const DistSymmFront<F>& front = L.dist.fronts[s];

        Matrix<F> localXT;
        localXT.View( localX, sn.localOffset1d, 0, sn.localSize1d, width );

        elem::DiagonalSolve
        ( LEFT, NORMAL, front.diag.LockedLocalMatrix(), localXT, true );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace cliq

template void cliq::numeric::DistDiagonalSolve
( const symbolic::SymmFact& S,
  const numeric::SymmFrontTree<float>& L,
        Matrix<float>& localX );

template void cliq::numeric::DistDiagonalSolve
( const symbolic::SymmFact& S,
  const numeric::SymmFrontTree<double>& L,
        Matrix<double>& localX );

template void cliq::numeric::DistDiagonalSolve
( const symbolic::SymmFact& S,
  const numeric::SymmFrontTree<Complex<float> >& L,
        Matrix<Complex<float> >& localX );

template void cliq::numeric::DistDiagonalSolve
( const symbolic::SymmFact& S,
  const numeric::SymmFrontTree<Complex<double> >& L,
        Matrix<Complex<double> >& localX );
