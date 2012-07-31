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

namespace cliq {

// For handling a vector distributed in a [VC,* ] manner over each node
// of the elimination tree
//
// TODO: Generalize to a set of vectors
//
template<typename F>
class DistNodalMultiVector
{
public:
    Matrix<F> localMultiVec;

    void Pull
    ( const DistMap& inverseMap, const DistSymmInfo& info,
      const DistMultiVector<F>& X );
    void Push
    ( const DistMap& inverseMap, const DistSymmInfo& info,
            DistMultiVector<F>& X ) const;

    DistNodalMultiVector();
    DistNodalMultiVector
    ( const DistMap& inverseMap, const DistSymmInfo& info,
      const DistMultiVector<F>& X );
};

} // namespace cliq
