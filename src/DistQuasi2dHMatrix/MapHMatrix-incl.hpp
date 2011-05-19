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

// C := alpha A B
template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
                      DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrix");
    if( this->_width != B._height )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( this->_numLevels != B._numLevels )
        throw std::logic_error("H-matrices must have same number of levels");
#endif
    // HERE    
#ifndef RELEASE
    PopCallStack();
#endif
}

// C := alpha A B + beta C
template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
  Scalar beta,        DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrix");
    if( this->_width  != B._height || 
        this->_height != C._height ||
        B._width      != C._width )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( this->_numLevels != B._numLevels || B._numLevels != C._numLevels )
        throw std::logic_error("H-matrices must have same number of levels");
#endif
    // HERE
#ifndef RELEASE
    PopCallStack();
#endif
}

