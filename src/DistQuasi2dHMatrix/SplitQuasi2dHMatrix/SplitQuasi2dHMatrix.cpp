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

#include "MapDenseMatrix-incl.hpp"
#include "MapVector-incl.hpp"
#include "Pack-incl.hpp"

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

// Create an empty split H-matrix
template<typename Scalar,bool Conjugated>
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::SplitQuasi2dHMatrix
( MPI_Comm comm )
: _height(0), _width(0), _numLevels(0), _maxRank(0), 
  _sourceOffset(0), _targetOffset(0), 
  /*_type(GENERAL),*/ _stronglyAdmissible(false),
  _xSizeSource(0), _xSizeTarget(0), _ySizeSource(0), _ySizeTarget(0),
  _zSize(0), _xSource(0), _xTarget(0), _ySource(0), _yTarget(0),
  _ownSourceSide(false), _comm(comm), _partner(0)
{ }

template<typename Scalar,bool Conjugated>
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::~SplitQuasi2dHMatrix()
{ }

template class psp::SplitQuasi2dHMatrix<float,false>;
template class psp::SplitQuasi2dHMatrix<float,true>;
template class psp::SplitQuasi2dHMatrix<double,false>;
template class psp::SplitQuasi2dHMatrix<double,true>;
template class psp::SplitQuasi2dHMatrix<std::complex<float>,false>;
template class psp::SplitQuasi2dHMatrix<std::complex<float>,true>;
template class psp::SplitQuasi2dHMatrix<std::complex<double>,false>;
template class psp::SplitQuasi2dHMatrix<std::complex<double>,true>;
