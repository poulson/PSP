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

#include "./MapDenseMatrix-incl.hpp"
#include "./MapVector-incl.hpp"
#include "./RedistQuasi2dHMatrix-incl.hpp"

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::DistQuasi2dHMatrix
( const Subcomms& subcomms )
: _height(0), _width(0), _numLevels(0), _maxRank(0), 
  _sourceOffset(0), _targetOffset(0), /*_symmetric(false),*/
  _stronglyAdmissible(false), _xSizeSource(0), _xSizeTarget(0),
  _ySizeSource(0), _ySizeTarget(0), _zSize(0), _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0), _subcomms(&subcomms), _level(0),
  _inSourceTeam(true), _inTargetTeam(true), _rootOfOtherTeam(0),
  _localSourceOffset(0), _localTargetOffset(0)
{ 
    _shell.type = EMPTY;
}

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::DistQuasi2dHMatrix
( const Subcomms& subcomms, unsigned level, 
  bool inSourceTeam, bool inTargetTeam, 
  int localSourceOffset, int localTargetOffset )
: _height(0), _width(0), _numLevels(0), _maxRank(0), 
  _sourceOffset(0), _targetOffset(0), /*_symmetric(false),*/
  _stronglyAdmissible(false), _xSizeSource(0), _xSizeTarget(0),
  _ySizeSource(0), _ySizeTarget(0), _zSize(0), _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0), _subcomms(&subcomms), _level(level),
  _inSourceTeam(inSourceTeam), _inTargetTeam(inTargetTeam),
  _localSourceOffset(localSourceOffset), _localTargetOffset(localTargetOffset)
{ 
    _shell.type = EMPTY;
}

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::~DistQuasi2dHMatrix()
{ 
    Clear();
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::Clear()
{
    _shell.Clear();
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::LocalHeight() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::LocalHeight");
#endif
    int teamSize = mpi::CommSize( _subcomms->Subcomm(_level) );
    int teamRank = mpi::CommRank( _subcomms->Subcomm(_level) );

    int localHeight;
    ComputeLocalDimensionRecursion
    ( localHeight, teamSize, teamRank, _xSizeTarget, _ySizeTarget, _zSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return localHeight;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::LocalWidth() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::LocalWidth");
#endif
    int teamSize = mpi::CommSize( _subcomms->Subcomm(_level) );
    int teamRank = mpi::CommRank( _subcomms->Subcomm(_level) );

    int localWidth;
    ComputeLocalDimensionRecursion
    ( localWidth, teamSize, teamRank, _xSizeSource, _ySizeSource, _zSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return localWidth;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::FirstLocalRow() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::FirstLocalRow");
#endif
    int teamSize = mpi::CommSize( _subcomms->Subcomm(_level) );
    int teamRank = mpi::CommRank( _subcomms->Subcomm(_level) );

    int firstLocalRow = 0;
    ComputeFirstLocalIndexRecursion
    ( firstLocalRow, teamSize, teamRank, _xSizeTarget, _ySizeTarget, _zSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return firstLocalRow;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::FirstLocalCol() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::FirstLocalCol");
#endif
    int teamSize = mpi::CommSize( _subcomms->Subcomm(_level) );
    int teamRank = mpi::CommRank( _subcomms->Subcomm(_level) );

    int firstLocalCol = 0;
    ComputeFirstLocalIndexRecursion
    ( firstLocalCol, teamSize, teamRank, _xSizeSource, _ySizeSource, _zSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return firstLocalCol;
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::Scale( Scalar alpha )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::Scale");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                shell.data.N->Child(t,s).Scale( alpha );
        break;
    case DIST_LOW_RANK:
        if( _inTargetTeam )
            hmatrix_tools::Scale( alpha, shell.data.DF->ULocal );
        break;
    case SPLIT_LOW_RANK:
        if( _inTargetTeam )
            hmatrix_tools::Scale( alpha, shell.data.SF->D );
        break;
    case LOW_RANK:
        hmatrix_tools::Scale( alpha, shell.data.F->U );
        break;
    case SPLIT_DENSE:
        if( _inSourceTeam )
            hmatrix_tools::Scale( alpha, shell.data.SD->D );
        break;
    case DENSE:
        hmatrix_tools::Scale( alpha, *shell.data.D );
        break;
    case EMPTY:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template class psp::DistQuasi2dHMatrix<float,false>;
template class psp::DistQuasi2dHMatrix<float,true>;
template class psp::DistQuasi2dHMatrix<double,false>;
template class psp::DistQuasi2dHMatrix<double,true>;
template class psp::DistQuasi2dHMatrix<std::complex<float>,false>;
template class psp::DistQuasi2dHMatrix<std::complex<float>,true>;
template class psp::DistQuasi2dHMatrix<std::complex<double>,false>;
template class psp::DistQuasi2dHMatrix<std::complex<double>,true>;
