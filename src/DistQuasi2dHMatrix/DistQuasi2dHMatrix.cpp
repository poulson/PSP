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

#include "./Ghost-incl.hpp"
#include "./MapDenseMatrix-incl.hpp"
//#include "./MapHMatrix-incl.hpp"
#include "./MapVector-incl.hpp"
#include "./RedistQuasi2dHMatrix-incl.hpp"

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::DistQuasi2dHMatrix
( const Subcomms& subcomms )
: _numLevels(0), _maxRank(0), 
  _sourceOffset(0), _targetOffset(0), /*_symmetric(false),*/
  _stronglyAdmissible(false), _xSizeSource(0), _xSizeTarget(0),
  _ySizeSource(0), _ySizeTarget(0), _zSize(0), _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0), _subcomms(&subcomms), _level(0),
  _inSourceTeam(true), _inTargetTeam(true), _rootOfOtherTeam(0),
  _localSourceOffset(0), _localTargetOffset(0)
{ 
    _block.type = EMPTY;
}

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::DistQuasi2dHMatrix
( int numLevels, int maxRank, bool stronglyAdmissible,
  int sourceOffset, int targetOffset,
  int xSizeSource, int xSizeTarget, int ySizeSource, int ySizeTarget,
  int zSize, int xSource, int xTarget, int ySource, int yTarget,
  const Subcomms& subcomms, unsigned level, 
  bool inSourceTeam, bool inTargetTeam, 
  int localSourceOffset, int localTargetOffset )
: _numLevels(numLevels), _maxRank(maxRank), 
  _sourceOffset(sourceOffset), _targetOffset(targetOffset), 
  /*_symmetric(false),*/
  _stronglyAdmissible(stronglyAdmissible), 
  _xSizeSource(xSizeSource), _xSizeTarget(xSizeTarget),
  _ySizeSource(ySizeSource), _ySizeTarget(ySizeTarget), _zSize(zSize), 
  _xSource(xSource), _xTarget(xTarget),
  _ySource(ySource), _yTarget(yTarget), _subcomms(&subcomms), _level(level),
  _inSourceTeam(inSourceTeam), _inTargetTeam(inTargetTeam),
  _localSourceOffset(localSourceOffset), _localTargetOffset(localTargetOffset)
{ 
    _block.type = EMPTY;
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
    _block.Clear();
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
                Dense& ULocal = _block.data.DF->ULocal;
                ULocal.Resize( ULocal.Height(), 0, ULocal.Height() );
            }
            if( _inSourceTeam )
            {
                Dense& VLocal = _block.data.DF->VLocal;
                VLocal.Resize( VLocal.Height(), 0, VLocal.Height() );
            }
        }
        else if( _inTargetTeam )
            hmatrix_tools::Scale( alpha, _block.data.DF->ULocal );
        break;
    case SPLIT_LOW_RANK:
        if( alpha == (Scalar)0 )
        {
            _block.data.SF->rank = 0;
            Dense& D = _block.data.SF->D;
            D.Resize( D.Height(), 0, D.Height() );
        }
        else if( _inTargetTeam )
            hmatrix_tools::Scale( alpha, _block.data.SF->D );
        break;
    case LOW_RANK:
        hmatrix_tools::Scale( alpha, *_block.data.F );
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
            hmatrix_tools::Scale( alpha, _block.data.SD->D );
        break;
    case DENSE:
        hmatrix_tools::Scale( alpha, *_block.data.D );
        break;

    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
    case EMPTY:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//
template<typename Scalar,bool Conjugated>
bool
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::Admissible() const
{
    return Admissible( _xSource, _xTarget, _ySource, _yTarget );
}

template<typename Scalar,bool Conjugated>
bool
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::Admissible
( int xSource, int xTarget, int ySource, int yTarget ) const
{
    if( _stronglyAdmissible )
        return std::max(std::abs(xSource-xTarget),std::abs(ySource-yTarget))>1;
    else
        return xSource != xTarget || ySource != yTarget;
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::WriteLocalStructure
( const std::string& basename ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::WriteLocalStructure");
#endif
    MPI_Comm comm = _subcomms->Subcomm( 0 );
    const int commRank = mpi::CommRank( comm );

    std::ostringstream os;
    os << basename << "-" << commRank;
    std::ofstream file( os.str().c_str() );
    WriteLocalStructureRecursion( file );
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//
template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::WriteLocalStructureRecursion
( std::ofstream& file ) const
{
    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    {
        file << "1 " 
             << _targetOffset << " " << _sourceOffset << " "
             << Height() << " " << Width() << "\n";
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).WriteLocalStructureRecursion( file );
        break;
    }

    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    {
        file << "1 "
             << _targetOffset << " " << _sourceOffset << " "
             << Height() << " " << Width() << "\n";
        const NodeGhost& nodeGhost = *_block.data.NG;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                nodeGhost.Child(t,s).WriteLocalStructureRecursion( file );
        break;
    }

    case DIST_LOW_RANK:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK:
    case LOW_RANK_GHOST:
        file << "5 "
             << _targetOffset << " " << _sourceOffset << " "
             << Height() << " " << Width() << "\n";
        break;

    case SPLIT_DENSE:
    case SPLIT_DENSE_GHOST:
    case DENSE:
    case DENSE_GHOST:
        file << "20 "
             << _targetOffset << " " << _sourceOffset << " "
             << Height() << " " << Width() << "\n";
        break;

    case EMPTY:
        break;
    }
}

template class psp::DistQuasi2dHMatrix<float,false>;
template class psp::DistQuasi2dHMatrix<float,true>;
template class psp::DistQuasi2dHMatrix<double,false>;
template class psp::DistQuasi2dHMatrix<double,true>;
template class psp::DistQuasi2dHMatrix<std::complex<float>,false>;
template class psp::DistQuasi2dHMatrix<std::complex<float>,true>;
template class psp::DistQuasi2dHMatrix<std::complex<double>,false>;
template class psp::DistQuasi2dHMatrix<std::complex<double>,true>;
