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
#include "./MultiplyDense-incl.hpp"
#include "./MultiplyHMat-incl.hpp"
#include "./MultiplyVector-incl.hpp"
#include "./RedistQuasi2dHMat-incl.hpp"

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMat<Scalar,Conjugated>::DistQuasi2dHMat
( const Teams& teams )
: _numLevels(0), _maxRank(0), 
  _sourceOffset(0), _targetOffset(0), /*_symmetric(false),*/
  _stronglyAdmissible(false), _xSizeSource(0), _xSizeTarget(0),
  _ySizeSource(0), _ySizeTarget(0), _zSize(0), _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0), _teams(&teams), _level(0),
  _inSourceTeam(true), _inTargetTeam(true), 
  _sourceRoot(0), _targetRoot(0),
  _localSourceOffset(0), _localTargetOffset(0),
  _beganRowSpaceComp(false), _beganColSpaceComp(false)
{ 
    _block.type = EMPTY;
}

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMat<Scalar,Conjugated>::DistQuasi2dHMat
( int numLevels, int maxRank, bool stronglyAdmissible,
  int sourceOffset, int targetOffset,
  int xSizeSource, int xSizeTarget, int ySizeSource, int ySizeTarget,
  int zSize, int xSource, int xTarget, int ySource, int yTarget,
  const Teams& teams, unsigned level, 
  bool inSourceTeam, bool inTargetTeam, 
  int sourceRoot, int targetRoot,
  int localSourceOffset, int localTargetOffset )
: _numLevels(numLevels), _maxRank(maxRank), 
  _sourceOffset(sourceOffset), _targetOffset(targetOffset), 
  /*_symmetric(false),*/
  _stronglyAdmissible(stronglyAdmissible), 
  _xSizeSource(xSizeSource), _xSizeTarget(xSizeTarget),
  _ySizeSource(ySizeSource), _ySizeTarget(ySizeTarget), _zSize(zSize), 
  _xSource(xSource), _xTarget(xTarget),
  _ySource(ySource), _yTarget(yTarget), _teams(&teams), _level(level),
  _inSourceTeam(inSourceTeam), _inTargetTeam(inTargetTeam),
  _sourceRoot(sourceRoot), _targetRoot(targetRoot),
  _localSourceOffset(localSourceOffset), _localTargetOffset(localTargetOffset),
  _beganRowSpaceComp(false), _beganColSpaceComp(false)
{ 
    _block.type = EMPTY;
}

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMat<Scalar,Conjugated>::~DistQuasi2dHMat()
{ 
    Clear();
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::Clear()
{
    _block.Clear();
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::LocalHeight() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::LocalHeight");
#endif
    int localHeight;
    if( _inTargetTeam )
    {
        int teamSize = mpi::CommSize( _teams->Team(_level) );
        int teamRank = mpi::CommRank( _teams->Team(_level) );
        ComputeLocalDimensionRecursion
        ( localHeight, teamSize, teamRank, _xSizeTarget, _ySizeTarget, _zSize );
    }
    else
        localHeight = 0;
#ifndef RELEASE
    PopCallStack();
#endif
    return localHeight;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::LocalWidth() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::LocalWidth");
#endif
    int localWidth;
    if( _inSourceTeam )
    {
        int teamSize = mpi::CommSize( _teams->Team(_level) );
        int teamRank = mpi::CommRank( _teams->Team(_level) );
        ComputeLocalDimensionRecursion
        ( localWidth, teamSize, teamRank, _xSizeSource, _ySizeSource, _zSize );
    }
    else
        localWidth = 0;
#ifndef RELEASE
    PopCallStack();
#endif
    return localWidth;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::FirstLocalRow() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::FirstLocalRow");
#endif
    int firstLocalRow = 0;
    if( _inTargetTeam )
    {
        int teamSize = mpi::CommSize( _teams->Team(_level) );
        int teamRank = mpi::CommRank( _teams->Team(_level) );
        ComputeFirstLocalIndexRecursion
        ( firstLocalRow, teamSize, teamRank, 
          _xSizeTarget, _ySizeTarget, _zSize );
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return firstLocalRow;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::FirstLocalCol() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::FirstLocalCol");
#endif
    int firstLocalCol = 0;
    if( _inSourceTeam )
    {
        int teamSize = mpi::CommSize( _teams->Team(_level) );
        int teamRank = mpi::CommRank( _teams->Team(_level) );
        ComputeFirstLocalIndexRecursion
        ( firstLocalCol, teamSize, teamRank, 
          _xSizeSource, _ySizeSource, _zSize );
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return firstLocalCol;
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::RequireRoot() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::RequireRoot");
#endif
    if( _level != 0 )
        throw std::logic_error("Not a root H-matrix as required.");
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::Rank() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Rank");
#endif
    int rank;
    switch( _block.type )
    {
    case DIST_LOW_RANK:
        rank = _block.data.DF->rank;
        break;
    case DIST_LOW_RANK_GHOST:
        rank = _block.data.DFG->rank;
        break;
    case SPLIT_LOW_RANK:
        rank = _block.data.SF->rank;
        break;
    case SPLIT_LOW_RANK_GHOST:
        rank = _block.data.SFG->rank;
        break;
    case LOW_RANK:
        rank = _block.data.F->Rank();
        break;
    case LOW_RANK_GHOST:
        rank = _block.data.FG->rank;
        break;
    default:
#ifndef RELEASE
        throw std::logic_error("Can only request rank of low-rank blocks");
#endif
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return rank;
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::SetGhostRank( int rank ) 
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::SetGhostRank");
#endif
    switch( _block.type )
    {
    case DIST_LOW_RANK_GHOST:
        _block.data.DFG->rank = rank;
        break;
    case SPLIT_LOW_RANK_GHOST:
        _block.data.SFG->rank = rank;
        break;
    case LOW_RANK_GHOST:
        _block.data.FG->rank = rank;
        break;
    default:
#ifndef RELEASE
        throw std::logic_error
        ("Can only set ghost rank of ghost low-rank blocks");
#endif
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::Scale( Scalar alpha )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Scale");
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
                Dense<Scalar>& ULocal = _block.data.DF->ULocal;
                ULocal.Resize( ULocal.Height(), 0, ULocal.Height() );
            }
            if( _inSourceTeam )
            {
                Dense<Scalar>& VLocal = _block.data.DF->VLocal;
                VLocal.Resize( VLocal.Height(), 0, VLocal.Height() );
            }
        }
        else if( _inTargetTeam )
            hmat_tools::Scale( alpha, _block.data.DF->ULocal );
        break;
    case SPLIT_LOW_RANK:
        if( alpha == (Scalar)0 )
        {
            _block.data.SF->rank = 0;
            Dense<Scalar>& D = _block.data.SF->D;
            D.Resize( D.Height(), 0, D.Height() );
        }
        else if( _inTargetTeam )
            hmat_tools::Scale( alpha, _block.data.SF->D );
        break;
    case LOW_RANK:
        hmat_tools::Scale( alpha, *_block.data.F );
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
            hmat_tools::Scale( alpha, _block.data.SD->D );
        break;
    case DENSE:
        hmat_tools::Scale( alpha, *_block.data.D );
        break;
    
    default:
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::Admissible() const
{
    return Admissible( _xSource, _xTarget, _ySource, _yTarget );
}

template<typename Scalar,bool Conjugated>
bool
psp::DistQuasi2dHMat<Scalar,Conjugated>::Admissible
( int xSource, int xTarget, int ySource, int yTarget ) const
{
    if( _stronglyAdmissible )
        return std::max(std::abs(xSource-xTarget),std::abs(ySource-yTarget))>1;
    else
        return xSource != xTarget || ySource != yTarget;
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::LatexWriteLocalStructure
( const std::string& basename ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::LatexWriteLocalStructure");
#endif
    MPI_Comm comm = _teams->Team( 0 );
    const int commRank = mpi::CommRank( comm );

    std::ostringstream os;
    os << basename << "-" << commRank << ".tex";
    std::ofstream file( os.str().c_str() );

    double scale = 12.8;
    file << "\\documentclass[11pt]{article}\n"
         << "\\usepackage{tikz}\n"
         << "\\begin{document}\n"
         << "\\begin{center}\n"
         << "\\begin{tikzpicture}[scale=" << scale << "]\n";
    LatexWriteLocalStructureRecursion( file, Height() );
    file << "\\end{tikzpicture}\n"
         << "\\end{center}\n"
         << "\\end{document}" << std::endl;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MScriptWriteLocalStructure
( const std::string& basename ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MScriptWriteLocalStructure");
#endif
    MPI_Comm comm = _teams->Team( 0 );
    const int commRank = mpi::CommRank( comm );

    std::ostringstream os;
    os << basename << "-" << commRank << ".dat";
    std::ofstream file( os.str().c_str() );
    MScriptWriteLocalStructureRecursion( file );
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMat<Scalar,Conjugated>::DistQuasi2dHMat()
: _numLevels(0), _maxRank(0), 
  _sourceOffset(0), _targetOffset(0), /*_symmetric(false),*/
  _stronglyAdmissible(false), _xSizeSource(0), _xSizeTarget(0),
  _ySizeSource(0), _ySizeTarget(0), _zSize(0), _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0), _teams(0), _level(0),
  _inSourceTeam(true), _inTargetTeam(true), 
  _sourceRoot(0), _targetRoot(0),
  _localSourceOffset(0), _localTargetOffset(0)
{ 
    _block.type = EMPTY;
}

namespace {

void FillBox
( std::ofstream& file, 
  double hStart, double vStart, double hStop, double vStop,
  const std::string& fillColor )
{
    file << "\\fill[" << fillColor << "] (" << hStart << "," << vStart
         << ") rectangle (" << hStop << "," << vStop << ");\n";
}

void DrawBox
( std::ofstream& file, 
  double hStart, double vStart, double hStop, double vStop,
  const std::string& drawColor )
{
    file << "\\draw[" << drawColor << "] (" << hStart << "," << vStart
         << ") rectangle (" << hStop << "," << vStop << ");\n";
}

} // anonymous namespace

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::LatexWriteLocalStructureRecursion
( std::ofstream& file, int globalHeight ) const
{
    const double invScale = globalHeight;
    const double hStart = _sourceOffset/invScale;
    const double hStop  = (_sourceOffset+Width())/invScale;
    const double vStart = (globalHeight-(_targetOffset + Height()))/invScale;
    const double vStop  = (globalHeight-_targetOffset)/invScale;

    const std::string lowRankColor = "green";
    const std::string lowRankEmptyColor = "cyan";
    const std::string lowRankGhostColor = "lightgray";
    const std::string denseColor = "red";
    const std::string denseGhostColor = "gray";
    const std::string borderColor = "black";

    switch( _block.type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).LatexWriteLocalStructureRecursion
                ( file, globalHeight );
        break;
    }

    case DIST_LOW_RANK:
    case SPLIT_LOW_RANK:
    case LOW_RANK:
    {
        const int rank = Rank();
        if( rank == 0 )
            FillBox( file, hStart, vStart, hStop, vStop, lowRankEmptyColor );
        else
            FillBox( file, hStart, vStart, hStop, vStop, lowRankColor );
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;
    }

    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
        FillBox( file, hStart, vStart, hStop, vStop, lowRankGhostColor );
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;

    case SPLIT_DENSE:
    case DENSE:
        FillBox( file, hStart, vStart, hStop, vStop, denseColor );
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;
    
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
        FillBox( file, hStart, vStart, hStop, vStop, denseGhostColor );
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;

    case EMPTY:
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;
    }
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::MScriptWriteLocalStructureRecursion
( std::ofstream& file ) const
{
    switch( _block.type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
    {
        file << "1 " 
             << _targetOffset << " " << _sourceOffset << " "
             << Height() << " " << Width() << "\n";
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MScriptWriteLocalStructureRecursion( file );
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

template class psp::DistQuasi2dHMat<float,false>;
template class psp::DistQuasi2dHMat<float,true>;
template class psp::DistQuasi2dHMat<double,false>;
template class psp::DistQuasi2dHMat<double,true>;
template class psp::DistQuasi2dHMat<std::complex<float>,false>;
template class psp::DistQuasi2dHMat<std::complex<float>,true>;
template class psp::DistQuasi2dHMat<std::complex<double>,false>;
template class psp::DistQuasi2dHMat<std::complex<double>,true>;
