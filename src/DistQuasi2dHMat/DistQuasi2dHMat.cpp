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

#include "./Adjoint-incl.hpp"
#include "./Conjugate-incl.hpp"
#include "./Copy-incl.hpp"
#include "./Ghost-incl.hpp"
#include "./MultiplyDense-incl.hpp"
#include "./MultiplyHMat-incl.hpp"
#include "./MultiplyVector-incl.hpp"
#include "./RedistQuasi2dHMat-incl.hpp"
#include "./Scale-incl.hpp"
#include "./SetToRandom-incl.hpp"
#include "./Transpose-incl.hpp"

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMat<Scalar,Conjugated>::DistQuasi2dHMat
( const Teams& teams )
: _numLevels(0), _maxRank(0), 
  _sourceOffset(0), _targetOffset(0), 
  _stronglyAdmissible(false), 
  _xSizeSource(0), _xSizeTarget(0),
  _ySizeSource(0), _ySizeTarget(0), _zSize(0), 
  _xSource(0), _xTarget(0), _ySource(0), _yTarget(0), 
  _teams(&teams), _level(0),
  _inSourceTeam(true), _inTargetTeam(true), 
  _sourceRoot(0), _targetRoot(0),
  _haveDenseUpdate(false), _storedDenseUpdate(false),
  _beganRowSpaceComp(false), _finishedRowSpaceComp(false),
  _beganColSpaceComp(false), _finishedColSpaceComp(false)
{ 
    _block.type = EMPTY;
}

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMat<Scalar,Conjugated>::DistQuasi2dHMat
( int numLevels, int maxRank, bool stronglyAdmissible, 
  int xSize, int ySize, int zSize,
  const Teams& teams )
: _numLevels(numLevels), _maxRank(maxRank), 
  _sourceOffset(0), _targetOffset(0), 
  _stronglyAdmissible(stronglyAdmissible), 
  _xSizeSource(xSize), _xSizeTarget(xSize),
  _ySizeSource(ySize), _ySizeTarget(ySize), _zSize(zSize), 
  _xSource(0), _xTarget(0), _ySource(0), _yTarget(0), 
  _teams(&teams), _level(0),
  _inSourceTeam(true), _inTargetTeam(true), 
  _sourceRoot(0), _targetRoot(0),
  _haveDenseUpdate(false), _storedDenseUpdate(false),
  _beganRowSpaceComp(false), _finishedRowSpaceComp(false),
  _beganColSpaceComp(false), _finishedColSpaceComp(false)
{ 
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::DistQuasi2dHMat");
#endif
    const int numTeamLevels = teams.NumLevels();
    if( numTeamLevels > numLevels )
        throw std::logic_error("Too many processes for this H-matrix depth");
    BuildTree();
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMat<Scalar,Conjugated>::DistQuasi2dHMat
( int numLevels, int maxRank, bool stronglyAdmissible, 
  int xSizeSource, int xSizeTarget, 
  int ySizeSource, int ySizeTarget, int zSize,
  const Teams& teams )
: _numLevels(numLevels), _maxRank(maxRank), 
  _sourceOffset(0), _targetOffset(0), 
  _stronglyAdmissible(stronglyAdmissible), 
  _xSizeSource(xSizeSource), _xSizeTarget(xSizeTarget),
  _ySizeSource(ySizeSource), _ySizeTarget(ySizeTarget), _zSize(zSize), 
  _xSource(0), _xTarget(0), _ySource(0), _yTarget(0), 
  _teams(&teams), _level(0),
  _inSourceTeam(true), _inTargetTeam(true), 
  _sourceRoot(0), _targetRoot(0),
  _haveDenseUpdate(false), _storedDenseUpdate(false),
  _beganRowSpaceComp(false), _finishedRowSpaceComp(false),
  _beganColSpaceComp(false), _finishedColSpaceComp(false)
{ 
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::DistQuasi2dHMat");
#endif
    const int numTeamLevels = teams.NumLevels();
    if( numTeamLevels > numLevels )
        throw std::logic_error("Too many processes for this H-matrix depth");
    BuildTree();
#ifndef RELEASE
    PopCallStack();
#endif
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
        int xSize = _xSizeTarget;
        int ySize = _ySizeTarget;
        int zSize = _zSize;
        ComputeLocalDimensionRecursion
        ( localHeight, xSize, ySize, zSize, teamSize, teamRank );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::LocalHeightPartner() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::LocalHeightPartner");
#endif
    int localHeightPartner;
    if( _inSourceTeam )
    {
        int teamSize = mpi::CommSize( _teams->Team(_level) );
        int teamRank = mpi::CommRank( _teams->Team(_level) );
        int xSize = _xSizeTarget;
        int ySize = _ySizeTarget;
        int zSize = _zSize;
        ComputeLocalDimensionRecursion
        ( localHeightPartner, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        localHeightPartner = 0;
#ifndef RELEASE
    PopCallStack();
#endif
    return localHeightPartner;
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
        int xSize = _xSizeSource;
        int ySize = _ySizeSource;
        int zSize = _zSize;
        ComputeLocalDimensionRecursion
        ( localWidth, xSize, ySize, zSize, teamSize, teamRank );
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
psp::DistQuasi2dHMat<Scalar,Conjugated>::LocalWidthPartner() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::LocalWidthPartner");
#endif
    int localWidthPartner;
    if( _inTargetTeam )
    {
        int teamSize = mpi::CommSize( _teams->Team(_level) );
        int teamRank = mpi::CommRank( _teams->Team(_level) );
        int xSize = _xSizeSource;
        int ySize = _ySizeSource;
        int zSize = _zSize;
        ComputeLocalDimensionRecursion
        ( localWidthPartner, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        localWidthPartner = 0;
#ifndef RELEASE
    PopCallStack();
#endif
    return localWidthPartner;
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
        ( firstLocalRow, _xSizeTarget, _ySizeTarget, _zSize,
          teamSize, teamRank );
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
        ( firstLocalCol, _xSizeSource, _ySizeSource, _zSize,
          teamSize, teamRank );
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return firstLocalCol;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::FirstLocalXTarget() const
{ return _xTarget << (_numLevels-1); }

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::FirstLocalXSource() const
{ return _xSource << (_numLevels-1); }

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::FirstLocalYTarget() const
{ return _yTarget << (_numLevels-1); }

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::FirstLocalYSource() const
{ return _ySource << (_numLevels-1); }

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::FirstLocalZTarget() const
{ return 0; }

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::FirstLocalZSource() const
{ return _zSize; }

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::LocalXTargetSize() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::LocalXTargetSize");
#endif
    int xSize;
    if( _inTargetTeam )
    {
        int localHeight;
        int teamSize = mpi::CommSize( _teams->Team(_level) );
        int teamRank = mpi::CommRank( _teams->Team(_level) );
        xSize = _xSizeTarget;
        int ySize = _ySizeTarget;
        int zSize = _zSize;
        ComputeLocalDimensionRecursion
        ( localHeight, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        xSize = 0;
#ifndef RELEASE
    PopCallStack();
#endif
    return xSize;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::LocalXSourceSize() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::LocalXSourceSize");
#endif
    int xSize;
    if( _inSourceTeam )
    {
        int localWidth;
        int teamSize = mpi::CommSize( _teams->Team(_level) );
        int teamRank = mpi::CommRank( _teams->Team(_level) );
        xSize = _xSizeSource;
        int ySize = _ySizeSource;
        int zSize = _zSize;
        ComputeLocalDimensionRecursion
        ( localWidth, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        xSize = 0;
#ifndef RELEASE
    PopCallStack();
#endif
    return xSize;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::LocalYTargetSize() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::LocalYTargetSize");
#endif
    int ySize;
    if( _inTargetTeam )
    {
        int localHeight;
        int teamSize = mpi::CommSize( _teams->Team(_level) );
        int teamRank = mpi::CommRank( _teams->Team(_level) );
        int xSize = _xSizeTarget;
        ySize = _ySizeTarget;
        int zSize = _zSize;
        ComputeLocalDimensionRecursion
        ( localHeight, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        ySize = 0;
#ifndef RELEASE
    PopCallStack();
#endif
    return ySize;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::LocalYSourceSize() const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::LocalYSourceSize");
#endif
    int ySize;
    if( _inSourceTeam )
    {
        int localWidth;
        int teamSize = mpi::CommSize( _teams->Team(_level) );
        int teamRank = mpi::CommRank( _teams->Team(_level) );
        int xSize = _xSizeSource;
        ySize = _ySizeSource;
        int zSize = _zSize;
        ComputeLocalDimensionRecursion
        ( localWidth, xSize, ySize, zSize, teamSize, teamRank );
    }
    else
        ySize = 0;
#ifndef RELEASE
    PopCallStack();
#endif
    return ySize;
}

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::LocalZTargetSize() const
{ return _zSize; }

template<typename Scalar,bool Conjugated>
int
psp::DistQuasi2dHMat<Scalar,Conjugated>::LocalZSourceSize() const
{ return _zSize; }

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
( const std::string basename ) const
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
( const std::string basename ) const
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
// Private static routines                                                    //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
const std::string
psp::DistQuasi2dHMat<Scalar,Conjugated>::BlockTypeString
( BlockType type )
{
    std::string s;
    switch( type )
    {
    case DIST_NODE:            s = "DIST_NODE";            break;
    case DIST_NODE_GHOST:      s = "DIST_NODE_GHOST";      break;
    case SPLIT_NODE:           s = "SPLIT_NODE";           break;
    case SPLIT_NODE_GHOST:     s = "SPLIT_NODE_GHOST";     break;
    case NODE:                 s = "NODE";                 break;
    case NODE_GHOST:           s = "NODE_GHOST";           break;
    case DIST_LOW_RANK:        s = "DIST_LOW_RANK";        break;
    case DIST_LOW_RANK_GHOST:  s = "DIST_LOW_RANK_GHOST";  break;
    case SPLIT_LOW_RANK:       s = "SPLIT_LOW_RANK";       break;
    case SPLIT_LOW_RANK_GHOST: s = "SPLIT_LOW_RANK_GHOST"; break;
    case LOW_RANK:             s = "LOW_RANK";             break;
    case LOW_RANK_GHOST:       s = "LOW_RANK_GHOST";       break;
    case SPLIT_DENSE:          s = "SPLIT_DENSE";          break;
    case SPLIT_DENSE_GHOST:    s = "SPLIT_DENSE_GHOST";    break;
    case DENSE:                s = "DENSE";                break;
    case DENSE_GHOST:          s = "DENSE_GHOST";          break;
    case EMPTY:                s = "EMPTY";                break;
    }
    return s;
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMat<Scalar,Conjugated>::DistQuasi2dHMat()
: _numLevels(0), _maxRank(0), 
  _sourceOffset(0), _targetOffset(0), 
  _stronglyAdmissible(false), 
  _xSizeSource(0), _xSizeTarget(0), 
  _ySizeSource(0), _ySizeTarget(0), _zSize(0),
  _xSource(0), _xTarget(0), 
  _ySource(0), _yTarget(0), 
  _teams(0), _level(0),
  _inSourceTeam(true), _inTargetTeam(true), 
  _sourceRoot(0), _targetRoot(0),
  _haveDenseUpdate(false), _storedDenseUpdate(false),
  _beganRowSpaceComp(false), _finishedRowSpaceComp(false),
  _beganColSpaceComp(false), _finishedColSpaceComp(false)
{ 
    _block.type = EMPTY;
}

template<typename Scalar,bool Conjugated>
psp::DistQuasi2dHMat<Scalar,Conjugated>::DistQuasi2dHMat
( int numLevels, int maxRank, bool stronglyAdmissible,
  int sourceOffset, int targetOffset,
  int xSizeSource, int xSizeTarget, int ySizeSource, int ySizeTarget,
  int zSize, int xSource, int xTarget, int ySource, int yTarget,
  const Teams& teams, int level, 
  bool inSourceTeam, bool inTargetTeam, 
  int sourceRoot, int targetRoot )
: _numLevels(numLevels), _maxRank(maxRank), 
  _sourceOffset(sourceOffset), _targetOffset(targetOffset), 
  _stronglyAdmissible(stronglyAdmissible), 
  _xSizeSource(xSizeSource), _xSizeTarget(xSizeTarget),
  _ySizeSource(ySizeSource), _ySizeTarget(ySizeTarget), _zSize(zSize), 
  _xSource(xSource), _xTarget(xTarget),
  _ySource(ySource), _yTarget(yTarget), 
  _teams(&teams), _level(level),
  _inSourceTeam(inSourceTeam), _inTargetTeam(inTargetTeam),
  _sourceRoot(sourceRoot), _targetRoot(targetRoot),
  _haveDenseUpdate(false), _storedDenseUpdate(false),
  _beganRowSpaceComp(false), _finishedRowSpaceComp(false),
  _beganColSpaceComp(false), _finishedColSpaceComp(false)
{ 
    _block.type = EMPTY;
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::BuildTree()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::BuildTree");
#endif
    MPI_Comm team = _teams->Team(_level);
    const int teamSize = mpi::CommSize( team );
    const int teamRank = mpi::CommRank( team );
    if( !_inSourceTeam && !_inTargetTeam )
        _block.type = EMPTY;
    else if( Admissible() ) // low rank
    {
        if( teamSize > 1 )
        {
            _block.type = DIST_LOW_RANK;
            _block.data.DF = new DistLowRank;
            _block.data.DF->rank = 0;
            _block.data.DF->ULocal.Resize( LocalHeight(), 0 );
            _block.data.DF->VLocal.Resize( LocalWidth(),  0 );
        }
        else if( _sourceRoot == _targetRoot )
        {
            _block.type = LOW_RANK;
            _block.data.F = new LowRank<Scalar,Conjugated>;
            _block.data.F->U.Resize( Height(), 0 );
            _block.data.F->V.Resize( Width(),  0 );
        }
        else
        {
            _block.type = SPLIT_LOW_RANK;
            _block.data.SF = new SplitLowRank;
            _block.data.SF->rank = 0;
            if( _inTargetTeam )
                _block.data.SF->D.Resize( Height(), 0 );
            else
                _block.data.SF->D.Resize( Width(), 0 );
        }
    }
    else if( _numLevels > 1 ) // recurse
    {
        _block.data.N = NewNode();
        Node& node = *_block.data.N;        

        if( teamSize >= 4 )
        {
            _block.type = DIST_NODE;

            const int subteam = teamRank/(teamSize/4);
            // Top-left block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = _targetRoot + t*(teamSize/4);
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = _sourceRoot + s*(teamSize/4);

                    node.children[s+4*t] =
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[0],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource, 2*_yTarget,
                          *_teams, _level+1,
                          _inSourceTeam && (s==subteam),
                          _inTargetTeam && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).BuildTree();
                }
            }
            // Top-right block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = _targetRoot + t*(teamSize/4);
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1];
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = _sourceRoot + s*(teamSize/4);

                    node.children[s+4*t] =
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[0],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource+1, 2*_yTarget,
                          *_teams, _level+1,
                          _inSourceTeam && (s==subteam),
                          _inTargetTeam && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).BuildTree();
                }
            }
            // Bottom-left block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1];
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = _targetRoot + t*(teamSize/4);
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = _sourceRoot + s*(teamSize/4);

                    node.children[s+4*t] =
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[1],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource, 2*_yTarget+1,
                          *_teams, _level+1,
                          _inSourceTeam && (s==subteam),
                          _inTargetTeam && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).BuildTree();
                }
            }
            // Bottom-right block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1];
                t<4; tOffset+=node.targetSizes[t],++t )
            {
                const int targetRoot = _targetRoot + t*(teamSize/4);
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1];
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    const int sourceRoot = _sourceRoot + s*(teamSize/4);

                    node.children[s+4*t] =
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[1],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource+1, 2*_yTarget+1,
                          *_teams, _level+1,
                          _inSourceTeam && (s==subteam),
                          _inTargetTeam && (t==subteam),
                          sourceRoot, targetRoot );
                    node.Child(t,s).BuildTree();
                }
            }
        }
        else if( teamSize == 2 )
        {
            _block.type = DIST_NODE;

            const bool inUpperTeam = ( teamRank == 1 );
            const bool inLeftSourceTeam = ( !inUpperTeam && _inSourceTeam );
            const bool inRightSourceTeam = ( inUpperTeam && _inSourceTeam );
            const bool inTopTargetTeam = ( !inUpperTeam && _inTargetTeam );
            const bool inBottomTargetTeam = ( inUpperTeam && _inTargetTeam );

            // Top-left block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] =
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[0],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource, 2*_yTarget,
                          *_teams, _level+1,
                          inLeftSourceTeam, inTopTargetTeam,
                          _sourceRoot, _targetRoot );
                    node.Child(t,s).BuildTree();
                }
            }
            // Top-right block
            for( int t=0,tOffset=0; t<2; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1];
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] =
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[0],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource+1, 2*_yTarget,
                          *_teams, _level+1,
                          inRightSourceTeam, inTopTargetTeam,
                          _sourceRoot+1, _targetRoot );
                    node.Child(t,s).BuildTree();
                }
            }
            // Bottom-left block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1];
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<2; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] =
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[0], node.yTargetSizes[1],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource, 2*_yTarget+1,
                          *_teams, _level+1,
                          inLeftSourceTeam, inBottomTargetTeam,
                          _sourceRoot, _targetRoot+1 );
                    node.Child(t,s).BuildTree();
                }
            }
            // Bottom-right block
            for( int t=2,tOffset=node.targetSizes[0]+node.targetSizes[1];
                 t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=2,sOffset=node.sourceSizes[0]+node.sourceSizes[1];
                     s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] =
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[1], node.yTargetSizes[1],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource+1, 2*_yTarget+1,
                          *_teams, _level+1,
                          inRightSourceTeam, inBottomTargetTeam,
                          _sourceRoot+1, _targetRoot+1 );
                    node.Child(t,s).BuildTree();
                }
            }
        }
        else // teamSize == 1 
        {
            _block.type = ( _sourceRoot==_targetRoot ? NODE : SPLIT_NODE );

            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] =
                        new DistQuasi2dHMat<Scalar,Conjugated>
                        ( _numLevels-1, _maxRank, _stronglyAdmissible,
                          _sourceOffset+sOffset, _targetOffset+tOffset,
                          node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                          node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                          _zSize,
                          2*_xSource+(s&1), 2*_xTarget+(t&1),
                          2*_ySource+(s/2), 2*_yTarget+(t/2),
                          *_teams, _level+1,
                          _inSourceTeam, _inTargetTeam,
                          _sourceRoot, _targetRoot );
                    node.Child(t,s).BuildTree();
                }
            }
        }
    }
    else // dense
    {
        if( _sourceRoot == _targetRoot )
        {
            _block.type = DENSE;
            _block.data.D = new Dense<Scalar>( Height(), Width() );
        }
        else
        {
            _block.type = SPLIT_DENSE;
            _block.data.SD = new SplitDense;
            if( _inSourceTeam )
                _block.data.SD->D.Resize( Height(), Width() );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
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
