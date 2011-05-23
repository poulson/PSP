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
    if( Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( _numLevels != B._numLevels )
        throw std::logic_error("H-matrices must have same number of levels");
    if( _zSize != B._zSize )
        throw std::logic_error("Mismatched z sizes");
    if( _level != B._level )
        throw std::logic_error("Mismatched levels");
#endif
    MapHMatrixContext context;
    MapMatrixInitialize( context, alpha, B, C );
    MapMatrixPrecompute( context, alpha, B, C );
    // TODO
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixInitialize
( MapHMatrixContext& context,
  Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
                      DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixInitialize");
#endif
    C._numLevels = _numLevels;
    C._maxRank = _maxRank;
    C._sourceOffset = _sourceOffset;
    C._targetOffset = _targetOffset;
    C._stronglyAdmissible = ( _stronglyAdmissible || B._stronglyAdmissible );

    C._xSizeSource = B._xSizeSource;
    C._ySizeSource = B._ySizeSource;
    C._xSizeTarget = _xSizeTarget;
    C._ySizeTarget = _ySizeTarget;
    C._zSize = _zSize;
    C._xSource = B._xSource;
    C._ySource = B._ySource;
    C._xTarget = _xTarget;
    C._yTarget = _yTarget;

    C._subcomms = _subcomms;
    C._level = _level;
    C._inSourceTeam = B._inSourceTeam;
    C._inTargetTeam = _inTargetTeam;
    if( C._inSourceTeam && !C._inTargetTeam )
        C._rootOfOtherTeam = _rootOfOtherTeam;
    else if( !C._inSourceTeam && C._inTargetTeam )
        C._rootOfOtherTeam = B._rootOfOtherTeam;
    C._localSourceOffset = B._localSourceOffset;
    C._localTargetOffset = _localTargetOffset;

    C._shell.Clear();
    context.Clear();
    MPI_Comm team = _subcomms->Subcomm( _level );
    const int teamSize = mpi::CommSize( team );
    if( C.Admissible() )
    {
        if( teamSize > 1 )
        {
            C._shell.type = DIST_LOW_RANK;
            C._shell.data.DF = new DistLowRank;

            context.shell.type = DIST_LOW_RANK;
            context.shell.data.DF = 
                new typename MapHMatrixContext::DistLowRankContext;
        }
        else // teamSize == 1
        {
            if( C._inSourceTeam && C._inTargetTeam )
            {
                C._shell.type = LOW_RANK;    
                C._shell.data.F = new LowRank;

                context.shell.type = LOW_RANK;
                context.shell.data.F = 
                    new typename MapHMatrixContext::LowRankContext;
            }
            else
            {
                C._shell.type = SPLIT_LOW_RANK;
                C._shell.data.SF = new SplitLowRank;

                context.shell.type = SPLIT_LOW_RANK;
                context.shell.data.SF = 
                    new typename MapHMatrixContext::SplitLowRankContext;
            }
        }
    }
    else if( C._numLevels > 1 )
    {
        context.shell.type = NODE;
        context.shell.data.N = new typename MapHMatrixContext::NodeContext;
        typename MapHMatrixContext::NodeContext& nodeContext = 
            *context.shell.data.N;

        if( teamSize > 1 )
        {
            C._shell.type = DIST_NODE;
            C._shell.data.N = C.NewNode();

            // TODO
            /*
            for( int t=0; t<4; ++t )
                for( int s=0; s<4; ++s )
                    // HERE
            */
        }
        else // teamSize == 1
        {
            if( C._inSourceTeam && C._inTargetTeam )
            {
                C._shell.type = NODE;
                C._shell.data.N = C.NewNode();
                // TODO
            }
            else
            {
                C._shell.type = SPLIT_NODE;
                C._shell.data.N = C.NewNode();
                // TODO
            }
        }
    }
    else
    {
        if( C._inSourceTeam && C._inTargetTeam )
        {
            C._shell.type = DENSE;
            C._shell.data.D = new Dense;

            context.shell.type = DENSE;
            context.shell.data.D = 
                new typename MapHMatrixContext::DenseContext;
        }
        else
        {
            C._shell.type = SPLIT_DENSE;
            C._shell.data.SD = new SplitDense;

            context.shell.type = SPLIT_DENSE;
            context.shell.data.SD = 
                new typename MapHMatrixContext::SplitDenseContext;
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixPrecompute
( MapHMatrixContext& context,
  Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
                      DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapMatrixPrecompute");
#endif
    // TODO
#ifndef RELEASE
    PopCallStack();
#endif
}
