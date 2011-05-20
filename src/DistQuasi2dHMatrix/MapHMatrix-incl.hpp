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
    if( this->_zSize != B._zSize )
        throw std::logic_error("Mismatched z sizes");
    if( this->_level != B._level )
        throw std::logic_error("Mismatched levels");
#endif
    MapHMatrixContext context;
    MapMatrixPrecompute( context, alpha, B, C );
    // TODO
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
    C._height = this->_height;
    C._width = B._width;
    C._numLevels = this->_numLevels;
    C._maxRank = this->_maxRank;
    C._sourceOffset = this->_sourceOffset;
    C._targetOffset = this->_targetOffset;
    C._stronglyAdmissible = 
        ( this->_stronglyAdmissible || B._stronglyAdmissible );

    C._xSizeSource = B._xSizeSource;
    C._ySizeSource = B._ySizeSource;
    C._xSizeTarget = this->_xSizeTarget;
    C._ySizeTarget = this->_ySizeTarget;
    C._zSize = this->_zSize;
    C._xSource = B._xSource;
    C._ySource = B._ySource;
    C._xTarget = this->_xTarget;
    C._yTarget = this->_yTarget;

    C._subcomms = this->_subcomms;
    C._level = this->_level;
    C._inSourceTeam = B._inSourceTeam;
    C._inTargetTeam = this->_inTargetTeam;
    if( C._inSourceTeam && !C._inTargetTeam )
        C._rootOfOtherTeam = this->_rootOfOtherTeam;
    else if( !C._inSourceTeam && C._inTargetTeam )
        C._rootOfOtherTeam = B._rootOfOtherTeam;
    C._localSourceOffset = B._localSourceOffset;
    C._localTargetOffset = this->_localTargetOffset;

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
            DistLowRank& FC = *C._shell.data.DF;

            context.shell.type = DIST_LOW_RANK;
            context.shell.data.DF = 
                new typename MapHMatrixContext::DistLowRankContext;
            typename MapHMatrixContext::DistLowRankContext& DFContext = 
                *context.shell.data.DF;

            if( this->_shell.type == DIST_LOW_RANK && 
                B._shell.type     == DIST_LOW_RANK )
            {
                // Form the local portion of A.V^[T/H] B.U in the product
                // A.U (A.V^[T/H] B.U) B.V^[T/H]
                const DistLowRank& FA = *this->_shell.data.DF;
                const DistLowRank& FB = *B._shell.data.DF;
                FC.rank = std::min( FA.rank, FB.rank );
                if( this->_inSourceTeam )
                {
#ifndef RELEASE
                    if( !B._inTargetTeam )
                        throw std::logic_error("Mismatched distributions");
#endif
                    Dense& Z = DFContext.Z;
                    Z.Resize( FA.rank, FB.rank, FA.rank );
                    const char option = ( Conjugated ? 'C' : 'T' );
                    blas::Gemm
                    ( option, 'N', FA.rank, FB.rank, FA.VLocal.Height(),
                      (Scalar)1, FA.VLocal.LockedBuffer(), FA.VLocal.LDim(),
                                 FB.ULocal.LockedBuffer(), FB.ULocal.LDim(),
                      (Scalar)0, Z.Buffer(),               Z.LDim() );
                }
            }
            else if( this->_shell.type == DIST_LOW_RANK &&
                     B._shell.type     == DIST_NODE )
            {
                // Precompute what we can of B^[T/H] A.V in the product 
                // A.U A.V^[T/H] B = A.U (B^[T/H] A.V)^[T/H]
                const DistLowRank& FA = *this->_shell.data.DF;
                FC.rank = FA.rank;
                if( this->_inSourceTeam )
                {
                    FC.VLocal.Resize
                    ( B.LocalWidth(), FA.rank, B.LocalWidth() );
                    hmatrix_tools::Scale( (Scalar)0, FC.VLocal );
                }
                if( Conjugated )
                    B.HermitianTransposeMapMatrixPrecompute
                    ( DFContext.context, (Scalar)1, FA.VLocal, FC.VLocal );
                else
                    B.TransposeMapMatrixPrecompute
                    ( DFContext.context, (Scalar)1, FA.VLocal, FC.VLocal );
            }
            else if( this->_shell.type == DIST_NODE &&
                     B._shell.type     == DIST_LOW_RANK )
            {
                // Precompute what we can of alpha A B.U in the product
                // (alpha A B.U) B.V^[T/H]
                const DistLowRank& FB = *B._shell.data.DF;
                FC.rank = FB.rank;
                if( this->_inSourceTeam )
                {
                    FC.ULocal.Resize
                    ( this->LocalHeight(), FB.rank, this->LocalHeight() );
                    hmatrix_tools::Scale( (Scalar)0, FC.ULocal );
                }
                this->MapMatrixPrecompute
                ( DFContext.context, alpha, FB.ULocal, FC.ULocal );
            }
            else if( this->_shell.type == DIST_NODE &&
                     B._shell.type     == DIST_NODE )
            {
                // TODO: Generate random vectors, Omega,  and precompute 
                //       B Omega
            }
#ifndef RELEASE
            else
                std::logic_error("Invalid H-matrix combination");
#endif
        }
        else // teamSize == 1
        {
            if( C._sourceOffset == C._targetOffset )
            {
                C._shell.type = LOW_RANK;    
                C._shell.data.F = new LowRank;
                LowRank& FC = *C._shell.data.F;

                context.shell.type = LOW_RANK;
                context.shell.data.F = 
                    new typename MapHMatrixContext::LowRankContext;
                typename MapHMatrixContext::LowRankContext& FContext = 
                    *context.shell.data.F;

                if( this->_shell.type == LOW_RANK &&
                    B._shell.type     == LOW_RANK )
                {
                    /*
                    const LowRank& FA = *this->_shell.data.F;
                    const LowRank& FB = *B._shell.data.F;
                    hmatrix_tools::MatrixMatrix
                    ( alpha, 
                    */
                }
                else if( this->_shell.type == LOW_RANK &&
                         B._shell.type     == NODE )
                {
                    // TODO
                }
                else if( this->_shell.type == NODE &&
                         B._shell.type     == LOW_RANK )
                {
                    // TODO
                }
                else if( this->_shell.type == NODE &&
                         B._shell.type     == NODE )
                {
                    // TODO
                }
                else if( this->_shell.type == SPLIT_LOW_RANK &&
                         B._shell.type     == SPLIT_LOW_RANK )
                {
                    // TODO
                }
                else if( this->_shell.type == SPLIT_LOW_RANK &&
                         B._shell.type     == SPLIT_NODE )
                {
                    // TODO
                }
                else if( this->_shell.type == SPLIT_NODE &&
                         B._shell.type     == SPLIT_LOW_RANK )
                {
                    // TODO
                }
                else if( this->_shell.type == SPLIT_NODE &&
                         B._shell.type     == SPLIT_NODE )
                {
                    // TODO
                }
#ifndef RELEASE
                else
                    throw std::logic_error("Invalid H-matrix combination");
#endif
            }
            else
            {
                C._shell.type = SPLIT_LOW_RANK;
                C._shell.data.SF = new SplitLowRank;
                // TODO
            }
        }
    }
    else if( C._numLevels > 1 )
    {
        if( teamSize >= 4 )
        {
            C._shell.type = DIST_NODE;
            C._shell.data.N = 
                new Node
                ( C._xSizeSource, C._xSizeTarget, 
                  C._ySizeSource, C._ySizeTarget, C._zSize );
            // TODO
        }
        else if( teamSize == 2 )
        {
            C._shell.type = DIST_NODE; 
            C._shell.data.N = 
                new Node
                ( C._xSizeSource, C._xSizeTarget,
                  C._ySizeSource, C._ySizeTarget, C._zSize );
            // TODO
        }
        else // teamSize == 1
        {
            if( C._sourceOffset == C._targetOffset )
            {
                C._shell.type = NODE;
                C._shell.data.N = 
                    new Node
                    ( C._xSizeSource, C._xSizeTarget,
                      C._ySizeSource, C._ySizeTarget, C._zSize );
                // TODO
            }
            else
            {
                C._shell.type = SPLIT_NODE;
                C._shell.data.N = 
                    new Node
                    ( C._xSizeSource, C._xSizeTarget,
                      C._ySizeSource, C._ySizeTarget, C._zSize );
                // TODO
            }
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}
