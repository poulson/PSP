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

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixPrecompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::MapMatrixPrecompute");
    if( XLocal.Type() != GENERAL )
        throw std::logic_error("Can only map general matrices.");
#endif
    // Clear the context
    switch( context._shell.type )
    {
    case NODE:
        delete context._shell.data.N; break;

    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
        delete context._shell.data.Z; break;
    }

    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        context._shell.type = NODE;
        context._shell.data.N = new typename MapDenseMatrixContext::NodeContext();

        const Node& node = *shell.data.N;
        typename MapDenseMatrixContext::NodeContext& nodeContext = 
            *context._shell.data.N;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            YLocalSub.View( YLocal, tOffset, 0, node.targetSizes[t], width );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                XLocalSub.LockedView
                ( XLocal, sOffset, 0, node.sourceSizes[s], width );
                node.Child(t,s).MapMatrixPrecompute
                ( nodeContext.Child(t,s), alpha, XLocalSub, YLocalSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
    {
        context._shell.type = SPLIT_LOW_RANK;
        context._shell.data.Z = new DenseMatrix<Scalar>();

        if( _ownSourceSide )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;

            if( Conjugated )
            {
                hmatrix_tools::MatrixHermitianTransposeMatrix
                ( alpha, SF.D, XLocal, Z );
            }
            else
            {
                hmatrix_tools::MatrixTransposeMatrix
                ( alpha, SF.D, XLocal, Z );
            }
        }
        break;
    }
    case SPLIT_DENSE:
    {
        context._shell.type = SPLIT_DENSE;
        context._shell.data.Z = new DenseMatrix<Scalar>();

        if( _ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            hmatrix_tools::MatrixMatrix( alpha, SD.D, XLocal, Z );
        }
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixPrecompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::TransposeMapMatrixPrecompute");
#endif
    // Clear the context
    switch( context._shell.type )
    {
    case NODE:
        delete context._shell.data.N; break;

    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
        delete context._shell.data.Z; break;
    }

    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        context._shell.type = NODE;
        context._shell.data.N = new typename MapDenseMatrixContext::NodeContext();
        typename MapDenseMatrixContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            XLocalSub.LockedView
            ( XLocal, tOffset, 0, node.targetSizes[t], width );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                YLocalSub.View
                ( YLocal, sOffset, 0, node.sourceSizes[s], width );
                node.Child(t,s).TransposeMapMatrixPrecompute
                ( nodeContext.Child(t,s), alpha, XLocalSub, YLocalSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
    {
        context._shell.type = SPLIT_LOW_RANK;
        context._shell.data.Z = new DenseMatrix<Scalar>();

        if( !_ownSourceSide )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;

            hmatrix_tools::MatrixTransposeMatrix( alpha, SF.D, XLocal, Z );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        context._shell.type = SPLIT_DENSE;
        context._shell.data.Z = new DenseMatrix<Scalar>();
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixPrecompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::HermitianTransposeMapMatrixPrecompute");
#endif
    // Clear the context
    switch( context._shell.type )
    {
    case NODE:
        delete context._shell.data.N; break;

    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
        delete context._shell.data.Z; break;
    }

    const int width = YLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        context._shell.type = NODE;
        context._shell.data.N = new typename MapDenseMatrixContext::NodeContext();
        typename MapDenseMatrixContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            XLocalSub.LockedView
            ( XLocal, tOffset, 0, node.targetSizes[t], width );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                YLocalSub.View
                ( YLocal, sOffset, 0, node.sourceSizes[s], width );
                node.Child(t,s).HermitianTransposeMapMatrixPrecompute
                ( nodeContext.Child(t,s), alpha, XLocalSub, YLocalSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
    {
        context._shell.type = SPLIT_LOW_RANK;
        context._shell.data.Z = new DenseMatrix<Scalar>();
        if( !_ownSourceSide )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            hmatrix_tools::MatrixHermitianTransposeMatrix
            ( alpha, SF.D, XLocal, Z );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        context._shell.type = SPLIT_DENSE;
        context._shell.data.Z = new DenseMatrix<Scalar>();
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixNaivePassData
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::MapMatrixNaivePassData");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        typename MapDenseMatrixContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            YLocalSub.View( YLocal, tOffset, 0, node.targetSizes[t], width );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                XLocalSub.LockedView
                ( XLocal, sOffset, 0, node.sourceSizes[s], width );
                node.Child(t,s).MapMatrixNaivePassData
                ( nodeContext.Child(t,s), alpha, XLocalSub, YLocalSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        DenseMatrix<Scalar>& Z = *context._shell.data.Z;
        // We can safely assume that SF.Z has its ldim equal to its height
        if( _ownSourceSide )
        {
            mpi::Send
            ( Z.LockedBuffer(), SF.rank*width, _partner, 0, _comm );
        }
        else
        {
            Z.Resize( SF.rank, width, SF.rank );
            mpi::Recv( Z.Buffer(), SF.rank*width, _partner, 0, _comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        DenseMatrix<Scalar>& Z = *context._shell.data.Z;
        // We can safely assume that SF.Z has its ldim equal to its height
        if( _ownSourceSide )
        {
            mpi::Send
            ( Z.LockedBuffer(), _height*width, _partner, 0, _comm );
        }
        else
        {
            Z.Resize( _height, width, _height );
            mpi::Recv( Z.Buffer(), _height*width, _partner, 0, _comm );
        }
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixNaivePassData
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::TransposeMapMatrixNaivePassData");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        typename MapDenseMatrixContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            XLocalSub.LockedView
            ( XLocal, tOffset, 0, node.targetSizes[t], width );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                YLocalSub.View
                ( YLocal, sOffset, 0, node.sourceSizes[s], width );
                node.Child(t,s).TransposeMapMatrixNaivePassData
                ( nodeContext.Child(t,s), alpha, XLocalSub, YLocalSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        DenseMatrix<Scalar>& Z = *context._shell.data.Z;
        if( !_ownSourceSide )
        {
            // We can safely assume Z's ldim is equal to its height
            mpi::Send
            ( Z.LockedBuffer(), SF.rank*width, _partner, 0, _comm );
        }
        else
        {
            Z.Resize( SF.rank, width, SF.rank );
            mpi::Recv( Z.Buffer(), SF.rank*width, _partner, 0, _comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        DenseMatrix<Scalar>& Z = *context._shell.data.Z;
        if( !_ownSourceSide )
        {
            if( XLocal.Height() != XLocal.LDim() )
            {
                // We must pack XLocal since it's not contiguous in memory
                Z.Resize( _height, width, _height );
                for( int j=0; j<width; ++j )
                {
                    std::memcpy
                    ( Z.Buffer(0,j), XLocal.LockedBuffer(0,j), 
                      _height*sizeof(Scalar) );
                }
                mpi::Send
                ( Z.LockedBuffer(), _height*width, _partner, 0, _comm );
            }
            else
            {
                mpi::Send
                ( XLocal.LockedBuffer(), _height*width, _partner, 0, _comm );
            }
        }
        else
        {
            Z.Resize( _height, width, _height );
            mpi::Recv( Z.Buffer(), _height*width, _partner, 0, _comm );
        }
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixNaivePassData
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("SplitQuasi2dHMatrix::HermitianTransposeMapMatrixNaivePassData");
#endif
    const int width = XLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        typename MapDenseMatrixContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            XLocalSub.LockedView
            ( XLocal, tOffset, 0, node.targetSizes[t], width );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                YLocalSub.View
                ( YLocal, sOffset, 0, node.sourceSizes[s], width );
                node.Child(t,s).HermitianTransposeMapMatrixNaivePassData
                ( nodeContext.Child(t,s), alpha, XLocalSub, YLocalSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
    {
        const SplitLowRankMatrix& SF = *shell.data.SF;
        DenseMatrix<Scalar>& Z = *context._shell.data.Z;
        if( !_ownSourceSide )
        {
            // We can safely assume Z's ldim is equal to its height
            mpi::Send( Z.LockedBuffer(), SF.rank*width, _partner, 0, _comm );
        }
        else
        {
            Z.Resize( SF.rank, width, SF.rank );
            mpi::Recv( Z.Buffer(), SF.rank*width, _partner, 0, _comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        DenseMatrix<Scalar>& Z = *context._shell.data.Z;
        if( !_ownSourceSide )
        {
            if( XLocal.LDim() != XLocal.Height() )
            {
                Z.Resize( _height, width, _height );
                for( int j=0; j<width; ++j )
                {
                    std::memcpy
                    ( Z.Buffer(0,j), XLocal.LockedBuffer(0,j), 
                      _height*sizeof(Scalar) );
                }
                mpi::Send
                ( Z.LockedBuffer(), _height*width, _partner, 0, _comm );
            }
            else
            {
                mpi::Send
                ( XLocal.LockedBuffer(), _height*width, _partner, 0, _comm );
            }
        }
        else
        {
            Z.Resize( _height, width, _height );
            mpi::Recv( Z.Buffer(), _height*width, _partner, 0, _comm );
        }
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::MapMatrixPostcompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::MapMatrixPostcompute");
#endif
    const int width = YLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        typename MapDenseMatrixContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            YLocalSub.View( YLocal, tOffset, 0, node.targetSizes[t], width );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                XLocalSub.LockedView
                ( XLocal, sOffset, 0, node.sourceSizes[s], width );
                node.Child(t,s).MapMatrixPostcompute
                ( nodeContext.Child(t,s), alpha, XLocalSub, YLocalSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
        if( !_ownSourceSide )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            const DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            hmatrix_tools::MatrixMatrix
            ( (Scalar)1, SF.D, Z, (Scalar)1, YLocal );
        }
        break;
    case SPLIT_DENSE:
        if( !_ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            const DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            const int localHeight = _height;
            for( int j=0; j<width; ++j )
            {
                Scalar* YLocalCol = YLocal.Buffer(0,j);
                const Scalar* ZCol = Z.LockedBuffer(0,j);
                for( int i=0; i<localHeight; ++i )
                    YLocalCol[i] += ZCol[i];
            }
        }
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapMatrixPostcompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::TransposeMapMatrixPostcompute");
#endif
    const int width = YLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        typename MapDenseMatrixContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            YLocalSub.View( YLocal, sOffset, 0, node.sourceSizes[s], width );
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                XLocalSub.LockedView
                ( XLocal, tOffset, 0, node.targetSizes[t], width );
                node.Child(t,s).TransposeMapMatrixPostcompute
                ( nodeContext.Child(t,s), alpha, XLocalSub, YLocalSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
        if( _ownSourceSide )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            if( Conjugated )
            {
                // YLocal += conj(V) Z
                hmatrix_tools::Conjugate( Z );
                hmatrix_tools::Conjugate( YLocal );
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, Z, (Scalar)1, YLocal );
                hmatrix_tools::Conjugate( YLocal );
            }
            else
            {
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, Z, (Scalar)1, YLocal );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            const DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            hmatrix_tools::MatrixTransposeMatrix
            ( alpha, SD.D, Z, (Scalar)1, YLocal );
        }
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapMatrixPostcompute
( MapDenseMatrixContext& context,
  Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                      DenseMatrix<Scalar>& YLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("SplitQuasi2dHMatrix::HermitianTransposeMapMatrixPostcompute");
#endif
    const int width = YLocal.Width();
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        typename MapDenseMatrixContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        DenseMatrix<Scalar> XLocalSub, YLocalSub;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            YLocalSub.View( YLocal, sOffset, 0, node.sourceSizes[s], width );
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                XLocalSub.LockedView
                ( XLocal, tOffset, 0, node.targetSizes[t], width );
                node.Child(t,s).HermitianTransposeMapMatrixPostcompute
                ( nodeContext.Child(t,s), alpha, XLocalSub, YLocalSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
#ifndef RELEASE
        throw std::logic_error("Symmetric case not yet supported");
#endif
        break;
    case SPLIT_LOW_RANK:
        if( _ownSourceSide )
        {
            const SplitLowRankMatrix& SF = *shell.data.SF;
            DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            if( Conjugated )
            {
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, Z, (Scalar)1, YLocal );
            }
            else
            {
                // YLocal += conj(V) Z
                hmatrix_tools::Conjugate( Z );
                hmatrix_tools::Conjugate( YLocal );
                hmatrix_tools::MatrixMatrix
                ( (Scalar)1, SF.D, Z, (Scalar)1, YLocal );
                hmatrix_tools::Conjugate( YLocal );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            const DenseMatrix<Scalar>& Z = *context._shell.data.Z;
            hmatrix_tools::MatrixHermitianTransposeMatrix
            ( alpha, SD.D, Z, (Scalar)1, YLocal );
        }
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

