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
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::MapVectorPrecompute
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::MapVectorPrecompute");
#endif
    // Clear the context
    switch( context._shell.type )
    {
    case NODE: 
        delete context._shell.data.N; break;

    case SPLIT_LOW_RANK: 
    case SPLIT_DENSE: 
        delete context._shell.data.z; break;
    }

    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        context._shell.type = NODE;
        context._shell.data.N = new typename MapVectorContext::NodeContext();
        typename MapVectorContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                xLocalSub.LockedView( xLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).MapVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocalSub, yLocalSub );
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
        context._shell.data.z = new Vector<Scalar>();

        if( _ownSourceSide )
        {
            const DenseMatrix<Scalar>& D = shell.data.SF->D;
            Vector<Scalar>& z = *context._shell.data.z;

            if( Conjugated )
                hmatrix_tools::MatrixHermitianTransposeVector
                ( alpha, D, xLocal, z );
            else
                hmatrix_tools::MatrixTransposeVector( alpha, D, xLocal, z );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        context._shell.type = SPLIT_DENSE;
        context._shell.data.z = new Vector<Scalar>();

        if( _ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            Vector<Scalar>& z = *context._shell.data.z;
            hmatrix_tools::MatrixVector( alpha, SD.D, xLocal, z );
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
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorPrecompute
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::TransposeMapVectorPrecompute");
#endif
    // Clear the context
    switch( context._shell.type )
    {
    case NODE: 
        delete context._shell.data.N; break;

    case SPLIT_LOW_RANK: 
    case SPLIT_DENSE: 
        delete context._shell.data.z; break;
    }

    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        context._shell.type = NODE;
        context._shell.data.N = new typename MapVectorContext::NodeContext();
        typename MapVectorContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).TransposeMapVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocalSub, yLocalSub );
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
        context._shell.data.z = new Vector<Scalar>();

        if( !_ownSourceSide )
        {
            const DenseMatrix<Scalar>& D = shell.data.SF->D;
            Vector<Scalar>& z = *context._shell.data.z;
            hmatrix_tools::MatrixTransposeVector( alpha, D, xLocal, z );
        }
        break;
    }
    case SPLIT_DENSE:
        context._shell.type = SPLIT_DENSE;
        context._shell.data.z = new Vector<Scalar>();
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::
HermitianTransposeMapVectorPrecompute
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::HermitianTransposeMapVectorPrecompute");
#endif
    // Clear the context
    switch( context._shell.type )
    {
    case NODE: 
        delete context._shell.data.N; break;

    case SPLIT_LOW_RANK: 
    case SPLIT_DENSE: 
        delete context._shell.data.z; break;
    }

    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        context._shell.type = NODE;
        context._shell.data.N = new typename MapVectorContext::NodeContext();
        typename MapVectorContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).HermitianTransposeMapVectorPrecompute
                ( nodeContext.Child(t,s), alpha, xLocalSub, yLocalSub );
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
        context._shell.data.z = new Vector<Scalar>();
        if( !_ownSourceSide )
        {
            const DenseMatrix<Scalar>& D = shell.data.SF->D;
            Vector<Scalar>& z = *context._shell.data.z;

            hmatrix_tools::MatrixHermitianTransposeVector
            ( alpha, D, xLocal, z );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        context._shell.type = SPLIT_DENSE;
        context._shell.data.z = new Vector<Scalar>();
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::MapVectorNaivePassData
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::MapVectorNaivePassData");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        typename MapVectorContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                xLocalSub.LockedView( xLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).MapVectorNaivePassData
                ( nodeContext.Child(t,s), alpha, xLocalSub, yLocalSub );
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
        Vector<Scalar>& z = *context._shell.data.z;
        if( _ownSourceSide )
        {
            mpi::Send( z.LockedBuffer(), SF.rank, _partner, 0, _comm );
        }
        else
        {
            z.Resize( SF.rank );
            mpi::Recv( z.Buffer(), SF.rank, _partner, 0, _comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        Vector<Scalar>& z = *context._shell.data.z;
        if( _ownSourceSide )
            mpi::Send( z.LockedBuffer(), _height, _partner, 0, _comm );
        else
        {
            z.Resize( _height );
            mpi::Recv( z.Buffer(), _height, _partner, 0, _comm );
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
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorNaivePassData
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::TransposeMapVectorNaivePassData");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        typename MapVectorContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).TransposeMapVectorNaivePassData
                ( nodeContext.Child(t,s), alpha, xLocalSub, yLocalSub );
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
        Vector<Scalar>& z = *context._shell.data.z;
        if( !_ownSourceSide )
            mpi::Send( z.LockedBuffer(), SF.rank, _partner, 0, _comm );
        else
        {
            z.Resize( SF.rank );
            mpi::Recv( z.Buffer(), SF.rank, _partner, 0, _comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        Vector<Scalar>& z = *context._shell.data.z;
        if( !_ownSourceSide )
            mpi::Send( xLocal.LockedBuffer(), _height, _partner, 0, _comm );
        else
        {
            z.Resize( _height );
            mpi::Recv( z.Buffer(), _height, _partner, 0, _comm );
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
HermitianTransposeMapVectorNaivePassData
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("SplitQuasi2dHMatrix::HermitianTransposeMapVectorNaivePassData");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        typename MapVectorContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).HermitianTransposeMapVectorNaivePassData
                ( nodeContext.Child(t,s), alpha, xLocalSub, yLocalSub );
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
        Vector<Scalar>& z = *context._shell.data.z;
        if( !_ownSourceSide )
            mpi::Send( z.LockedBuffer(), SF.rank, _partner, 0, _comm );
        else
        {
            z.Resize( SF.rank );
            mpi::Recv( z.Buffer(), SF.rank, _partner, 0, _comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        Vector<Scalar>& z = *context._shell.data.z;
        if( !_ownSourceSide )
            mpi::Send( xLocal.LockedBuffer(), _height, _partner, 0, _comm );
        else
        {
            z.Resize( _height );
            mpi::Recv( z.Buffer(), _height, _partner, 0, _comm );
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
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::MapVectorPostcompute
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::MapVectorPostcompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        typename MapVectorContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                xLocalSub.LockedView( xLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).MapVectorPostcompute
                ( nodeContext.Child(t,s), alpha, xLocalSub, yLocalSub );
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
            const Vector<Scalar>& z = *context._shell.data.z;

            hmatrix_tools::MatrixVector
            ( (Scalar)1, SF.D, z, (Scalar)1, yLocal );
        }
        break;
    case SPLIT_DENSE:
        if( !_ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            const Vector<Scalar>& z = *context._shell.data.z;

            const int localHeight = _height;
            const Scalar* zBuffer = z.LockedBuffer();
            Scalar* yLocalBuffer = yLocal.Buffer();
            for( int i=0; i<localHeight; ++i )
                yLocalBuffer[i] += zBuffer[i];
        }
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorPostcompute
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::TransposeMapVectorPostcompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        typename MapVectorContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
                node.Child(t,s).TransposeMapVectorPostcompute
                ( nodeContext.Child(t,s), alpha, xLocalSub, yLocalSub );
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
            Vector<Scalar>& z = *context._shell.data.z;
            if( Conjugated )
            {
                // yLocal += conj(V) z
                hmatrix_tools::Conjugate( z );
                hmatrix_tools::Conjugate( yLocal );
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, z, (Scalar)1, yLocal );
                hmatrix_tools::Conjugate( yLocal );
            }
            else
            {
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, z, (Scalar)1, yLocal );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            const Vector<Scalar>& z = *context._shell.data.z;
            hmatrix_tools::MatrixTransposeVector
            ( alpha, SD.D, z, (Scalar)1, yLocal );
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
HermitianTransposeMapVectorPostcompute
( MapVectorContext& context,
  Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack
    ("SplitQuasi2dHMatrix::HermitianTransposeMapVectorPostcompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        typename MapVectorContext::NodeContext& nodeContext = 
            *context._shell.data.N;

        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
                node.Child(t,s).HermitianTransposeMapVectorPostcompute
                ( nodeContext.Child(t,s), alpha, xLocalSub, yLocalSub );
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
            Vector<Scalar>& z = *context._shell.data.z;

            if( Conjugated )
            {
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, z, (Scalar)1, yLocal );
            }
            else
            {
                // yLocal += conj(V) z
                hmatrix_tools::Conjugate( z );
                hmatrix_tools::Conjugate( yLocal );
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, z, (Scalar)1, yLocal );
                hmatrix_tools::Conjugate( yLocal );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            const Vector<Scalar>& z = *context._shell.data.z;

            hmatrix_tools::MatrixHermitianTransposeVector
            ( alpha, SD.D, z, (Scalar)1, yLocal );
        }
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

