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
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::MapVectorPrecompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                xLocalSub.LockedView( xLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).MapVectorPrecompute
                ( alpha, xLocalSub, yLocalSub );
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
            if( Conjugated )
            {
                hmatrix_tools::MatrixHermitianTransposeVector
                ( alpha, SF.D, xLocal, SF.z );
            }
            else
            {
                hmatrix_tools::MatrixTransposeVector
                ( alpha, SF.D, xLocal, SF.z );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            hmatrix_tools::MatrixVector( alpha, SD.D, xLocal, SD.z );
        }
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::TransposeMapVectorPrecompute
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::TransposeMapVectorPrecompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).TransposeMapVectorPrecompute
                ( alpha, xLocalSub, yLocalSub );
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
            hmatrix_tools::MatrixTransposeVector( alpha, SF.D, xLocal, SF.z );
        }
        break;
    case SPLIT_DENSE:
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
( Scalar alpha, const Vector<Scalar>& xLocal,
                      Vector<Scalar>& yLocal ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::HermitianTransposeMapVectorPrecompute");
#endif
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).HermitianTransposeMapVectorPrecompute
                ( alpha, xLocalSub, yLocalSub );
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
            hmatrix_tools::MatrixHermitianTransposeVector
            ( alpha, SF.D, xLocal, SF.z );
        }
        break;
    case SPLIT_DENSE:
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::SplitQuasi2dHMatrix<Scalar,Conjugated>::MapVectorNaivePassData
( Scalar alpha, const Vector<Scalar>& xLocal,
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
        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                xLocalSub.LockedView( xLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).MapVectorNaivePassData
                ( alpha, xLocalSub, yLocalSub );
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
        if( _ownSourceSide )
            mpi::Send( SF.z.LockedBuffer(), SF.rank, _partner, 0, _comm );
        else
        {
            SF.z.Resize( SF.rank );
            mpi::Recv( SF.z.Buffer(), SF.rank, _partner, 0, _comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        if( _ownSourceSide )
            mpi::Send( SD.z.LockedBuffer(), _height, _partner, 0, _comm );
        else
        {
            SD.z.Resize( _height );
            mpi::Recv( SD.z.Buffer(), _height, _partner, 0, _comm );
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
( Scalar alpha, const Vector<Scalar>& xLocal,
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
        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).TransposeMapVectorNaivePassData
                ( alpha, xLocalSub, yLocalSub );
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
        if( !_ownSourceSide )
            mpi::Send( SF.z.LockedBuffer(), SF.rank, _partner, 0, _comm );
        else
        {
            SF.z.Resize( SF.rank );
            mpi::Recv( SF.z.Buffer(), SF.rank, _partner, 0, _comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        if( !_ownSourceSide )
            mpi::Send( xLocal.LockedBuffer(), _height, _partner, 0, _comm );
        else
        {
            SD.z.Resize( _height );
            mpi::Recv( SD.z.Buffer(), _height, _partner, 0, _comm );
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
( Scalar alpha, const Vector<Scalar>& xLocal,
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
        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).HermitianTransposeMapVectorNaivePassData
                ( alpha, xLocalSub, yLocalSub );
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
        if( !_ownSourceSide )
            mpi::Send( SF.z.LockedBuffer(), SF.rank, _partner, 0, _comm );
        else
        {
            SF.z.Resize( SF.rank );
            mpi::Recv( SF.z.Buffer(), SF.rank, _partner, 0, _comm );
        }
        break;
    }
    case SPLIT_DENSE:
    {
        const SplitDenseMatrix& SD = *shell.data.SD;
        if( !_ownSourceSide )
            mpi::Send( xLocal.LockedBuffer(), _height, _partner, 0, _comm );
        else
        {
            SD.z.Resize( _height );
            mpi::Recv( SD.z.Buffer(), _height, _partner, 0, _comm );
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
( Scalar alpha, const Vector<Scalar>& xLocal,
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
        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            yLocalSub.View( yLocal, tOffset, node.targetSizes[t] );
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                xLocalSub.LockedView( xLocal, sOffset, node.sourceSizes[s] );
                node.Child(t,s).MapVectorPostcompute
                ( alpha, xLocalSub, yLocalSub );
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
            hmatrix_tools::MatrixVector
            ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocal );
        }
        break;
    case SPLIT_DENSE:
        if( !_ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            const int localHeight = _height;
            const Scalar* zBuffer = SD.z.LockedBuffer();
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
( Scalar alpha, const Vector<Scalar>& xLocal,
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
        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
                node.Child(t,s).TransposeMapVectorPostcompute
                ( alpha, xLocalSub, yLocalSub );
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
            if( Conjugated )
            {
                // yLocal += conj(V) z
                hmatrix_tools::Conjugate( SF.z );
                hmatrix_tools::Conjugate( yLocal );
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocal );
                hmatrix_tools::Conjugate( yLocal );
            }
            else
            {
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocal );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            hmatrix_tools::MatrixTransposeVector
            ( alpha, SD.D, SD.z, (Scalar)1, yLocal );
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
( Scalar alpha, const Vector<Scalar>& xLocal,
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
        const Node& node = *shell.data.N;
        Vector<Scalar> xLocalSub, yLocalSub;
        for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
        {
            yLocalSub.View( yLocal, sOffset, node.sourceSizes[s] );
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                xLocalSub.LockedView( xLocal, tOffset, node.targetSizes[t] );
                node.Child(t,s).HermitianTransposeMapVectorPostcompute
                ( alpha, xLocalSub, yLocalSub );
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
            if( Conjugated )
            {
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocal );
            }
            else
            {
                // yLocal += conj(V) z
                hmatrix_tools::Conjugate( SF.z );
                hmatrix_tools::Conjugate( yLocal );
                hmatrix_tools::MatrixVector
                ( (Scalar)1, SF.D, SF.z, (Scalar)1, yLocal );
                hmatrix_tools::Conjugate( yLocal );
            }
        }
        break;
    case SPLIT_DENSE:
        if( _ownSourceSide )
        {
            const SplitDenseMatrix& SD = *shell.data.SD;
            hmatrix_tools::MatrixHermitianTransposeVector
            ( alpha, SD.D, SD.z, (Scalar)1, yLocal );
        }
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

