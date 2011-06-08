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
psp::Quasi2dHMat<Scalar,Conjugated>::Multiply
( Scalar alpha, const Dense<Scalar>& B, 
  Scalar beta,        Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Multiply (D := H D + D)");
#endif
    hmat_tools::Scale( beta, C );
    switch( _block.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_block.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            Dense<Scalar> CSub;
            CSub.View( C, tOffset, 0, node.targetSizes[t], C.Width() );

            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                Dense<Scalar> BSub;
                BSub.LockedView
                ( B, sOffset, 0, node.sourceSizes[s], B.Width() );

                node.Child(t,s).Multiply( alpha, BSub, (Scalar)1, CSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateWithNodeSymmetric( alpha, B, C );
        break;
    case LOW_RANK:
        hmat_tools::Multiply( alpha, *_block.data.F, B, (Scalar)1, C );
        break;
    case DENSE:
        hmat_tools::Multiply( alpha, *_block.data.D, B, (Scalar)1, C );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::Multiply
( Scalar alpha, const Dense<Scalar>& B,
                      Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Multiply (D := H D)");
#endif
    C.SetType( GENERAL );
    C.Resize( Height(), B.Width() );
    Multiply( alpha, B, 0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::TransposeMultiply
( Scalar alpha, const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::TransposeMultiply (D := H^T D + D)");
#endif
    hmat_tools::Scale( beta, C );
    switch( _block.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_block.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Dense<Scalar> CSub;
            CSub.View( C, tOffset, 0, node.sourceSizes[t], C.Width() );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Dense<Scalar> BSub;
                BSub.LockedView
                ( B, sOffset, 0, node.targetSizes[s], B.Width() );

                node.Child(s,t).TransposeMultiply
                ( alpha, BSub, (Scalar)1, CSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateWithNodeSymmetric( alpha, B, C );
        break;
    case LOW_RANK:
        hmat_tools::TransposeMultiply
        ( alpha, *_block.data.F, B, (Scalar)1, C );
        break;
    case DENSE:
        hmat_tools::TransposeMultiply
        ( alpha, *_block.data.D, B, (Scalar)1, C );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::TransposeMultiply
( Scalar alpha, const Dense<Scalar>& B,
                      Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::TransposeMultiply (D := H^T D)");
#endif
    C.SetType( GENERAL );
    C.Resize( Width(), B.Width() );
    TransposeMultiply( alpha, B, 0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::AdjointMultiply
( Scalar alpha, const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::AdjointMultiply (D := H^H D + D)");
#endif
    hmat_tools::Scale( beta, C );
    switch( _block.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_block.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Dense<Scalar> CSub;
            CSub.View( C, tOffset, 0, node.sourceSizes[t], C.Width() );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Dense<Scalar> BSub;
                BSub.LockedView
                ( B, sOffset, 0, node.targetSizes[s], B.Width() );

                node.Child(s,t).AdjointMultiply
                ( alpha, BSub, (Scalar)1, CSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        Dense<Scalar> BConj;
        hmat_tools::Conjugate( B, BConj );
        hmat_tools::Conjugate( C );
        UpdateWithNodeSymmetric( alpha, B, C );
        hmat_tools::Conjugate( C );
        break;
    }
    case LOW_RANK:
        hmat_tools::AdjointMultiply( alpha, *_block.data.F, B, (Scalar)1, C );
        break;
    case DENSE:
        hmat_tools::AdjointMultiply( alpha, *_block.data.D, B, (Scalar)1, C );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// This version allows for temporary in-place conjugation of B
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::AdjointMultiply
( Scalar alpha, Dense<Scalar>& B,
  Scalar beta,  Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::AdjointMultiply (D := H^H D + D, non-const)");
#endif
    hmat_tools::Scale( beta, C );
    switch( _block.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_block.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Dense<Scalar> CSub;
            CSub.View( C, tOffset, 0, node.sourceSizes[t], C.Width() );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Dense<Scalar> BSub;
                BSub.LockedView
                ( B, sOffset, 0, node.targetSizes[s], B.Width() );

                node.Child(s,t).AdjointMultiply
                ( alpha, BSub, (Scalar)1, CSub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        hmat_tools::Conjugate( B );
        hmat_tools::Conjugate( C );
        UpdateWithNodeSymmetric( alpha, B, C );
        hmat_tools::Conjugate( B );
        hmat_tools::Conjugate( C );
        break;
    case LOW_RANK:
        hmat_tools::AdjointMultiply( alpha, *_block.data.F, B, (Scalar)1, C );
        break;
    case DENSE:
        hmat_tools::AdjointMultiply( alpha, *_block.data.D, B, (Scalar)1, C );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::AdjointMultiply
( Scalar alpha, const Dense<Scalar>& B,
                      Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::AdjointMultiply (D := H^H D)");
#endif
    C.SetType( GENERAL );
    C.Resize( Width(), B.Width() );
    AdjointMultiply( alpha, B, 0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

// This version allows for temporary in-place conjugation of B
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::AdjointMultiply
( Scalar alpha, Dense<Scalar>& B,
                Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::AdjointMultiply (D := H^H D, non-const)");
#endif
    C.SetType( GENERAL );
    C.Resize( Width(), B.Width() );
    AdjointMultiply( alpha, B, 0, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

