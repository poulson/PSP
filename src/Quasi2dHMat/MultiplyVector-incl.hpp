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
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Multiply (y := H x + y)");
#endif
    hmat_tools::Scale( beta, y );
    switch( _block.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_block.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, tOffset, node.targetSizes[t] );

            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sOffset, node.sourceSizes[s] );

                node.Child(t,s).Multiply( alpha, xSub, (Scalar)1, ySub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateVectorWithNodeSymmetric( alpha, x, y );
        break;
    case LOW_RANK:
        hmat_tools::Multiply( alpha, *_block.data.F, x, (Scalar)1, y );
        break;
    case DENSE:
        hmat_tools::Multiply( alpha, *_block.data.D, x, (Scalar)1, y );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::Multiply
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Multiply (y := H x)");
#endif
    y.Resize( Height() );
    Multiply( alpha, x, 0, y );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::TransposeMultiply
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::TransposeMultiply (y := H^T x + y)");
#endif
    hmat_tools::Scale( beta, y );
    switch( _block.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_block.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, tOffset, node.sourceSizes[t] );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sOffset, node.targetSizes[s] );

                node.Child(s,t).TransposeMultiply
                ( alpha, xSub, (Scalar)1, ySub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateVectorWithNodeSymmetric( alpha, x, y );
        break;
    case LOW_RANK:
        hmat_tools::TransposeMultiply( alpha, *_block.data.F, x, (Scalar)1, y );
        break;
    case DENSE:
        hmat_tools::TransposeMultiply( alpha, *_block.data.D, x, (Scalar)1, y );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::TransposeMultiply
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::TransposeMultiply (y := H^T x)");
#endif
    y.Resize( Width() );
    TransposeMultiply( alpha, x, 0, y );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::AdjointMultiply
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::AdjointMultiply (y := H^H x + y)");
#endif
    hmat_tools::Scale( beta, y );
    switch( _block.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_block.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, tOffset, node.sourceSizes[t] );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sOffset, node.targetSizes[s] );

                node.Child(s,t).AdjointMultiply
                ( alpha, xSub, (Scalar)1, ySub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        Vector<Scalar> xConj;
        hmat_tools::Conjugate( x, xConj );
        hmat_tools::Conjugate( y );
        UpdateVectorWithNodeSymmetric( Conj(alpha), xConj, y ); 
        hmat_tools::Conjugate( y );
        break;
    }
    case LOW_RANK:
        hmat_tools::AdjointMultiply
        ( alpha, *_block.data.F, x, (Scalar)1, y );
        break;
    case DENSE:
        hmat_tools::AdjointMultiply
        ( alpha, *_block.data.D, x, (Scalar)1, y );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Having a non-const x allows us to conjugate x in place for the 
// NODE_SYMMETRIC updates.
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::AdjointMultiply
( Scalar alpha, Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack
    ("Quasi2dHMat::AdjointMultiply (y := H^H x + y, non-const)");
#endif
    hmat_tools::Scale( beta, y );
    switch( _block.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_block.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, tOffset, node.sourceSizes[t] );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sOffset, node.targetSizes[s] );

                node.Child(s,t).AdjointMultiply
                ( alpha, xSub, (Scalar)1, ySub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        hmat_tools::Conjugate( x );
        hmat_tools::Conjugate( y );
        UpdateVectorWithNodeSymmetric( Conj(alpha), x, y ); 
        hmat_tools::Conjugate( x );
        hmat_tools::Conjugate( y );
        break;
    case LOW_RANK:
        hmat_tools::AdjointMultiply( alpha, *_block.data.F, x, (Scalar)1, y );
        break;
    case DENSE:
        hmat_tools::AdjointMultiply( alpha, *_block.data.D, x, (Scalar)1, y );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::AdjointMultiply
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::AdjointMultiply (y := H^H x)");
#endif
    y.Resize( Width() );
    AdjointMultiply( alpha, x, 0, y );
#ifndef RELEASE
    PopCallStack();
#endif
}

// This version allows for temporary in-place conjugation of x
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::AdjointMultiply
( Scalar alpha, Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::AdjointMultiply (y := H^H x, non-const)");
#endif
    y.Resize( Width() );
    AdjointMultiply( alpha, x, 0, y );
#ifndef RELEASE
    PopCallStack();
#endif
}

