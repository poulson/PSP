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
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapVector
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::MapVector (y := H x + y)");
#endif
    hmatrix_tools::Scale( beta, y );
    switch( _shell.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_shell.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, tOffset, node.targetSizes[t] );

            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sOffset, node.sourceSizes[s] );

                node.Child(t,s).MapVector( alpha, xSub, (Scalar)1, ySub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateVectorWithNodeSymmetric( alpha, x, y );
        break;
    case LOW_RANK:
        hmatrix_tools::MatrixVector( alpha, *_shell.data.F, x, (Scalar)1, y );
        break;
    case DENSE:
        hmatrix_tools::MatrixVector( alpha, *_shell.data.D, x, (Scalar)1, y );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapVector
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::MapVector (y := H x)");
#endif
    y.Resize( Height() );
    MapVector( alpha, x, 0, y );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::TransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::TransposeMapVector (y := H^T x + y)");
#endif
    hmatrix_tools::Scale( beta, y );
    switch( _shell.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_shell.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, tOffset, node.sourceSizes[t] );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sOffset, node.targetSizes[s] );

                node.Child(s,t).TransposeMapVector
                ( alpha, xSub, (Scalar)1, ySub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        UpdateVectorWithNodeSymmetric( alpha, x, y );
        break;
    case LOW_RANK:
        hmatrix_tools::MatrixTransposeVector
        ( alpha, *_shell.data.F, x, (Scalar)1, y );
        break;
    case DENSE:
        hmatrix_tools::MatrixTransposeVector
        ( alpha, *_shell.data.D, x, (Scalar)1, y );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::TransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::TransposeMapVector (y := H^T x)");
#endif
    y.Resize( Width() );
    TransposeMapVector( alpha, x, 0, y );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack
    ("Quasi2dHMatrix::HermitianTransposeMapVector (y := H^H x + y)");
#endif
    hmatrix_tools::Scale( beta, y );
    switch( _shell.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_shell.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, tOffset, node.sourceSizes[t] );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sOffset, node.targetSizes[s] );

                node.Child(s,t).HermitianTransposeMapVector
                ( alpha, xSub, (Scalar)1, ySub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        Vector<Scalar> xConj;
        hmatrix_tools::Conjugate( x, xConj );
        hmatrix_tools::Conjugate( y );
        UpdateVectorWithNodeSymmetric( Conj(alpha), xConj, y ); 
        hmatrix_tools::Conjugate( y );
        break;
    }
    case LOW_RANK:
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, *_shell.data.F, x, (Scalar)1, y );
        break;
    case DENSE:
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, *_shell.data.D, x, (Scalar)1, y );
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
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack
    ("Quasi2dHMatrix::HermitianTransposeMapVector (y := H^H x + y, non-const)");
#endif
    hmatrix_tools::Scale( beta, y );
    switch( _shell.type )
    {
    case NODE:
    {
        // Loop over all 16 children, summing in each row
        Node& node = *_shell.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sourceSizes[t],++t )
        {
            Vector<Scalar> ySub;
            ySub.View( y, tOffset, node.sourceSizes[t] );

            for( int s=0,sOffset=0; s<4; sOffset+=node.targetSizes[s],++s )
            {
                Vector<Scalar> xSub;
                xSub.LockedView( x, sOffset, node.targetSizes[s] );

                node.Child(s,t).HermitianTransposeMapVector
                ( alpha, xSub, (Scalar)1, ySub );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
        hmatrix_tools::Conjugate( x );
        hmatrix_tools::Conjugate( y );
        UpdateVectorWithNodeSymmetric( Conj(alpha), x, y ); 
        hmatrix_tools::Conjugate( x );
        hmatrix_tools::Conjugate( y );
        break;
    case LOW_RANK:
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, *_shell.data.F, x, (Scalar)1, y );
        break;
    case DENSE:
        hmatrix_tools::MatrixHermitianTransposeVector
        ( alpha, *_shell.data.D, x, (Scalar)1, y );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::HermitianTransposeMapVector (y := H^H x)");
#endif
    y.Resize( Width() );
    HermitianTransposeMapVector( alpha, x, 0, y );
#ifndef RELEASE
    PopCallStack();
#endif
}

// This version allows for temporary in-place conjugation of x
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeMapVector
( Scalar alpha, Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack
    ("Quasi2dHMatrix::HermitianTransposeMapVector (y := H^H x, non-const)");
#endif
    y.Resize( Width() );
    HermitianTransposeMapVector( alpha, x, 0, y );
#ifndef RELEASE
    PopCallStack();
#endif
}

