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

// A := inv(A) using recursive Schur complements
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::DirectInvert()
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::DirectInvert");
    if( this->Height() != this->Width() )
        throw std::logic_error("Cannot invert non-square matrices");
    if( this->IsLowRank() )
        throw std::logic_error("Cannot invert low-rank matrices");
#endif
    switch( _shell.type )
    {
    case NODE:
    {
        // We will form the inverse in the original matrix, so we only need to
        // create a temporary matrix.
        Quasi2dHMatrix<Scalar,Conjugated> B; 
        B.CopyFrom( *this );

        // Initialize our soon-to-be inverse as the identity
        this->SetToIdentity();

        Node& nodeA = *_shell.data.N;
        Node& nodeB = *B._shell.data.N;

        for( int l=0; l<4; ++l )
        {
            // A_ll := inv(B_ll)
            nodeA.Child(l,l).CopyFrom( nodeB.Child(l,l) );
            nodeA.Child(l,l).DirectInvert();

            // NOTE: Can be skipped for upper-triangular matrices
            for( int j=0; j<l; ++j )
            {
                // A_lj := A_ll A_lj
                Quasi2dHMatrix<Scalar,Conjugated> C;
                C.CopyFrom( nodeA.Child(l,j) );
                nodeA.Child(l,l).MapMatrix( (Scalar)1, C, nodeA.Child(l,j) );
            }

            // NOTE: Can be skipped for lower-triangular matrices
            for( int j=l+1; j<4; ++j )
            {
                // B_lj := A_ll B_lj
                Quasi2dHMatrix<Scalar,Conjugated> C;
                C.CopyFrom( nodeB.Child(l,j) );
                nodeA.Child(l,l).MapMatrix( (Scalar)1, C, nodeB.Child(l,j) );
            }

            for( int i=l+1; i<4; ++i )
            {
                // NOTE: Can be skipped for upper triangular matrices.
                for( int j=0; j<=l; ++j )
                {
                    // A_ij -= B_il A_lj
                    nodeB.Child(i,l).MapMatrix
                    ( (Scalar)-1, nodeA.Child(l,j), 
                      (Scalar)1,  nodeA.Child(i,j) );
                }
                // NOTE: Can be skipped for either lower or upper-triangular
                //       matrices, effectively decoupling the diagonal block
                //       inversions.
                for( int j=l+1; j<4; ++j )
                {
                    // B_ij -= B_il B_lj
                    nodeB.Child(i,l).MapMatrix
                    ( (Scalar)-1, nodeB.Child(l,j),
                      (Scalar)1,  nodeB.Child(i,j) );
                }
            }
        }

        // NOTE: Can be skipped for lower-triangular matrices.
        for( int l=3; l>=0; --l )
        {
            for( int i=l-1; i>=0; --i )
            {
                // NOTE: For upper-triangular matrices, change the loop to
                //       for( int j=l; j<4; ++j )
                for( int j=0; j<4; ++j )
                {
                    // A_ij -= B_il A_lj
                    nodeB.Child(i,l).MapMatrix
                    ( (Scalar)-1, nodeA.Child(l,j),
                      (Scalar)1,  nodeA.Child(i,j) );
                }
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
#ifndef RELEASE
        throw std::logic_error("Symmetric inversion not yet supported.");
#endif
        break;
    }
    case DENSE:
        hmatrix_tools::Invert( *_shell.data.D );
        break;
    case LOW_RANK:
    {
#ifndef RELEASE
        throw std::logic_error("Mistake in inversion code.");
#endif
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := inv(A) using Schulz iterations, X_k+1 := (2I - X_k A) X_k
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::SchulzInvert
( int numIterations, 
  typename RealBase<Scalar>::type theta, 
  typename RealBase<Scalar>::type confidence )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::SchulzInvert");
    if( this->Height() != this->Width() )
        throw std::logic_error("Cannot invert non-square matrices");
    if( this->IsLowRank() )
        throw std::logic_error("Cannot invert low-rank matrices");
    if( theta <= 1 )
        throw std::logic_error("Theta must be > 1");
    if( confidence <= 0 )
        throw std::logic_error("Confidence must be positive");
#endif
    if( numIterations <= 0 )
        throw std::logic_error("Must use at least 1 iteration.");

    const Scalar estimate = 
        hmatrix_tools::EstimateTwoNorm( *this, theta, confidence );
    const Scalar alpha = ((Scalar)2) / (estimate*estimate);

    // Initialize X_0 := alpha A^H
    Quasi2dHMatrix<Scalar,Conjugated> X;
    X.HermitianTransposeFrom( *this );
    X.Scale( alpha );

    for( int k=0; k<numIterations; ++k )
    {
        // Form Z := 2I - X_k A
        Quasi2dHMatrix<Scalar,Conjugated> Z;        
        X.MapMatrix( (Scalar)-1, *this, Z );
        Z.AddConstantToDiagonal( (Scalar)2 );

        // Form X_k+1 := Z X_k = (2I - X_k A) X_k
        Quasi2dHMatrix<Scalar,Conjugated> XCopy;
        XCopy.CopyFrom( X );
        Z.MapMatrix( (Scalar)1, XCopy, X );
    }

    this->CopyFrom( X );
#ifndef RELEASE
    PopCallStack();
#endif
}

