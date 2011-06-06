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
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B,
                      Quasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::MapMatrix H := H H");
    if( Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( _numLevels != B._numLevels )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( _zSize != B._zSize )
        throw std::logic_error("Mismatched z size");
#endif
    const Quasi2d& A = *this;

    C._numLevels = A._numLevels;
    C._maxRank = A._maxRank;
    C._sourceOffset = B._sourceOffset;
    C._targetOffset = A._targetOffset;
    C._symmetric = false;
    C._stronglyAdmissible = ( A._stronglyAdmissible || B._stronglyAdmissible );

    C._xSizeSource = B._xSizeSource;
    C._ySizeSource = B._ySizeSource;
    C._xSizeTarget = A._xSizeTarget;
    C._ySizeTarget = A._ySizeTarget;
    C._zSize = A._zSize;
    C._xSource = B._xSource;
    C._ySource = B._ySource;
    C._xTarget = A._xTarget;
    C._yTarget = A._yTarget;

    C._block.Clear();
    if( C.Admissible() )
    {
        C._block.type = LOW_RANK;
        C._block.data.F = new LowRankMatrix<Scalar,Conjugated>;
        if( A.IsLowRank() && B.IsLowRank() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.F, *B._block.data.F, *C._block.data.F );
        else if( A.IsLowRank() && B.IsHierarchical() )
        {
            hmatrix_tools::Copy( A._block.data.F->U, C._block.data.F->U );
            if( Conjugated )
                B.AdjointMapMatrix
                ( Conj(alpha), A._block.data.F->V, C._block.data.F->V );
            else
                B.TransposeMapMatrix
                ( alpha, A._block.data.F->V, C._block.data.F->V );
        }
        else if( A.IsLowRank() && B.IsDense() )
        {
            hmatrix_tools::Copy( A._block.data.F->U, C._block.data.F->U );
            if( Conjugated )
                hmatrix_tools::MatrixAdjointMatrix
                ( Conj(alpha), *B._block.data.D, A._block.data.F->V, 
                  C._block.data.F->V );
            else
                hmatrix_tools::MatrixTransposeMatrix
                ( alpha, *B._block.data.D, A._block.data.F->V,
                  C._block.data.F->V );
        }
        else if( A.IsHierarchical() && B.IsLowRank() )
        {
            // C.F.U := alpha A B.F.U
            MapMatrix( alpha, B._block.data.F->U, C._block.data.F->U );
            // C.F.V := B.F.V
            hmatrix_tools::Copy( B._block.data.F->V, C._block.data.F->V );
        }
        else if( A.IsHierarchical() && B.IsHierarchical() )
        {
            // C.F := alpha H H
            const int oversampling = 4; // TODO: Lift this definition
            hmatrix_tools::MatrixMatrix
            ( oversampling, alpha, *this, B, *C._block.data.F );
        }
        else if( A.IsDense() && B.IsLowRank() )
        {
            // C.F.U := alpha A B.F.U
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.D, 
              B._block.data.F->U, C._block.data.F->U );
            // C.F.V := B.F.V
            hmatrix_tools::Copy( B._block.data.F->V, C._block.data.F->V );
        }
        else if( A.IsDense() && B.IsDense() )
            hmatrix_tools::MatrixMatrix
            ( C.MaxRank(),
              alpha, *A._block.data.D, *B._block.data.D, *C._block.data.F );
#ifndef RELEASE
        else
            std::logic_error("Invalid H-matrix combination");
#endif
    }
    else if( C.NumLevels() > 1 )
    {
        // A product of two matrices will be assumed non-symmetric.
        C._block.type = NODE;
        C._block.data.N = 
            new Node
            ( C._xSizeSource, C._xSizeTarget, C._ySizeSource, C._ySizeTarget,
              C._zSize );

#ifndef RELEASE
        if( A.IsDense() || B.IsDense() )
            throw std::logic_error("Invalid H-matrix combination");
#endif
        if( A.IsLowRank() && B.IsLowRank() )
        {
            // Form W := alpha A B
            LowRank W;
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.F, *B._block.data.F, W );

            // Form C :~= W
            C.ImportLowRankMatrix( W );
        }
        else if( A.IsLowRank() && B.IsHierarchical() )
        {
            // Form W := alpha A B
            LowRank W;
            hmatrix_tools::Copy( A._block.data.F->U, W.U );
            if( Conjugated )
                B.AdjointMapMatrix
                ( Conj(alpha), A._block.data.F->V, W.V );
            else
                B.TransposeMapMatrix
                ( alpha, A._block.data.F->V, W.V );

            // Form C :=~ W
            C.ImportLowRankMatrix( W );
        }
        else if( A.IsHierarchical() && B.IsLowRank() )
        {
            // Form W := alpha A B    
            LowRank W;
            MapMatrix( alpha, B._block.data.F->U, W.U );
            hmatrix_tools::Copy( B._block.data.F->V, W.V );

            // Form C :=~ W
            C.ImportLowRankMatrix( W );
        }
        else
        {
#ifndef RELEASE
            if( A.Symmetric() || B.Symmetric() )
                throw std::logic_error("Unsupported h-matrix multipy case.");
#endif
            const Node& nodeA = *A._block.data.N;
            const Node& nodeB = *B._block.data.N;
            Node& nodeC = *C._block.data.N;

            for( int t=0; t<4; ++t )
            {
                for( int s=0; s<4; ++s )
                {
                    // Create the H-matrix here
                    nodeC.children[s+4*t] = new Quasi2d;

                    // Initialize the [t,s] box of C with the first product
                    nodeA.Child(t,0).MapMatrix
                    ( alpha, nodeB.Child(0,s), nodeC.Child(t,s) );

                    // Add the other three products onto it
                    for( int u=1; u<4; ++u )
                        nodeA.Child(t,u).MapMatrix
                        ( alpha, nodeB.Child(u,s), 1, nodeC.Child(t,s) );
                }
            }
        }
    }
    else /* C is dense */
    {
#ifndef RELEASE
        if( A.IsHierarchical() || B.IsHierarchical() )
            throw std::logic_error("Invalid combination of H-matrices.");
#endif
        C._block.type = DENSE;
        C._block.data.D = new Dense;

        if( A.IsDense() && B.IsDense() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.D, *B._block.data.D, *C._block.data.D );
        else if( A.IsDense() && B.IsLowRank() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.D, *B._block.data.F, *C._block.data.D );
        else if( A.IsLowRank() && B.IsDense() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.F, *B._block.data.D, *C._block.data.D );
        else /* both low-rank */
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.F, *B._block.data.F, *C._block.data.D );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// C := alpha A B + beta C
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::MapMatrix
( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B,
  Scalar beta,        Quasi2dHMatrix<Scalar,Conjugated>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::MapMatrix (H := H H + H)");
    if( Width() != B.Height() || 
        Height() != C.Height() || B.Width() != C.Width() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( NumLevels() != B.NumLevels() || 
        NumLevels() != C.NumLevels() )
        throw std::logic_error
        ("Can only multiply H-matrices with same number of levels.");
    if( C.Symmetric() )
        throw std::logic_error("Symmetric updates not yet supported.");
#endif
    const Quasi2d& A = *this;
    if( C.Admissible() )
    {
        if( A.IsLowRank() && B.IsLowRank() )
        {
            // W := alpha A.F B.F
            LowRank W;
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.F, *B._block.data.F, W );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._block.data.F );
        }
        else if( A.IsLowRank() && B.IsHierarchical() )
        {
            // W := alpha A.F B
            LowRank W;
            hmatrix_tools::Copy( A._block.data.F->U, W.U );
            if( Conjugated )
                B.AdjointMapMatrix
                ( Conj(alpha), A._block.data.F->V, W.V );
            else
                B.TransposeMapMatrix
                ( alpha, A._block.data.F->V, W.V );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._block.data.F );
        }
        else if( A.IsLowRank() && B.IsDense() )
        {
            // W := alpha A.F B.D
            LowRank W;
            hmatrix_tools::Copy( A._block.data.F->U, W.U );
            if( Conjugated )
                hmatrix_tools::MatrixAdjointMatrix
                ( Conj(alpha), *B._block.data.D, A._block.data.F->V, W.V );
            else
                hmatrix_tools::MatrixTransposeMatrix
                ( alpha, *B._block.data.D, A._block.data.F->V, W.V );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._block.data.F );
        }
        else if( A.IsHierarchical() && B.IsLowRank() )
        {
            // W := alpha A B.F
            LowRank W;
            MapMatrix( alpha, B._block.data.F->U, W.U );
            hmatrix_tools::Copy( B._block.data.F->V, W.V );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._block.data.F );
        }
        else if( A.IsHierarchical() && B.IsHierarchical() )
        {
            // W := alpha A B
            LowRank W;
            const int oversampling = 4; // TODO: Lift this definition
            hmatrix_tools::MatrixMatrix
            ( oversampling, alpha, *this, B, W );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._block.data.F );
        }
        else if( A.IsDense() && B.IsLowRank() )
        {
            // W := alpha A.D B.F
            LowRank W;
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.D, B._block.data.F->U, W.U );
            hmatrix_tools::Copy( B._block.data.F->V, W.V );

            // C.F :=~ W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._block.data.F );
        }
        else if( A.IsDense() && B.IsDense() )
            hmatrix_tools::MatrixMatrix
            ( C.MaxRank(),
              alpha, *A._block.data.D, *B._block.data.D, 
              beta, *C._block.data.F );
#ifndef RELEASE
        else
            std::logic_error("Invalid H-matrix combination.");
#endif
    }
    else if( C.NumLevels() > 1 )
    {

#ifndef RELEASE
        if( A.IsDense() || B.IsDense() )
            throw std::logic_error("Invalid H-matrix combination");
#endif
        if( A.IsLowRank() && B.IsLowRank() )
        {
            // Form W := alpha A B 
            LowRank W;
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.F, *B._block.data.F, W );

            // C :~= W + beta C
            C.Scale( beta );
            C.UpdateWithLowRankMatrix( (Scalar)1, W );
        }
        else if( A.IsLowRank() && B.IsHierarchical() )
        {
            // Form W := alpha A B
            LowRank W;
            hmatrix_tools::Copy( A._block.data.F->U, W.U );
            if( Conjugated )
                B.AdjointMapMatrix
                ( Conj(alpha), A._block.data.F->V, W.V );
            else
                B.TransposeMapMatrix
                ( alpha, A._block.data.F->V, W.V );

            // C :~= W + beta C
            C.Scale( beta );
            C.UpdateWithLowRankMatrix( (Scalar)1, W );
        }
        else if( A.IsHierarchical() && B.IsLowRank() )
        {
            // Form W := alpha A B    
            LowRank W;
            MapMatrix( alpha, B._block.data.F->U, W.U );
            hmatrix_tools::Copy( B._block.data.F->V, W.V );

            // Form C :~= W + beta C
            C.Scale( beta );
            C.UpdateWithLowRankMatrix( (Scalar)1, W );
        }
        else
        {
            if( A.Symmetric() || B.Symmetric() )
                throw std::logic_error("Unsupported h-matrix multipy case.");
            else 
            {
                const Node& nodeA = *A._block.data.N;
                const Node& nodeB = *B._block.data.N;
                Node& nodeC = *C._block.data.N;

                for( int t=0; t<4; ++t )
                {
                    for( int s=0; s<4; ++s )
                    {
                        // Scale the [t,s] box of C in the first product
                        nodeA.Child(t,0).MapMatrix
                        ( alpha, nodeB.Child(0,s), beta, nodeC.Child(t,s) ); 
        
                        // Add the other three products onto it
                        for( int u=1; u<4; ++u )
                            nodeA.Child(t,u).MapMatrix
                            ( alpha, nodeB.Child(u,s), 
                              (Scalar)1, nodeC.Child(t,s) ); 
                    }
                }
            }
        }
    }
    else /* C is dense */
    {
#ifndef RELEASE
        if( A.IsHierarchical() || B.IsHierarchical() )
            throw std::logic_error("Invalid combination of H-matrices.");
#endif
        if( A.IsDense() && B.IsDense() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.D, *B._block.data.D, 
              beta, *C._block.data.D );
        else if( A.IsDense() && B.IsLowRank() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.D, *B._block.data.F, 
              beta, *C._block.data.D );
        else if( A.IsLowRank() && B.IsDense() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.F, *B._block.data.D, 
              beta, *C._block.data.D );
        else /* both low-rank */
            hmatrix_tools::MatrixMatrix
            ( alpha, *A._block.data.F, *B._block.data.F, 
              beta, *C._block.data.D );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

