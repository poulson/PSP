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
    if( this->Width() != B.Height() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( this->NumLevels() != B.NumLevels() )
        throw std::logic_error
        ("Can only multiply H-matrices with same number of levels.");
#endif
    C._height = this->Height();
    C._width = B.Width();
    C._numLevels = this->NumLevels();
    C._maxRank = this->MaxRank();
    C._sourceOffset = B.SourceOffset();
    C._targetOffset = this->TargetOffset();
    C._symmetric = false;
    C._stronglyAdmissible = 
        ( this->StronglyAdmissible() || B.StronglyAdmissible() );

    C._xSizeSource = B.XSizeSource();
    C._ySizeSource = B.YSizeSource();
    C._xSizeTarget = this->XSizeTarget();
    C._ySizeTarget = this->YSizeTarget();
    C._zSize = this->ZSize();
    C._xSource = B.XSource();
    C._ySource = B.YSource();
    C._xTarget = this->XTarget();
    C._yTarget = this->YTarget();

    C._shell.Clear();
    if( C.Admissible() )
    {
        C._shell.type = LOW_RANK;
        C._shell.data.F = new LowRankMatrix<Scalar,Conjugated>;
        if( this->IsLowRank() && B.IsLowRank() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.F, *C._shell.data.F );
        else if( this->IsLowRank() && B.IsHierarchical() )
        {
            hmatrix_tools::Copy( _shell.data.F->U, C._shell.data.F->U );
            if( Conjugated )
                B.HermitianTransposeMapMatrix
                ( Conj(alpha), _shell.data.F->V, C._shell.data.F->V );
            else
                B.TransposeMapMatrix
                ( alpha, _shell.data.F->V, C._shell.data.F->V );
        }
        else if( this->IsLowRank() && B.IsDense() )
        {
            hmatrix_tools::Copy( _shell.data.F->U, C._shell.data.F->U );
            if( Conjugated )
                hmatrix_tools::MatrixHermitianTransposeMatrix
                ( Conj(alpha), *B._shell.data.D, _shell.data.F->V, 
                  C._shell.data.F->V );
            else
                hmatrix_tools::MatrixTransposeMatrix
                ( alpha, *B._shell.data.D, _shell.data.F->V,
                  C._shell.data.F->V );
        }
        else if( this->IsHierarchical() && B.IsLowRank() )
        {
            // C.F.U := alpha A B.F.U
            this->MapMatrix( alpha, B._shell.data.F->U, C._shell.data.F->U );
            // C.F.V := B.F.V
            hmatrix_tools::Copy( B._shell.data.F->V, C._shell.data.F->V );
        }
        else if( this->IsHierarchical() && B.IsHierarchical() )
        {
            // C.F := alpha H H
            const int oversampling = 4; // lift this definition
            hmatrix_tools::MatrixMatrix
            ( oversampling, alpha, *this, B, *C._shell.data.F );
        }
        else if( this->IsDense() && B.IsLowRank() )
        {
            // C.F.U := alpha A B.F.U
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.D, B._shell.data.F->U, C._shell.data.F->U );
            // C.F.V := B.F.V
            hmatrix_tools::Copy( B._shell.data.F->V, C._shell.data.F->V );
        }
        else if( this->IsDense() && B.IsDense() )
            hmatrix_tools::MatrixMatrix
            ( C.MaxRank(),
              alpha, *_shell.data.D, *B._shell.data.D, *C._shell.data.F );
#ifndef RELEASE
        else
            std::logic_error("Invalid H-matrix combination.");
#endif
    }
    else if( C.NumLevels() > 1 )
    {
        // A product of two matrices will be assumed non-symmetric.
        C._shell.type = NODE;
        C._shell.data.N = 
            new Node
            ( C._xSizeSource, C._xSizeTarget, C._ySizeSource, C._ySizeTarget,
              C._zSize );

#ifndef RELEASE
        if( this->IsDense() || B.IsDense() )
            throw std::logic_error("Invalid H-matrix combination");
#endif
        if( this->IsLowRank() && B.IsLowRank() )
        {
            // Form W := alpha A B
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.F, W );

            // Form C :~= W
            C.ImportLowRankMatrix( W );
        }
        else if( this->IsLowRank() && B.IsHierarchical() )
        {
            // Form W := alpha A B
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::Copy( _shell.data.F->U, W.U );
            if( Conjugated )
                B.HermitianTransposeMapMatrix
                ( Conj(alpha), _shell.data.F->V, W.V );
            else
                B.TransposeMapMatrix
                ( alpha, _shell.data.F->V, W.V );

            // Form C :=~ W
            C.ImportLowRankMatrix( W );
        }
        else if( this->IsHierarchical() && B.IsLowRank() )
        {
            // Form W := alpha A B    
            LowRankMatrix<Scalar,Conjugated> W;
            this->MapMatrix( alpha, B._shell.data.F->U, W.U );
            hmatrix_tools::Copy( B._shell.data.F->V, W.V );

            // Form C :=~ W
            C.ImportLowRankMatrix( W );
        }
        else
        {
#ifndef RELEASE
            if( this->Symmetric() || B.Symmetric() )
                throw std::logic_error("Unsupported h-matrix multipy case.");
#endif
            const Node& nodeA = *this->_shell.data.N;
            const Node& nodeB = *B._shell.data.N;
            Node& nodeC = *C._shell.data.N;

            for( int t=0; t<4; ++t )
            {
                for( int s=0; s<4; ++s )
                {
                    // Create the H-matrix here
                    nodeC.children[s+4*t] = 
                        new Quasi2dHMatrix<Scalar,Conjugated>;

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
        if( this->IsHierarchical() || B.IsHierarchical() )
            throw std::logic_error("Invalid combination of H-matrices.");
#endif
        C._shell.type = DENSE;
        C._shell.data.D = new DenseMatrix<Scalar>;

        if( this->IsDense() && B.IsDense() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.D, *B._shell.data.D, *C._shell.data.D );
        else if( this->IsDense() && B.IsLowRank() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.D, *B._shell.data.F, *C._shell.data.D );
        else if( this->IsLowRank() && B.IsDense() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.D, *C._shell.data.D );
        else /* both low-rank */
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.F, *C._shell.data.D );
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
    if( this->Width() != B.Height() || 
        this->Height() != C.Height() || B.Width() != C.Width() )
        throw std::logic_error("Attempted nonconformal matrix-matrix multiply");
    if( this->NumLevels() != B.NumLevels() || 
        this->NumLevels() != C.NumLevels() )
        throw std::logic_error
        ("Can only multiply H-matrices with same number of levels.");
    if( C.Symmetric() )
        throw std::logic_error("Symmetric updates not yet supported.");
#endif
    if( C.Admissible() )
    {
        if( this->IsLowRank() && B.IsLowRank() )
        {
            // W := alpha A.F B.F
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.F, W );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._shell.data.F );
        }
        else if( this->IsLowRank() && B.IsHierarchical() )
        {
            // W := alpha A.F B
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::Copy( _shell.data.F->U, W.U );
            if( Conjugated )
                B.HermitianTransposeMapMatrix
                ( Conj(alpha), _shell.data.F->V, W.V );
            else
                B.TransposeMapMatrix
                ( alpha, _shell.data.F->V, W.V );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._shell.data.F );
        }
        else if( this->IsLowRank() && B.IsDense() )
        {
            // W := alpha A.F B.D
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::Copy( _shell.data.F->U, W.U );
            if( Conjugated )
                hmatrix_tools::MatrixHermitianTransposeMatrix
                ( Conj(alpha), *B._shell.data.D, _shell.data.F->V, W.V );
            else
                hmatrix_tools::MatrixTransposeMatrix
                ( alpha, *B._shell.data.D, _shell.data.F->V, W.V );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._shell.data.F );
        }
        else if( this->IsHierarchical() && B.IsLowRank() )
        {
            // W := alpha A B.F
            LowRankMatrix<Scalar,Conjugated> W;
            this->MapMatrix( alpha, B._shell.data.F->U, W.U );
            hmatrix_tools::Copy( B._shell.data.F->V, W.V );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._shell.data.F );
        }
        else if( this->IsHierarchical() && B.IsHierarchical() )
        {
            // W := alpha A B
            LowRankMatrix<Scalar,Conjugated> W;
            const int oversampling = 4; // lift this definition
            hmatrix_tools::MatrixMatrix
            ( oversampling, alpha, *this, B, W );

            // C.F :~= W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._shell.data.F );
        }
        else if( this->IsDense() && B.IsLowRank() )
        {
            // W := alpha A.D B.F
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.D, B._shell.data.F->U, W.U );
            hmatrix_tools::Copy( B._shell.data.F->V, W.V );

            // C.F :=~ W + beta C.F
            hmatrix_tools::MatrixUpdateRounded
            ( C.MaxRank(), (Scalar)1, W, beta, *C._shell.data.F );
        }
        else if( this->IsDense() && B.IsDense() )
            hmatrix_tools::MatrixMatrix
            ( C.MaxRank(),
              alpha, *_shell.data.D, *B._shell.data.D, beta, *C._shell.data.F );
#ifndef RELEASE
        else
            std::logic_error("Invalid H-matrix combination.");
#endif
    }
    else if( C.NumLevels() > 1 )
    {

#ifndef RELEASE
        if( this->IsDense() || B.IsDense() )
            throw std::logic_error("Invalid H-matrix combination");
#endif
        if( this->IsLowRank() && B.IsLowRank() )
        {
            // Form W := alpha A B 
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.F, W );

            // C :~= W + beta C
            C.Scale( beta );
            C.UpdateWithLowRankMatrix( (Scalar)1, W );
        }
        else if( this->IsLowRank() && B.IsHierarchical() )
        {
            // Form W := alpha A B
            LowRankMatrix<Scalar,Conjugated> W;
            hmatrix_tools::Copy( _shell.data.F->U, W.U );
            if( Conjugated )
                B.HermitianTransposeMapMatrix
                ( Conj(alpha), _shell.data.F->V, W.V );
            else
                B.TransposeMapMatrix
                ( alpha, _shell.data.F->V, W.V );

            // C :~= W + beta C
            C.Scale( beta );
            C.UpdateWithLowRankMatrix( (Scalar)1, W );
        }
        else if( this->IsHierarchical() && B.IsLowRank() )
        {
            // Form W := alpha A B    
            LowRankMatrix<Scalar,Conjugated> W;
            this->MapMatrix( alpha, B._shell.data.F->U, W.U );
            hmatrix_tools::Copy( B._shell.data.F->V, W.V );

            // Form C :~= W + beta C
            C.Scale( beta );
            C.UpdateWithLowRankMatrix( (Scalar)1, W );
        }
        else
        {
            if( this->Symmetric() || B.Symmetric() )
                throw std::logic_error("Unsupported h-matrix multipy case.");
            else 
            {
                const Node& nodeA = *this->_shell.data.N;
                const Node& nodeB = *B._shell.data.N;
                Node& nodeC = *C._shell.data.N;

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
        if( this->IsHierarchical() || B.IsHierarchical() )
            throw std::logic_error("Invalid combination of H-matrices.");
#endif
        if( this->IsDense() && B.IsDense() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.D, *B._shell.data.D, beta, *C._shell.data.D );
        else if( this->IsDense() && B.IsLowRank() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.D, *B._shell.data.F, beta, *C._shell.data.D );
        else if( this->IsLowRank() && B.IsDense() )
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.D, beta, *C._shell.data.D );
        else /* both low-rank */
            hmatrix_tools::MatrixMatrix
            ( alpha, *_shell.data.F, *B._shell.data.F, beta, *C._shell.data.D );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

