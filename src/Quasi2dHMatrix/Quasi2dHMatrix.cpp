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
#include "psp.hpp"

#include "./Invert-incl.hpp"
#include "./MapDenseMatrix-incl.hpp"
#include "./MapHMatrix-incl.hpp"
#include "./MapVector-incl.hpp"
#include "./Pack-incl.hpp"

//----------------------------------------------------------------------------//
// Public static routines                                                     //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void 
psp::Quasi2dHMatrix<Scalar,Conjugated>::BuildMapOnQuadrant
( int* map, int& index, int level, int numLevels,
  int xSize, int ySize, int zSize, int thisXSize, int thisYSize )
{
    if( level == numLevels-1 )
    {
        // Stamp these indices into the buffer 
        for( int k=0; k<zSize; ++k )
        {
            for( int j=0; j<thisYSize; ++j )
            {
                int* thisRow = &map[k*xSize*ySize+j*xSize];
                for( int i=0; i<thisXSize; ++i )
                    thisRow[i] = index++;
            }
        }
    }
    else
    {
        const int leftWidth = thisXSize/2;
        const int rightWidth = thisXSize - leftWidth;
        const int bottomHeight = thisYSize/2;
        const int topHeight = thisYSize - bottomHeight;

        // Recurse on the lower-left quadrant 
        BuildMapOnQuadrant
        ( &map[0], index, level+1, numLevels,
          xSize, ySize, zSize, leftWidth, bottomHeight );
        // Recurse on the lower-right quadrant
        BuildMapOnQuadrant
        ( &map[leftWidth], index, level+1, numLevels,
          xSize, ySize, zSize, rightWidth, bottomHeight );
        // Recurse on the upper-left quadrant
        BuildMapOnQuadrant
        ( &map[bottomHeight*xSize], index, level+1, numLevels,
          xSize, ySize, zSize, leftWidth, topHeight );
        // Recurse on the upper-right quadrant
        BuildMapOnQuadrant
        ( &map[bottomHeight*xSize+leftWidth], index, level+1, numLevels,
          xSize, ySize, zSize, rightWidth, topHeight );
    }
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::BuildNaturalToHierarchicalMap
( std::vector<int>& map, int xSize, int ySize, int zSize, int numLevels )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::BuildNaturalToHierarchicalMap");
#endif
    map.resize( xSize*ySize*zSize );

    // Fill the mapping from the 'natural' x-y-z ordering
    int index = 0;
    BuildMapOnQuadrant
    ( &map[0], index, 0, numLevels, xSize, ySize, zSize, xSize, ySize );
#ifndef RELEASE
    if( index != xSize*ySize*zSize )
        throw std::logic_error("Map recursion is incorrect.");
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

// Create an empty H-matrix
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix()
: AbstractHMatrix<Scalar>(),
  _xSizeSource(0), _xSizeTarget(0),
  _ySizeSource(0), _ySizeTarget(0),
  _zSize(0),
  _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0)
{ }

// Create a square top-level H-matrix
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix
( const LowRankMatrix<Scalar,Conjugated>& F,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSize, int ySize, int zSize )
: AbstractHMatrix<Scalar>
  (F.Height(),F.Width(),numLevels,maxRank,0,0,false,stronglyAdmissible),
  _xSizeSource(xSize), _xSizeTarget(xSize),
  _ySizeSource(ySize), _ySizeTarget(ySize),
  _zSize(zSize),
  _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0)
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Quasi2dHMatrix");
#endif
    ImportLowRankMatrix( F );
#ifndef RELEASE
    PopCallStack();
#endif
}
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix
( const SparseMatrix<Scalar>& S,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSize, int ySize, int zSize )
: AbstractHMatrix<Scalar>
  (S.height,S.width,numLevels,maxRank,0,0,S.symmetric,stronglyAdmissible),
  _xSizeSource(xSize), _xSizeTarget(xSize),
  _ySizeSource(ySize), _ySizeTarget(ySize),
  _zSize(zSize),
  _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0)
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Quasi2dHMatrix");
#endif
    ImportSparseMatrix( S );
#ifndef RELEASE
    PopCallStack();
#endif
}

// Create a potentially non-square non-top-level H-matrix
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix
( const LowRankMatrix<Scalar,Conjugated>& F,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int zSize,
  int xSource, int xTarget,
  int ySource, int yTarget,
  int sourceOffset, int targetOffset )
: AbstractHMatrix<Scalar>
  (xSizeTarget*ySizeTarget*zSize,xSizeSource*ySizeSource*zSize,
   numLevels,maxRank,sourceOffset,targetOffset,false,stronglyAdmissible),
  _xSizeSource(xSizeSource), _xSizeTarget(xSizeTarget),
  _ySizeSource(ySizeSource), _ySizeTarget(ySizeTarget),
  _zSize(zSize),
  _xSource(xSource), _xTarget(xTarget),
  _ySource(ySource), _yTarget(yTarget)
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Quasi2dHMatrix");
#endif
    ImportLowRankMatrix( F );
#ifndef RELEASE
    PopCallStack();
#endif
}
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix
( const SparseMatrix<Scalar>& S,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int zSize,
  int xSource, int xTarget,
  int ySource, int yTarget,
  int sourceOffset, int targetOffset )
: AbstractHMatrix<Scalar>
  (xSizeTarget*ySizeTarget*zSize,xSizeSource*ySizeSource*zSize,
   numLevels,maxRank,sourceOffset,targetOffset,
   (S.symmetric && sourceOffset==targetOffset),stronglyAdmissible),
  _xSizeSource(xSizeSource), _xSizeTarget(xSizeTarget),
  _ySizeSource(ySizeSource), _ySizeTarget(ySizeTarget), 
  _zSize(zSize),
  _xSource(xSource), _xTarget(xTarget),
  _ySource(ySource), _yTarget(yTarget)
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Quasi2dHMatrix");
#endif
    ImportSparseMatrix( S, targetOffset, sourceOffset );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::~Quasi2dHMatrix()
{ }

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::Clear()
{
    _shell.Clear();
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::Print
( const std::string& tag ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Print");
#endif
    Dense I( this->Width(), this->Width() );
    std::memset( I.Buffer(), 0, I.LDim()*I.Width() );
    for( int j=0; j<this->Width(); ++j )
        I.Set(j,j,(Scalar)1);

    Dense HFull;
    this->MapMatrix( (Scalar)1, I, HFull );
    HFull.Print( tag );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::WriteStructure
( const std::string& filename ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::WriteStructure");
#endif
    std::ofstream file( filename.c_str() );
    WriteStructureRecursion( file );
#ifndef RELEASE
    PopCallStack();
#endif
}

/*\
|*| Computational routines specific to Quasi2dHMatrix
\*/

// A := B
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::CopyFrom
( const Quasi2dHMatrix<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::CopyFrom");
#endif
    this->_height = B.Height();
    this->_width = B.Width();
    this->_numLevels = B.NumLevels();
    this->_maxRank = B.MaxRank();
    this->_sourceOffset = B.SourceOffset();
    this->_targetOffset = B.TargetOffset();
    this->_symmetric = B.Symmetric();
    this->_stronglyAdmissible = B.StronglyAdmissible();
    this->_xSizeSource = B.XSizeSource();
    this->_xSizeTarget = B.XSizeTarget();
    this->_ySizeSource = B.YSizeSource();
    this->_ySizeTarget = B.YSizeTarget();
    this->_zSize = B.ZSize();
    this->_xSource = B.XSource();
    this->_xTarget = B.XTarget();
    this->_ySource = B.YSource();
    this->_yTarget = B.YTarget();

    this->_shell.Clear();
    this->_shell.type = B._shell.type;
    switch( _shell.type )
    {
    case NODE:
    {
        this->_shell.data.N = 
            new Node
            ( this->_xSizeSource, this->_xSizeTarget,
              this->_ySizeSource, this->_ySizeTarget, this->_zSize );
        Node& nodeA = *this->_shell.data.N;
        const Node& nodeB = *B._shell.data.N;
        for( int i=0; i<16; ++i )
        {
            nodeA.children[i] = new Quasi2d;
            nodeA.children[i]->CopyFrom( *nodeB.children[i] );
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        this->_shell.data.NS = 
            new NodeSymmetric
            ( this->_xSizeSource, this->_ySizeSource, this->_zSize );
        NodeSymmetric& nodeA = *this->_shell.data.NS;
        const NodeSymmetric& nodeB = *B._shell.data.NS;
        for( int i=0; i<10; ++i )
        {
            nodeA.children[i] = new Quasi2d;
            nodeA.children[i]->CopyFrom( *nodeB.children[i] );
        }
        break;
    }
    case LOW_RANK:
        this->_shell.data.F = new LowRank;
        hmatrix_tools::Copy( *B._shell.data.F, *this->_shell.data.F );
        break;
    case DENSE:
        this->_shell.data.D = new Dense;
        hmatrix_tools::Copy( *B._shell.data.D, *this->_shell.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := Conj(A)
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::Conjugate()
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Conjugate");
#endif
    switch( this->_shell.type )
    {
    case NODE:
        for( int i=0; i<16; ++i )
            this->_shell.data.N->children[i]->Conjugate();
        break;
    case NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
            this->_shell.data.NS->children[i]->Conjugate();
        break;
    case LOW_RANK:
        hmatrix_tools::Conjugate( this->_shell.data.F->U );
        hmatrix_tools::Conjugate( this->_shell.data.F->V );
        break;
    case DENSE:
        hmatrix_tools::Conjugate( *this->_shell.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := Conj(B)
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::ConjugateFrom
( const Quasi2dHMatrix<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::ConjugateFrom");
#endif
    this->_height = B.Height();
    this->_width = B.Width();
    this->_numLevels = B.NumLevels();
    this->_maxRank = B.MaxRank();
    this->_sourceOffset = B.SourceOffset();
    this->_targetOffset = B.TargetOffset();
    this->_symmetric = B.Symmetric();
    this->_stronglyAdmissible = B.StronglyAdmissible();
    this->_xSizeSource = B.XSizeSource();
    this->_xSizeTarget = B.XSizeTarget();
    this->_ySizeSource = B.YSizeSource();
    this->_ySizeTarget = B.YSizeTarget();
    this->_zSize = B.ZSize();
    this->_xSource = B.XSource();
    this->_xTarget = B.XTarget();
    this->_ySource = B.YSource();
    this->_yTarget = B.YTarget();

    this->_shell.Clear();
    this->_shell.type = B._shell.type;
    switch( this->_shell.type )
    {
    case NODE:
        for( int i=0; i<16; ++i )
        {
            this->_shell.data.N->children[i]->ConjugateFrom
            ( *B._shell.data.N->children[i] );
        }
        break;
    case NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
        {
            this->_shell.data.NS->children[i]->ConjugateFrom
            ( *B._shell.data.NS->children[i] );
        }
        break;
    case LOW_RANK:
        hmatrix_tools::Conjugate( B._shell.data.F->U, this->_shell.data.F->U );
        hmatrix_tools::Conjugate( B._shell.data.F->V, this->_shell.data.F->V );
        break;
    case DENSE:
        hmatrix_tools::Conjugate( *B._shell.data.D, *this->_shell.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := B^T
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::TransposeFrom
( const Quasi2dHMatrix<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::TransposeFrom");
#endif
    this->_height = B.Width();
    this->_width = B.Height();
    this->_numLevels = B.NumLevels();
    this->_maxRank = B.MaxRank();
    this->_sourceOffset = B.TargetOffset();
    this->_targetOffset = B.SourceOffset();
    this->_symmetric = B.Symmetric();
    this->_stronglyAdmissible = B.StronglyAdmissible();
    this->_xSizeSource = B.XSizeTarget();
    this->_xSizeTarget = B.XSizeSource();
    this->_ySizeSource = B.YSizeTarget();
    this->_ySizeTarget = B.YSizeSource();
    this->_zSize = B.ZSize();
    this->_xSource = B.XTarget();
    this->_xTarget = B.XSource();
    this->_ySource = B.YTarget();
    this->_yTarget = B.YSource();

    Shell& shell = this->_shell;
    shell.Clear();
    shell.type = B._shell.type;
    switch( shell.type )
    {
    case NODE:
    {
        shell.data.N = 
            new Node
            ( this->_xSizeSource, this->_xSizeTarget,
              this->_ySizeSource, this->_ySizeTarget, this->_zSize );
        Node& nodeA = *shell.data.N;
        const Node& nodeB = *B._shell.data.N;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<4; ++s )
            {
                nodeA.children[s+4*t] = new Quasi2d;
                nodeA.Child(t,s).TransposeFrom( nodeB.Child(s,t) );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        shell.data.NS = 
            new NodeSymmetric
            ( this->_xSizeSource, this->_ySizeSource, this->_zSize );
        NodeSymmetric& nodeA = *shell.data.NS;
        const NodeSymmetric& nodeB = *B._shell.data.NS;
        for( int i=0; i<10; ++i )
        {
            nodeA.children[i] = new Quasi2d;
            nodeA.children[i]->CopyFrom( *nodeB.children[i] );
        }
        break;
    }
    case LOW_RANK:
    {
        shell.data.F = new LowRank;
        hmatrix_tools::Transpose( *B._shell.data.F, *shell.data.F );
        break;
    }
    case DENSE:
    {
        shell.data.D = new Dense;
        hmatrix_tools::Transpose( *B._shell.data.D, *shell.data.D );
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := B^H
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::HermitianTransposeFrom
( const Quasi2dHMatrix<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::HermitianTransposeFrom");
#endif
    this->_height = B.Width();
    this->_width = B.Height();
    this->_numLevels = B.NumLevels();
    this->_maxRank = B.MaxRank();
    this->_sourceOffset = B.TargetOffset();
    this->_targetOffset = B.SourceOffset();
    this->_symmetric = B.Symmetric();
    this->_stronglyAdmissible = B.StronglyAdmissible();
    this->_xSizeSource = B.XSizeTarget();
    this->_xSizeTarget = B.XSizeSource();
    this->_ySizeSource = B.YSizeTarget();
    this->_ySizeTarget = B.YSizeSource();
    this->_zSize = B.ZSize();
    this->_xSource = B.XTarget();
    this->_xTarget = B.XSource();
    this->_ySource = B.YTarget();
    this->_yTarget = B.YSource();

    Shell& shell = this->_shell;
    shell.Clear();
    shell.type = B._shell.type;
    switch( B._shell.type )
    {
    case NODE:
    {
        shell.data.N = 
            new Node
            ( this->_xSizeSource, this->_xSizeTarget,
              this->_ySizeSource, this->_ySizeTarget, this->_zSize );
        Node& nodeA = *shell.data.N;
        const Node& nodeB = *B._shell.data.N;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<4; ++s )
            {
                nodeA.children[s+4*t] = new Quasi2d;
                nodeA.Child(t,s).HermitianTransposeFrom( nodeB.Child(s,t) );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        shell.data.NS = 
            new NodeSymmetric
            ( this->_xSizeSource, this->_ySizeSource, this->_zSize );
        NodeSymmetric& nodeA = *shell.data.NS;
        const NodeSymmetric& nodeB = *B._shell.data.NS;
        for( int i=0; i<10; ++i )
        {
            nodeA.children[i] = new Quasi2d;
            nodeA.children[i]->ConjugateFrom( *nodeB.children[i] );
        }
        break;
    }
    case LOW_RANK:
    {
        shell.data.F = new LowRank;
        hmatrix_tools::HermitianTranspose( *B._shell.data.F, *shell.data.F );
        break;
    }
    case DENSE:
    {
        shell.data.D = new Dense;
        hmatrix_tools::HermitianTranspose( *B._shell.data.D, *shell.data.D );
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := alpha A
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::Scale( Scalar alpha )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Scale");
#endif
    Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        Node& nodeA = *shell.data.N;
        for( int i=0; i<16; ++i )
            nodeA.children[i]->Scale( alpha );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *shell.data.NS;
        for( int i=0; i<10; ++i )
            nodeA.children[i]->Scale( alpha );
        break;
    }
    case LOW_RANK:
        hmatrix_tools::Scale( alpha, *shell.data.F );
        break;
    case DENSE:
        hmatrix_tools::Scale( alpha, *shell.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := I
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::SetToIdentity()
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::SetToIdentity");
#endif
    Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        Node& nodeA = *shell.data.N;
        for( int i=0; i<4; ++i )
        {
            for( int j=0; j<4; ++j )
            {
                if( i == j )
                    nodeA.Child(i,j).SetToIdentity();
                else
                    nodeA.Child(i,j).Scale( (Scalar)0 );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *shell.data.NS;
        for( int i=0; i<4; ++i )
        {
            for( int j=0; j<i; ++j )
                nodeA.Child(i,j).Scale( (Scalar)0 );
            nodeA.Child(i,i).SetToIdentity();
        }
        break;
    }
    case LOW_RANK:
    {
#ifndef RELEASE
        throw std::logic_error("Error in SetToIdentity logic.");
#endif
        break;
    }
    case DENSE:
    {
        Dense& D = *shell.data.D;
        hmatrix_tools::Scale( (Scalar)0, D );
        for( int j=0; j<D.Width(); ++j )
        {
            if( j < D.Height() )
                D.Set( j, j, (Scalar)1 );
            else
                break;
        }
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := A + alpha I
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::AddConstantToDiagonal
( Scalar alpha )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::AddConstantToDiagonal");
#endif
    Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        Node& nodeA = *shell.data.N;
        for( int i=0; i<4; ++i )
            nodeA.Child(i,i).AddConstantToDiagonal( alpha );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *shell.data.NS;
        for( int i=0; i<4; ++i )
            nodeA.Child(i,i).AddConstantToDiagonal( alpha );
        break;
    }
    case LOW_RANK:
#ifndef RELEASE
        throw std::logic_error("Mistake in logic");
#endif
        break;
    case DENSE:
    {
        Scalar* DBuffer = shell.data.D->Buffer();
        const int m = shell.data.D->Height();
        const int n = shell.data.D->Width();
        const int DLDim = shell.data.D->LDim();
        for( int j=0; j<std::min(m,n); ++j )
            DBuffer[j+j*DLDim] += alpha;
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := alpha B + A
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::UpdateWith
( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::UpdateWith");
#endif
    Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        Node& nodeA = *shell.data.N;
        const Node& nodeB = *B._shell.data.N;
        for( int i=0; i<16; ++i )
            nodeA.children[i]->UpdateWith( alpha, *nodeB.children[i] );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *shell.data.NS;
        const NodeSymmetric& nodeB = *B._shell.data.NS;
        for( int i=0; i<10; ++i )
            nodeA.children[i]->UpdateWith( alpha, *nodeB.children[i] );
        break;
    }
    case LOW_RANK:
        hmatrix_tools::MatrixUpdateRounded
        ( this->_maxRank, 
          alpha, *B._shell.data.F, (Scalar)1, *shell.data.F );
        break;
    case DENSE:
        hmatrix_tools::MatrixUpdate
        ( alpha, *B._shell.data.D, (Scalar)1, *shell.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
bool
psp::Quasi2dHMatrix<Scalar,Conjugated>::Admissible() const
{
    return Admissible
           ( this->_xSource, this->_xTarget, this->_ySource, this->_yTarget );
}

template<typename Scalar,bool Conjugated>
bool
psp::Quasi2dHMatrix<Scalar,Conjugated>::Admissible
( int xSource, int xTarget, int ySource, int yTarget ) const
{
    if( this->_stronglyAdmissible )
        return std::max(std::abs(xSource-xTarget),std::abs(ySource-yTarget))>1;
    else
        return xSource != xTarget || ySource != yTarget;
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::ImportLowRankMatrix
( const LowRankMatrix<Scalar,Conjugated>& F )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::ImportLowRankMatrix");
#endif
    Shell& shell = this->_shell;
    shell.Clear();

    if( Admissible( this->_xSource, this->_xTarget, 
                    this->_ySource, this->_yTarget ) )
    {
        shell.type = LOW_RANK;
        shell.data.F = new LowRank;
        hmatrix_tools::Copy( F.U, shell.data.F->U );
        hmatrix_tools::Copy( F.V, shell.data.F->V );
    }
    else if( this->_numLevels > 1 )
    {
        if( this->_symmetric && this->_sourceOffset == this->_targetOffset )
        {
            shell.type = NODE_SYMMETRIC;
            shell.data.NS = 
                new NodeSymmetric
                ( this->_xSizeSource, this->_ySizeSource, this->_zSize );
            NodeSymmetric& node = *shell.data.NS;

            int child = 0;
            const int parentOffset = this->_targetOffset;
            LowRank FSub;
            for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
            {
                FSub.U.LockedView( F.U, tOffset, 0, node.sizes[t], F.Rank() );

                for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
                {
                    FSub.V.LockedView
                    ( F.V, sOffset, 0, node.sizes[s], F.Rank() );

                    node.children[child++] = 
                      new Quasi2d
                      ( FSub, 
                        this->_numLevels-1, this->_maxRank, 
                        this->_stronglyAdmissible,
                        node.xSizes[s&1], node.xSizes[t&1],
                        node.ySizes[s/2], node.ySizes[t/2],
                        this->_zSize,
                        2*this->_xSource+(s&1), 2*this->_xTarget+(t&1),
                        2*this->_ySource+(s/2), 2*this->_yTarget+(t/2),
                        sOffset+parentOffset, tOffset+parentOffset );
                }
            }
        }
        else
        {
            shell.type = NODE;
            shell.data.N = 
                new Node
                ( this->_xSizeSource, this->_xSizeTarget,
                  this->_ySizeSource, this->_ySizeTarget, this->_zSize );
            Node& node = *shell.data.N;

            LowRank FSub;
            const int parentSourceOffset = this->_sourceOffset;
            const int parentTargetOffset = this->_targetOffset;
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                FSub.U.LockedView
                ( F.U, tOffset, 0, node.targetSizes[t], F.Rank() );

                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    FSub.V.LockedView
                    ( F.V, sOffset, 0, node.sourceSizes[s], F.Rank() );

                    node.children[s+4*t] = 
                      new Quasi2d
                      ( FSub,
                        this->_numLevels-1, this->_maxRank,
                        this->_stronglyAdmissible,
                        node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                        node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                        this->_zSize,
                        2*this->_xSource+(s&1), 2*this->_xTarget+(t&1),
                        2*this->_ySource+(s/2), 2*this->_yTarget+(t/2),
                        sOffset+parentSourceOffset, 
                        tOffset+parentTargetOffset );
                }
            }
        }
    }
    else
    {
        shell.type = DENSE;
        shell.data.D = new Dense( this->_height, this->_width );
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, this->_height, this->_width, F.Rank(),
          1, F.U.LockedBuffer(), F.U.LDim(),
             F.V.LockedBuffer(), F.V.LDim(),
          0, shell.data.D->Buffer(), shell.data.D->LDim() );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::UpdateWithLowRankMatrix
( Scalar alpha, 
  const LowRankMatrix<Scalar,Conjugated>& F )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::UpdateWithLowRankMatrix");
#endif
    Shell& shell = this->_shell;
    if( Admissible( this->_xSource, this->_xTarget, 
                    this->_ySource, this->_yTarget ) )
    {
        hmatrix_tools::MatrixUpdateRounded
        ( this->MaxRank(), alpha, F, (Scalar)1, *shell.data.F );
    }
    else if( this->_numLevels > 1 )
    {
        if( this->_symmetric )
        {
            NodeSymmetric& node = *shell.data.NS;
            LowRank FSub;
            for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
            {
                FSub.U.LockedView( F.U, tOffset, 0, node.sizes[t], F.Rank() );
                for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
                {
                    FSub.V.LockedView
                    ( F.V, sOffset, 0, node.sizes[s],  F.Rank() );
                    node.Child(t,s).UpdateWithLowRankMatrix( alpha, FSub );
                }
            }
        }
        else
        {
            Node& node = *shell.data.N;
            LowRank FSub;
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                FSub.U.LockedView
                ( F.U, tOffset, 0, node.targetSizes[t], F.Rank() );
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    FSub.V.LockedView
                    ( F.V, sOffset, 0, node.sourceSizes[s],  F.Rank() );
                    node.Child(t,s).UpdateWithLowRankMatrix( alpha, FSub );
                }
            }
        }
    }
    else
    {
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, this->_height, this->_width, F.Rank(),
          alpha, F.U.LockedBuffer(), F.U.LDim(),
                 F.V.LockedBuffer(), F.V.LDim(),
          1, shell.data.D->Buffer(), shell.data.D->LDim() );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::ImportSparseMatrix
( const SparseMatrix<Scalar>& S, int iOffset, int jOffset )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::ImportSparseMatrix");
#endif
    Shell& shell = this->_shell;
    shell.Clear();

    if( Admissible( this->_xSource, this->_xTarget, 
                    this->_ySource, this->_yTarget ) )
    {
        shell.type = LOW_RANK;
        shell.data.F = new LowRank;
        hmatrix_tools::ConvertSubmatrix
        ( *shell.data.F, S, iOffset, jOffset, this->_height, this->_width );
    }
    else if( this->NumLevels() > 1 )
    {
        if( this->_symmetric && this->_sourceOffset == this->_targetOffset )
        {
            shell.type = NODE_SYMMETRIC;
            shell.data.NS = 
                new NodeSymmetric
                ( this->_xSizeSource, this->_ySizeSource, this->_zSize );
            NodeSymmetric& node = *shell.data.NS;

            int child = 0;
            const int parentOffset = this->_targetOffset;
            for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
            {
                for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
                {
                    node.children[child++] = 
                      new Quasi2d
                      ( S, 
                        this->_numLevels-1, this->_maxRank,
                        this->_stronglyAdmissible,
                        node.xSizes[s&1], node.xSizes[t&1],
                        node.ySizes[s/2], node.ySizes[t/2],
                        this->_zSize,
                        2*this->_xSource+(s&1), 2*this->_xTarget+(t&1),
                        2*this->_ySource+(s/2), 2*this->_yTarget+(t/2),
                        sOffset+parentOffset, tOffset+parentOffset );
                }
            }
        }
        else
        {
            shell.type = NODE;
            shell.data.N = 
                new Node
                ( this->_xSizeSource, this->_xSizeTarget,
                  this->_ySizeSource, this->_ySizeTarget, this->_zSize );
            Node& node = *shell.data.N;

            const int parentSourceOffset = this->_sourceOffset;
            const int parentTargetOffset = this->_targetOffset;
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] = 
                      new Quasi2d
                      ( S,
                        this->_numLevels-1, this->_maxRank,
                        this->_stronglyAdmissible,
                        node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                        node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                        this->_zSize,
                        2*this->_xSource+(s&1), 2*this->_xTarget+(t&1),
                        2*this->_ySource+(s/2), 2*this->_yTarget+(t/2),
                        sOffset+parentSourceOffset, 
                        tOffset+parentTargetOffset );
                }
            }
        }
    }
    else
    {
        shell.type = DENSE;
        shell.data.D = new Dense;
        hmatrix_tools::ConvertSubmatrix
        ( *shell.data.D, S, iOffset, jOffset, this->_height, this->_width );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// y += alpha A x
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::UpdateVectorWithNodeSymmetric
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::UpdateVectorWithNodeSymmetric");
#endif
    NodeSymmetric& node = *this->_shell.data.NS;

    // Loop over the 10 children in the lower triangle, summing in each row
    for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
    {
        Vector<Scalar> ySub;
        ySub.View( y, tOffset, node.sizes[t] );

        for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
        {
            Vector<Scalar> xSub;
            xSub.LockedView( x, sOffset, node.sizes[s] );

            node.Child(t,s).MapVector( alpha, xSub, (Scalar)1, ySub );
        }
    }

    // Loop over the 6 children in the strictly lower triangle, summing in
    // each row
    for( int s=0,tOffset=0; s<4; tOffset+=node.sizes[s],++s )
    {
        Vector<Scalar> ySub;
        ySub.View( y, tOffset, node.sizes[s] );

        for( int t=s+1,sOffset=tOffset+node.sizes[s]; t<4; 
             sOffset+=node.sizes[t],++t )
        {
            Vector<Scalar> xSub;
            xSub.LockedView( x, sOffset, node.sizes[t] );

            node.Child(t,s).TransposeMapVector
            ( alpha, xSub, (Scalar)1, ySub );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// C += alpha A B
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::UpdateMatrixWithNodeSymmetric
( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::UpdateMatrixWithNodeSymmetric");
#endif
    NodeSymmetric& node = *_shell.data.NS;

    // Loop over the 10 children in the lower triangle, summing in each row
    for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
    {
        Dense CSub;
        CSub.View( C, tOffset, 0, node.sizes[t], C.Width() );

        for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
        {
            Dense BSub;
            BSub.LockedView( B, sOffset, 0, node.sizes[s], B.Width() );

            node.Child(t,s).MapMatrix( alpha, BSub, (Scalar)1, CSub );
        }
    }

    // Loop over the 6 children in the strictly lower triangle, summing in
    // each row
    for( int s=0,tOffset=0; s<4; tOffset+=node.sizes[s],++s )
    {
        Dense CSub;
        CSub.View( C, tOffset, 0, node.sizes[s], C.Width() );

        for( int t=s+1,sOffset=tOffset+node.sizes[s]; t<4; 
             sOffset+=node.sizes[t],++t )
        {
            Dense BSub;
            BSub.LockedView( B, sOffset, 0, node.sizes[t], B.Width() );

            node.Child(t,s).TransposeMapMatrix
            ( alpha, BSub, (Scalar)1, CSub );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::WriteStructureRecursion
( std::ofstream& file ) const
{
    const Shell& shell = this->_shell;
    switch( shell.type )
    {
    case NODE:
    {
        file << "1 " 
             << this->TargetOffset() << " " << this->SourceOffset() << " "
             << this->TargetSize() << " " << this->SourceSize() 
             << std::endl;
        const Node& node = *shell.data.N;
        for( unsigned child=0; child<node.children.size(); ++child )
            node.children[child]->WriteStructureRecursion( file );
        break;
    }
    case NODE_SYMMETRIC:
    {
        file << "1 " 
             << this->TargetOffset() << " " << this->SourceOffset() << " "
             << this->TargetSize() << " " << this->SourceSize() 
             << std::endl;
        const NodeSymmetric& node = *shell.data.NS;
        for( unsigned child=0; child<node.children.size(); ++child )
            node.children[child]->WriteStructureRecursion( file );
        break;
    }
    case LOW_RANK:
        file << "5 " 
             << this->TargetOffset() << " " << this->SourceOffset() << " "
             << this->TargetSize() << " " << this->SourceSize() 
             << std::endl;
        break;
    case DENSE:
        file << "20 " 
             << this->TargetOffset() << " " << this->SourceOffset() << " "
             << this->TargetSize() << " " << this->SourceSize() 
             << std::endl;
        break;
    }
}

template class psp::Quasi2dHMatrix<float,false>;
template class psp::Quasi2dHMatrix<float,true>;
template class psp::Quasi2dHMatrix<double,false>;
template class psp::Quasi2dHMatrix<double,true>;
template class psp::Quasi2dHMatrix<std::complex<float>,false>;
template class psp::Quasi2dHMatrix<std::complex<float>,true>;
template class psp::Quasi2dHMatrix<std::complex<double>,false>;
template class psp::Quasi2dHMatrix<std::complex<double>,true>;
