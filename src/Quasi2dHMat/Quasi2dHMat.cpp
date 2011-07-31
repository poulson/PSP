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
#include "./MultiplyDense-incl.hpp"
#include "./MultiplyHMat-incl.hpp"
#include "./MultiplyVector-incl.hpp"
#include "./Pack-incl.hpp"
#include "./SetToRandom-incl.hpp"

//----------------------------------------------------------------------------//
// Public static routines                                                     //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void 
psp::Quasi2dHMat<Scalar,Conjugated>::BuildMapOnQuadrant
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
psp::Quasi2dHMat<Scalar,Conjugated>::BuildNaturalToHierarchicalMap
( std::vector<int>& map, int xSize, int ySize, int zSize, int numLevels )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::BuildNaturalToHierarchicalMap");
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
psp::Quasi2dHMat<Scalar,Conjugated>::Quasi2dHMat()
: _numLevels(0),
  _maxRank(0),
  _sourceOffset(0), _targetOffset(0),
  _symmetric(false), 
  _stronglyAdmissible(false),
  _xSizeSource(0), _xSizeTarget(0),
  _ySizeSource(0), _ySizeTarget(0),
  _zSize(0),
  _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0)
{ }

// Create a square top-level H-matrix
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMat<Scalar,Conjugated>::Quasi2dHMat
( int numLevels, int maxRank, bool symmetric, bool stronglyAdmissible,
  int xSize, int ySize, int zSize )
: _numLevels(numLevels),
  _maxRank(maxRank),
  _sourceOffset(0), _targetOffset(0),
  _symmetric(symmetric), 
  _stronglyAdmissible(stronglyAdmissible),
  _xSizeSource(xSize), _xSizeTarget(xSize),
  _ySizeSource(ySize), _ySizeTarget(ySize),
  _zSize(zSize),
  _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0)
{ }
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMat<Scalar,Conjugated>::Quasi2dHMat
( const LowRank<Scalar,Conjugated>& F,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSize, int ySize, int zSize )
: _numLevels(numLevels),
  _maxRank(maxRank),
  _sourceOffset(0), _targetOffset(0),
  _symmetric(false), 
  _stronglyAdmissible(stronglyAdmissible),
  _xSizeSource(xSize), _xSizeTarget(xSize),
  _ySizeSource(ySize), _ySizeTarget(ySize),
  _zSize(zSize),
  _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0)
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Quasi2dHMat");
#endif
    ImportLowRank( F );
#ifndef RELEASE
    PopCallStack();
#endif
}
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMat<Scalar,Conjugated>::Quasi2dHMat
( const Sparse<Scalar>& S,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSize, int ySize, int zSize )
: _numLevels(numLevels),
  _maxRank(maxRank),
  _sourceOffset(0), _targetOffset(0),
  _symmetric(S.symmetric),
  _stronglyAdmissible(stronglyAdmissible),
  _xSizeSource(xSize), _xSizeTarget(xSize),
  _ySizeSource(ySize), _ySizeTarget(ySize),
  _zSize(zSize),
  _xSource(0), _xTarget(0),
  _ySource(0), _yTarget(0)
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Quasi2dHMat");
#endif
    ImportSparse( S );
#ifndef RELEASE
    PopCallStack();
#endif
}

// Create a potentially non-square non-top-level H-matrix
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMat<Scalar,Conjugated>::Quasi2dHMat
( int numLevels, int maxRank, bool symmetric, bool stronglyAdmissible,
  int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int zSize,
  int xSource, int xTarget,
  int ySource, int yTarget,
  int sourceOffset, int targetOffset )
: _numLevels(numLevels),
  _maxRank(maxRank),
  _sourceOffset(sourceOffset), _targetOffset(targetOffset),
  _symmetric(symmetric),
  _stronglyAdmissible(stronglyAdmissible),
  _xSizeSource(xSizeSource), _xSizeTarget(xSizeTarget),
  _ySizeSource(ySizeSource), _ySizeTarget(ySizeTarget),
  _zSize(zSize),
  _xSource(xSource), _xTarget(xTarget),
  _ySource(ySource), _yTarget(yTarget)
{ }
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMat<Scalar,Conjugated>::Quasi2dHMat
( const LowRank<Scalar,Conjugated>& F,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int zSize,
  int xSource, int xTarget,
  int ySource, int yTarget,
  int sourceOffset, int targetOffset )
: _numLevels(numLevels),
  _maxRank(maxRank),
  _sourceOffset(sourceOffset), _targetOffset(targetOffset),
  _symmetric(false),
  _stronglyAdmissible(stronglyAdmissible),
  _xSizeSource(xSizeSource), _xSizeTarget(xSizeTarget),
  _ySizeSource(ySizeSource), _ySizeTarget(ySizeTarget),
  _zSize(zSize),
  _xSource(xSource), _xTarget(xTarget),
  _ySource(ySource), _yTarget(yTarget)
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Quasi2dHMat");
#endif
    ImportLowRank( F );
#ifndef RELEASE
    PopCallStack();
#endif
}
template<typename Scalar,bool Conjugated>
psp::Quasi2dHMat<Scalar,Conjugated>::Quasi2dHMat
( const Sparse<Scalar>& S,
  int numLevels, int maxRank, bool stronglyAdmissible,
  int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int zSize,
  int xSource, int xTarget,
  int ySource, int yTarget,
  int sourceOffset, int targetOffset )
: _numLevels(numLevels),
  _maxRank(maxRank),
  _sourceOffset(sourceOffset), _targetOffset(targetOffset),
  _symmetric(S.symmetric && sourceOffset==targetOffset),
  _stronglyAdmissible(stronglyAdmissible),
  _xSizeSource(xSizeSource), _xSizeTarget(xSizeTarget),
  _ySizeSource(ySizeSource), _ySizeTarget(ySizeTarget), 
  _zSize(zSize),
  _xSource(xSource), _xTarget(xTarget),
  _ySource(ySource), _yTarget(yTarget)
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Quasi2dHMat");
#endif
    ImportSparse( S, targetOffset, sourceOffset );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
psp::Quasi2dHMat<Scalar,Conjugated>::~Quasi2dHMat()
{ }

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::Clear()
{
    _block.Clear();
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::Print( const std::string tag ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Print");
#endif
    const int n = Width();
    Dense<Scalar> I( n, n );
    std::memset( I.Buffer(), 0, I.LDim()*n );
    for( int j=0; j<n; ++j )
        I.Set(j,j,(Scalar)1);

    Dense<Scalar> HFull;
    Multiply( (Scalar)1, I, HFull );
    HFull.Print( tag );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::LatexWriteStructure
( const std::string filebase ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::LatexWriteStructure");
#endif

    std::ofstream file( (filebase+".tex").c_str() );
    double scale = 12.8;
    file << "\\documentclass[11pt]{article}\n"
         << "\\usepackage{tikz}\n"
         << "\\begin{document}\n"
         << "\\begin{center}\n"
         << "\\begin{tikzpicture}[scale=" << scale << "]\n";
    LatexWriteStructureRecursion( file, Height() );
    file << "\\end{tikzpicture}\n"
         << "\\end{center}\n"
         << "\\end{document}" << std::endl;
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::MScriptWriteStructure
( const std::string filebase ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::MScriptWriteStructure");
#endif
    std::ofstream file( (filebase+".dat").c_str() );
    MScriptWriteStructureRecursion( file );
#ifndef RELEASE
    PopCallStack();
#endif
}

/*\
|*| Computational routines specific to Quasi2dHMat
\*/

// A := B
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::CopyFrom
( const Quasi2dHMat<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::CopyFrom");
#endif
    _numLevels = B.NumLevels();
    _maxRank = B.MaxRank();
    _sourceOffset = B.SourceOffset();
    _targetOffset = B.TargetOffset();
    _symmetric = B.Symmetric();
    _stronglyAdmissible = B.StronglyAdmissible();
    _xSizeSource = B.XSizeSource();
    _xSizeTarget = B.XSizeTarget();
    _ySizeSource = B.YSizeSource();
    _ySizeTarget = B.YSizeTarget();
    _zSize = B.ZSize();
    _xSource = B.XSource();
    _xTarget = B.XTarget();
    _ySource = B.YSource();
    _yTarget = B.YTarget();

    _block.Clear();
    _block.type = B._block.type;
    switch( _block.type )
    {
    case NODE:
    {
        _block.data.N = NewNode();
        Node& nodeA = *_block.data.N;
        const Node& nodeB = *B._block.data.N;
        for( int i=0; i<16; ++i )
        {
            nodeA.children[i] = new Quasi2dHMat<Scalar,Conjugated>;
            nodeA.children[i]->CopyFrom( *nodeB.children[i] );
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        _block.data.NS = NewNodeSymmetric();
        NodeSymmetric& nodeA = *_block.data.NS;
        const NodeSymmetric& nodeB = *B._block.data.NS;
        for( int i=0; i<10; ++i )
        {
            nodeA.children[i] = new Quasi2dHMat<Scalar,Conjugated>;
            nodeA.children[i]->CopyFrom( *nodeB.children[i] );
        }
        break;
    }
    case LOW_RANK:
        _block.data.F = new LowRank<Scalar,Conjugated>;
        hmat_tools::Copy( *B._block.data.F, *_block.data.F );
        break;
    case DENSE:
        _block.data.D = new Dense<Scalar>;
        hmat_tools::Copy( *B._block.data.D, *_block.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := Conj(A)
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::Conjugate()
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Conjugate");
#endif
    switch( _block.type )
    {
    case NODE:
        for( int i=0; i<16; ++i )
            _block.data.N->children[i]->Conjugate();
        break;
    case NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
            _block.data.NS->children[i]->Conjugate();
        break;
    case LOW_RANK:
        hmat_tools::Conjugate( _block.data.F->U );
        hmat_tools::Conjugate( _block.data.F->V );
        break;
    case DENSE:
        hmat_tools::Conjugate( *_block.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := Conj(B)
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::ConjugateFrom
( const Quasi2dHMat<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::ConjugateFrom");
#endif
    Quasi2dHMat<Scalar,Conjugated>& A = *this;

    A._numLevels = B.NumLevels();
    A._maxRank = B.MaxRank();
    A._sourceOffset = B.SourceOffset();
    A._targetOffset = B.TargetOffset();
    A._symmetric = B.Symmetric();
    A._stronglyAdmissible = B.StronglyAdmissible();
    A._xSizeSource = B.XSizeSource();
    A._xSizeTarget = B.XSizeTarget();
    A._ySizeSource = B.YSizeSource();
    A._ySizeTarget = B.YSizeTarget();
    A._zSize = B.ZSize();
    A._xSource = B.XSource();
    A._xTarget = B.XTarget();
    A._ySource = B.YSource();
    A._yTarget = B.YTarget();

    A._block.Clear();
    A._block.type = B._block.type;
    switch( A._block.type )
    {
    case NODE:
        for( int i=0; i<16; ++i )
        {
            A._block.data.N->children[i]->ConjugateFrom
            ( *B._block.data.N->children[i] );
        }
        break;
    case NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
        {
            A._block.data.NS->children[i]->ConjugateFrom
            ( *B._block.data.NS->children[i] );
        }
        break;
    case LOW_RANK:
        hmat_tools::Conjugate( B._block.data.F->U, A._block.data.F->U );
        hmat_tools::Conjugate( B._block.data.F->V, A._block.data.F->V );
        break;
    case DENSE:
        hmat_tools::Conjugate( *B._block.data.D, *A._block.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := B^T
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::TransposeFrom
( const Quasi2dHMat<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::TransposeFrom");
#endif
    Quasi2dHMat<Scalar,Conjugated>& A = *this;

    A._numLevels = B.NumLevels();
    A._maxRank = B.MaxRank();
    A._sourceOffset = B.TargetOffset();
    A._targetOffset = B.SourceOffset();
    A._symmetric = B.Symmetric();
    A._stronglyAdmissible = B.StronglyAdmissible();
    A._xSizeSource = B.XSizeTarget();
    A._xSizeTarget = B.XSizeSource();
    A._ySizeSource = B.YSizeTarget();
    A._ySizeTarget = B.YSizeSource();
    A._zSize = B.ZSize();
    A._xSource = B.XTarget();
    A._xTarget = B.XSource();
    A._ySource = B.YTarget();
    A._yTarget = B.YSource();

    A._block.Clear();
    A._block.type = B._block.type;
    switch( A._block.type )
    {
    case NODE:
    {
        A._block.data.N = NewNode();
        Node& nodeA = *A._block.data.N;
        const Node& nodeB = *B._block.data.N;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<4; ++s )
            {
                nodeA.children[s+4*t] = new Quasi2dHMat<Scalar,Conjugated>;
                nodeA.Child(t,s).TransposeFrom( nodeB.Child(s,t) );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        A._block.data.NS = NewNodeSymmetric();
        NodeSymmetric& nodeA = *A._block.data.NS;
        const NodeSymmetric& nodeB = *B._block.data.NS;
        for( int i=0; i<10; ++i )
        {
            nodeA.children[i] = new Quasi2dHMat<Scalar,Conjugated>;
            nodeA.children[i]->CopyFrom( *nodeB.children[i] );
        }
        break;
    }
    case LOW_RANK:
    {
        A._block.data.F = new LowRank<Scalar,Conjugated>;
        hmat_tools::Transpose( *B._block.data.F, *A._block.data.F );
        break;
    }
    case DENSE:
    {
        A._block.data.D = new Dense<Scalar>;
        hmat_tools::Transpose( *B._block.data.D, *A._block.data.D );
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
psp::Quasi2dHMat<Scalar,Conjugated>::AdjointFrom
( const Quasi2dHMat<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::AdjointFrom");
#endif
    Quasi2dHMat<Scalar,Conjugated>& A = *this;

    A._numLevels = B.NumLevels();
    A._maxRank = B.MaxRank();
    A._sourceOffset = B.TargetOffset();
    A._targetOffset = B.SourceOffset();
    A._symmetric = B.Symmetric();
    A._stronglyAdmissible = B.StronglyAdmissible();
    A._xSizeSource = B.XSizeTarget();
    A._xSizeTarget = B.XSizeSource();
    A._ySizeSource = B.YSizeTarget();
    A._ySizeTarget = B.YSizeSource();
    A._zSize = B.ZSize();
    A._xSource = B.XTarget();
    A._xTarget = B.XSource();
    A._ySource = B.YTarget();
    A._yTarget = B.YSource();

    A._block.Clear();
    A._block.type = B._block.type;
    switch( B._block.type )
    {
    case NODE:
    {
        A._block.data.N = NewNode();
        Node& nodeA = *A._block.data.N;
        const Node& nodeB = *B._block.data.N;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<4; ++s )
            {
                nodeA.children[s+4*t] = new Quasi2dHMat<Scalar,Conjugated>;
                nodeA.Child(t,s).AdjointFrom( nodeB.Child(s,t) );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        A._block.data.NS = NewNodeSymmetric();
        NodeSymmetric& nodeA = *A._block.data.NS;
        const NodeSymmetric& nodeB = *B._block.data.NS;
        for( int i=0; i<10; ++i )
        {
            nodeA.children[i] = new Quasi2dHMat<Scalar,Conjugated>;
            nodeA.children[i]->ConjugateFrom( *nodeB.children[i] );
        }
        break;
    }
    case LOW_RANK:
    {
        A._block.data.F = new LowRank<Scalar,Conjugated>;
        hmat_tools::Adjoint( *B._block.data.F, *A._block.data.F );
        break;
    }
    case DENSE:
    {
        A._block.data.D = new Dense<Scalar>;
        hmat_tools::Adjoint( *B._block.data.D, *A._block.data.D );
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
psp::Quasi2dHMat<Scalar,Conjugated>::Scale( Scalar alpha )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Scale");
#endif
    switch( _block.type )
    {
    case NODE:
    {
        Node& nodeA = *_block.data.N;
        for( int i=0; i<16; ++i )
            nodeA.children[i]->Scale( alpha );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *_block.data.NS;
        for( int i=0; i<10; ++i )
            nodeA.children[i]->Scale( alpha );
        break;
    }
    case LOW_RANK:
        hmat_tools::Scale( alpha, *_block.data.F );
        break;
    case DENSE:
        hmat_tools::Scale( alpha, *_block.data.D );
        break;
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// A := I
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::SetToIdentity()
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::SetToIdentity");
#endif
    switch( _block.type )
    {
    case NODE:
    {
        Node& nodeA = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                if( s == t )
                    nodeA.Child(t,s).SetToIdentity();
                else
                    nodeA.Child(t,s).Scale( (Scalar)0 );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *_block.data.NS;
        for( int t=0; t<4; ++t )
        {
            for( int s=0; s<t; ++s )
                nodeA.Child(t,s).Scale( (Scalar)0 );
            nodeA.Child(t,t).SetToIdentity();
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
        Dense<Scalar>& D = *_block.data.D;
        hmat_tools::Scale( (Scalar)0, D );
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
psp::Quasi2dHMat<Scalar,Conjugated>::AddConstantToDiagonal
( Scalar alpha )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::AddConstantToDiagonal");
#endif
    switch( _block.type )
    {
    case NODE:
    {
        Node& nodeA = *_block.data.N;
        for( int i=0; i<4; ++i )
            nodeA.Child(i,i).AddConstantToDiagonal( alpha );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *_block.data.NS;
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
        Scalar* DBuffer = _block.data.D->Buffer();
        const int m = _block.data.D->Height();
        const int n = _block.data.D->Width();
        const int DLDim = _block.data.D->LDim();
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
psp::Quasi2dHMat<Scalar,Conjugated>::UpdateWith
( Scalar alpha, const Quasi2dHMat<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::UpdateWith");
#endif
    Quasi2dHMat<Scalar,Conjugated>& A = *this;

    switch( A._block.type )
    {
    case NODE:
    {
        Node& nodeA = *A._block.data.N;
        const Node& nodeB = *B._block.data.N;
        for( int i=0; i<16; ++i )
            nodeA.children[i]->UpdateWith( alpha, *nodeB.children[i] );
        break;
    }
    case NODE_SYMMETRIC:
    {
        NodeSymmetric& nodeA = *A._block.data.NS;
        const NodeSymmetric& nodeB = *B._block.data.NS;
        for( int i=0; i<10; ++i )
            nodeA.children[i]->UpdateWith( alpha, *nodeB.children[i] );
        break;
    }
    case LOW_RANK:
        hmat_tools::RoundedUpdate
        ( this->_maxRank, 
          alpha, *B._block.data.F, (Scalar)1, *A._block.data.F );
        break;
    case DENSE:
        hmat_tools::Update
        ( alpha, *B._block.data.D, (Scalar)1, *A._block.data.D );
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
psp::Quasi2dHMat<Scalar,Conjugated>::Admissible() const
{
    return Admissible( _xSource, _xTarget, _ySource, _yTarget );
}

template<typename Scalar,bool Conjugated>
bool
psp::Quasi2dHMat<Scalar,Conjugated>::Admissible
( int xSource, int xTarget, int ySource, int yTarget ) const
{
    if( _stronglyAdmissible )
        return std::max(std::abs(xSource-xTarget),std::abs(ySource-yTarget))>1;
    else
        return xSource != xTarget || ySource != yTarget;
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::ImportLowRank
( const LowRank<Scalar,Conjugated>& F )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::ImportLowRank");
#endif
    _block.Clear();
    if( Admissible() )
    {
        _block.type = LOW_RANK;
        _block.data.F = new LowRank<Scalar,Conjugated>;
        hmat_tools::Copy( F.U, _block.data.F->U );
        hmat_tools::Copy( F.V, _block.data.F->V );
    }
    else if( _numLevels > 1 )
    {
        if( _symmetric && _sourceOffset == _targetOffset )
        {
            _block.type = NODE_SYMMETRIC;
            _block.data.NS = NewNodeSymmetric();
            NodeSymmetric& node = *_block.data.NS;

            int child = 0;
            const int parentOffset = _targetOffset;
            LowRank<Scalar,Conjugated> FSub;
            for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
            {
                FSub.U.LockedView( F.U, tOffset, 0, node.sizes[t], F.Rank() );

                for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
                {
                    FSub.V.LockedView
                    ( F.V, sOffset, 0, node.sizes[s], F.Rank() );

                    node.children[child++] = 
                      new Quasi2dHMat<Scalar,Conjugated>
                      ( FSub, 
                        _numLevels-1, _maxRank, 
                        _stronglyAdmissible,
                        node.xSizes[s&1], node.xSizes[t&1],
                        node.ySizes[s/2], node.ySizes[t/2],
                        _zSize,
                        2*_xSource+(s&1), 2*_xTarget+(t&1),
                        2*_ySource+(s/2), 2*_yTarget+(t/2),
                        sOffset+parentOffset, tOffset+parentOffset );
                }
            }
        }
        else
        {
            _block.type = NODE;
            _block.data.N = NewNode();
            Node& node = *_block.data.N;

            LowRank<Scalar,Conjugated> FSub;
            const int parentSourceOffset = _sourceOffset;
            const int parentTargetOffset = _targetOffset;
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                FSub.U.LockedView
                ( F.U, tOffset, 0, node.targetSizes[t], F.Rank() );

                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    FSub.V.LockedView
                    ( F.V, sOffset, 0, node.sourceSizes[s], F.Rank() );

                    node.children[s+4*t] = 
                      new Quasi2dHMat<Scalar,Conjugated>
                      ( FSub,
                        _numLevels-1, _maxRank,
                        _stronglyAdmissible,
                        node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                        node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                        _zSize,
                        2*_xSource+(s&1), 2*_xTarget+(t&1),
                        2*_ySource+(s/2), 2*_yTarget+(t/2),
                        sOffset+parentSourceOffset, 
                        tOffset+parentTargetOffset );
                }
            }
        }
    }
    else
    {
        _block.type = DENSE;
        _block.data.D = new Dense<Scalar>( Height(), Width() );
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, Height(), Width(), F.Rank(),
          1, F.U.LockedBuffer(), F.U.LDim(),
             F.V.LockedBuffer(), F.V.LDim(),
          0, _block.data.D->Buffer(), _block.data.D->LDim() );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::UpdateWithLowRank
( Scalar alpha, 
  const LowRank<Scalar,Conjugated>& F )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::UpdateWithLowRank");
#endif
    if( Admissible() )
        hmat_tools::RoundedUpdate
        ( _maxRank, alpha, F, (Scalar)1, *_block.data.F );
    else if( _numLevels > 1 )
    {
        if( _symmetric )
        {
            NodeSymmetric& node = *_block.data.NS;
            LowRank<Scalar,Conjugated> FSub;
            for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
            {
                FSub.U.LockedView( F.U, tOffset, 0, node.sizes[t], F.Rank() );
                for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
                {
                    FSub.V.LockedView
                    ( F.V, sOffset, 0, node.sizes[s],  F.Rank() );
                    node.Child(t,s).UpdateWithLowRank( alpha, FSub );
                }
            }
        }
        else
        {
            Node& node = *_block.data.N;
            LowRank<Scalar,Conjugated> FSub;
            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                FSub.U.LockedView
                ( F.U, tOffset, 0, node.targetSizes[t], F.Rank() );
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    FSub.V.LockedView
                    ( F.V, sOffset, 0, node.sourceSizes[s],  F.Rank() );
                    node.Child(t,s).UpdateWithLowRank( alpha, FSub );
                }
            }
        }
    }
    else
    {
        const char option = ( Conjugated ? 'C' : 'T' );
        blas::Gemm
        ( 'N', option, Height(), Width(), F.Rank(),
          alpha, F.U.LockedBuffer(), F.U.LDim(),
                 F.V.LockedBuffer(), F.V.LDim(),
          1, _block.data.D->Buffer(), _block.data.D->LDim() );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::ImportSparse
( const Sparse<Scalar>& S, int iOffset, int jOffset )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::ImportSparse");
#endif
    _block.Clear();
    if( Admissible() )
    {
        _block.type = LOW_RANK;
        _block.data.F = new LowRank<Scalar,Conjugated>;
        hmat_tools::ConvertSubmatrix
        ( *_block.data.F, S, iOffset, jOffset, Height(), Width() );
    }
    else if( _numLevels > 1 )
    {
        if( _symmetric && _sourceOffset == _targetOffset )
        {
            _block.type = NODE_SYMMETRIC;
            _block.data.NS = NewNodeSymmetric();
            NodeSymmetric& node = *_block.data.NS;

            int child = 0;
            for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
            {
                for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
                {
                    node.children[child++] = 
                      new Quasi2dHMat<Scalar,Conjugated>
                      ( S, 
                        _numLevels-1, _maxRank,
                        _stronglyAdmissible,
                        node.xSizes[s&1], node.xSizes[t&1],
                        node.ySizes[s/2], node.ySizes[t/2],
                        _zSize,
                        2*_xSource+(s&1), 2*_xTarget+(t&1),
                        2*_ySource+(s/2), 2*_yTarget+(t/2),
                        sOffset+_targetOffset, tOffset+_targetOffset );
                }
            }
        }
        else
        {
            _block.type = NODE;
            _block.data.N = NewNode();
            Node& node = *_block.data.N;

            for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
            {
                for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
                {
                    node.children[s+4*t] = 
                      new Quasi2dHMat<Scalar,Conjugated>
                      ( S,
                        _numLevels-1, _maxRank, _stronglyAdmissible,
                        node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                        node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                        _zSize,
                        2*_xSource+(s&1), 2*_xTarget+(t&1),
                        2*_ySource+(s/2), 2*_yTarget+(t/2),
                        sOffset+_sourceOffset, tOffset+_targetOffset );
                }
            }
        }
    }
    else
    {
        _block.type = DENSE;
        _block.data.D = new Dense<Scalar>;
        hmat_tools::ConvertSubmatrix
        ( *_block.data.D, S, iOffset, jOffset, Height(), Width() );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// y += alpha A x
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::UpdateVectorWithNodeSymmetric
( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::UpdateVectorWithNodeSymmetric");
#endif
    NodeSymmetric& node = *_block.data.NS;

    // Loop over the 10 children in the lower triangle, summing in each row
    for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
    {
        Vector<Scalar> ySub;
        ySub.View( y, tOffset, node.sizes[t] );

        for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
        {
            Vector<Scalar> xSub;
            xSub.LockedView( x, sOffset, node.sizes[s] );

            node.Child(t,s).Multiply( alpha, xSub, (Scalar)1, ySub );
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

            node.Child(t,s).TransposeMultiply
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
psp::Quasi2dHMat<Scalar,Conjugated>::UpdateWithNodeSymmetric
( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::UpdateWithNodeSymmetric");
#endif
    const Quasi2dHMat<Scalar,Conjugated>& A = *this;
    NodeSymmetric& node = *A._block.data.NS;

    // Loop over the 10 children in the lower triangle, summing in each row
    for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
    {
        Dense<Scalar> CSub;
        CSub.View( C, tOffset, 0, node.sizes[t], C.Width() );

        for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
        {
            Dense<Scalar> BSub;
            BSub.LockedView( B, sOffset, 0, node.sizes[s], B.Width() );

            node.Child(t,s).Multiply( alpha, BSub, (Scalar)1, CSub );
        }
    }

    // Loop over the 6 children in the strictly lower triangle, summing in
    // each row
    for( int s=0,tOffset=0; s<4; tOffset+=node.sizes[s],++s )
    {
        Dense<Scalar> CSub;
        CSub.View( C, tOffset, 0, node.sizes[s], C.Width() );

        for( int t=s+1,sOffset=tOffset+node.sizes[s]; t<4; 
             sOffset+=node.sizes[t],++t )
        {
            Dense<Scalar> BSub;
            BSub.LockedView( B, sOffset, 0, node.sizes[t], B.Width() );

            node.Child(t,s).TransposeMultiply
            ( alpha, BSub, (Scalar)1, CSub );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

namespace {

void FillBox
( std::ofstream& file,
  double hStart, double vStart, double hStop, double vStop,
  const std::string& fillColor )
{
    file << "\\fill[" << fillColor << "] (" << hStart << "," << vStart
         << ") rectangle (" << hStop << "," << vStop << ");\n";
}

void DrawBox
( std::ofstream& file,
  double hStart, double vStart, double hStop, double vStop,
  const std::string& drawColor )
{
    file << "\\draw[" << drawColor << "] (" << hStart << "," << vStart
         << ") rectangle (" << hStop << "," << vStop << ");\n";
}

} // anonymous namespace

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::LatexWriteStructureRecursion
( std::ofstream& file, int globalHeight ) const
{
    const double invScale = globalHeight; 
    const double hStart = _sourceOffset/invScale;
    const double hStop  = (_sourceOffset+Width())/invScale;
    const double vStart = (globalHeight-(_targetOffset + Height()))/invScale;
    const double vStop  = (globalHeight-_targetOffset)/invScale;

    const std::string lowRankColor = "green";
    const std::string lowRankEmptyColor = "cyan";
    const std::string denseColor = "red";
    const std::string borderColor = "black";

    switch( _block.type )
    {
    case NODE:
    {
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).LatexWriteStructureRecursion
                ( file, globalHeight );
        break;
    }
    case NODE_SYMMETRIC:
    {
        const NodeSymmetric& node = *_block.data.NS;
        for( unsigned child=0; child<node.children.size(); ++child )
            node.children[child]->LatexWriteStructureRecursion
            ( file, globalHeight );
        break;
    }
    case LOW_RANK:
    {
        const int rank = _block.data.F->Rank();
        if( rank == 0 )
            FillBox( file, hStart, vStart, hStop, vStop, lowRankEmptyColor );
        else
            FillBox( file, hStart, vStart, hStop, vStop, lowRankColor );
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;
    }
    case DENSE:
        FillBox( file, hStart, vStart, hStop, vStop, denseColor );
        DrawBox( file, hStart, vStart, hStop, vStop, borderColor );
        break;
    }
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::MScriptWriteStructureRecursion
( std::ofstream& file ) const
{
    switch( _block.type )
    {
    case NODE:
    {
        file << "1 " 
             << _targetOffset << " " << _sourceOffset << " "
             << Height() << " " << Width() << "\n";
        const Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).MScriptWriteStructureRecursion( file );
        break;
    }
    case NODE_SYMMETRIC:
    {
        file << "1 " 
             << _targetOffset << " " << _sourceOffset << " "
             << Height() << " " << Width() << "\n";
        const NodeSymmetric& node = *_block.data.NS;
        for( unsigned child=0; child<node.children.size(); ++child )
            node.children[child]->MScriptWriteStructureRecursion( file );
        break;
    }
    case LOW_RANK:
        file << "5 " 
             << _targetOffset << " " << _sourceOffset << " "
             << Height() << " " << Width() << "\n";
        break;
    case DENSE:
        file << "20 " 
             << _targetOffset << " " << _sourceOffset << " "
             << Height() << " " << Width() << "\n";
        break;
    }
}

template class psp::Quasi2dHMat<float,false>;
template class psp::Quasi2dHMat<float,true>;
template class psp::Quasi2dHMat<double,false>;
template class psp::Quasi2dHMat<double,true>;
template class psp::Quasi2dHMat<std::complex<float>,false>;
template class psp::Quasi2dHMat<std::complex<float>,true>;
template class psp::Quasi2dHMat<std::complex<double>,false>;
template class psp::Quasi2dHMat<std::complex<double>,true>;
