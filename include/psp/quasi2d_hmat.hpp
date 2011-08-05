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
#ifndef PSP_QUASI2D_HMAT_HPP
#define PSP_QUASI2D_HMAT_HPP 1

#include "psp/building_blocks/abstract_hmat.hpp"
#include "psp/hmat_tools.hpp"

namespace psp {

// Forward declare friend classes
template<typename Scalar,bool Conjugated> class DistQuasi2dHMat;

template<typename Scalar,bool Conjugated=true>
class Quasi2dHMat : public AbstractHMat<Scalar>
{
    friend class DistQuasi2dHMat<Scalar,Conjugated>;
private:
    /*
     * Private static member functions
     */
    static void BuildMapOnQuadrant
    ( int* map, int& index, int level, int numLevels,
      int xSize, int ySize, int zSize, int thisXSize, int thisYSize );

    /*
     * Private data structures
     */
    struct Node
    {
        std::vector<Quasi2dHMat*> children;
        int xSourceSizes[2];
        int ySourceSizes[2];
        int sourceSizes[4];
        int xTargetSizes[2];
        int yTargetSizes[2];
        int targetSizes[4];
        Node
        ( int xSizeSource, int xSizeTarget,
          int ySizeSource, int ySizeTarget,
          int zSize );
        ~Node();
        Quasi2dHMat& Child( int i, int j );
        const Quasi2dHMat& Child( int i, int j ) const;
    };
    Node* NewNode() const;

    struct NodeSymmetric
    {
        std::vector<Quasi2dHMat*> children;
        int xSizes[2];
        int ySizes[2];
        int sizes[4];
        NodeSymmetric( int xSize, int ySize, int zSize );
        ~NodeSymmetric();
        Quasi2dHMat& Child( int i, int j );
        const Quasi2dHMat& Child( int i, int j ) const;
    };
    NodeSymmetric* NewNodeSymmetric() const;

    enum BlockType 
    { 
        NODE, 
        NODE_SYMMETRIC, 
        LOW_RANK, 
        DENSE 
    };

    struct Block
    {
        BlockType type;
        union Data
        {
            Node* N;
            NodeSymmetric* NS;
            LowRank<Scalar,Conjugated>* F;
            Dense<Scalar>* D;
            Data() { std::memset( this, 0, sizeof(Data) ); }
        } data;
        Block();
        ~Block();
        void Clear();
    };

    /*
     * Private member data
     */
    int _numLevels;
    int _maxRank;
    int _sourceOffset, _targetOffset;
    bool _symmetric;
    bool _stronglyAdmissible;

    int _xSizeSource, _xSizeTarget;
    int _ySizeSource, _ySizeTarget;
    int _zSize;
    int _xSource, _xTarget;
    int _ySource, _yTarget;
    Block _block;

    /*
     * Private non-static member functions
     */
    void PackedSizeRecursion( std::size_t& packedSize ) const;
    void PackRecursion( byte*& head ) const;

    bool Admissible() const;
    bool Admissible( int xSource, int xTarget, int ySource, int yTarget ) const;

    void ImportLowRank
    ( const LowRank<Scalar,Conjugated>& F );
    
    void UpdateWithLowRank
    ( Scalar alpha,
      const LowRank<Scalar,Conjugated>& F );

    void ImportSparse
    ( const Sparse<Scalar>& S, int iOffset=0, int jOffset=0 );

    void UnpackRecursion( const byte*& head );

   // y += alpha A x
    void UpdateVectorWithNodeSymmetric
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // C += alpha A B
    void UpdateWithNodeSymmetric
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const;

    void LatexWriteStructureRecursion
    ( std::ofstream& file, int globalHeight ) const;

    void MScriptWriteStructureRecursion( std::ofstream& file ) const;

public:    

    /*
     * Public static member functions
     */
    static int SampleRank( int approxRank ) { return approxRank + 4; }

    static void BuildNaturalToHierarchicalMap
    ( std::vector<int>& map, int xSize, int ySize, int zSize, int numLevels );

    /*
     * Public non-static member functions
     */
    Quasi2dHMat();

    // Create a square top-level H-matrix
    //
    // The weak admissibility criterion is:
    //     max(dist_x(A,B),dist_y(A,B)) >= 1
    //
    // The strong admissibility criterion is:
    //     max(dist_x(A,B),dist_y(A,B)) > 1
    //
    Quasi2dHMat
    ( int numLevels, int maxRank, bool symmetric, bool stronglyAdmissible,
      int xSize, int ySize, int zSize );
    Quasi2dHMat
    ( const LowRank<Scalar,Conjugated>& F,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSize, int ySize, int zSize );
    Quasi2dHMat
    ( const Sparse<Scalar>& S,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSize, int ySize, int zSize );
    
    // Create a potentially non-square non-top-level H-matrix
    Quasi2dHMat
    ( int numLevels, int maxRank, bool symmetric, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int zSize,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );
    Quasi2dHMat
    ( const LowRank<Scalar,Conjugated>& F,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int zSize,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );
    Quasi2dHMat
    ( const Sparse<Scalar>& S,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int zSize,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );

    // Reconstruct an H-matrix from its packed form
    Quasi2dHMat( const std::vector<byte>& packedHMat );

    ~Quasi2dHMat();
    void Clear();

    void SetToRandom();

    // Fulfillments of AbstractHMat
    virtual int Height() const;
    virtual int Width() const;
    virtual int NumLevels() const;
    virtual int MaxRank() const;
    virtual int SourceOffset() const;
    virtual int TargetOffset() const;
    virtual bool Symmetric() const;
    virtual bool StronglyAdmissible() const;

    // Routines useful for packing and unpacking the Quasi2dHMat to/from
    // a contiguous buffer.
    std::size_t PackedSize() const;
    std::size_t Pack( byte* packedHMat ) const;
    std::size_t Pack( std::vector<byte>& packedHMat ) const;
    std::size_t Unpack( const byte* packedHMat );
    std::size_t Unpack( const std::vector<byte>& packedHMat );

    int XSizeSource() const { return _xSizeSource; }
    int XSizeTarget() const { return _xSizeTarget; }
    int YSizeSource() const { return _ySizeSource; }
    int YSizeTarget() const { return _ySizeTarget; }
    int ZSize() const { return _zSize; }
    int XSource() const { return _xSource; }
    int YSource() const { return _ySource; }
    int XTarget() const { return _xTarget; }
    int YTarget() const { return _yTarget; }

    bool IsDense() const { return _block.type == DENSE; }
    bool IsHierarchical() const
    { return _block.type == NODE || _block.type == NODE_SYMMETRIC; }
    bool IsLowRank() const { return _block.type == LOW_RANK; }

    /* 
     * Write a representation of the H-matrix structure to file. 
     */
    // Compile this output with pdflatex+TikZ
    void LatexWriteStructure( const std::string filebase ) const;
    // This can be visualized with util/PlotHStructure.m and Octave/Matlab
    void MScriptWriteStructure( const std::string filebase ) const;

    //------------------------------------------------------------------------//
    // Fulfillments of AbstractHMat interface                                 //
    //------------------------------------------------------------------------//

    // Multiply the H-matrix by identity and print the result
    virtual void Print( const std::string tag ) const;

    // y := alpha H x + beta y
    virtual void Multiply
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A x
    virtual void Multiply
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // y := alpha A^T x + beta y
    virtual void TransposeMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A^T x
    virtual void TransposeMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // y := alpha A^H x + beta y
    virtual void AdjointMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A^H x + beta y (temporarily conjugate x in-place)
    void AdjointMultiply
    ( Scalar alpha, Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A^H x
    virtual void AdjointMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // y := alpha A^H x (temporarily conjugate x in-place)
    void AdjointMultiply
    ( Scalar alpha, Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // C := alpha A B + beta C
    virtual void Multiply
    ( Scalar alpha, const Dense<Scalar>& B, 
      Scalar beta,        Dense<Scalar>& C ) const;

    // C := alpha A B
    virtual void Multiply
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const;
    
    // C := alpha A^T B + beta C
    virtual void TransposeMultiply
    ( Scalar alpha, const Dense<Scalar>& B, 
      Scalar beta,        Dense<Scalar>& C ) const;

    // C := alpha A^T B
    virtual void TransposeMultiply
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const;
    
    // C := alpha A^H B + beta C
    virtual void AdjointMultiply
    ( Scalar alpha, const Dense<Scalar>& B, 
      Scalar beta,        Dense<Scalar>& C ) const;

    // C := alpha A^H B + beta C (temporarily conjugate B in place)
    void AdjointMultiply
    ( Scalar alpha, Dense<Scalar>& B, 
      Scalar beta,  Dense<Scalar>& C ) const;

    // C := alpha A^H B
    virtual void AdjointMultiply
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const;
    
    // C := alpha A^H B (temporarily conjugate B in place)
    void AdjointMultiply
    ( Scalar alpha, Dense<Scalar>& B, Dense<Scalar>& C ) const;
    
    //------------------------------------------------------------------------//
    // Computational routines specific to Quasi2dHMat                         //
    //------------------------------------------------------------------------//

    // A := B
    void CopyFrom( const Quasi2dHMat<Scalar,Conjugated>& B );
    
    // A := conj(A)
    void Conjugate();

    // A := conj(B)
    void ConjugateFrom( const Quasi2dHMat<Scalar,Conjugated>& B );

    // A := B^T
    void TransposeFrom( const Quasi2dHMat<Scalar,Conjugated>& B );

    // A := B^H
    void AdjointFrom( const Quasi2dHMat<Scalar,Conjugated>& B );

    // A := alpha A
    void Scale( Scalar alpha );

    // A := I
    void SetToIdentity();

    // A := A + alpha I
    void AddConstantToDiagonal( Scalar alpha );

    // A :~= alpha B + A
    void UpdateWith( Scalar alpha, const Quasi2dHMat<Scalar,Conjugated>& B );

    // C :~= alpha A B
    void Multiply
    ( Scalar alpha, const Quasi2dHMat<Scalar,Conjugated>& B, 
                          Quasi2dHMat<Scalar,Conjugated>& C ) const;

    // C :~= alpha A B + beta C
    void Multiply
    ( Scalar alpha, const Quasi2dHMat<Scalar,Conjugated>& B, 
      Scalar beta,        Quasi2dHMat<Scalar,Conjugated>& C ) const;

    // A :~= inv(A) using recursive Schur complements
    void DirectInvert();

    // A :~= inv(A) using Schulz iteration, 
    //     X_k+1 = X_k (2I - A X_k) = (2I - X_k A) X_k,
    // where X_k -> inv(A) if X_0 = alpha A^H,
    // with 0 < alpha < 2/||A||_2^2.
    //
    // Require the condition number estimation to be accurate enough that
    //   Estimate <= ||A||_2 <= theta Estimate, where 1 < theta,
    // with probability at least 1-10^{-confidence}.
    //
    // The values for theta and confidence are currently hardcoded.
    void SchulzInvert
    ( int numIterations, 
      typename RealBase<Scalar>::type theta=1.5, 
      typename RealBase<Scalar>::type confidence=6 );
};

} // namespace psp

//----------------------------------------------------------------------------//
// Inlined implementations                                                    //
//----------------------------------------------------------------------------//

namespace psp {

template<typename Scalar,bool Conjugated>
inline int
Quasi2dHMat<Scalar,Conjugated>::Height() const
{
    return _xSizeTarget*_ySizeTarget*_zSize;
}

template<typename Scalar,bool Conjugated>
inline int
Quasi2dHMat<Scalar,Conjugated>::Width() const
{
    return _xSizeSource*_ySizeSource*_zSize;
}

template<typename Scalar,bool Conjugated>
inline int
Quasi2dHMat<Scalar,Conjugated>::NumLevels() const
{
    return _numLevels;
}

template<typename Scalar,bool Conjugated>
inline int
Quasi2dHMat<Scalar,Conjugated>::MaxRank() const
{
    return _maxRank;
}

template<typename Scalar,bool Conjugated>
inline int
Quasi2dHMat<Scalar,Conjugated>::SourceOffset() const
{
    return _sourceOffset;
}

template<typename Scalar,bool Conjugated>
inline int
Quasi2dHMat<Scalar,Conjugated>::TargetOffset() const
{
    return _targetOffset;
}

template<typename Scalar,bool Conjugated>
inline bool
Quasi2dHMat<Scalar,Conjugated>::Symmetric() const
{
    return _symmetric;
}

template<typename Scalar,bool Conjugated>
inline bool
Quasi2dHMat<Scalar,Conjugated>::StronglyAdmissible() const
{
    return _stronglyAdmissible;
}

template<typename Scalar,bool Conjugated>
inline
Quasi2dHMat<Scalar,Conjugated>::Node::Node
( int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int zSize )
: children(16)
{
    xSourceSizes[0] = xSizeSource/2;
    xSourceSizes[1] = xSizeSource - xSourceSizes[0];
    ySourceSizes[0] = ySizeSource/2;
    ySourceSizes[1] = ySizeSource - ySourceSizes[0];
            
    sourceSizes[0] = xSourceSizes[0]*ySourceSizes[0]*zSize;
    sourceSizes[1] = xSourceSizes[1]*ySourceSizes[0]*zSize;
    sourceSizes[2] = xSourceSizes[0]*ySourceSizes[1]*zSize;
    sourceSizes[3] = xSourceSizes[1]*ySourceSizes[1]*zSize;

    xTargetSizes[0] = xSizeTarget/2;
    xTargetSizes[1] = xSizeTarget - xTargetSizes[0];
    yTargetSizes[0] = ySizeTarget/2;
    yTargetSizes[1] = ySizeTarget - yTargetSizes[0];

    targetSizes[0] = xTargetSizes[0]*yTargetSizes[0]*zSize;
    targetSizes[1] = xTargetSizes[1]*yTargetSizes[0]*zSize;
    targetSizes[2] = xTargetSizes[0]*yTargetSizes[1]*zSize;
    targetSizes[3] = xTargetSizes[1]*yTargetSizes[1]*zSize;
}

template<typename Scalar,bool Conjugated>
inline
Quasi2dHMat<Scalar,Conjugated>::Node::~Node()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline Quasi2dHMat<Scalar,Conjugated>& 
Quasi2dHMat<Scalar,Conjugated>::Node::Child( int i, int j )
{ 
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Node::Child");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > 3 || j > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[j+4*i]; 
}

template<typename Scalar,bool Conjugated>
inline const Quasi2dHMat<Scalar,Conjugated>& 
Quasi2dHMat<Scalar,Conjugated>::Node::Child( int i, int j ) const
{ 
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Node::Child");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > 3 || j > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[j+4*i]; 
}

template<typename Scalar,bool Conjugated>
inline typename Quasi2dHMat<Scalar,Conjugated>::Node*
Quasi2dHMat<Scalar,Conjugated>::NewNode() const
{
    return 
        new typename Quasi2dHMat<Scalar,Conjugated>::Node
        ( _xSizeSource, _xSizeTarget, _ySizeSource, _ySizeTarget, _zSize );
}

template<typename Scalar,bool Conjugated>
inline
Quasi2dHMat<Scalar,Conjugated>::NodeSymmetric::NodeSymmetric
( int xSize, int ySize, int zSize )
: children(10)
{
    xSizes[0] = xSize/2;
    xSizes[1] = xSize - xSizes[0];
    ySizes[0] = ySize/2;
    ySizes[1] = ySize - ySizes[0];

    sizes[0] = xSizes[0]*ySizes[0]*zSize;
    sizes[1] = xSizes[1]*ySizes[0]*zSize;
    sizes[2] = xSizes[0]*ySizes[1]*zSize;
    sizes[3] = xSizes[1]*ySizes[1]*zSize;
}

template<typename Scalar,bool Conjugated>
inline
Quasi2dHMat<Scalar,Conjugated>::NodeSymmetric::~NodeSymmetric()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline Quasi2dHMat<Scalar,Conjugated>& 
Quasi2dHMat<Scalar,Conjugated>::NodeSymmetric::Child( int i, int j )
{ 
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::NodeSymmetric::Child");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > 3 || j > 3 )
        throw std::logic_error("Indices out of bounds");
    if( j > i )
        throw std::logic_error("Index outside of lower triangle");
    if( children.size() != 10 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[(i*(i+1))/2 + j]; 
}

template<typename Scalar,bool Conjugated>
inline const Quasi2dHMat<Scalar,Conjugated>& 
Quasi2dHMat<Scalar,Conjugated>::NodeSymmetric::Child( int i, int j ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::NodeSymmetric::Child");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > 3 || j > 3 )
        throw std::logic_error("Indices out of bounds");
    if( j > i )
        throw std::logic_error("Index outside of lower triangle");
    if( children.size() != 10 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[(i*(i+1))/2 + j];
}

template<typename Scalar,bool Conjugated>
inline typename Quasi2dHMat<Scalar,Conjugated>::NodeSymmetric*
Quasi2dHMat<Scalar,Conjugated>::NewNodeSymmetric() const
{
    return 
        new typename Quasi2dHMat<Scalar,Conjugated>::NodeSymmetric
        ( _xSizeSource, _ySizeSource, _zSize );
}

template<typename Scalar,bool Conjugated>
inline
Quasi2dHMat<Scalar,Conjugated>::Block::Block()
: type(NODE), data() 
{ }

template<typename Scalar,bool Conjugated>
inline
Quasi2dHMat<Scalar,Conjugated>::Block::~Block()
{
    Clear();
}

template<typename Scalar,bool Conjugated>
inline void
Quasi2dHMat<Scalar,Conjugated>::Block::Clear()
{
    switch( type )
    {
    case NODE:           delete data.N;  break;
    case NODE_SYMMETRIC: delete data.NS; break;
    case LOW_RANK:       delete data.F;  break;
    case DENSE:          delete data.D;  break;
    }
    type = NODE;
    data.N = 0;
}

} // namespace psp

#endif // PSP_QUASI2D_HMAT_HPP
