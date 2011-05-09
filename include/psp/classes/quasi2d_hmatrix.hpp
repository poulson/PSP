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
#ifndef PSP_QUASI2D_HMATRIX_HPP
#define PSP_QUASI2D_HMATRIX_HPP 1

#include "psp/classes/abstract_hmatrix.hpp"
#include "psp/hmatrix_tools.hpp"

namespace psp {

// Forward declare friend classes
template<typename Scalar,bool Conjugated> class SplitQuasi2dHMatrix;
template<typename Scalar,bool Conjugated> class DistQuasi2dHMatrix;

template<typename Scalar,bool Conjugated>
class Quasi2dHMatrix : public AbstractHMatrix<Scalar>
{
private:
    static void PackedSizeRecursion
    ( std::size_t& packedSize, 
      const Quasi2dHMatrix<Scalar,Conjugated>& H );

    static void PackRecursion
    ( byte*& head, const Quasi2dHMatrix<Scalar,Conjugated>& H );

    static void BuildMapOnQuadrant
    ( int* map, int& index, int level, int numLevels,
      int xSize, int ySize, int zSize, int thisXSize, int thisYSize );

    struct Node
    {
        std::vector<Quasi2dHMatrix*> children;
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
        Quasi2dHMatrix& Child( int i, int j );
        const Quasi2dHMatrix& Child( int i, int j ) const;
    };

    struct NodeSymmetric
    {
        std::vector<Quasi2dHMatrix*> children;
        int xSizes[2];
        int ySizes[2];
        int sizes[4];
        NodeSymmetric( int xSize, int ySize, int zSize );
        ~NodeSymmetric();
        Quasi2dHMatrix& Child( int i, int j );
        const Quasi2dHMatrix& Child( int i, int j ) const;
    };

    enum ShellType 
    { 
        NODE, 
        NODE_SYMMETRIC, 
        LOW_RANK, 
        DENSE 
    };

    struct Shell
    {
        ShellType type;
        union Data
        {
            Node* node;
            NodeSymmetric* nodeSymmetric;
            LowRankMatrix<Scalar,Conjugated>* F;
            DenseMatrix<Scalar>* D;
            Data() { std::memset( this, 0, sizeof(Data) ); }
        } data;
        Shell();
        ~Shell();
    };

    // Data specific to our quasi-2d H-matrix
    int _xSizeSource, _xSizeTarget;
    int _ySizeSource, _ySizeTarget;
    int _zSize;
    int _xSource, _xTarget;
    int _ySource, _yTarget;
    Shell _shell;

    bool Admissible() const;
    bool Admissible( int xSource, int xTarget, int ySource, int yTarget ) const;

    void ImportLowRankMatrix
    ( const LowRankMatrix<Scalar,Conjugated>& F );
    
    void UpdateWithLowRankMatrix
    ( Scalar alpha,
      const LowRankMatrix<Scalar,Conjugated>& F );

    void ImportSparseMatrix
    ( const SparseMatrix<Scalar>& S, int iOffset=0, int jOffset=0 );

    void UnpackRecursion
    ( const byte*& head, Quasi2dHMatrix<Scalar,Conjugated>& H );

    // y += alpha A x
    void UpdateVectorWithNodeSymmetric
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // C += alpha A B
    void UpdateMatrixWithNodeSymmetric
    ( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const;

    void WriteStructureRecursion( std::ofstream& file ) const;
    
public:    
    friend class SplitQuasi2dHMatrix<Scalar,Conjugated>;
    friend class DistQuasi2dHMatrix<Scalar,Conjugated>;

    static void BuildNaturalToHierarchicalMap
    ( std::vector<int>& map, int xSize, int ySize, int zSize, int numLevels );

    // Default constructor
    Quasi2dHMatrix();

    // Create a square top-level H-matrix
    //
    // The weak admissibility criterion is:
    //     max(dist_x(A,B),dist_y(A,B)) >= 1
    //
    // The strong admissibility criterion is:
    //     max(dist_x(A,B),dist_y(A,B)) > 1
    //
    Quasi2dHMatrix
    ( const LowRankMatrix<Scalar,Conjugated>& F,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSize, int ySize, int zSize );
    Quasi2dHMatrix
    ( const SparseMatrix<Scalar>& S,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSize, int ySize, int zSize );
    
    // Create a potentially non-square non-top-level H-matrix
    Quasi2dHMatrix
    ( const LowRankMatrix<Scalar,Conjugated>& F,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int zSize,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );
    Quasi2dHMatrix
    ( const SparseMatrix<Scalar>& S,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int zSize,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );

    // Reconstruct an H-matrix from its packed form
    Quasi2dHMatrix( const std::vector<byte>& packedHMatrix );

    ~Quasi2dHMatrix();

    // Routines useful for packing and unpacking the Quasi2dHMatrix to/from
    // a contiguous buffer.
    std::size_t PackedSize() const;
    std::size_t Pack( byte* packedHMatrix ) const;
    std::size_t Pack( std::vector<byte>& packedHMatrix ) const;
    std::size_t Unpack( const byte* packedHMatrix );
    std::size_t Unpack( const std::vector<byte>& packedHMatrix );

    int XSizeSource() const { return _xSizeSource; }
    int XSizeTarget() const { return _xSizeTarget; }
    int YSizeSource() const { return _ySizeSource; }
    int YSizeTarget() const { return _ySizeTarget; }
    int ZSize() const { return _zSize; }
    int SourceSize() const { return _xSizeSource*_ySizeSource*_zSize; }
    int TargetSize() const { return _xSizeTarget*_ySizeTarget*_zSize; }
    int XSource() const { return _xSource; }
    int YSource() const { return _ySource; }
    int XTarget() const { return _xTarget; }
    int YTarget() const { return _yTarget; }

    bool IsDense() const { return _shell.type == DENSE; }
    bool IsHierarchical() const
    { return _shell.type == NODE || _shell.type == NODE_SYMMETRIC; }
    bool IsLowRank() const { return _shell.type == LOW_RANK; }

    // Write a representation of the H-matrix structure to file. It can be
    // visualized with util/PlotHStructure.m
    void WriteStructure( const std::string& filename ) const;

    //------------------------------------------------------------------------//
    // Fulfillments of AbstractHMatrix interface                              //
    //------------------------------------------------------------------------//

    // Multiply the H-matrix by identity and print the result
    virtual void Print( const std::string& tag ) const;

    // y := alpha H x + beta y
    virtual void MapVector
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A x
    virtual void MapVector
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // y := alpha A^T x + beta y
    virtual void TransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A^T x
    virtual void TransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // y := alpha A^H x + beta y
    virtual void HermitianTransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A^H x + beta y (temporarily conjugate x in-place)
    void HermitianTransposeMapVector
    ( Scalar alpha, Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A^H x
    virtual void HermitianTransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // y := alpha A^H x (temporarily conjugate x in-place)
    void HermitianTransposeMapVector
    ( Scalar alpha, Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // C := alpha A B + beta C
    virtual void MapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, 
      Scalar beta, DenseMatrix<Scalar>& C ) const;

    // C := alpha A B
    virtual void MapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const;
    
    // C := alpha A^T B + beta C
    virtual void TransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, 
      Scalar beta, DenseMatrix<Scalar>& C ) const;

    // C := alpha A^T B
    virtual void TransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const;
    
    // C := alpha A^H B + beta C
    virtual void HermitianTransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, 
      Scalar beta, DenseMatrix<Scalar>& C ) const;

    // C := alpha A^H B + beta C (temporarily conjugate B in place)
    void HermitianTransposeMapMatrix
    ( Scalar alpha, DenseMatrix<Scalar>& B, 
      Scalar beta,  DenseMatrix<Scalar>& C ) const;

    // C := alpha A^H B
    virtual void HermitianTransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const;
    
    // C := alpha A^H B (temporarily conjugate B in place)
    void HermitianTransposeMapMatrix
    ( Scalar alpha, DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const;
    
    //------------------------------------------------------------------------//
    // Computational routines specific to Quasi2dHMatrix                      //
    //------------------------------------------------------------------------//

    // A := B
    void CopyFrom( const Quasi2dHMatrix<Scalar,Conjugated>& B );
    
    // A := conj(A)
    void Conjugate();

    // A := conj(B)
    void ConjugateFrom( const Quasi2dHMatrix<Scalar,Conjugated>& B );

    // A := B^T
    void TransposeFrom( const Quasi2dHMatrix<Scalar,Conjugated>& B );

    // A := B^H
    void HermitianTransposeFrom( const Quasi2dHMatrix<Scalar,Conjugated>& B );

    // A := alpha A
    void Scale( Scalar alpha );

    // A := I
    void SetToIdentity();

    // A := A + alpha I
    void AddConstantToDiagonal( Scalar alpha );

    // A :~= alpha B + A
    void UpdateWith( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B );

    // C :~= alpha A B
    void MapMatrix
    ( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B, 
                          Quasi2dHMatrix<Scalar,Conjugated>& C ) const;

    // C :~= alpha A B + beta C
    void MapMatrix
    ( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B, 
      Scalar beta,        Quasi2dHMatrix<Scalar,Conjugated>& C ) const;

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
    void SchulzInvert( int numIterations );
};

} // namespace psp

//----------------------------------------------------------------------------//
// Inlined implementations                                                    //
//----------------------------------------------------------------------------//

namespace psp {

template<typename Scalar,bool Conjugated>
inline
Quasi2dHMatrix<Scalar,Conjugated>::Node::Node
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
Quasi2dHMatrix<Scalar,Conjugated>::Node::~Node()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline Quasi2dHMatrix<Scalar,Conjugated>& 
Quasi2dHMatrix<Scalar,Conjugated>::Node::Child( int i, int j )
{ 
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Node::Child");
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
inline const Quasi2dHMatrix<Scalar,Conjugated>& 
Quasi2dHMatrix<Scalar,Conjugated>::Node::Child( int i, int j ) const
{ 
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Node::Child");
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
inline
Quasi2dHMatrix<Scalar,Conjugated>::NodeSymmetric::NodeSymmetric
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
Quasi2dHMatrix<Scalar,Conjugated>::NodeSymmetric::~NodeSymmetric()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline Quasi2dHMatrix<Scalar,Conjugated>& 
Quasi2dHMatrix<Scalar,Conjugated>::NodeSymmetric::Child( int i, int j )
{ 
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::NodeSymmetric::Child");
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
inline const Quasi2dHMatrix<Scalar,Conjugated>& 
Quasi2dHMatrix<Scalar,Conjugated>::NodeSymmetric::Child( int i, int j ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::NodeSymmetric::Child");
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
inline
Quasi2dHMatrix<Scalar,Conjugated>::Shell::Shell()
: type(NODE), data() 
{ }

template<typename Scalar,bool Conjugated>
inline
Quasi2dHMatrix<Scalar,Conjugated>::Shell::~Shell()
{
    switch( type )
    {
    case NODE:           delete data.node; break;
    case NODE_SYMMETRIC: delete data.nodeSymmetric; break;
    case LOW_RANK:       delete data.F; break;
    case DENSE:          delete data.D; break;
    }
}

} // namespace psp

#endif // PSP_QUASI2D_HMATRIX_HPP