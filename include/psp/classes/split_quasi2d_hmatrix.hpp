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
#ifndef PSP_SPLIT_QUASI2D_HMATRIX_HPP
#define PSP_SPLIT_QUASI2D_HMATRIX_HPP 1

#include "psp/classes/quasi2d_hmatrix.hpp"

namespace psp {

template<typename Scalar,bool Conjugated>
class SplitQuasi2dHMatrix
{
private:
    static void PackedSizesRecursion
    ( std::size_t& sourceSize, std::size_t& targetSize, 
      const Quasi2dHMatrix<Scalar,Conjugated>& H );

    static void PackRecursion
    ( byte*& sourceHead, byte*& targetHead,
      int sourceRank, int targetRank,
      const Quasi2dHMatrix<Scalar,Conjugated>& H );

    struct SplitLowRankMatrix
    {
        int rank;
        DenseMatrix<Scalar> D;
        mutable Vector<Scalar> z;
        mutable DenseMatrix<Scalar> Z;
    };

    struct SplitDenseMatrix
    {
        DenseMatrix<Scalar> D;
        mutable Vector<Scalar> z;
        mutable DenseMatrix<Scalar> Z;
    };

    struct Node
    {
        std::vector<SplitQuasi2dHMatrix*> children;
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
        SplitQuasi2dHMatrix& Child( int i, int j );
        const SplitQuasi2dHMatrix& Child( int i, int j ) const;
    };

    struct NodeSymmetric
    {
        std::vector<SplitQuasi2dHMatrix*> children;
        int xSizes[2];
        int ySizes[2];
        int sizes[4];
        NodeSymmetric( int xSize, int ySize, int zSize );
        ~NodeSymmetric();
        SplitQuasi2dHMatrix& Child( int i, int j );
        const SplitQuasi2dHMatrix& Child( int i, int j ) const;
    };

    enum ShellType 
    { 
        NODE, 
        NODE_SYMMETRIC, 
        SPLIT_LOW_RANK, 
        SPLIT_DENSE
    };

    struct Shell
    {
        ShellType type;
        union Data
        {
            Node* N;
            NodeSymmetric* NS;
            SplitLowRankMatrix* SF;
            SplitDenseMatrix* SD;
            Data() { std::memset( this, 0, sizeof(Data) ); }
        } data;
        Shell();
        ~Shell();
    };

    int _height, _width;
    int _numLevels;
    int _maxRank;
    int _sourceOffset, _targetOffset;
    // TODO: Make use of MatrixType
    bool _stronglyAdmissible;

    int _xSizeSource, _xSizeTarget;
    int _ySizeSource, _ySizeTarget;
    int _zSize;
    int _xSource, _xTarget;
    int _ySource, _yTarget;
    Shell _shell;

    bool _ownSourceSide;
    int _localOffset;
    MPI_Comm _comm;
    int _partner;

    // Temporary storage
    mutable Vector<Scalar> _z;
    mutable DenseMatrix<Scalar> _Z;

    bool Admissible( int xSource, int xTarget, int ySource, int yTarget ) const;

    void UnpackRecursion
    ( const byte*& head, SplitQuasi2dHMatrix<Scalar,Conjugated>& H );

    void MapVectorPrecompute
    ( Scalar alpha, const Vector<Scalar>& xLocal ) const;
    void MapVectorNaivePassData() const;
    void MapVectorPostcompute( Vector<Scalar>& yLocal ) const;

    void MapMatrixPrecompute
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal ) const;
    void MapMatrixNaivePassData( const DenseMatrix<Scalar>& XLocal ) const;
    void MapMatrixPostcompute( DenseMatrix<Scalar>& YLocal ) const;

    void TransposeMapVectorPrecompute
    ( Scalar alpha, const Vector<Scalar>& xLocal ) const;
    void TransposeMapVectorNaivePassData
    ( const Vector<Scalar>& xLocal ) const;
    void TransposeMapVectorPostcompute
    ( Scalar alpha, Vector<Scalar>& yLocal ) const;

    void TransposeMapMatrixPrecompute
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal ) const;
    void TransposeMapMatrixNaivePassData
    ( const DenseMatrix<Scalar>& XLocal ) const;
    void TransposeMapMatrixPostcompute
    ( Scalar alpha, DenseMatrix<Scalar>& YLocal ) const;

    void HermitianTransposeMapVectorPrecompute
    ( Scalar alpha, const Vector<Scalar>& xLocal ) const;
    void HermitianTransposeMapVectorNaivePassData
    ( const Vector<Scalar>& xLocal ) const;
    void HermitianTransposeMapVectorPostcompute
    ( Scalar alpha, Vector<Scalar>& yLocal ) const;

    void HermitianTransposeMapMatrixPrecompute
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal ) const;
    void HermitianTransposeMapMatrixNaivePassData
    ( const DenseMatrix<Scalar>& XLocal ) const;
    void HermitianTransposeMapMatrixPostcompute
    ( Scalar alpha, DenseMatrix<Scalar>& YLocal ) const;
public:
    friend class DistQuasi2dHMatrix<Scalar,Conjugated>;

    static std::pair<std::size_t,std::size_t> PackedSizes
    ( const Quasi2dHMatrix<Scalar,Conjugated>& H );
    
    static std::size_t PackedSourceSize
    ( const Quasi2dHMatrix<Scalar,Conjugated>& H );

    static std::size_t PackedTargetSize
    ( const Quasi2dHMatrix<Scalar,Conjugated>& H );

    static std::pair<std::size_t,std::size_t> Pack
    ( byte* packedSourceSide, byte* packedTargetSide,
      int sourceRank, int targetRank,
      const Quasi2dHMatrix<Scalar,Conjugated>& H );

    SplitQuasi2dHMatrix( MPI_Comm comm );

    SplitQuasi2dHMatrix
    ( const byte* packedHalf, MPI_Comm comm );

    ~SplitQuasi2dHMatrix();

    std::size_t Unpack( const byte* packedHalf, MPI_Comm comm );
};

} // namespace psp

//----------------------------------------------------------------------------//
// Inlined implementations                                                    //
//----------------------------------------------------------------------------//

namespace psp {

// NOTE: The following implementations are practically identical to those of 
//       Quasi2dHMatrix. However, it is probably not worth coupling the two in
//       order to prevent code duplication.
template<typename Scalar,bool Conjugated>
inline
SplitQuasi2dHMatrix<Scalar,Conjugated>::Node::Node
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
SplitQuasi2dHMatrix<Scalar,Conjugated>::Node::~Node()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline SplitQuasi2dHMatrix<Scalar,Conjugated>&
SplitQuasi2dHMatrix<Scalar,Conjugated>::Node::Child( int i, int j )
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::Node::Child");
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
inline const SplitQuasi2dHMatrix<Scalar,Conjugated>&
SplitQuasi2dHMatrix<Scalar,Conjugated>::Node::Child( int i, int j ) const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::Node::Child");
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
SplitQuasi2dHMatrix<Scalar,Conjugated>::NodeSymmetric::NodeSymmetric
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
SplitQuasi2dHMatrix<Scalar,Conjugated>::NodeSymmetric::~NodeSymmetric()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline SplitQuasi2dHMatrix<Scalar,Conjugated>&
SplitQuasi2dHMatrix<Scalar,Conjugated>::NodeSymmetric::Child( int i, int j )
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::NodeSymmetric::Child");
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
inline const SplitQuasi2dHMatrix<Scalar,Conjugated>&
SplitQuasi2dHMatrix<Scalar,Conjugated>::NodeSymmetric::Child( int i, int j ) 
const
{
#ifndef RELEASE
    PushCallStack("SplitQuasi2dHMatrix::NodeSymmetric::Child");
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
SplitQuasi2dHMatrix<Scalar,Conjugated>::Shell::Shell()
: type(NODE), data() 
{ }

template<typename Scalar,bool Conjugated>
inline
SplitQuasi2dHMatrix<Scalar,Conjugated>::Shell::~Shell()
{
    switch( type )
    {
    case NODE:           delete data.N;  break;
    case NODE_SYMMETRIC: delete data.NS; break;
    case SPLIT_LOW_RANK: delete data.SF; break;
    case SPLIT_DENSE:    delete data.SD; break;
    }
}

} // namespace psp

#endif // PSP_SPLIT_QUASI2D_HMATRIX_HPP
