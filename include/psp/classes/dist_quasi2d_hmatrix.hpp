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
#ifndef PSP_DIST_QUASI2D_HMATRIX_HPP
#define PSP_DIST_QUASI2D_HMATRIX_HPP 1

#include "psp/classes/quasi2d_hmatrix.hpp"
#include "psp/classes/split_quasi2d_hmatrix.hpp"
#include "psp/classes/subcomms.hpp"

namespace psp {

// We will enforce the requirement that is a power of 2 numbers or processes, 
// but not more than 4^{numLevels-1}.
template<typename Scalar,bool Conjugated>
class DistQuasi2dHMatrix
{
private:
    static void PackedSizesRecursion
    ( std::vector<std::size_t>& packedSizes,
      const std::vector<int>& localSizes,
      int sourceRankOffset, int targetRankOffset, int teamSize,
      const Quasi2dHMatrix<Scalar,Conjugated>& H );

    static void PackRecursion
    ( std::vector<byte**>& headPointers,
      const std::vector<int>& localSizes,
      int sourceRankOffset, int targetRankOffset, int teamSize,
      const Quasi2dHMatrix<Scalar,Conjugated>& H );

    static void ComputeLocalDimensionRecursion
    ( int& localDim, int p, int rank, int xSize, int ySize, int zSize );

    static void ComputeFirstLocalIndexRecursion
    ( int& firstLocalIndex, int p, int rank, int xSize, int ySize, int zSize );

    static void ComputeLocalSizesRecursion
    ( int* localSizes, int teamSize, int xSize, int ySize, int zSize );

    struct DistLowRankMatrix
    {
        int rank;
        DenseMatrix<Scalar> ULocal, VLocal;
        mutable Vector<Scalar> z;
        mutable DenseMatrix<Scalar> Z;
    };

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
        std::vector<DistQuasi2dHMatrix*> children;
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
        DistQuasi2dHMatrix& Child( int i, int j );
        const DistQuasi2dHMatrix& Child( int i, int j ) const;
    };

    struct NodeSymmetric
    {
        std::vector<DistQuasi2dHMatrix*> children;
        int xSizes[2];
        int ySizes[2];
        int sizes[4];
        NodeSymmetric( int xSize, int ySize, int zSize );
        ~NodeSymmetric();
        DistQuasi2dHMatrix& Child( int i, int j );
        const DistQuasi2dHMatrix& Child( int i, int j ) const;
    };

    enum ShellType 
    { 
        NODE,                // recurse
        NODE_SYMMETRIC,      // recurse symmetrically
        DIST_LOW_RANK,       // each side is distributed
        SPLIT_QUASI2D,       // split between two processes
        SPLIT_LOW_RANK,      // each side is given to a different process
        SPLIT_DENSE,         // split between two processes
        QUASI2D,             // serial
        LOW_RANK,            // serial
        DENSE,               // serial
        EMPTY
    };

    struct Shell
    {
        ShellType type;
        union Data
        {
            Node* N;
            NodeSymmetric* NS;
            DistLowRankMatrix* DF;
            SplitQuasi2dHMatrix<Scalar,Conjugated>* SH;
            SplitLowRankMatrix* SF;
            SplitDenseMatrix* SD;
            Quasi2dHMatrix<Scalar,Conjugated>* H;
            LowRankMatrix<Scalar,Conjugated>* F;
            DenseMatrix<Scalar>* D;
            Data() { std::memset( this, 0, sizeof(Data) ); }
        } data;
        Shell();
        ~Shell();
    };

    int _height, _width;
    int _numLevels;
    int _maxRank;
    int _sourceOffset, _targetOffset;
    bool _stronglyAdmissible;

    int _xSizeSource, _xSizeTarget;
    int _ySizeSource, _ySizeTarget;
    int _zSize;
    int _xSource, _xTarget;
    int _ySource, _yTarget;
    Shell _shell;

    const Subcomms* _subcomms;
    unsigned _level;
    bool _inSourceTeam;
    bool _inTargetTeam;
    int _rootOfOtherTeam; // only applies if in only source or target team
    int _localSourceOffset;
    int _localTargetOffset;

    void UnpackRecursion
    ( const byte*& head, DistQuasi2dHMatrix<Scalar,Conjugated>& H,
      int sourceRankOffset, int targetRankOffset );

    // Ensure that the default constructor is not accessible, a communicator
    // must be supplied
    DistQuasi2dHMatrix();

    void MapVectorPrecompute
    ( Scalar alpha, const Vector<Scalar>& xLocal, 
                          Vector<Scalar>& yLocal ) const;
    void MapVectorSummations
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorSummationsCount
    ( std::vector<int>& sizes,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorSummationsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets, 
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorSummationsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorNaiveSummations
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorPassData
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorNaivePassData
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorBroadcasts
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorBroadcastsCount
    ( std::vector<int>& sizes,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorBroadcastsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorBroadcastsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorNaiveBroadcasts
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorPostcompute
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    void MapMatrixPrecompute
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixSummations
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixSummationsCount
    ( std::vector<int>& sizes,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixSummationsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixSummationsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixNaiveSummations
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixPassData
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixNaivePassData
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixBroadcasts
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixBroadcastsCount
    ( std::vector<int>& sizes,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixBroadcastsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixBroadcastsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixNaiveBroadcasts
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixPostcompute
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;

    void TransposeMapVectorPrecompute
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorSummations
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorSummationsCount
    ( std::vector<int>& sizes,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorSummationsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorSummationsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorNaiveSummations
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorPassData
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorNaivePassData
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorBroadcasts
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorBroadcastsCount
    ( std::vector<int>& sizes,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorBroadcastsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorBroadcastsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorNaiveBroadcasts
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorPostcompute
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    void TransposeMapMatrixPrecompute
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixSummations
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixSummationsCount
    ( std::vector<int>& sizes,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixSummationsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixSummationsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixNaiveSummations
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixPassData
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixNaivePassData
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixBroadcasts
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixBroadcastsCount
    ( std::vector<int>& sizes,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixBroadcastsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixBroadcastsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixNaiveBroadcasts
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixPostcompute
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;

    void HermitianTransposeMapVectorPrecompute
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void HermitianTransposeMapVectorSummations
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void HermitianTransposeMapVectorNaiveSummations
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void HermitianTransposeMapVectorPassData
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void HermitianTransposeMapVectorNaivePassData
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void HermitianTransposeMapVectorBroadcasts
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void HermitianTransposeMapVectorNaiveBroadcasts
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void HermitianTransposeMapVectorPostcompute
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    void HermitianTransposeMapMatrixPrecompute
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void HermitianTransposeMapMatrixSummations
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void HermitianTransposeMapMatrixNaiveSummations
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void HermitianTransposeMapMatrixPassData
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void HermitianTransposeMapMatrixNaivePassData
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void HermitianTransposeMapMatrixBroadcasts
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void HermitianTransposeMapMatrixNaiveBroadcasts
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void HermitianTransposeMapMatrixPostcompute
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;

public:
    static std::size_t PackedSizes
    ( std::vector<std::size_t>& packedSizes,
      const Quasi2dHMatrix<Scalar,Conjugated>& H, const Subcomms& subcomms );

    static std::size_t Pack
    ( std::vector<byte*>& packedPieces, 
      const Quasi2dHMatrix<Scalar,Conjugated>& H, const Subcomms& subcomms );

    static int ComputeLocalHeight
    ( int p, int rank, const Quasi2dHMatrix<Scalar,Conjugated>& H );

    static int ComputeLocalWidth
    ( int p, int rank, const Quasi2dHMatrix<Scalar,Conjugated>& H );

    static int ComputeFirstLocalRow
    ( int p, int rank, const Quasi2dHMatrix<Scalar,Conjugated>& H );

    static int ComputeFirstLocalCol
    ( int p, int rank, const Quasi2dHMatrix<Scalar,Conjugated>& H );

    static void ComputeLocalSizes
    ( std::vector<int>& localSizes, 
      const Quasi2dHMatrix<Scalar,Conjugated>& H );

    DistQuasi2dHMatrix( const Subcomms& subcomms );
    DistQuasi2dHMatrix
    ( const Subcomms& subcomms, unsigned level, 
      bool inSourceTeam, bool inTargetTeam, 
      int localSourceOffset=0, int localTargetOffset=0 );
    DistQuasi2dHMatrix( const byte* packedPiece, const Subcomms& subcomms );
    ~DistQuasi2dHMatrix();

    int LocalHeight() const;
    int LocalWidth() const;

    int FirstLocalRow() const;
    int FirstLocalCol() const;

    // Unpack this process's portion of the DistQuasi2dHMatrix
    std::size_t Unpack
    ( const byte* packedDistHMatrix, const Subcomms& subcomms );

    // y := alpha H x
    void MapVector
    ( Scalar alpha, const Vector<Scalar>& xLocal, 
                          Vector<Scalar>& yLocal ) const;

    // y := alpha H x + beta y
    void MapVector
    ( Scalar alpha, const Vector<Scalar>& xLocal, 
      Scalar beta,        Vector<Scalar>& yLocal ) const;

    // Y := alpha H X
    void MapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;

    // Y := alpha H X + beta Y
    void MapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
      Scalar beta,        DenseMatrix<Scalar>& YLocal ) const;

    // y := alpha H^T x
    void TransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& xLocal, 
                          Vector<Scalar>& yLocal ) const;

    // y := alpha H^T x + beta y
    void TransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& xLocal, 
      Scalar beta,        Vector<Scalar>& yLocal ) const;

    // Y := alpha H^T X
    void TransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;

    // Y := alpha H^T X + beta Y
    void TransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
      Scalar beta,        DenseMatrix<Scalar>& YLocal ) const;

    // y := alpha H' x
    void HermitianTransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    // y := alpha H' x + beta y
    void HermitianTransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& xLocal,
      Scalar beta,        Vector<Scalar>& yLocal ) const;

    // Y := alpha H' X
    void HermitianTransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;

    // Y := alpha H' X + beta Y
    void HermitianTransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& XLocal,
      Scalar beta,        DenseMatrix<Scalar>& YLocal ) const;
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
DistQuasi2dHMatrix<Scalar,Conjugated>::Node::Node
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
DistQuasi2dHMatrix<Scalar,Conjugated>::Node::~Node()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline DistQuasi2dHMatrix<Scalar,Conjugated>&
DistQuasi2dHMatrix<Scalar,Conjugated>::Node::Child( int i, int j )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::Node::Child");
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
inline const DistQuasi2dHMatrix<Scalar,Conjugated>&
DistQuasi2dHMatrix<Scalar,Conjugated>::Node::Child( int i, int j ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::Node::Child");
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
DistQuasi2dHMatrix<Scalar,Conjugated>::NodeSymmetric::NodeSymmetric
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
DistQuasi2dHMatrix<Scalar,Conjugated>::NodeSymmetric::~NodeSymmetric()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline DistQuasi2dHMatrix<Scalar,Conjugated>&
DistQuasi2dHMatrix<Scalar,Conjugated>::NodeSymmetric::Child( int i, int j )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::NodeSymmetric::Child");
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
inline const DistQuasi2dHMatrix<Scalar,Conjugated>&
DistQuasi2dHMatrix<Scalar,Conjugated>::NodeSymmetric::Child( int i, int j )
const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::NodeSymmetric::Child");
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
DistQuasi2dHMatrix<Scalar,Conjugated>::Shell::Shell()
: type(EMPTY), data() 
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::Shell::~Shell()
{ 
    switch( type )
    {
    case NODE:           delete data.N;  break;
    case NODE_SYMMETRIC: delete data.NS; break;
    case DIST_LOW_RANK:  delete data.DF; break;
    case SPLIT_QUASI2D:  delete data.SH; break;
    case SPLIT_LOW_RANK: delete data.SF; break;
    case SPLIT_DENSE:    delete data.SD; break;
    case QUASI2D:        delete data.H;  break;
    case LOW_RANK:       delete data.F;  break;
    case DENSE:          delete data.D;  break;
    case EMPTY: break;
    }
}

} // namespace psp

#endif // PSP_DIST_QUASI2D_HMATRIX_HPP
