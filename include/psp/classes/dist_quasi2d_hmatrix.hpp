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

template<typename Scalar,bool Conjugated>
class DistQuasi2dHMatrix
{
private:
    /*
     * Private static functions
     */
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

    /*
     * Private data structures
     */
    typedef SplitQuasi2dHMatrix<Scalar,Conjugated> SplitQuasi2d;

    struct DistLowRankMatrix
    {
        int rank;
        DenseMatrix<Scalar> ULocal, VLocal;
    };

    struct SplitLowRankMatrix
    {
        int rank;
        DenseMatrix<Scalar> D;
    };

    struct SplitDenseMatrix
    {
        DenseMatrix<Scalar> D;
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
        DistQuasi2dHMatrix& Child( int t, int s );
        const DistQuasi2dHMatrix& Child( int t, int s ) const;
    };

    enum ShellType 
    { 
        NODE,                // recurse
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
            DistLowRankMatrix* DF;
            SplitQuasi2d* SH;
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

    struct MapVectorContext
    {
        struct NodeContext
        {
            std::vector<MapVectorContext*> children;
            NodeContext();
            ~NodeContext();
            MapVectorContext& Child( int t, int s );
            const MapVectorContext& Child( int t, int s ) const;
        };
        struct ContextShell
        {
            ShellType type;
            union Data
            {
                NodeContext* N;     
                typename SplitQuasi2d::MapVectorContext* SH;
                Vector<Scalar>* z;
                Data() { std::memset( this, 0, sizeof(Data) ); }
            } data;
            ContextShell();
            ~ContextShell();
        };
        ContextShell _shell;
    };
    
    struct MapDenseMatrixContext
    {
        struct NodeContext
        {
            std::vector<MapDenseMatrixContext*> children;
            NodeContext();
            ~NodeContext();
            MapDenseMatrixContext& Child( int t, int s );
            const MapDenseMatrixContext& Child( int t, int s ) const;
        };
        struct ContextShell
        {
            ShellType type;
            union Data
            {
                NodeContext* N;     
                typename SplitQuasi2d::MapDenseMatrixContext* SH;
                DenseMatrix<Scalar>* Z;
                Data() { std::memset( this, 0, sizeof(Data) ); }
            } data;
            ContextShell();
            ~ContextShell();
        };
        ContextShell _shell;
    };

    // HERE
    /*
    struct MapHMatrixContext
    {
        // TODO
    };
    */

    /*
     * Private data
     */
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

    /*
     * Private non-static member functions
     */
    
    // Ensure that the default constructor is not accessible, a communicator
    // must be supplied
    DistQuasi2dHMatrix();

    void UnpackRecursion
    ( const byte*& head, DistQuasi2dHMatrix<Scalar,Conjugated>& H,
      int sourceRankOffset, int targetRankOffset );

    void MapVectorPrecompute
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal, 
                          Vector<Scalar>& yLocal ) const;
    void MapVectorSummations
    ( MapVectorContext& context ) const;
    void MapVectorSummationsCount
    ( std::vector<int>& sizes, MapVectorContext& context ) const;
    void MapVectorSummationsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets, 
      MapVectorContext& context ) const;
    void MapVectorSummationsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapVectorContext& context ) const;
    void MapVectorNaiveSummations
    ( MapVectorContext& context ) const;
    void MapVectorPassData
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorNaivePassData
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void MapVectorBroadcasts
    ( MapVectorContext& context ) const;
    void MapVectorBroadcastsCount
    ( std::vector<int>& sizes, MapVectorContext& context ) const;
    void MapVectorBroadcastsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapVectorContext& context ) const;
    void MapVectorBroadcastsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapVectorContext& context ) const;
    void MapVectorNaiveBroadcasts
    ( MapVectorContext& context ) const;
    void MapVectorPostcompute
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    void MapMatrixPrecompute
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixSummations
    ( MapDenseMatrixContext& context ) const;
    void MapMatrixSummationsCount
    ( std::vector<int>& sizes, MapDenseMatrixContext& context ) const;
    void MapMatrixSummationsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void MapMatrixSummationsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void MapMatrixNaiveSummations
    ( MapDenseMatrixContext& context ) const;
    void MapMatrixPassData
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixNaivePassData
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixBroadcasts
    ( MapDenseMatrixContext& context ) const;
    void MapMatrixBroadcastsCount
    ( std::vector<int>& sizes, MapDenseMatrixContext& context ) const;
    void MapMatrixBroadcastsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void MapMatrixBroadcastsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void MapMatrixNaiveBroadcasts
    ( MapDenseMatrixContext& context ) const;
    void MapMatrixPostcompute
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;

    void TransposeMapVectorPrecompute
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorSummations
    ( MapVectorContext& context ) const;
    void TransposeMapVectorSummationsCount
    ( std::vector<int>& sizes, MapVectorContext& context ) const;
    void TransposeMapVectorSummationsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapVectorContext& context ) const;
    void TransposeMapVectorSummationsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapVectorContext& context ) const;
    void TransposeMapVectorNaiveSummations
    ( MapVectorContext& context ) const;
    void TransposeMapVectorPassData
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorNaivePassData
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorBroadcasts
    ( MapVectorContext& context ) const;
    void TransposeMapVectorBroadcastsCount
    ( std::vector<int>& sizes, MapVectorContext& context ) const;
    void TransposeMapVectorBroadcastsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapVectorContext& context ) const;
    void TransposeMapVectorBroadcastsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapVectorContext& context ) const;
    void TransposeMapVectorNaiveBroadcasts
    ( MapVectorContext& context ) const;
    void TransposeMapVectorPostcompute
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    void TransposeMapMatrixPrecompute
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixSummations
    ( MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixSummationsCount
    ( std::vector<int>& sizes, MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixSummationsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixSummationsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixNaiveSummations
    ( MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixPassData
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixNaivePassData
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixBroadcasts
    ( MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixBroadcastsCount
    ( std::vector<int>& sizes, MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixBroadcastsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixBroadcastsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixNaiveBroadcasts
    ( MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixPostcompute
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;

    void HermitianTransposeMapVectorPrecompute
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void HermitianTransposeMapVectorSummations
    ( MapVectorContext& context ) const;
    void HermitianTransposeMapVectorNaiveSummations
    ( MapVectorContext& context ) const;
    void HermitianTransposeMapVectorPassData
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void HermitianTransposeMapVectorNaivePassData
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void HermitianTransposeMapVectorBroadcasts
    ( MapVectorContext& context ) const;
    void HermitianTransposeMapVectorNaiveBroadcasts
    ( MapVectorContext& context ) const;
    void HermitianTransposeMapVectorPostcompute
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    void HermitianTransposeMapMatrixPrecompute
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void HermitianTransposeMapMatrixSummations
    ( MapDenseMatrixContext& context ) const;
    void HermitianTransposeMapMatrixNaiveSummations
    ( MapDenseMatrixContext& context ) const;
    void HermitianTransposeMapMatrixPassData
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void HermitianTransposeMapMatrixNaivePassData
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void HermitianTransposeMapMatrixBroadcasts
    ( MapDenseMatrixContext& context ) const;
    void HermitianTransposeMapMatrixNaiveBroadcasts
    ( MapDenseMatrixContext& context ) const;
    void HermitianTransposeMapMatrixPostcompute
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;

public:
    /*
     * Public static member functions
     */
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

    /*
     * Public non-static member functions
     */
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

/*
 * Private structure member functions
 */
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
DistQuasi2dHMatrix<Scalar,Conjugated>::Node::Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::Node::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[s+4*t];
}

template<typename Scalar,bool Conjugated>
inline const DistQuasi2dHMatrix<Scalar,Conjugated>&
DistQuasi2dHMatrix<Scalar,Conjugated>::Node::Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::Node::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[s+4*t];
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

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext::
NodeContext::NodeContext()
: children(16)
{
    for( int i=0; i<16; ++i )
        children[i] = new MapVectorContext();
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext::
NodeContext::~NodeContext()
{
    for( int i=0; i<16; ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline typename DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext&
DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext::
NodeContext::Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorContext::NodeContext::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[s+4*t];
}

template<typename Scalar,bool Conjugated>
inline const typename DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext&
DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext::
NodeContext::Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMatrix::MapVectorContext::NodeContext::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[s+4*t];
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext::
ContextShell::ContextShell()
: type(EMPTY), data()
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext::
ContextShell::~ContextShell()
{
    switch( type )
    {
    case NODE: 
        delete data.N; break;

    case SPLIT_QUASI2D: 
        delete data.SH; break;

    case DIST_LOW_RANK:  
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
        delete data.z; break;

    case QUASI2D:
    case LOW_RANK:
    case DENSE:
    case EMPTY: 
        break;
    }
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext::
NodeContext::NodeContext()
: children(16)
{
    for( int i=0; i<16; ++i )
        children[i] = new MapDenseMatrixContext();
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext::
NodeContext::~NodeContext()
{
    for( int i=0; i<16; ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline typename DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext&
DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext::
NodeContext::Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::MapDenseMatrixContext::NodeContext::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[s+4*t];
}

template<typename Scalar,bool Conjugated>
inline const typename 
DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext&
DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext::
NodeContext::Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::MapDenseMatrixContext::NodeContext::Child");
    if( t < 0 || s < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( t > 3 || s > 3 )
        throw std::logic_error("Indices out of bounds");
    if( children.size() != 16 )
        throw std::logic_error("children array not yet set up");
    PopCallStack();
#endif
    return *children[s+4*t];
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext::
ContextShell::ContextShell()
: type(EMPTY), data()
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext::
ContextShell::~ContextShell()
{
    switch( type )
    {
    case NODE: 
        delete data.N; break;

    case SPLIT_QUASI2D: 
        delete data.SH; break;

    case DIST_LOW_RANK:  
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
        delete data.Z; break;

    case QUASI2D:
    case LOW_RANK:
    case DENSE:
    case EMPTY: 
        break;
    }
}

} // namespace psp

#endif // PSP_DIST_QUASI2D_HMATRIX_HPP
