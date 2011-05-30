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

#include "psp/building_blocks/mpi.hpp"
#include "psp/quasi2d_hmatrix.hpp"

namespace psp {

template<typename Scalar,bool Conjugated>
class DistQuasi2dHMatrix
{
public:
    /*
     * Public data structures
     */
    class Subcomms
    {
    private:
        std::vector<MPI_Comm> _subcomms;
    public:
        Subcomms( MPI_Comm comm );
        ~Subcomms();

        unsigned NumLevels() const;
        MPI_Comm Subcomm( unsigned level ) const;
    };

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
    ( int numLevels, int maxRank, bool stronglyAdmissible, 
      int sourceOffset, int targetOffset,
      int xSizeSource, int xSizeTarget, int ySizeSource, int ySizeTarget,
      int zSize, int xSource, int xTarget, int ySource, int yTarget,
      const Subcomms& subcomms, unsigned level, 
      bool inSourceTeam, bool inTargetTeam, 
      int localSourceOffset=0, int localTargetOffset=0 );
    DistQuasi2dHMatrix( const byte* packedPiece, const Subcomms& subcomms );
    ~DistQuasi2dHMatrix();
    void Clear();

    int Height() const;
    int Width() const;

    int LocalHeight() const;
    int LocalWidth() const;

    int FirstLocalRow() const;
    int FirstLocalCol() const;

    void RequireRoot() const;

    // Print out the structure of the tree that we're aware of
    void WriteLocalStructure( const std::string& basename ) const;

    // Unpack this process's portion of the DistQuasi2dHMatrix
    std::size_t Unpack
    ( const byte* packedDistHMatrix, const Subcomms& subcomms );

    // Union the structure known in each block row and column at each level.
    void FormGhostNodes();

    // Return to the minimal local structure
    void PruneGhostNodes();

    bool Ghosted() const;

    // A := alpha A
    void Scale( Scalar alpha );

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

    // C := alpha A B 
    void MapMatrix
    ( Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
                          DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const;

    // C := alpha A B + beta C
    void MapMatrix
    ( Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
      Scalar beta,        DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const;

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

    // C := alpha A^T B
    void TransposeMapMatrix
    ( Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
                          DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const;

    // C := alpha A^T B + beta C
    void TransposeMapMatrix
    ( Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
      Scalar beta,        DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const;

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

    // C := alpha A' B
    void HermitianTransposeMapMatrix
    ( Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
                          DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const;

    // C := alpha A' B + beta C
    void HermitianTransposeMapMatrix
    ( Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
      Scalar beta,        DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const;

private:
    /*
     * Private data structures
     */

    struct DistLowRankMatrix
    {
        int rank;
        DenseMatrix<Scalar> ULocal, VLocal;
    };

    struct DistLowRankMatrixGhost
    {
        int rank;
        int sourceRoot, targetRoot;
    };

    struct SplitLowRankMatrix
    {
        int rank;
        DenseMatrix<Scalar> D;
    };

    struct SplitLowRankMatrixGhost
    {
        int rank;
        int sourceOwner, targetOwner;
    };

    struct LowRankMatrixGhost
    {
        int rank;
        int owner;
    };

    struct SplitDenseMatrix
    {
        DenseMatrix<Scalar> D;
    };

    struct SplitDenseMatrixGhost
    {
        int sourceOwner, targetOwner;
    };

    struct DenseMatrixGhost
    {
        int owner;
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
    Node* NewNode() const;

    struct NodeGhost : public Node
    {
        int sourceRoot, targetRoot;    
        NodeGhost
        ( int xSizeSource, int xSizeTarget,
          int ySizeSource, int ySizeTarget,
          int zSize,
          int sRoot, int tRoot );
    };
    NodeGhost* NewNodeGhost( int sRoot, int tRoot ) const;

    enum BlockType 
    { 
        DIST_NODE,        // each side is distributed
        DIST_NODE_GHOST,  //
        SPLIT_NODE,       // each side is owned by a single process
        SPLIT_NODE_GHOST, // 
        NODE,             // serial
        NODE_GHOST,       //

        DIST_LOW_RANK,        // each side is distributed
        DIST_LOW_RANK_GHOST,  //
        SPLIT_LOW_RANK,       // each side is given to a different process
        SPLIT_LOW_RANK_GHOST, //
        LOW_RANK,             // serial
        LOW_RANK_GHOST,       //

        SPLIT_DENSE,       // split between two processes
        SPLIT_DENSE_GHOST, //
        DENSE,             // serial
        DENSE_GHOST,       //

        EMPTY
    };

    struct Block
    {
        BlockType type;
        union Data
        {
            Node* N;
            NodeGhost* NG;

            DistLowRankMatrix* DF;
            DistLowRankMatrixGhost* DFG;
            SplitLowRankMatrix* SF;
            SplitLowRankMatrixGhost* SFG;
            LowRankMatrix<Scalar,Conjugated>* F;
            LowRankMatrixGhost* FG;

            SplitDenseMatrix* SD;
            SplitDenseMatrixGhost* SDG;
            DenseMatrix<Scalar>* D;
            DenseMatrixGhost* DG;

            Data() { std::memset( this, 0, sizeof(Data) ); }
        } data;
        Block();
        ~Block();
        void Clear();
    };

    struct BlockId
    {
        int level;
        int sourceOffset, targetOffset;
    };

    struct MapVectorContext
    {
        struct DistNodeContext
        {
            std::vector<MapVectorContext*> children;
            DistNodeContext();
            ~DistNodeContext();
            MapVectorContext& Child( int t, int s );
            const MapVectorContext& Child( int t, int s ) const;
        };
        // For now this will be the same as the DistNodeContext, but we will
        // eventually have it combine all precompute data into a single buffer.
        typedef DistNodeContext SplitNodeContext;

        struct ContextBlock
        {
            BlockType type;
            union Data
            {
                DistNodeContext* DN;
                SplitNodeContext* SN;
                Vector<Scalar>* z;
                Data() { std::memset( this, 0, sizeof(Data) ); }
            } data;
            ContextBlock();
            ~ContextBlock();
            void Clear();
        };
        ContextBlock block;
        void Clear();
    };
    
    struct MapDenseMatrixContext
    {
        struct DistNodeContext
        {
            std::vector<MapDenseMatrixContext*> children;
            DistNodeContext();
            ~DistNodeContext();
            MapDenseMatrixContext& Child( int t, int s );
            const MapDenseMatrixContext& Child( int t, int s ) const;
        };
        typedef DistNodeContext SplitNodeContext;

        struct ContextBlock
        {
            BlockType type;
            union Data
            {
                DistNodeContext* DN;
                SplitNodeContext* SN;
                DenseMatrix<Scalar>* Z;
                Data() { std::memset( this, 0, sizeof(Data) ); }
            } data;
            ContextBlock();
            ~ContextBlock();
            void Clear();
        };
        ContextBlock block;
        void Clear();
    };

    struct MapHMatrixContext
    {
        template<typename T1,typename T2>
        class MemoryMap 
        {
        private:
            std::map<T1,T2*> _map;
        public:
            // NOTE: Insertion with the same key without manual deletion
            //       will cause a memory leak.
            T2*& operator[]( T1 key )
            {
                return _map[key];
            }

            ~MemoryMap()
            {
                typename std::map<T1,T2*>::iterator it;
                for( it=_map.begin(); it!=_map.end(); it++ )
                    delete (*it).second;
            }
        };

        /*
         * Structs for the different types of matrix blocks. Some store several
         * types of updates.
         */
        struct DistNodeContext
        {
            std::vector<MapHMatrixContext*> children;
            DistNodeContext();
            ~DistNodeContext();
            MapHMatrixContext& Child( int t, int s );
            const MapHMatrixContext& Child( int t, int s ) const;

            // For low-rank updates
            MemoryMap<int,MapDenseMatrixContext> denseContextMap;
            MemoryMap<int,DenseMatrix<Scalar> > ULocalMap, VLocalMap;
            MemoryMap<int,DenseMatrix<Scalar> > ZMap;
        };

        struct SplitNodeContext
        {
            std::vector<MapHMatrixContext*> children;
            SplitNodeContext();
            ~SplitNodeContext();
            MapHMatrixContext& Child( int t, int s );
            const MapHMatrixContext& Child( int t, int s ) const;

            // For low-rank updates
            MemoryMap<int,MapDenseMatrixContext> denseContextMap;
            MemoryMap<int,DenseMatrix<Scalar> > UOrVMap;
        };

        struct NodeContext
        {
            std::vector<MapHMatrixContext*> children;
            NodeContext();
            ~NodeContext();
            MapHMatrixContext& Child( int t, int s );
            const MapHMatrixContext& Child( int t, int s ) const;

            // For low-rank updates
            MemoryMap<int,MapDenseMatrixContext> denseContextMap;
            MemoryMap<int,DenseMatrix<Scalar> > UMap, VMap;
        };

        struct DistLowRankContext
        {
            // For low-rank updates
            MemoryMap<int,MapDenseMatrixContext> denseContextMap;
            MemoryMap<int,DenseMatrix<Scalar> > ULocalMap, VLocalMap;

            // For temporary inner products
            MemoryMap<int,DenseMatrix<Scalar> > ZMap;
        };

        struct SplitLowRankContext
        {
            // For low-rank updates
            MemoryMap<int,MapDenseMatrixContext> denseContextMap;
            MemoryMap<int,DenseMatrix<Scalar> > UOrVMap;

            // For dense updates
            MemoryMap<int,DenseMatrix<Scalar> > DMap;
        };

        struct LowRankContext
        {
            // For low-rank updates
            MemoryMap<int,MapDenseMatrixContext> denseContextMap;
            MemoryMap<int,DenseMatrix<Scalar> > UMap, VMap;

            // For dense updates
            MemoryMap<int,DenseMatrix<Scalar> > DMap;

            // For temporary inner products
            MemoryMap<int,DenseMatrix<Scalar> > ZMap;
        };

        struct SplitDenseContext
        {
            // For low-rank updates
            MemoryMap<int,DenseMatrix<Scalar> > UOrVMap;

            // For dense updates
            MemoryMap<int,DenseMatrix<Scalar> > DMap;

            // For temporary inner products
            MemoryMap<int,DenseMatrix<Scalar> > ZMap;
        };

        struct DenseContext
        {
            // For low-rank updates
            MemoryMap<int,DenseMatrix<Scalar> > UMap, VMap;

            // For dense updates
            MemoryMap<int,DenseMatrix<Scalar> > DMap;

            // For temporary inner products
            MemoryMap<int,DenseMatrix<Scalar> > ZMap;
        };

        /*
         * Wrapper for storing different types of contexts.
         */
        struct ContextBlock
        {
            BlockType type;
            union Data
            {
                DistNodeContext* DN;
                SplitNodeContext* SN;
                NodeContext* N;

                DistLowRankContext* DF;
                SplitLowRankContext* SF;
                LowRankContext* F;

                SplitDenseContext* SD;
                DenseContext* D;
            } data;
            ContextBlock();
            ~ContextBlock();
            void Clear();
        };
        ContextBlock block;
        void Clear();
    };

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
     * Private non-static member functions
     */
    bool Admissible() const;
    bool Admissible( int xSource, int xTarget, int ySource, int yTarget ) const;

    void WriteLocalStructureRecursion( std::ofstream& file ) const;
    
    // Ensure that the default constructor is not accessible, a communicator
    // must be supplied
    DistQuasi2dHMatrix();

    void UnpackRecursion
    ( const byte*& head, int sourceRankOffset, int targetRankOffset );

    void FillStructureRecursion
    ( std::vector< std::set<int> >& sourceStructure,
      std::vector< std::set<int> >& targetStructure ) const;

    void FindGhostNodesRecursion
    ( std::vector< std::vector<BlockId> >& blockIds,
      const std::vector< std::set<int> >& sourceStructure,
      const std::vector< std::set<int> >& targetStructure,
      int sourceRoot, int targetRoot );

    void GetRank( const BlockId& blockId, int& rank ) const;
    void SetGhostRank( const BlockId& blockId, const int rank );

    void MapVectorInitialize
    ( MapVectorContext& context ) const;
    void MapVectorPrecompute
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal, 
                          Vector<Scalar>& yLocal ) const;
    void MapVectorSummations
    ( MapVectorContext& context ) const;
    void MapVectorSummationsCount
    ( std::vector<int>& sizes ) const;
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
    void MapVectorBroadcasts
    ( MapVectorContext& context ) const;
    void MapVectorBroadcastsCount
    ( std::vector<int>& sizes ) const;
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

    void MapMatrixInitialize
    ( MapDenseMatrixContext& context ) const;
    void MapMatrixPrecompute
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixSummations
    ( MapDenseMatrixContext& context, int width ) const;
    void MapMatrixSummationsCount
    ( std::vector<int>& sizes, int width ) const;
    void MapMatrixSummationsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void MapMatrixSummationsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void MapMatrixNaiveSummations
    ( MapDenseMatrixContext& context, int width ) const;
    void MapMatrixPassData
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;
    void MapMatrixBroadcasts
    ( MapDenseMatrixContext& context, int width ) const;
    void MapMatrixBroadcastsCount
    ( std::vector<int>& sizes, int width ) const;
    void MapMatrixBroadcastsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void MapMatrixBroadcastsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context, int width ) const;
    void MapMatrixNaiveBroadcasts
    ( MapDenseMatrixContext& context, int width ) const;
    void MapMatrixPostcompute
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal, 
                          DenseMatrix<Scalar>& YLocal ) const;

    void MapMatrixPrecompute
    ( MapHMatrixContext& context,
      Scalar alpha, const DistQuasi2dHMatrix<Scalar,Conjugated>& B,
                          DistQuasi2dHMatrix<Scalar,Conjugated>& C ) const;
    // TODO

    void TransposeMapVectorInitialize
    ( MapVectorContext& context ) const;
    void TransposeMapVectorPrecompute
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMapVectorSummations
    ( MapVectorContext& context ) const;
    void TransposeMapVectorSummationsCount
    ( std::vector<int>& sizes ) const;
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
    void TransposeMapVectorBroadcasts
    ( MapVectorContext& context ) const;
    void TransposeMapVectorBroadcastsCount
    ( std::vector<int>& sizes ) const;
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

    void TransposeMapMatrixInitialize
    ( MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixPrecompute
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixSummations
    ( MapDenseMatrixContext& context, int width ) const;
    void TransposeMapMatrixSummationsCount
    ( std::vector<int>& sizes, int width ) const;
    void TransposeMapMatrixSummationsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixSummationsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixNaiveSummations
    ( MapDenseMatrixContext& context, int width ) const;
    void TransposeMapMatrixPassData
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void TransposeMapMatrixBroadcasts
    ( MapDenseMatrixContext& context, int width ) const;
    void TransposeMapMatrixBroadcastsCount
    ( std::vector<int>& sizes, int width ) const;
    void TransposeMapMatrixBroadcastsPack
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context ) const;
    void TransposeMapMatrixBroadcastsUnpack
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets,
      MapDenseMatrixContext& context, int width ) const;
    void TransposeMapMatrixNaiveBroadcasts
    ( MapDenseMatrixContext& context, int width ) const;
    void TransposeMapMatrixPostcompute
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;

    void HermitianTransposeMapVectorInitialize
    ( MapVectorContext& context ) const;
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
    void HermitianTransposeMapVectorBroadcasts
    ( MapVectorContext& context ) const;
    void HermitianTransposeMapVectorNaiveBroadcasts
    ( MapVectorContext& context ) const;
    void HermitianTransposeMapVectorPostcompute
    ( MapVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    void HermitianTransposeMapMatrixInitialize
    ( MapDenseMatrixContext& context ) const;
    void HermitianTransposeMapMatrixPrecompute
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void HermitianTransposeMapMatrixSummations
    ( MapDenseMatrixContext& context, int width ) const;
    void HermitianTransposeMapMatrixNaiveSummations
    ( MapDenseMatrixContext& context, int width ) const;
    void HermitianTransposeMapMatrixPassData
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;
    void HermitianTransposeMapMatrixBroadcasts
    ( MapDenseMatrixContext& context, int width ) const;
    void HermitianTransposeMapMatrixNaiveBroadcasts
    ( MapDenseMatrixContext& context, int width ) const;
    void HermitianTransposeMapMatrixPostcompute
    ( MapDenseMatrixContext& context,
      Scalar alpha, const DenseMatrix<Scalar>& XLocal,
                          DenseMatrix<Scalar>& YLocal ) const;

    /*
     * Private data
     */
    int _numLevels;
    int _maxRank;
    int _sourceOffset, _targetOffset;
    bool _stronglyAdmissible;

    int _xSizeSource, _xSizeTarget;
    int _ySizeSource, _ySizeTarget;
    int _zSize;
    int _xSource, _xTarget;
    int _ySource, _yTarget;
    Block _block;

    const Subcomms* _subcomms;
    unsigned _level;
    bool _inSourceTeam;
    bool _inTargetTeam;
    int _rootOfOtherTeam; // only applies if in only source or target team
    int _localSourceOffset;
    int _localTargetOffset;

    // Create shortened names for convenience in implementations.
    typedef DenseMatrix<Scalar> Dense;
    typedef DenseMatrixGhost DenseGhost;
    typedef LowRankMatrix<Scalar,Conjugated> LowRank;
    typedef LowRankMatrixGhost LowRankGhost;
    typedef Quasi2dHMatrix<Scalar,Conjugated> Quasi2d;
    typedef SplitDenseMatrix SplitDense;
    typedef SplitDenseMatrixGhost SplitDenseGhost;
    typedef SplitLowRankMatrix SplitLowRank;
    typedef SplitLowRankMatrixGhost SplitLowRankGhost;
    typedef DistLowRankMatrix DistLowRank;
    typedef DistLowRankMatrixGhost DistLowRankGhost;
    typedef DistQuasi2dHMatrix<Scalar,Conjugated> DistQuasi2d;
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
inline typename DistQuasi2dHMatrix<Scalar,Conjugated>::Node*
DistQuasi2dHMatrix<Scalar,Conjugated>::NewNode() const
{
    return 
        new Node
        ( _xSizeSource, _xSizeTarget, _ySizeSource, _ySizeTarget, _zSize );
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::NodeGhost::NodeGhost
( int xSizeSource, int xSizeTarget,
  int ySizeSource, int ySizeTarget,
  int zSize,
  int sRoot, int tRoot )
: Node(xSizeSource,xSizeTarget,ySizeSource,ySizeTarget,zSize),
  sourceRoot(sRoot), targetRoot(tRoot)
{ }

template<typename Scalar,bool Conjugated>
inline typename DistQuasi2dHMatrix<Scalar,Conjugated>::NodeGhost*
DistQuasi2dHMatrix<Scalar,Conjugated>::NewNodeGhost( int sRoot, int tRoot ) 
const
{
    return 
        new NodeGhost
        ( _xSizeSource, _xSizeTarget, _ySizeSource, _ySizeTarget, _zSize,
          sRoot, tRoot );
}


template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::Block::Block()
: type(EMPTY), data() 
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::Block::~Block()
{ 
    Clear();
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMatrix<Scalar,Conjugated>::Block::Clear()
{
    switch( type )
    {
    case DIST_NODE:
    case SPLIT_NODE:
    case NODE:
        delete data.N; break;

    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
        delete data.NG; break;

    case DIST_LOW_RANK:  delete data.DF; break;
    case SPLIT_LOW_RANK: delete data.SF; break;
    case LOW_RANK:       delete data.F;  break;

    case DIST_LOW_RANK_GHOST:  delete data.DFG; break;
    case SPLIT_LOW_RANK_GHOST: delete data.SFG; break;
    case LOW_RANK_GHOST:       delete data.FG;  break;

    case SPLIT_DENSE: delete data.SD; break;
    case DENSE:       delete data.D;  break;

    case SPLIT_DENSE_GHOST: delete data.SDG; break;
    case DENSE_GHOST:       delete data.DG;  break;

    case EMPTY: break;
    }
    type = EMPTY;
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext::
DistNodeContext::DistNodeContext()
: children(16)
{
    for( int i=0; i<16; ++i )
        children[i] = new MapVectorContext;
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext::
DistNodeContext::~DistNodeContext()
{
    for( int i=0; i<16; ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline typename DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext&
DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext::
DistNodeContext::Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::MapVectorContext::DistNodeContext::Child");
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
DistNodeContext::Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::MapVectorContext::DistNodeContext::Child");
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
ContextBlock::ContextBlock()
: type(EMPTY), data()
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext::
ContextBlock::~ContextBlock()
{
    Clear();
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext::
ContextBlock::Clear()
{
    switch( type )
    {
    case DIST_NODE: 
        delete data.DN; break;

    case SPLIT_NODE:
        delete data.SN; break;

    case DIST_LOW_RANK:  
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
        delete data.z; break;

    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
    case NODE:
    case LOW_RANK:
    case DENSE:
    case EMPTY: 
        break;
    }
    type = EMPTY;
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMatrix<Scalar,Conjugated>::MapVectorContext::Clear()
{
    block.Clear();
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext::
DistNodeContext::DistNodeContext()
: children(16)
{
    for( int i=0; i<16; ++i )
        children[i] = new MapDenseMatrixContext;
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext::
DistNodeContext::~DistNodeContext()
{
    for( int i=0; i<16; ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline typename DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext&
DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext::
DistNodeContext::Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::MapDenseMatrixContext::DistNodeContext::Child");
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
DistNodeContext::Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::MapDenseMatrixContext::DistNodeContext::Child");
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
ContextBlock::ContextBlock()
: type(EMPTY), data()
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext::
ContextBlock::~ContextBlock()
{
    Clear();
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext::
ContextBlock::Clear()
{
    switch( type )
    {
    case DIST_NODE: 
        delete data.DN; break;

    case SPLIT_NODE:
        delete data.SN; break;

    case DIST_LOW_RANK:  
    case SPLIT_LOW_RANK:
    case SPLIT_DENSE:
        delete data.Z; break;

    case DIST_NODE_GHOST:
    case SPLIT_NODE_GHOST:
    case NODE_GHOST:
    case DIST_LOW_RANK_GHOST:
    case SPLIT_LOW_RANK_GHOST:
    case LOW_RANK_GHOST:
    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
    case NODE:
    case LOW_RANK:
    case DENSE:
    case EMPTY: 
        break;
    }
    type = EMPTY;
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMatrix<Scalar,Conjugated>::MapDenseMatrixContext::Clear()
{
    block.Clear();
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
DistNodeContext::DistNodeContext()
: children(16)
{
    for( int i=0; i<16; ++i )
        children[i] = new MapHMatrixContext;
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
DistNodeContext::~DistNodeContext()
{
    for( int i=0; i<16; ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline typename DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext&
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
DistNodeContext::Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::MapHMatrixContext::DistNodeContext::Child");
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
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext&
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
DistNodeContext::Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::MapHMatrixContext::DistNodeContext::Child");
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
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
SplitNodeContext::SplitNodeContext()
: children(16)
{
    for( int i=0; i<16; ++i )
        children[i] = new MapHMatrixContext;
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
SplitNodeContext::~SplitNodeContext()
{
    for( int i=0; i<16; ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline typename DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext&
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
SplitNodeContext::Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::MapHMatrixContext::SplitNodeContext::Child");
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
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext&
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
SplitNodeContext::Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::MapHMatrixContext::SplitNodeContext::Child");
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
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
NodeContext::NodeContext()
: children(16)
{
    for( int i=0; i<16; ++i )
        children[i] = new MapHMatrixContext;
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
NodeContext::~NodeContext()
{
    for( int i=0; i<16; ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline typename DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext&
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
NodeContext::Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::MapHMatrixContext::NodeContext::Child");
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
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext&
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
NodeContext::Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack
    ("DistQuasi2dHMatrix::MapHMatrixContext::NodeContext::Child");
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
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
ContextBlock::ContextBlock()
: type(EMPTY), data()
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
ContextBlock::~ContextBlock()
{
    Clear();
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::
ContextBlock::Clear()
{
    switch( type )
    {
    case DIST_NODE: 
    case DIST_NODE_GHOST:
        delete data.DN; break;

    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
        delete data.SN; break;

    case NODE:
    case NODE_GHOST:
        delete data.N; break;

    case DIST_LOW_RANK:  
    case DIST_LOW_RANK_GHOST:
        delete data.DF; break;

    case SPLIT_LOW_RANK:
    case SPLIT_LOW_RANK_GHOST:
        delete data.SF;

    case LOW_RANK:
    case LOW_RANK_GHOST:
        delete data.F;

    case SPLIT_DENSE:
    case SPLIT_DENSE_GHOST:
        delete data.SD; break;

    case DENSE:
    case DENSE_GHOST:
        delete data.D; break;

    case EMPTY: 
        break;
    }
    type = EMPTY;
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMatrix<Scalar,Conjugated>::MapHMatrixContext::Clear()
{
    block.Clear();
}

/*
 * Public member functions
 */

template<typename Scalar,bool Conjugated>
inline int
DistQuasi2dHMatrix<Scalar,Conjugated>::Height() const
{
    return _xSizeTarget*_ySizeTarget*_zSize;
}

template<typename Scalar,bool Conjugated>
inline int
DistQuasi2dHMatrix<Scalar,Conjugated>::Width() const
{
    return _xSizeSource*_ySizeSource*_zSize;
}

/*
 * Public structures member functions
 */

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::Subcomms::Subcomms( MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("Subcomms::Subcomms");
#endif
    const int rank = mpi::CommRank( comm );
    const int p = mpi::CommSize( comm );
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");

    // Simple (yet slow) method for computing the number of subcommunicators
    unsigned numLevels = 1;
    unsigned teamSize = p;
    while( teamSize != 1 )
    {
        if( teamSize >= 4 )
            teamSize >>= 2;
        else // teamSize == 2
            teamSize = 1;
        ++numLevels;
    }

    _subcomms.resize( numLevels );
    mpi::CommDup( comm, _subcomms[0] );
    teamSize = p;
    for( unsigned i=1; i<numLevels; ++i )
    {
        if( teamSize >= 4 )
            teamSize >>= 2;
        else
            teamSize = 1;
        const int color = rank/teamSize;
        const int key = rank - color*teamSize;
        mpi::CommSplit( comm, color, key, _subcomms[i] );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::Subcomms::~Subcomms()
{
#ifndef RELEASE
    PushCallStack("Subcomms::~Subcomms");
#endif
    for( unsigned i=0; i<_subcomms.size(); ++i )
        mpi::CommFree( _subcomms[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
inline unsigned
DistQuasi2dHMatrix<Scalar,Conjugated>::Subcomms::NumLevels() const
{
    return _subcomms.size();
}

template<typename Scalar,bool Conjugated>
inline MPI_Comm
DistQuasi2dHMatrix<Scalar,Conjugated>::Subcomms::Subcomm
( unsigned level ) const
{
    return _subcomms[std::min(level,(unsigned)_subcomms.size()-1)];
}

} // namespace psp

#endif // PSP_DIST_QUASI2D_HMATRIX_HPP
