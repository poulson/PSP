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
#ifndef PSP_DIST_QUASI2D_HMAT_HPP
#define PSP_DIST_QUASI2D_HMAT_HPP 1

#include "psp/building_blocks/mpi.hpp"
#include "psp/building_blocks/memory_map.hpp"
#include "psp/quasi2d_hmat.hpp"

namespace psp {

template<typename Scalar,bool Conjugated>
class DistQuasi2dHMat
{
public:
    /*
     * Public data structures
     */
    class Teams
    {
    private:
        std::vector<MPI_Comm> _teams, _rootTeams;
    public:
        Teams( MPI_Comm comm );
        ~Teams();

        unsigned NumLevels() const;
        unsigned NumRootLevels() const;
        MPI_Comm Team( unsigned level ) const;
        MPI_Comm RootTeam( unsigned inverseLevel ) const;

        void TreeSummations
        (       std::vector<Scalar>& buffer, 
          const std::vector<int>& sizes,
          const std::vector<int>& offsets,
                bool customCollective=true ) const;

        void TreeBroadcasts
        (       std::vector<Scalar>& buffer,
          const std::vector<int>& sizes,
          const std::vector<int>& offsets,
                bool customCollective=true ) const;
    };

    /*
     * Public static member functions
     */
    static std::size_t PackedSizes
    ( std::vector<std::size_t>& packedSizes,
      const Quasi2dHMat<Scalar,Conjugated>& H, const Teams& teams );

    static std::size_t Pack
    ( std::vector<byte*>& packedPieces, 
      const Quasi2dHMat<Scalar,Conjugated>& H, const Teams& teams );

    static int ComputeLocalHeight
    ( int p, int rank, const Quasi2dHMat<Scalar,Conjugated>& H );

    static int ComputeLocalWidth
    ( int p, int rank, const Quasi2dHMat<Scalar,Conjugated>& H );

    static int ComputeFirstLocalRow
    ( int p, int rank, const Quasi2dHMat<Scalar,Conjugated>& H );

    static int ComputeFirstLocalCol
    ( int p, int rank, const Quasi2dHMat<Scalar,Conjugated>& H );

    static void ComputeLocalSizes
    ( std::vector<int>& localSizes, 
      const Quasi2dHMat<Scalar,Conjugated>& H );

    /*
     * Public non-static member functions
     */
    DistQuasi2dHMat( const Teams& teams );
    DistQuasi2dHMat
    ( int numLevels, int maxRank, bool stronglyAdmissible, 
      int sourceOffset, int targetOffset,
      int xSizeSource, int xSizeTarget, int ySizeSource, int ySizeTarget,
      int zSize, int xSource, int xTarget, int ySource, int yTarget,
      const Teams& teams, unsigned level, 
      bool inSourceTeam, bool inTargetTeam, 
      int sourceRoot, int targetRoot,
      int localSourceOffset=0, int localTargetOffset=0 );
    DistQuasi2dHMat( const byte* packedPiece, const Teams& teams );
    ~DistQuasi2dHMat();
    void Clear();

    int Height() const;
    int Width() const;
    int MaxRank() const;

    int LocalHeight() const;
    int LocalWidth() const;

    int FirstLocalRow() const;
    int FirstLocalCol() const;

    void RequireRoot() const;

    /*
     * Routines for visualizing the locally known H-matrix structure
     */
    // Compile this output with pdflatex+TikZ
    void LatexWriteLocalStructure( const std::string& basename ) const;
    // This can be visualized with util/PlotHStructure.m and Octave/Matlab
    void MScriptWriteLocalStructure( const std::string& basename ) const;

    // Unpack this process's portion of the DistQuasi2dHMat
    std::size_t Unpack
    ( const byte* packedDistHMat, const Teams& teams );

    // Union the structure known in each block row and column at each level.
    void FormGhostNodes();

    // Return to the minimal local structure
    void PruneGhostNodes();

    bool Ghosted() const;

    // A := alpha A
    void Scale( Scalar alpha );

    // y := alpha H x
    void Multiply
    ( Scalar alpha, const Vector<Scalar>& xLocal, 
                          Vector<Scalar>& yLocal ) const;

    // y := alpha H x + beta y
    void Multiply
    ( Scalar alpha, const Vector<Scalar>& xLocal, 
      Scalar beta,        Vector<Scalar>& yLocal ) const;

    // Y := alpha H X
    void Multiply
    ( Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;

    // Y := alpha H X + beta Y
    void Multiply
    ( Scalar alpha, const Dense<Scalar>& XLocal,
      Scalar beta,        Dense<Scalar>& YLocal ) const;

    // C := alpha A B 
    void Multiply
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C ) const;

    // C := alpha A B + beta C
    void Multiply
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
      Scalar beta,        DistQuasi2dHMat<Scalar,Conjugated>& C ) const;

    // y := alpha H^T x
    void TransposeMultiply
    ( Scalar alpha, const Vector<Scalar>& xLocal, 
                          Vector<Scalar>& yLocal ) const;

    // y := alpha H^T x + beta y
    void TransposeMultiply
    ( Scalar alpha, const Vector<Scalar>& xLocal, 
      Scalar beta,        Vector<Scalar>& yLocal ) const;

    // Y := alpha H^T X
    void TransposeMultiply
    ( Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;

    // Y := alpha H^T X + beta Y
    void TransposeMultiply
    ( Scalar alpha, const Dense<Scalar>& XLocal,
      Scalar beta,        Dense<Scalar>& YLocal ) const;

    // C := alpha A^T B
    void TransposeMultiply
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C ) const;

    // C := alpha A^T B + beta C
    void TransposeMultiply
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
      Scalar beta,        DistQuasi2dHMat<Scalar,Conjugated>& C ) const;

    // y := alpha H' x
    void AdjointMultiply
    ( Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    // y := alpha H' x + beta y
    void AdjointMultiply
    ( Scalar alpha, const Vector<Scalar>& xLocal,
      Scalar beta,        Vector<Scalar>& yLocal ) const;

    // Y := alpha H' X
    void AdjointMultiply
    ( Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;

    // Y := alpha H' X + beta Y
    void AdjointMultiply
    ( Scalar alpha, const Dense<Scalar>& XLocal,
      Scalar beta,        Dense<Scalar>& YLocal ) const;

    // C := alpha A' B
    void AdjointMultiply
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C ) const;

    // C := alpha A' B + beta C
    void AdjointMultiply
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
      Scalar beta,        DistQuasi2dHMat<Scalar,Conjugated>& C ) const;

private:
    /*
     * Private data structures
     */

    struct DistLowRank
    {
        int rank;
        Dense<Scalar> ULocal, VLocal;
    };

    struct DistLowRankGhost
    {
        int rank;
    };

    struct SplitLowRank
    {
        int rank;
        Dense<Scalar> D;
    };

    struct SplitLowRankGhost
    {
        int rank;
    };

    struct LowRankGhost
    {
        int rank;
    };

    struct SplitDense
    {
        Dense<Scalar> D;
    };

    struct Node
    {
        std::vector<DistQuasi2dHMat*> children;
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
        DistQuasi2dHMat& Child( int t, int s );
        const DistQuasi2dHMat& Child( int t, int s ) const;
    };
    Node* NewNode() const;

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

            DistLowRank* DF;
            DistLowRankGhost* DFG;
            SplitLowRank* SF;
            SplitLowRankGhost* SFG;
            LowRank<Scalar,Conjugated>* F;
            LowRankGhost* FG;

            SplitDense* SD;
            Dense<Scalar>* D;

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

    // TODO: Merge this with all of the MultiplyVector routines and create 
    //       a full-fledged class.
    struct MultiplyVectorContext
    {
        struct DistNode
        {
            std::vector<MultiplyVectorContext*> children;
            DistNode();
            ~DistNode();
            MultiplyVectorContext& Child( int t, int s );
            const MultiplyVectorContext& Child( int t, int s ) const;
        };
        typedef DistNode SplitNode;

        struct Block
        {
            BlockType type;
            union Data
            {
                DistNode* DN;
                SplitNode* SN;
                Vector<Scalar>* z;
                Data() { std::memset( this, 0, sizeof(Data) ); }
            } data;
            Block();
            ~Block();
            void Clear();
        };
        Block block;
        void Clear();
    };

    // TODO: Merge this with all of the MultiplyDense routines and create 
    //       a full-fledged class.
    struct MultiplyDenseContext
    {
        struct DistNode
        {
            std::vector<MultiplyDenseContext*> children;
            DistNode();
            ~DistNode();
            MultiplyDenseContext& Child( int t, int s );
            const MultiplyDenseContext& Child( int t, int s ) const;
        };
        typedef DistNode SplitNode;

        struct Block
        {
            BlockType type;
            union Data
            {
                DistNode* DN;
                SplitNode* SN;
                Dense<Scalar>* Z;
                Data() { std::memset( this, 0, sizeof(Data) ); }
            } data;
            Block();
            ~Block();
            void Clear();
        };
        int numRhs;
        Block block;
        void Clear();
    };

    /*
     * Private static functions
     */
    static void PackedSizesRecursion
    ( std::vector<std::size_t>& packedSizes,
      const std::vector<int>& localSizes,
      int sourceRankOffset, int targetRankOffset, int teamSize,
      const Quasi2dHMat<Scalar,Conjugated>& H );

    static void PackRecursion
    ( std::vector<byte**>& headPointers,
      const std::vector<int>& localSizes,
      int sourceRankOffset, int targetRankOffset, int teamSize,
      const Quasi2dHMat<Scalar,Conjugated>& H );

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

    void LatexWriteLocalStructureRecursion
    ( std::ofstream& file, int globalheight ) const;
    void MScriptWriteLocalStructureRecursion( std::ofstream& file ) const;
    
    // This default constructure is purposely not publically accessible
    // because many routines are not functional without _teams set.
    DistQuasi2dHMat();

    void UnpackRecursion( const byte*& head );

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

    //
    // H-matrix/vector multiplication
    //
    void MultiplyVectorInitialize( MultiplyVectorContext& context ) const;
    void MultiplyVectorPrecompute
    ( MultiplyVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal, 
                          Vector<Scalar>& yLocal ) const;
    void MultiplyVectorSummations
    ( MultiplyVectorContext& context, bool customCollective=true ) const;
    void MultiplyVectorSummationsCount( std::vector<int>& sizes ) const;
    void MultiplyVectorSummationsPack
    ( const MultiplyVectorContext& context, 
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyVectorSummationsUnpack
    ( MultiplyVectorContext& context, 
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyVectorNaiveSummations( MultiplyVectorContext& context ) const;
    void MultiplyVectorPassData( MultiplyVectorContext& context ) const;
    void MultiplyVectorPassDataSplitNodeCount( std::size_t& bufferSize ) const;
    void MultiplyVectorPassDataSplitNodePack
    ( MultiplyVectorContext& context, byte*& head ) const;
    void MultiplyVectorPassDataSplitNodeUnpack
    ( MultiplyVectorContext& context, const byte*& head ) const;
    void MultiplyVectorBroadcasts
    ( MultiplyVectorContext& context, bool customCollective=true ) const;
    void MultiplyVectorBroadcastsCount( std::vector<int>& sizes ) const;
    void MultiplyVectorBroadcastsPack
    ( const MultiplyVectorContext& context,
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyVectorBroadcastsUnpack
    ( MultiplyVectorContext& context,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyVectorNaiveBroadcasts( MultiplyVectorContext& context ) const;
    void MultiplyVectorPostcompute
    ( MultiplyVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    //
    // H-matrix/dense-matrix multiplication
    //
    void MultiplyDenseInitialize
    ( MultiplyDenseContext& context, int numRhs ) const;
    void MultiplyDensePrecompute
    ( MultiplyDenseContext& context,
      Scalar alpha, const Dense<Scalar>& XLocal, 
                          Dense<Scalar>& YLocal ) const;
    void MultiplyDenseSummations
    ( MultiplyDenseContext& context, bool customCollective=true ) const;
    void MultiplyDenseSummationsCount
    ( std::vector<int>& sizes, int numRhs ) const;
    void MultiplyDenseSummationsPack
    ( const MultiplyDenseContext& context, 
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyDenseSummationsUnpack
    ( MultiplyDenseContext& context, 
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyDenseNaiveSummations( MultiplyDenseContext& context ) const;
    void MultiplyDensePassData( MultiplyDenseContext& context ) const;
    void MultiplyDensePassDataSplitNodePack
    ( MultiplyDenseContext& context, byte*& head ) const;
    void MultiplyDensePassDataSplitNodeUnpack
    ( MultiplyDenseContext& context, const byte*& head ) const;
    void MultiplyDenseBroadcasts
    ( MultiplyDenseContext& context, bool customCollective=true ) const;
    void MultiplyDenseBroadcastsCount
    ( std::vector<int>& sizes, int numRhs ) const;
    void MultiplyDenseBroadcastsPack
    ( const MultiplyDenseContext& context, 
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyDenseBroadcastsUnpack
    ( MultiplyDenseContext& context, 
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyDenseNaiveBroadcasts( MultiplyDenseContext& context ) const;
    void MultiplyDensePostcompute
    ( MultiplyDenseContext& context,
      Scalar alpha, const Dense<Scalar>& XLocal, 
                          Dense<Scalar>& YLocal ) const;
    // Extra fine-grain routines for use within H-matrix/H-matrix multiplication
    void MultiplyDensePassDataCount
    ( std::vector<int>& sendSizes, 
      std::vector<int>& recvSizes, int numRhs ) const;
    void MultiplyDensePassDataPack
    ( MultiplyDenseContext& context, 
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyDensePassDataUnpack
    ( MultiplyDenseContext& context,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;

    //
    // H-matrix/H-matrix multiplication
    //
    void MultiplyHMatSetUp
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C ) const;
    void MultiplyHMatMainPrecompute
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C ) const;

    void MultiplyHMatMainSummations
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            bool customCollective=true ) const;
    // To be called from A
    void MultiplyHMatMainSummationsCountA( std::vector<int>& sizes ) const;
    void MultiplyHMatMainSummationsPackA
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets ) const; 
    void MultiplyHMatMainSummationsUnpackA
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    // To be called from B
    void MultiplyHMatMainSummationsCountB( std::vector<int>& sizes ) const;
    void MultiplyHMatMainSummationsPackB
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets ) const; 
    void MultiplyHMatMainSummationsUnpackB
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    // To be called from C
    void MultiplyHMatMainSummationsCountC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
      const DistQuasi2dHMat<Scalar,Conjugated>& C,
      std::vector<int>& sizes ) const;
    void MultiplyHMatMainSummationsPackC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
      const DistQuasi2dHMat<Scalar,Conjugated>& C,
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const; 
    void MultiplyHMatMainSummationsUnpackC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;

    void MultiplyHMatMainPassData
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C ) const;
    // To be called from A
    void MultiplyHMatMainPassDataCountA
    ( std::vector<int>& sendSizes, std::vector<int>& recvSizes ) const;
    void MultiplyHMatMainPassDataPackA
    ( std::vector<Scalar>& sendBuffer, std::vector<int>& offsets ) const;
    void MultiplyHMatMainPassDataUnpackA
    ( const std::vector<Scalar>& recvBuffer, std::vector<int>& offsets ) const;
    // To be called from A
    void MultiplyHMatMainPassDataCountB
    ( std::vector<int>& sendSizes, std::vector<int>& recvSizes ) const;
    void MultiplyHMatMainPassDataPackB
    ( std::vector<Scalar>& sendBuffer, std::vector<int>& offsets ) const;
    void MultiplyHMatMainPassDataUnpackB
    ( const std::vector<Scalar>& recvBuffer, std::vector<int>& offsets ) const;
    // To be called from A
    void MultiplyHMatMainPassDataCountC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
      const DistQuasi2dHMat<Scalar,Conjugated>& C,
      std::vector<int>& sendSizes, std::vector<int>& recvSizes ) const;
    void MultiplyHMatMainPassDataPackC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      std::vector<Scalar>& sendBuffer, std::vector<int>& offsets ) const;
    void MultiplyHMatMainPassDataUnpackC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      const std::vector<Scalar>& recvBuffer, std::vector<int>& offsets ) const;

    void MultiplyHMatMainBroadcasts
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            bool customCollective=true ) const;
    // To be called from A
    void MultiplyHMatMainBroadcastsCountA( std::vector<int>& sizes ) const;
    void MultiplyHMatMainBroadcastsPackA
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyHMatMainBroadcastsUnpackA
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    // To be called from B
    void MultiplyHMatMainBroadcastsCountB( std::vector<int>& sizes ) const;
    void MultiplyHMatMainBroadcastsPackB
    ( std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyHMatMainBroadcastsUnpackB
    ( const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    // To be called from C
    void MultiplyHMatMainBroadcastsCountC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
      const DistQuasi2dHMat<Scalar,Conjugated>& C, 
      std::vector<int>& sizes ) const;
    void MultiplyHMatMainBroadcastsPackC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
      const DistQuasi2dHMat<Scalar,Conjugated>& C,
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyHMatMainBroadcastsUnpackC
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;

    void MultiplyHMatMainPostcompute
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C ) const;
    void MultiplyHMatMainPostcomputeA() const; // to be called from A
    void MultiplyHMatMainPostcomputeB() const; // to be called from B
    void MultiplyHMatMainPostcomputeC
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C ) const;
    void MultiplyHMatMainPostcomputeCCleanup(); // to be called from C

    void MultiplyHMatFHHPrecompute
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C ) const;
    void MultiplyHMatFHHSummations
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C,
      bool customCollective=true ) const;
    void MultiplyHMatFHHSummationsCount
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            std::vector<int>& sizes ) const;
    void MultiplyHMatFHHSummationsPack
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyHMatFHHSummationsUnpack
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyHMatFHHPassData
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C ) const;
    void MultiplyHMatFHHPassDataCount
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            std::vector<int>& sendSizes, std::vector<int>& recvSizes ) const;
    void MultiplyHMatFHHPassDataPack
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            std::vector<Scalar>& sendBuffer, std::vector<int>& offsets ) const;
    void MultiplyHMatFHHPassDataUnpack
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      const std::vector<Scalar>& recvBuffer, std::vector<int>& offsets ) const;
    void MultiplyHMatFHHBroadcasts
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C,
      bool customCollective=true ) const;
    void MultiplyHMatFHHBroadcastsCount
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            std::vector<int>& sizes ) const;
    void MultiplyHMatFHHBroadcastsPack
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
            std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyHMatFHHBroadcastsUnpack
    ( const DistQuasi2dHMat<Scalar,Conjugated>& B,
            DistQuasi2dHMat<Scalar,Conjugated>& C,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void MultiplyHMatFHHPostcompute
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C ) const;
    void MultiplyHMatFHHPostcomputeC
    ( Scalar alpha, const DistQuasi2dHMat<Scalar,Conjugated>& B,
                          DistQuasi2dHMat<Scalar,Conjugated>& C ) const;
    void MultiplyHMatFHHPostcomputeCCleanup(); // to be called from C

    // To be called from C
    void MultiplyHMatQR();
    void MultiplyHMatQRLowRankResize( int rank );
    void MultiplyHMatQRLowRankImport( int rank );
    void MultiplyHMatQRImportU( int rank, const Dense<Scalar>& U );
    void MultiplyHMatQRImportV( int rank, const Dense<Scalar>& V );

    //
    // Transpose H-matrix/vector multiplication
    //
    void TransposeMultiplyVectorInitialize
    ( MultiplyVectorContext& context ) const;
    void TransposeMultiplyVectorPrecompute
    ( MultiplyVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void TransposeMultiplyVectorSummations
    ( MultiplyVectorContext& context, bool customCollective=true ) const;
    void TransposeMultiplyVectorSummationsCount
    ( std::vector<int>& sizes ) const;
    void TransposeMultiplyVectorSummationsPack
    ( const MultiplyVectorContext& context, 
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void TransposeMultiplyVectorSummationsUnpack
    ( MultiplyVectorContext& context, 
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void TransposeMultiplyVectorNaiveSummations
    ( MultiplyVectorContext& context ) const;
    void TransposeMultiplyVectorPassData
    ( MultiplyVectorContext& context,
      const Vector<Scalar>& xLocal ) const;
    void TransposeMultiplyVectorPassDataSplitNodeCount
    ( std::size_t& bufferSize ) const;
    void TransposeMultiplyVectorPassDataSplitNodePack
    ( MultiplyVectorContext& context,
      const Vector<Scalar>& xLocal, byte*& head ) const;
    void TransposeMultiplyVectorPassDataSplitNodeUnpack
    ( MultiplyVectorContext& context, const byte*& head ) const;
    void TransposeMultiplyVectorBroadcasts
    ( MultiplyVectorContext& context, bool customCollective=true ) const;
    void TransposeMultiplyVectorBroadcastsCount
    ( std::vector<int>& sizes ) const;
    void TransposeMultiplyVectorBroadcastsPack
    ( const MultiplyVectorContext& context,
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void TransposeMultiplyVectorBroadcastsUnpack
    ( MultiplyVectorContext& context, 
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void TransposeMultiplyVectorNaiveBroadcasts
    ( MultiplyVectorContext& context ) const;
    void TransposeMultiplyVectorPostcompute
    ( MultiplyVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    //
    // Transpose H-matrix/dense-matrix multiplication
    //
    void TransposeMultiplyDenseInitialize
    ( MultiplyDenseContext& context, int numRhs ) const;
    void TransposeMultiplyDensePrecompute
    ( MultiplyDenseContext& context,
      Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;
    void TransposeMultiplyDenseSummations
    ( MultiplyDenseContext& context, bool customCollective=true ) const;
    void TransposeMultiplyDenseSummationsCount
    ( std::vector<int>& sizes, int numRhs ) const;
    void TransposeMultiplyDenseSummationsPack
    ( const MultiplyDenseContext& context,
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void TransposeMultiplyDenseSummationsUnpack
    ( MultiplyDenseContext& context,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void TransposeMultiplyDenseNaiveSummations
    ( MultiplyDenseContext& context ) const;
    void TransposeMultiplyDensePassData
    ( MultiplyDenseContext& context,
      const Dense<Scalar>& XLocal ) const;
    void TransposeMultiplyDensePassDataSplitNodePack
    ( MultiplyDenseContext& context,
      const Dense<Scalar>& XLocal, byte*& head ) const;
    void TransposeMultiplyDensePassDataSplitNodeUnpack
    ( MultiplyDenseContext& context, const byte*& head ) const;
    void TransposeMultiplyDenseBroadcasts
    ( MultiplyDenseContext& context, bool customCollective=true ) const;
    void TransposeMultiplyDenseBroadcastsCount
    ( std::vector<int>& sizes, int numRhs ) const;
    void TransposeMultiplyDenseBroadcastsPack
    ( const MultiplyDenseContext& context,
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void TransposeMultiplyDenseBroadcastsUnpack
    ( MultiplyDenseContext& context,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void TransposeMultiplyDenseNaiveBroadcasts
    ( MultiplyDenseContext& context ) const;
    void TransposeMultiplyDensePostcompute
    ( MultiplyDenseContext& context,
      Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;
    // Extra fine-grain routines for use within H-matrix/H-matrix multiplication
    void TransposeMultiplyDensePassDataCount
    ( std::vector<int>& sendSizes, 
      std::vector<int>& recvSizes, int numRhs ) const;
    void TransposeMultiplyDensePassDataPack
    ( MultiplyDenseContext& context, const Dense<Scalar>& XLocal,
      std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;
    void TransposeMultiplyDensePassDataUnpack
    ( MultiplyDenseContext& context,
      const std::vector<Scalar>& buffer, std::vector<int>& offsets ) const;

    //
    // Adjoint H-matrix/vector multiplication
    //
    void AdjointMultiplyVectorInitialize
    ( MultiplyVectorContext& context ) const;
    void AdjointMultiplyVectorPrecompute
    ( MultiplyVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;
    void AdjointMultiplyVectorSummations
    ( MultiplyVectorContext& context, bool customCollective=true ) const;
    void AdjointMultiplyVectorNaiveSummations
    ( MultiplyVectorContext& context ) const;
    void AdjointMultiplyVectorPassData
    ( MultiplyVectorContext& context, const Vector<Scalar>& xLocal ) const;
    void AdjointMultiplyVectorBroadcasts
    ( MultiplyVectorContext& context, bool customCollective=true ) const;
    void AdjointMultiplyVectorNaiveBroadcasts
    ( MultiplyVectorContext& context ) const;
    void AdjointMultiplyVectorPostcompute
    ( MultiplyVectorContext& context,
      Scalar alpha, const Vector<Scalar>& xLocal,
                          Vector<Scalar>& yLocal ) const;

    //
    // Adjoint H-matrix/dense-matrix multiplication
    //
    void AdjointMultiplyDenseInitialize
    ( MultiplyDenseContext& context, int numRhs ) const;
    void AdjointMultiplyDensePrecompute
    ( MultiplyDenseContext& context,
      Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;
    void AdjointMultiplyDenseSummations
    ( MultiplyDenseContext& context, bool customCollective=true ) const;
    void AdjointMultiplyDenseNaiveSummations
    ( MultiplyDenseContext& context ) const;
    void AdjointMultiplyDensePassData
    ( MultiplyDenseContext& context, const Dense<Scalar>& XLocal ) const;
    void AdjointMultiplyDenseBroadcasts
    ( MultiplyDenseContext& context, bool customCollective=true ) const;
    void AdjointMultiplyDenseNaiveBroadcasts
    ( MultiplyDenseContext& context ) const;
    void AdjointMultiplyDensePostcompute
    ( MultiplyDenseContext& context,
      Scalar alpha, const Dense<Scalar>& XLocal,
                          Dense<Scalar>& YLocal ) const;

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

    const Teams* _teams;
    unsigned _level;
    bool _inSourceTeam, _inTargetTeam;
    int _sourceRoot, _targetRoot;
    int _localSourceOffset, _localTargetOffset;

    // For temporary products in an H-matrix/H-matrix multiplication. 
    // These are only needed for the C in C += alpha A B
    mutable MemoryMap<int,MultiplyDenseContext> 
        _mainContextMap, _colFHHContextMap, _rowFHHContextMap;
    mutable MemoryMap<int,Dense<Scalar> > 
        _UMap, _VMap, _DMap, _ZMap, _colXMap, _rowXMap;

    // For the reuse of the computation of T1 = H Omega1 and T2 = H' Omega2 in 
    // order to capture the column and row space, respectively, of H. These 
    // variables are mutable since they do not effect the usage of the logical 
    // state of the class and simply help avoid redundant computation.
    mutable bool _beganRowSpaceComp, _beganColSpaceComp;
    mutable Dense<Scalar> _colOmega, _rowOmega, _colT, _rowT;
    mutable MultiplyDenseContext _colContext, _rowContext;
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
DistQuasi2dHMat<Scalar,Conjugated>::Node::Node
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
DistQuasi2dHMat<Scalar,Conjugated>::Node::~Node()
{
    for( unsigned i=0; i<children.size(); ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline DistQuasi2dHMat<Scalar,Conjugated>&
DistQuasi2dHMat<Scalar,Conjugated>::Node::Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Node::Child");
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
inline const DistQuasi2dHMat<Scalar,Conjugated>&
DistQuasi2dHMat<Scalar,Conjugated>::Node::Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Node::Child");
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
inline typename DistQuasi2dHMat<Scalar,Conjugated>::Node*
DistQuasi2dHMat<Scalar,Conjugated>::NewNode() const
{
    return 
        new Node
        ( _xSizeSource, _xSizeTarget, _ySizeSource, _ySizeTarget, _zSize );
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::Block::Block()
: type(EMPTY), data() 
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::Block::~Block()
{ 
    Clear();
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::Block::Clear()
{
    switch( type )
    {
    case DIST_NODE:
    case DIST_NODE_GHOST:
    case SPLIT_NODE:
    case SPLIT_NODE_GHOST:
    case NODE:
    case NODE_GHOST:
        delete data.N; break;

    case DIST_LOW_RANK:  delete data.DF; break;
    case SPLIT_LOW_RANK: delete data.SF; break;
    case LOW_RANK:       delete data.F;  break;

    case DIST_LOW_RANK_GHOST:  delete data.DFG; break;
    case SPLIT_LOW_RANK_GHOST: delete data.SFG; break;
    case LOW_RANK_GHOST:       delete data.FG;  break;

    case SPLIT_DENSE: delete data.SD; break;
    case DENSE:       delete data.D;  break;

    case SPLIT_DENSE_GHOST:
    case DENSE_GHOST:
    case EMPTY: 
        break;
    }
    type = EMPTY;
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::DistNode::DistNode()
: children(16)
{
    for( int i=0; i<16; ++i )
        children[i] = new MultiplyVectorContext;
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::DistNode::~DistNode()
{
    for( int i=0; i<16; ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline typename DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext&
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::DistNode::
Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorContext::DistNode::Child");
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
inline const typename DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext&
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::DistNode::
Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyVectorContext::DistNode::Child");
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
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::Block::Block()
: type(EMPTY), data()
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::Block::~Block()
{
    Clear();
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::Block::Clear()
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
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyVectorContext::Clear()
{
    block.Clear();
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::DistNode::DistNode()
: children(16)
{
    for( int i=0; i<16; ++i )
        children[i] = new MultiplyDenseContext;
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::DistNode::~DistNode()
{
    for( int i=0; i<16; ++i )
        delete children[i];
    children.clear();
}

template<typename Scalar,bool Conjugated>
inline typename DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext&
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::DistNode::
Child( int t, int s )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseContext::DistNode::Child");
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
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext&
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::DistNode::
Child( int t, int s ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::MultiplyDenseContext::DistNode::Child");
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
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::Block::Block()
: type(EMPTY), data()
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::Block::~Block()
{
    Clear();
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::Block::Clear()
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
DistQuasi2dHMat<Scalar,Conjugated>::MultiplyDenseContext::Clear()
{
    block.Clear();
}

/*
 * Public member functions
 */

template<typename Scalar,bool Conjugated>
inline int
DistQuasi2dHMat<Scalar,Conjugated>::Height() const
{
    return _xSizeTarget*_ySizeTarget*_zSize;
}

template<typename Scalar,bool Conjugated>
inline int
DistQuasi2dHMat<Scalar,Conjugated>::Width() const
{
    return _xSizeSource*_ySizeSource*_zSize;
}

template<typename Scalar,bool Conjugated>
inline int
DistQuasi2dHMat<Scalar,Conjugated>::MaxRank() const
{
    return _maxRank;
}

/*
 * Public structures member functions
 */

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::Teams::Teams( MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("Teams::Teams");
#endif
    const int rank = mpi::CommRank( comm );
    const int p = mpi::CommSize( comm );
    if( !(p && !(p & (p-1))) )
        throw std::logic_error("Must use a power of two number of processes");

    // Simple (yet slow) method for computing the number of teams
    // (and how many we're the root of)
    unsigned numLevels=1, numRootLevels=1;
    unsigned teamSize=p;
    while( teamSize != 1 )
    {
        ++numLevels;
        if( rank % teamSize == 0 )
            ++numRootLevels;

        if( teamSize >= 4 )
            teamSize >>= 2;
        else // teamSize == 2
            teamSize = 1;
    }

    _teams.resize( numLevels );
    mpi::CommDup( comm, _teams[0] );
    teamSize = p;
    for( unsigned level=1; level<numLevels; ++level )
    {
        if( teamSize >= 4 )
            teamSize >>= 2;
        else
            teamSize = 1;
        const int color = rank/teamSize;
        const int key = rank - color*teamSize;
        mpi::CommSplit( comm, color, key, _teams[level] );
    }

    _rootTeams.resize( numRootLevels );
    mpi::CommDup( _teams[numLevels-1], _rootTeams[0] );
    for( unsigned inverseLevel=1; inverseLevel<numLevels; ++inverseLevel )
    {
        const int level = numLevels-1-inverseLevel;
        teamSize = mpi::CommSize( _teams[level] );
        if( inverseLevel < numRootLevels )
        {
            const int color = (rank/teamSize) / 4;
            const int key = (rank/teamSize) % 4;
            mpi::CommSplit( comm, color, key, _rootTeams[inverseLevel] );
        }
        else
        {
            const int color = MPI_UNDEFINED;
            const int key = 0;
            MPI_Comm dummy;
            mpi::CommSplit( comm, color, key, dummy );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMat<Scalar,Conjugated>::Teams::~Teams()
{
#ifndef RELEASE
    PushCallStack("Teams::~Teams");
#endif
    for( unsigned i=0; i<_teams.size(); ++i )
        mpi::CommFree( _teams[i] );
    for( unsigned i=0; i<_rootTeams.size(); ++i )
        mpi::CommFree( _rootTeams[i] );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
inline unsigned
DistQuasi2dHMat<Scalar,Conjugated>::Teams::NumLevels() const
{
    return _teams.size();
}

template<typename Scalar,bool Conjugated>
inline unsigned
DistQuasi2dHMat<Scalar,Conjugated>::Teams::NumRootLevels() const
{
    return _rootTeams.size();
}

template<typename Scalar,bool Conjugated>
inline MPI_Comm
DistQuasi2dHMat<Scalar,Conjugated>::Teams::Team
( unsigned level ) const
{
    return _teams[std::min(level,(unsigned)_teams.size()-1)];
}

template<typename Scalar,bool Conjugated>
inline MPI_Comm
DistQuasi2dHMat<Scalar,Conjugated>::Teams::RootTeam
( unsigned inverseLevel ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Teams::RootTeam");
    if( inverseLevel >= _rootTeams.size() )
        throw std::logic_error("Invalid root team request");
    PopCallStack();
#endif
    return _rootTeams[inverseLevel];
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::Teams::TreeSummations
(       std::vector<Scalar>& buffer, 
  const std::vector<int>& sizes,
  const std::vector<int>& offsets,
        bool customCollective ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Teams::TreeSummations");
#endif
    const int numLevels = NumLevels();
    const int totalSize = buffer.size();
    if( numLevels == 1 || totalSize == 0 )
        return;

    const int numReduces = numLevels-1;
    if( customCollective )
    {
        // Use O(log(p)) custom method: 
        // - Reduce onto the root of our smallest nontrivial communicator
        // - Continue using the root teams to finish the reduction
        MPI_Comm smallTeam = Team( numLevels-2 );
        const int smallTeamRank = mpi::CommRank( smallTeam );
        if( smallTeamRank == 0 )
            mpi::Reduce
            ( (const Scalar*)MPI_IN_PLACE, &buffer[0], totalSize,
              0, MPI_SUM, smallTeam );
        else
            mpi::Reduce( &buffer[0], 0, totalSize, 0, MPI_SUM, smallTeam );

        int partialSize = totalSize - sizes[numReduces-1];
        const int numRootLevels = NumRootLevels();
        for( int i=1; i<numRootLevels; ++i )
        {
            if( partialSize == 0 )
                break;
            MPI_Comm rootTeam = RootTeam( i );
            const int rootTeamRank = mpi::CommRank( rootTeam );
            if( rootTeamRank == 0 )
                mpi::Reduce
                ( (const Scalar*)MPI_IN_PLACE, &buffer[0], partialSize,
                  0, MPI_SUM, rootTeam );
            else
                mpi::Reduce( &buffer[0], 0, partialSize, 0, MPI_SUM, rootTeam );
            partialSize -= sizes[numReduces-1-i];
        }
    }
    else
    {
        for( int i=0; i<numReduces; ++i )
        {
            if( sizes[i] != 0 )
            {
                MPI_Comm team = Team( i );
                const int teamRank = mpi::CommRank( team );
                if( teamRank == 0 )
                    mpi::Reduce
                    ( (const Scalar*)MPI_IN_PLACE, &buffer[offsets[i]], 
                      sizes[i], 0, MPI_SUM, team );
                else
                    mpi::Reduce
                    ( &buffer[offsets[i]], 0, sizes[i], 0, MPI_SUM, team );
            }
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
inline void
DistQuasi2dHMat<Scalar,Conjugated>::Teams::TreeBroadcasts
(       std::vector<Scalar>& buffer, 
  const std::vector<int>& sizes,
  const std::vector<int>& offsets,
        bool customCollective ) const
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Teams::TreeBroadcasts");
#endif
    const int numLevels = NumLevels();
    const int totalSize = buffer.size();
    if( numLevels == 1 || totalSize == 0 )
        return;

    const int numBroadcasts = numLevels-1;
    if( customCollective )
    {
        // Use O(log(p)) custom method: 
        // - Broadcasts from the root of our smallest nontrivial communicator
        // - Continue broadcasting from the roots of the root teams
        MPI_Comm smallTeam = Team( numLevels-2 );
        mpi::Broadcast( &buffer[0], totalSize, 0, smallTeam );

        int partialSize = totalSize - sizes[numBroadcasts-1];
        const int numRootLevels = NumRootLevels();
        for( int i=1; i<numRootLevels; ++i )
        {
            if( partialSize == 0 )
                break;
            MPI_Comm rootTeam = RootTeam( i );
            mpi::Broadcast( &buffer[0], partialSize, 0, rootTeam );
            partialSize -= sizes[numBroadcasts-1-i];
        }
    }
    else
    {
        // Use a simple O(log^2(p)) method
        for( int i=0; i<numBroadcasts; ++i )
        {
            if( sizes[i] != 0 )
            {
                MPI_Comm team = Team( i );
                mpi::Broadcast( &buffer[offsets[i]], sizes[i], 0, team );
            }
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_DIST_QUASI2D_HMAT_HPP
