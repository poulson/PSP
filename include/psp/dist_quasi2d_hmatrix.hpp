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

#include "psp/quasi2d_hmatrix.hpp"
#include "psp/subcomms.hpp"
#include "psp/shared_quasi2d_hmatrix.hpp"
#include "psp/shared_low_rank_matrix.hpp"
#include "psp/shared_dense_matrix.hpp"
#include "psp/dist_low_rank_matrix.hpp"
#include "psp/dist_shared_low_rank_matrix.hpp"

namespace psp {

// We will enforce the requirement that is a power of 2 numbers or processes, 
// but not more than 4^{numLevels-1}.
template<typename Scalar,bool Conjugated>
class DistQuasi2dHMatrix
{
private:
    static void PackedSizesRecursion
    ( std::vector<std::size_t>& packedSizes,
      const std::vector<int>& localHeights,
      const std::vector<int>& localWidths,
      int sourceRankOffset, int targetRankOffset, int teamSize,
      const Quasi2dHMatrix<Scalar,Conjugated>& H );

    static void PackRecursion
    ( std::vector<byte**>& headPointers,
      const std::vector<int>& localHeights,
      const std::vector<int>& localWidths,
      int sourceRankOffset, int targetRankOffset, int teamSize,
      const Quasi2dHMatrix<Scalar,Conjugated>& H );

    static void ComputeLocalDimensionRecursion
    ( int& localDim, int p, int rank, int xSize, int ySize, int zSize );

    static void ComputeLocalSizesRecursion
    ( int* localHeights, int* localWidths, int teamSize,
      int xSizeSource, int xSizeTarget, int ySizeSource, int ySizeTarget,
      int zSize );

    struct Node
    {
        std::vector<DistQuasi2dHMatrix*> children;
        int xSourceSizes[2];
        int ySourceSizes[2];
        int sourceSizes[2];
        int xTargetSizes[2];
        int yTargetSizes[2];
        int targetSizes[2];
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
        NODE,                 // recurse
        NODE_SYMMETRIC,       // recurse symmetrically
        DIST_SHARED_LOW_RANK, // each side is distributed to different teams
        DIST_LOW_RANK,        // both sides are distributed to same team
        SHARED_QUASI2D,       // shared between two processes
        SHARED_LOW_RANK,      // each side is given to one process
        SHARED_DENSE,         // shared between two processes
        QUASI2D,              // serial
        LOW_RANK,             // serial
        DENSE,                // serial
        EMPTY
    };

    struct Shell
    {
        ShellType type;
        union Data
        {
            Node* node;
            NodeSymmetric* nodeSymmetric;
            DistSharedLowRankMatrix<Scalar,Conjugated>* DSF;
            DistLowRankMatrix<Scalar,Conjugated>* DF;
            SharedQuasi2dHMatrix<Scalar,Conjugated>* SH;
            SharedLowRankMatrix<Scalar,Conjugated>* SF;
            SharedDenseMatrix<Scalar>* SD;
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
    bool _symmetric; // TODO: Replace with MatrixType
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

    void UnpackRecursion
    ( const byte*& head, DistQuasi2dHMatrix<Scalar,Conjugated>& H );

    // Ensure that the default constructor is not accessible, a communicator
    // must be supplied
    DistQuasi2dHMatrix();

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

    static void ComputeLocalSizes
    ( std::vector<int>& localHeights, std::vector<int>& localWidths,
      const Quasi2dHMatrix<Scalar,Conjugated>& H );

    DistQuasi2dHMatrix( const Subcomms& subcomms );
    DistQuasi2dHMatrix
    ( const Subcomms& subcomms, unsigned level, 
      bool inSourceTeam, bool inTargetTeam );
    DistQuasi2dHMatrix( const byte* packedPiece, const Subcomms& subcomms );
    ~DistQuasi2dHMatrix();

    int LocalHeight() const;
    int LocalWidth() const;

    // Unpack this process's portion of the DistQuasi2dHMatrix
    std::size_t Unpack
    ( const byte* packedDistHMatrix, const Subcomms& subcomms );

    // TODO: Figure out whether to act on PETSc's Vec or our own distributed
    //       vector class.
    /*
    void MapVector
    ( Scalar alpha, const DistVector<Scalar>& x, DistVector<Scalar>& y ) const;
    */
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
: type(NODE), data() 
{ }

template<typename Scalar,bool Conjugated>
inline
DistQuasi2dHMatrix<Scalar,Conjugated>::Shell::~Shell()
{ 
    switch( type )
    {
    case NODE:                 delete data.node; break;
    case NODE_SYMMETRIC:       delete data.nodeSymmetric; break;
    case DIST_SHARED_LOW_RANK: delete data.DSF; break;
    case DIST_LOW_RANK:        delete data.DF; break;
    case SHARED_QUASI2D:       delete data.SH; break;
    case SHARED_LOW_RANK:      delete data.SF; break;
    case SHARED_DENSE:         delete data.SD; break;
    case QUASI2D:              delete data.H; break;
    case LOW_RANK:             delete data.F; break;
    case DENSE:                delete data.D; break;
    case EMPTY: break;
    }
}

} // namespace psp

#endif // PSP_DIST_QUASI2D_HMATRIX_HPP
