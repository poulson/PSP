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
#include "psp/shared_quasi2d_hmatrix.hpp"
#include "psp/shared_low_rank_matrix.hpp"
#include "psp/shared_dense_matrix.hpp"

namespace psp {

// We will enforce the requirement that is a power of 2 numbers or processes, 
// but not more than 4^{numLevels-1}.
template<typename Scalar,bool Conjugated>
class DistQuasi2dHMatrix
{
private:
    static void PackedSizesRecursion
    ( std::vector<std::size_t>& scatteredSizes,
      int rank, int sourceRankOffset, int targetRankOffset, int teamSize,
      const Quasi2dHMatrix<Scalar,Conjugated>& H );

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
        NODE, 
        NODE_SYMMETRIC, 
        DIST_LOW_RANK, 
        SHARED_LOW_RANK, 
        SHARED_QUASI2D, 
        QUASI2D 
    };
    // NOTE: We may need to expand the list of dense shell types for triangular
    //       matrices in order to get better load balancing.

    struct Shell
    {
        ShellType type;
        union Data
        {
            Node* node;
            NodeSymmetric* nodeSymmetric;
            DistLowRankMatrix<Scalar,Conjugated>* DF;
            SharedLowRankMatrix<Scalar,Conjugated>* SF;
            SharedQuasi2dHMatrix<Scalar,Conjugated>* SH;
            Quasi2dHMatrix<Scalar,Conjugated>* H;

            Data() { std::memset( this, 0, sizeof(Data) ); }
        } data;

        Shell() : type(NODE), data() { }

        ~Shell()
        {
            switch( type )
            {
            case NODE:            delete data.node; break;
            case NODE_SYMMETRIC:  delete data.nodeSymmetric; break;
            case DIST_LOW_RANK:   delete data.DF; break;
            case SHARED_LOW_RANK: delete data.SF; break;
            case SHARED_QUASI2D:  delete data.SH; break;
            case QUASI2D:         delete data.H; break;
            }
        }
    };

    int _localHeight, _localWidth;
    // TODO: Finish filling in member variables

public:

    static void PackedSizes
    ( std::vector<std::size_t>& scatteredSizes,
      const Quasi2dHMatrix<Scalar,Conjugated>& H, MPI_Comm comm );

    int LocalHeight() const;
    int LocalWidth() const;

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

template<typename Scalar,bool Conjugated>
inline int
DistQuasi2dHMatrix<Scalar,Conjugated>::LocalHeight() const
{ 
    return _localHeight;
}

template<typename Scalar,bool Conjugated>
inline int
DistQuasi2dHMatrix<Scalar,Conjugated>::LocalWidth() const
{
    return _localWidth;
}

} // namespace psp

#endif // PSP_DIST_QUASI2D_HMATRIX_HPP
