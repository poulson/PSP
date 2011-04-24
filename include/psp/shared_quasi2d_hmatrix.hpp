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
#ifndef PSP_SHARED_QUASI2D_HMATRIX_HPP
#define PSP_SHARED_QUASI2D_HMATRIX_HPP 1

#include "psp/quasi2d_hmatrix.hpp"
#include "psp/shared_low_rank_matrix.hpp"
#include "psp/shared_dense_matrix.hpp"

namespace psp {

template<typename Scalar,bool Conjugated>
class SharedQuasi2dHMatrix
{
private:
    struct Node
    {
        std::vector<SharedQuasi2dHMatrix*> children;
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

        SharedQuasi2dHMatrix& Child( int i, int j );
        const SharedQuasi2dHMatrix& Child( int i, int j ) const;
    };

    struct NodeSymmetric
    {
        std::vector<SharedQuasi2dHMatrix*> children;
        int xSizes[2];
        int ySizes[2];
        int sizes[4];

        NodeSymmetric( int xSize, int ySize, int zSize );
        ~NodeSymmetric();

        SharedQuasi2dHMatrix& Child( int i, int j );
        const SharedQuasi2dHMatrix& Child( int i, int j ) const;
    };

    enum ShellType 
    { 
        NODE, 
        NODE_SYMMETRIC, 
        SHARED_LOW_RANK, 
        SHARED_DENSE, 
        DENSE 
    };

    struct Shell
    {
        ShellType type;
        union Data
        {
            Node* node;
            NodeSymmetric* nodeSymmetric;
            SharedLowRankMatrix<Scalar,Conjugated>* SF;
            SharedDenseMatrix<Scalar>* SD;
            DenseMatrix<Scalar>* D;

            Data() { std::memset( this, 0, sizeof(Data) ); }
        } data;

        Shell() : type(NODE), data() { }

        ~Shell()
        {
            switch( type )
            {
            case NODE:            delete data.node; break;
            case NODE_SYMMETRIC:  delete data.nodeSymmetric; break;
            case SHARED_LOW_RANK: delete data.SF; break;
            case SHARED_DENSE:    delete data.SD; break;
            case DENSE:           delete data.D; break;
            }
        }
    };

    // TODO: Fill in member variables

public:
    // TODO
};

} // namespace psp

#endif // PSP_SHARED_QUASI2D_HMATRIX_HPP
