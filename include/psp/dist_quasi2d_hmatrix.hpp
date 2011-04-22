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

namespace psp {

template<typename Scalar,bool Conjugated>
class DistQuasi2dHMatrix
{
private:
    enum ShellType 
    { NODE, NODE_SYMMETRIC, BLACK_BOX, 
      LEFT_HALF_OF_LOW_RANK, RIGHT_HALF_OF_LOW_RANK, 
      LEFT_TEAM_DENSE, RIGHT_TEAM_DENSE };
    // NOTE: We may need to expand the list of dense shell types for triangular
    //       matrices in order to get better load balancing.

    void PreCompute
    ( Scalar alpha, const DistVector<Scalar>& x, DistVector<Scalar>& y );
    void Communicate
    ( Scalar alpha, const DistVector<Scalar>& x, DistVector<Scalar>& y );
    void PostCompute
    ( Scalar alpha, const DistVector<Scalar>& x, DistVector<Scalar>& y );

public:
    struct PackedSharedLowRanks
    {
        int numShared;

        // Dimensions of full low-rank matrices
        int* heights;
        int* widths;
        int* ranks;

        // Which side we own for each half and who the partners are
        bool* sides;
        int* pairedRanks;

        // The buffer that all of the entries are packed into. The j'th half
        // of a low-rank matrix is stored at &buffer[offsets[j]].
        int* offsets;
        Scalar* buffer;
    };

    struct PackedSharedDenses
    {
        int numShared;

        // Dimensions of shared dense matrices
        int* heights;
        int* widths;

        // Whether or not we own the data and who the partners are
        bool* mine;
        int* pairedRanks;

        // The buffer where all of the entries are packed. The j'th dense
        // matrix (if we own it) can be accessed at &buffer[offsets[j]].
        int* offsets;
        Scalar* buffer;
    };

    // We should build the distributed H-matrix from the pieces rather than
    // from an entire serial H-matrix.
    //
    // The packed dense matrices imply the existence of the transposed 
    // pairing that requires no data at construction but does require
    // communication in the middle phase of a matvec.
    DistQuasi2dHMatrix
    ( int numLevels, bool stronglyAdmissible, int xSize, int ySize, int zSize,
      const Quasi2dHMatrix<Scalar,Conjugated>& myLeaf,
      const PackedSharedLowRanks& packedSharedLowRanks,
      const PackedSharedDenses& packedSharedDenses );

    void MapVector
    ( Scalar alpha, const DistVector<Scalar>& x, DistVector<Scalar>& y );
};

} // namespace psp

#endif // PSP_DIST_QUASI2D_HMATRIX_HPP
