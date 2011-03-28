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
#ifndef PSP_HMATRIX_QUASI2D_HPP
#define PSP_HMATRIX_QUASI2D_HPP 1

#include "psp/hmatrix_tools.hpp"

namespace psp {

template<typename Scalar>
class HMatrix_Quasi2d
{
    enum MatrixType { NODE, NODE_SYMMETRIC, DENSE, DENSE_SYMMETRIC, FACTOR };

    struct MatrixShell 
    {
        MatrixType matrixType;        

        // Only one of the following will be active
        std::vector<MatrixShell> children;
        DenseMatrix<Scalar> D;
        FactorMatrix<Scalar> F;
    };

    const bool _symmetric;
    const int _xSize, _ySize, _zSize;
    const int _numLevels;
    const bool _stronglyAdmissible;
    MatrixShell _rootShell;

    bool Admissible( int xSource, int ySource, int xTarget, int yTarget ) const;

    void
    RecursiveConstruction
    ( MatrixShell& shell,
      const SparseMatrix<Scalar>& S, 
      int level,
      int xSource, int ySource, 
      int xTarget, int yTarget,
      int sourceOffset, int xSizeSource, int ySizeSource,
      int targetOffset, int xSizeTarget, int ySizeTarget );

public:
    // This will convert a sparse matrix over an xSize x ySize x zSize domain
    // that is quasi-2d hierarchically ordered into an H-matrix with the
    // specified number of levels before dense storage. 
    //
    // The weak admissibility criterion is:
    //     max(dist_x(A,B),dist_y(A,B)) >= 1
    //
    // The strong admissibility criterion is:
    //     max(dist_x(A,B),dist_y(A,B)) > 1
    //
    HMatrix_Quasi2d
    ( const SparseMatrix<Scalar>& S,
      int xSize, int ySize, int zSize, int numLevels, bool stronglyAdmissible );

    ~HMatrix_Quasi2d();

};

} // namespace psp

#endif // PSP_HMATRIX_QUASI2D_HPP
