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
    const int _m, _n;

    enum ShellType { NODE, NODE_SYMMETRIC, DENSE, FACTOR };

    struct MatrixShell; // forward declaration for Node(Symmetric)Data

    struct NodeData
    {
        std::vector<MatrixShell> children;
        int xSourceSizes[2];
        int ySourceSizes[2];
        int sourceSizes[4];
        int xTargetSizes[2];
        int yTargetSizes[2];
        int targetSizes[4];
    };

    struct NodeSymmetricData
    {
        std::vector<MatrixShell> children;
        int xSizes[2];
        int ySizes[2];
        int sizes[4];
    };

    struct DenseData
    {
        DenseMatrix<Scalar> D;
    };

    struct FactorData
    {
        FactorMatrix<Scalar,false> F; // F = U V^T
    };

    struct MatrixShell 
    {
        ShellType type;        
        union
        {
            NodeData* node;
            NodeSymmetricData* nodeSymmetric;
            DenseData* dense;
            FactorData* factor;
        } u;
        MatrixShell() { }
        ~MatrixShell() 
        { 
            switch( type )
            {
                case NODE:           delete u.node;          break;
                case NODE_SYMMETRIC: delete u.nodeSymmetric; break;
                case DENSE:          delete u.dense;         break;
                case FACTOR:         delete u.factor;        break;
            }
        }
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

    // y := alpha H x + beta y
    void RecursiveMatrixVector
    ( Scalar alpha, const MatrixShell& shell, 
                    const Vector<Scalar>& x,
      Scalar beta,        Vector<Scalar>& y ) const;

    // y := alpha H^T x + beta y
    void RecursiveMatrixTransposeVector
    ( Scalar alpha, const MatrixShell& shell, 
                    const Vector<Scalar>& x,
      Scalar beta,        Vector<Scalar>& y ) const;

    // C := alpha H B + beta C
    void RecursiveMatrixMatrix
    ( Scalar alpha, const MatrixShell& shell,
                    const DenseMatrix<Scalar>& B,
      Scalar beta,        DenseMatrix<Scalar>& C ) const;

    // C := alpha H^T B + beta C
    void RecursiveMatrixTransposeMatrix
    ( Scalar alpha, const MatrixShell& shell,
                    const DenseMatrix<Scalar>& B,
      Scalar beta,        DenseMatrix<Scalar>& C ) const;

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

    const int Height() const;
    const int Width() const;

    // y := alpha H x + beta y
    void MapVector
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha H x
    void MapVector
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // y := alpha H^T x + beta y
    void TransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha H^T x
    void TransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // C := alpha H B + beta C
    void MapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, 
      Scalar beta, DenseMatrix<Scalar>& C ) const;

    // C := alpha H B
    void MapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const;
    
    // C := alpha H^T B + beta C
    void TransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, 
      Scalar beta, DenseMatrix<Scalar>& C ) const;

    // C := alpha H^T B
    void TransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const;
};

} // namespace psp

#endif // PSP_HMATRIX_QUASI2D_HPP
