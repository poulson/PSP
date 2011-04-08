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
#ifndef PSP_QUASI2D_HMATRIX_HPP
#define PSP_QUASI2D_HMATRIX_HPP 1

#include "psp/abstract_hmatrix.hpp"
#include "psp/hmatrix_tools.hpp"

namespace psp {

template<typename Scalar,bool Conjugated>
class Quasi2dHMatrix : public AbstractHMatrix<Scalar>
{
// Put the public section first since the private section depends upon 
// this class's public data structures.
public:    
    struct NodeData
    {
        std::vector<Quasi2dHMatrix*> children;
        int xSourceSizes[2];
        int ySourceSizes[2];
        int sourceSizes[4];
        int xTargetSizes[2];
        int yTargetSizes[2];
        int targetSizes[4];

        ~NodeData()
        {
            const int numChildren = children.size(); 
            for( int i=0; i<numChildren; ++i )
                delete children[i];
            children.clear();
        }
    };

    struct NodeSymmetricData
    {
        std::vector<Quasi2dHMatrix*> children;
        int xSizes[2];
        int ySizes[2];
        int sizes[4];

        ~NodeSymmetricData()
        {
            const int numChildren = children.size();
            for( int i=0; i<numChildren; ++i )
                delete children[i];
            children.clear();
        }
    };

    enum ShellType { NODE, NODE_SYMMETRIC, DENSE, FACTOR };

    struct Shell
    {
        ShellType type;
        union 
        {
            NodeData* node;
            NodeSymmetricData* nodeSymmetric;
            DenseMatrix<Scalar>* D;
            FactorMatrix<Scalar,Conjugated>* F;
        } data;
    };

    static void BuildNaturalToHierarchicalMap
    ( std::vector<int>& map, int xSize, int ySize, int zSize, int numLevels );

    // Default constructor
    Quasi2dHMatrix();

    // Create a square top-level H-matrix
    //
    // The weak admissibility criterion is:
    //     max(dist_x(A,B),dist_y(A,B)) >= 1
    //
    // The strong admissibility criterion is:
    //     max(dist_x(A,B),dist_y(A,B)) > 1
    //
    Quasi2dHMatrix
    ( const FactorMatrix<Scalar,Conjugated>& F,
      int numLevels, bool stronglyAdmissible,
      int xSize, int ySize, int zSize );
    Quasi2dHMatrix
    ( const SparseMatrix<Scalar>& S,
      int numLevels, bool stronglyAdmissible,
      int xSize, int ySize, int zSize );
    
    // Create a potentially non-square non-top-level H-matrix
    Quasi2dHMatrix
    ( const FactorMatrix<Scalar,Conjugated>& F,
      int numLevels, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int zSize,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );
    Quasi2dHMatrix
    ( const SparseMatrix<Scalar>& S,
      int numLevels, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int zSize,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );

    ~Quasi2dHMatrix();

    int XSizeSource() const { return _xSizeSource; }
    int XSizeTarget() const { return _xSizeTarget; }
    int YSizeSource() const { return _ySizeSource; }
    int YSizeTarget() const { return _ySizeTarget; }
    int ZSize() const { return _zSize; }
    int XSource() const { return _xSource; }
    int YSource() const { return _ySource; }

          Shell& GetShell() { return _shell; }
    const Shell& GetShell() const { return _shell; }

    // TODO
    /*
    void ConformWith
    ( const Quasi2dHMatrix<Scalar,Conjugate>& A, 
      const Quasi2dHMatrix<Scalar,Conjugate>& B );
    */

    //------------------------------------------------------------------------//
    // Fulfillments of AbstractHMatrix interface                              //
    //------------------------------------------------------------------------//

    // y := alpha H x + beta y
    virtual void MapVector
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A x
    virtual void MapVector
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // y := alpha A^T x + beta y
    virtual void TransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A^T x
    virtual void TransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // y := alpha A^H x + beta y
    virtual void HermitianTransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A^H x + beta y (temporarily conjugate x in-place)
    void HermitianTransposeMapVector
    ( Scalar alpha, Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const;

    // y := alpha A^H x
    virtual void HermitianTransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // y := alpha A^H x (temporarily conjugate x in-place)
    void HermitianTransposeMapVector
    ( Scalar alpha, Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // C := alpha A B + beta C
    virtual void MapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, 
      Scalar beta, DenseMatrix<Scalar>& C ) const;

    // C := alpha A B
    virtual void MapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const;
    
    // C := alpha A^T B + beta C
    virtual void TransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, 
      Scalar beta, DenseMatrix<Scalar>& C ) const;

    // C := alpha A^T B
    virtual void TransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const;
    
    // C := alpha A^H B + beta C
    virtual void HermitianTransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, 
      Scalar beta, DenseMatrix<Scalar>& C ) const;

    // C := alpha A^H B + beta C (temporarily conjugate B in place)
    void HermitianTransposeMapMatrix
    ( Scalar alpha, DenseMatrix<Scalar>& B, 
      Scalar beta,  DenseMatrix<Scalar>& C ) const;

    // C := alpha A^H B
    virtual void HermitianTransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const;
    
    // C := alpha A^H B (temporarily conjugate B in place)
    void HermitianTransposeMapMatrix
    ( Scalar alpha, DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const;
    
    //------------------------------------------------------------------------//
    // Computational routines specific to Quasi2dHMatrix                      //
    //------------------------------------------------------------------------//

    // TODO
    /*
    void MapMatrix
    ( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B, 
                          Quasi2dHMatrix<Scalar,Conjugated>& C );
    */
    /*
    void MapMatrix
    ( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B, 
      Scalar beta,        Quasi2dHMatrix<Scalar,Conjugated>& C );
    */

private:
    // Data specific to our quasi-2d H-matrix
    int _xSizeSource, _xSizeTarget;
    int _ySizeSource, _ySizeTarget;
    int _zSize;
    int _xSource, _xTarget;
    int _ySource, _yTarget;
    Shell _shell;

    bool Admissible( int xSource, int xTarget, int ySource, int yTarget ) const;

    // Since a constructor cannot call another constructor, have both of our
    // constructors call this routine.
    void ImportFactorMatrix( const FactorMatrix<Scalar,Conjugated>& F );
    void ImportSparseMatrix( const SparseMatrix<Scalar>& S );

    // y += alpha A x
    void UpdateVectorWithNodeSymmetric
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const;

    // C += alpha A B
    void UpdateMatrixWithNodeSymmetric
    ( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const;
};

} // namespace psp

#endif // PSP_QUASI2D_HMATRIX_HPP
