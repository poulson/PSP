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
    struct Node
    {
        std::vector<Quasi2dHMatrix*> children;
        int xSourceSizes[2];
        int ySourceSizes[2];
        int sourceSizes[4];
        int xTargetSizes[2];
        int yTargetSizes[2];
        int targetSizes[4];

        Node
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

        ~Node()
        {
            for( unsigned i=0; i<children.size(); ++i )
                delete children[i];
            children.clear();
        }

        Quasi2dHMatrix& Child( int i, int j )
        { 
#ifndef RELEASE
            if( i < 0 || j < 0 )
                throw std::logic_error("Child indices must be non-negative");
            if( i > 3 || j > 3 )
                throw std::logic_error("Child indices out of bounds");
            if( children.size() != 16 )
                throw std::logic_error("children array not yet set up");
#endif
            return *children[j+4*i]; 
        }

        const Quasi2dHMatrix& Child( int i, int j ) const
        { 
#ifndef RELEASE
            if( i < 0 || j < 0 )
                throw std::logic_error("Child indices must be non-negative");
            if( i > 3 || j > 3 )
                throw std::logic_error("Child indices out of bounds");
            if( children.size() != 16 )
                throw std::logic_error("children array not yet set up");
#endif
            return *children[j+4*i]; 
        }
    };

    struct NodeSymmetric
    {
        std::vector<Quasi2dHMatrix*> children;
        int xSizes[2];
        int ySizes[2];
        int sizes[4];

        NodeSymmetric( int xSize, int ySize, int zSize )
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

        ~NodeSymmetric()
        {
            for( unsigned i=0; i<children.size(); ++i )
                delete children[i];
            children.clear();
        }

        Quasi2dHMatrix& Child( int i, int j )
        { 
#ifndef RELEASE
            if( i < 0 || j < 0 )
                throw std::logic_error("Child indices must be non-negative");
            if( i > 3 || j > 3 )
                throw std::logic_error("Child indices out of bounds");
            if( j > i )
                throw std::logic_error("Child index outside of lower triangle");
            if( children.size() != 10 )
                throw std::logic_error("children array not yet set up");
#endif
            return *children[(i*(i+1))/2 + j]; 
        }

        const Quasi2dHMatrix& Child( int i, int j ) const
        {
#ifndef RELEASE
            if( i < 0 || j < 0 )
                throw std::logic_error("Child indices must be non-negative");
            if( i > 3 || j > 3 )
                throw std::logic_error("Child indices out of bounds");
            if( j > i )
                throw std::logic_error("Child index outside of lower triangle");
            if( children.size() != 10 )
                throw std::logic_error("children array not yet set up");
#endif
            return *children[(i*(i+1))/2 + j];
        }
    };

    enum ShellType { NODE, NODE_SYMMETRIC, DENSE, LOW_RANK };

    struct Shell
    {
        ShellType type;
        union Data
        {
            Node* node;
            NodeSymmetric* nodeSymmetric;
            DenseMatrix<Scalar>* D;
            LowRankMatrix<Scalar,Conjugated>* F;

            Data() { std::memset( this, 0, sizeof(Data) ); }
        } data;

        Shell() : type(NODE), data() { }

        ~Shell()
        {
            switch( type )
            {
            case NODE:           delete data.node; break;
            case NODE_SYMMETRIC: delete data.nodeSymmetric; break;
            case LOW_RANK:       delete data.F; break;
            case DENSE:          delete data.D; break;
            }
        }
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
    ( const LowRankMatrix<Scalar,Conjugated>& F,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSize, int ySize, int zSize );
    Quasi2dHMatrix
    ( const SparseMatrix<Scalar>& S,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSize, int ySize, int zSize );
    
    // Create a potentially non-square non-top-level H-matrix
    Quasi2dHMatrix
    ( const LowRankMatrix<Scalar,Conjugated>& F,
      int numLevels, int maxRank, bool stronglyAdmissible,
      int xSizeSource, int xSizeTarget,
      int ySizeSource, int ySizeTarget,
      int zSize,
      int xSource, int xTarget,
      int ySource, int yTarget,
      int sourceOffset, int targetOffset );
    Quasi2dHMatrix
    ( const SparseMatrix<Scalar>& S,
      int numLevels, int maxRank, bool stronglyAdmissible,
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
    int XTarget() const { return _xTarget; }
    int YTarget() const { return _yTarget; }

          Shell& GetShell() { return _shell; }
    const Shell& GetShell() const { return _shell; }

    bool IsDense() const { return _shell.type == DENSE; }
    bool IsHierarchical() const
    { return _shell.type == NODE || _shell.type == NODE_SYMMETRIC; }
    bool IsLowRank() const { return _shell.type == LOW_RANK; }

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

    // A := B
    void CopyFrom( const Quasi2dHMatrix<Scalar,Conjugated>& B );

    // A := alpha A
    void Scale( Scalar alpha );

    // A := I
    void SetToIdentity();

    // A :~= alpha B + A
    void UpdateWith( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B );

    // C :~= alpha A B
    void MapMatrix
    ( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B, 
                          Quasi2dHMatrix<Scalar,Conjugated>& C ) const;

    // C :~= alpha A B + beta C
    void MapMatrix
    ( Scalar alpha, const Quasi2dHMatrix<Scalar,Conjugated>& B, 
      Scalar beta,        Quasi2dHMatrix<Scalar,Conjugated>& C ) const;

    // A :~= inv(A)
    void Invert();

private:
    // Data specific to our quasi-2d H-matrix
    int _xSizeSource, _xSizeTarget;
    int _ySizeSource, _ySizeTarget;
    int _zSize;
    int _xSource, _xTarget;
    int _ySource, _yTarget;
    Shell _shell;

    bool Admissible( int xSource, int xTarget, int ySource, int yTarget ) const;

    void ImportLowRankMatrix
    ( const LowRankMatrix<Scalar,Conjugated>& F, int iOffset=0, int jOffset=0 );
    void ImportSparseMatrix
    ( const SparseMatrix<Scalar>& S, int iOffset=0, int jOffset=0 );

    void UpdateWithLowRankMatrix
    ( Scalar alpha,
      const LowRankMatrix<Scalar,Conjugated>& F, int iOffset=0, int jOffset=0 );

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
