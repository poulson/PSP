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
#ifndef PSP_ABSTRACT_HMATRIX_HPP
#define PSP_ABSTRACT_HMATRIX_HPP 1

#include "psp/hmatrix_tools.hpp"

namespace psp {

template<typename Scalar>
class AbstractHMatrix
{
public:
    virtual const int Height() const = 0;
    virtual const int Width() const = 0;

    // y := alpha A x + beta y
    virtual void MapVector
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const = 0;

    // y := alpha A x
    virtual void MapVector
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const = 0;

    // y := alpha A^T x + beta y
    virtual void TransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const = 0;

    // y := alpha A^T x
    virtual void TransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const = 0;

    // y := alpha A^H x + beta y
    virtual void HermitianTransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const = 0;

    // y := alpha A^H x
    virtual void HermitianTransposeMapVector
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const = 0;

    // C := alpha A B + beta C
    virtual void MapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, 
      Scalar beta, DenseMatrix<Scalar>& C ) const = 0;

    // C := alpha A B
    virtual void MapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const = 0;
    
    // C := alpha A^T B + beta C
    virtual void TransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, 
      Scalar beta, DenseMatrix<Scalar>& C ) const = 0;

    // C := alpha A^T B
    virtual void TransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const = 0;
    
    // C := alpha A^H B + beta C
    virtual void HermitianTransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, 
      Scalar beta, DenseMatrix<Scalar>& C ) const = 0;

    // C := alpha A^H B
    virtual void HermitianTransposeMapMatrix
    ( Scalar alpha, const DenseMatrix<Scalar>& B, DenseMatrix<Scalar>& C ) 
    const = 0;
};

} // namespace psp

#endif // PSP_ABSTRACT_HMATRIX_HPP
