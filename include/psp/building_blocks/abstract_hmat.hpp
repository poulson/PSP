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
#ifndef PSP_ABSTRACT_HMAT_HPP
#define PSP_ABSTRACT_HMAT_HPP 1

#include "psp/building_blocks/dense.hpp"
#include "psp/building_blocks/vector.hpp"

#include "psp/building_blocks/low_rank.hpp"

namespace psp {

template<typename Scalar>
class AbstractHMat
{
public:
    /*
     * Public virtual member functions
     */
    virtual int Height() const = 0; 
    virtual int Width() const = 0;
    virtual int NumLevels() const = 0;
    virtual int MaxRank() const = 0;
    virtual int SourceOffset() const = 0;
    virtual int TargetOffset() const = 0;
    virtual bool Symmetric() const = 0;
    virtual bool StronglyAdmissible() const = 0;

    // Display the equivalent dense matrix
    virtual void Print( const std::string tag ) const = 0;

    // y := alpha A x + beta y
    virtual void Multiply
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const = 0;

    // y := alpha A x
    virtual void Multiply
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const = 0;

    // y := alpha A^T x + beta y
    virtual void TransposeMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const = 0;

    // y := alpha A^T x
    virtual void TransposeMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const = 0;

    // y := alpha A^H x + beta y
    virtual void AdjointMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Scalar beta, Vector<Scalar>& y ) 
    const = 0;

    // y := alpha A^H x
    virtual void AdjointMultiply
    ( Scalar alpha, const Vector<Scalar>& x, Vector<Scalar>& y ) const = 0;

    // C := alpha A B + beta C
    virtual void Multiply
    ( Scalar alpha, const Dense<Scalar>& B, 
      Scalar beta,        Dense<Scalar>& C ) const = 0;

    // C := alpha A B
    virtual void Multiply
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const = 0;
    
    // C := alpha A^T B + beta C
    virtual void TransposeMultiply
    ( Scalar alpha, const Dense<Scalar>& B, 
      Scalar beta,        Dense<Scalar>& C ) const = 0;

    // C := alpha A^T B
    virtual void TransposeMultiply
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const = 0;
    
    // C := alpha A^H B + beta C
    virtual void AdjointMultiply
    ( Scalar alpha, const Dense<Scalar>& B, 
      Scalar beta,        Dense<Scalar>& C ) const = 0;

    // C := alpha A^H B
    virtual void AdjointMultiply
    ( Scalar alpha, const Dense<Scalar>& B, Dense<Scalar>& C ) const = 0;
};

} // namespace psp

#endif // PSP_ABSTRACT_HMAT_HPP
