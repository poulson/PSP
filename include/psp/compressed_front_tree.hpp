/*
   Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
   of a sweeping preconditioner for 3d Helmholtz equations.

   Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and
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
#ifndef PSP_COMPRESSED_FRONT_TREE_HPP
#define PSP_COMPRESSED_FRONT_TREE_HPP 1

namespace psp {

// Only keep track of the left and bottom-right piece of the fronts
// (with the bottom-right piece stored in workspace) since only the left side
// needs to be kept after the factorization is complete.
//
// As soon as they are factored we also compress the factorization using a 
// Kronecker product compression scheme. The exception is for the leaf level
// fronts, where the 'B' matrix can be stored sparsely.
//

template<typename F>
struct LocalCompressedFront
{
    Matrix<F> frontL;
    mutable Matrix<F> work;

    int sT, sB, depth;
    std::vector<Matrix<F> > AGreens, BGreens;
    std::vector<Matrix<F> > ACoefficients, BCoefficients;

    // TODO: Add possibility of storing B sparsly at the leaf level
    //       Typical sparse-matrix storage scheme?
};

template<typename F>
struct LocalCompressedFrontTree
{
    std::vector<LocalCompressedFront<F> > fronts;
};

template<typename F>
struct DistCompressedFront
{
    mutable DistMatrix<F> frontL;
    mutable DistMatrix<F> work2d;
    mutable DistMatrix<F,VC,STAR> work1d;

    int sT, sB, depth;
    const Grid* grid;
    std::vector<DistMatrix<F> > AGreens, BGreens;
    std::vector<DistMatrix<F,STAR,STAR> > ACoefficients, BCoefficients;
};

template<typename F>
struct DistCompressedFrontTree
{
    std::vector<DistCompressedFront<F> > fronts;
};

template<typename F>
struct CompressedFrontTree
{
    LocalCompressedFrontTree<F> local;
    DistCompressedFrontTree<F> dist;
};

} // namespace psp

#endif // PSP_COMPRESSED_FRONT_TREE_HPP
