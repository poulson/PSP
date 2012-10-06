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
#ifndef PSP_DIST_COMPRESSED_FRONT_TREE_HPP
#define PSP_DIST_COMPRESSED_FRONT_TREE_HPP 1

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
    bool isLeaf;
    std::vector<int> BRows, BCols;
    std::vector<F> BValues;
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
    bool isHermitian;
    std::vector<LocalCompressedFront<F> > localFronts;
    std::vector<DistCompressedFront<F> > distFronts;

    void MemoryInfo
    ( double& numLocalEntries, double& minLocalEntries, double& maxLocalEntries,
      double& numGlobalEntries ) const;
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F>
inline void
DistCompressedFrontTree<F>::MemoryInfo
( double& numLocalEntries, double& minLocalEntries, double& maxLocalEntries,
  double& numGlobalEntries ) const
{
#ifndef RELEASE
    PushCallStack("DistCompressedFrontTree::MemoryInfo");
#endif
    numLocalEntries = numGlobalEntries = 0;
    const int numLocalFronts = localFronts.size();
    const int numDistFronts = distFronts.size();
    const Grid& grid = *(distFronts.back().grid);
    mpi::Comm comm = grid.Comm();

    for( int s=0; s<numLocalFronts; ++s )
    {
        const LocalCompressedFront<F>& front = localFronts[s];
        numLocalEntries += front.frontL.MemorySize();
        numLocalEntries += front.work.MemorySize();

        const int numAGreens = front.AGreens.size();
        const int numBGreens = front.BGreens.size();
        const int numACoefficients = front.ACoefficients.size();
        const int numBCoefficients = front.BCoefficients.size();
        for( int t=0; t<numAGreens; ++t )
            numLocalEntries += front.AGreens[t].MemorySize();
        for( int t=0; t<numBGreens; ++t )
            numLocalEntries += front.BGreens[t].MemorySize();
        for( int t=0; t<numACoefficients; ++t )
            numLocalEntries += front.ACoefficients[t].MemorySize();
        for( int t=0; t<numBCoefficients; ++t )
            numLocalEntries += front.BCoefficients[t].MemorySize();
        // We now use the convention that four integers are equal to one 
        // double-precision complex floating point number
        numLocalEntries += 1.5*front.BValues.size();
    }
    for( int s=1; s<numDistFronts; ++s )
    {
        const DistCompressedFront<F>& front = distFronts[s];
        numLocalEntries += front.frontL.AllocatedMemory();
        numLocalEntries += front.work1d.AllocatedMemory();
        numLocalEntries += front.work2d.AllocatedMemory();

        const int numAGreens = front.AGreens.size();
        const int numBGreens = front.BGreens.size();
        const int numACoefficients = front.ACoefficients.size();
        const int numBCoefficients = front.BCoefficients.size();
        for( int t=0; t<numAGreens; ++t )
            numLocalEntries += front.AGreens[t].AllocatedMemory();
        for( int t=0; t<numBGreens; ++t )
            numLocalEntries += front.BGreens[t].AllocatedMemory();
        for( int t=0; t<numACoefficients; ++t )
            numLocalEntries += front.ACoefficients[t].AllocatedMemory();
        for( int t=0; t<numBCoefficients; ++t )
            numLocalEntries += front.BCoefficients[t].AllocatedMemory();
    }

    mpi::AllReduce( &numLocalEntries, &minLocalEntries, 1, mpi::MIN, comm );
    mpi::AllReduce( &numLocalEntries, &maxLocalEntries, 1, mpi::MAX, comm );
    mpi::AllReduce( &numLocalEntries, &numGlobalEntries, 1, mpi::SUM, comm );
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_DIST_COMPRESSED_FRONT_TREE_HPP
