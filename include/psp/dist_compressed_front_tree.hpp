/*
   Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and 
   The University of Texas at Austin
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
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
struct CompressedFront
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
    std::vector<CompressedFront<F> > localFronts;
    std::vector<DistCompressedFront<F> > distFronts;

    void MemoryInfo
    ( double& numLocalEntries, double& minLocalEntries, double& maxLocalEntries,
      double& numGlobalEntries ) const;

    void TopLeftMemoryInfo
    ( double& numLocalEntries, double& minLocalEntries, double& maxLocalEntries,
      double& numGlobalEntries ) const;

    void BottomLeftMemoryInfo
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
        const CompressedFront<F>& front = localFronts[s];
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

template<typename F>
inline void
DistCompressedFrontTree<F>::TopLeftMemoryInfo
( double& numLocalEntries, double& minLocalEntries, double& maxLocalEntries,
  double& numGlobalEntries ) const
{
#ifndef RELEASE
    PushCallStack("DistCompressedFrontTree::TopLeftMemoryInfo");
#endif
    numLocalEntries = numGlobalEntries = 0;
    const int numLocalFronts = localFronts.size();
    const int numDistFronts = distFronts.size();
    const Grid& grid = *(distFronts.back().grid);
    mpi::Comm comm = grid.Comm();

    for( int s=0; s<numLocalFronts; ++s )
    {
        const CompressedFront<F>& front = localFronts[s];
        Matrix<F> FTL,
                  FBL;
        elem::LockedPartitionDown
        ( front.frontL, FTL,
                        FBL, front.frontL.Width() );
        numLocalEntries += FTL.Height()*FTL.Width();

        const int numAGreens = front.AGreens.size();
        const int numACoefficients = front.ACoefficients.size();
        for( int t=0; t<numAGreens; ++t )
            numLocalEntries += front.AGreens[t].MemorySize();
        for( int t=0; t<numACoefficients; ++t )
            numLocalEntries += front.ACoefficients[t].MemorySize();
    }
    for( int s=1; s<numDistFronts; ++s )
    {
        const DistCompressedFront<F>& front = distFronts[s];
        DistMatrix<F> FTL(grid),
                      FBL(grid);
        elem::LockedPartitionDown
        ( front.frontL, FTL,
                        FBL, front.frontL.Width() );
        numLocalEntries += FTL.LocalHeight()*FTL.LocalWidth();

        const int numAGreens = front.AGreens.size();
        const int numACoefficients = front.ACoefficients.size();
        for( int t=0; t<numAGreens; ++t )
            numLocalEntries += front.AGreens[t].AllocatedMemory();
        for( int t=0; t<numACoefficients; ++t )
            numLocalEntries += front.ACoefficients[t].AllocatedMemory();
    }

    mpi::AllReduce( &numLocalEntries, &minLocalEntries, 1, mpi::MIN, comm );
    mpi::AllReduce( &numLocalEntries, &maxLocalEntries, 1, mpi::MAX, comm );
    mpi::AllReduce( &numLocalEntries, &numGlobalEntries, 1, mpi::SUM, comm );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void
DistCompressedFrontTree<F>::BottomLeftMemoryInfo
( double& numLocalEntries, double& minLocalEntries, double& maxLocalEntries,
  double& numGlobalEntries ) const
{
#ifndef RELEASE
    PushCallStack("DistCompressedFrontTree::BottomLeftMemoryInfo");
#endif
    numLocalEntries = numGlobalEntries = 0;
    const int numLocalFronts = localFronts.size();
    const int numDistFronts = distFronts.size();
    const Grid& grid = *(distFronts.back().grid);
    mpi::Comm comm = grid.Comm();

    for( int s=0; s<numLocalFronts; ++s )
    {
        const CompressedFront<F>& front = localFronts[s];

        Matrix<F> FTL,
                  FBL;
        elem::LockedPartitionDown
        ( front.frontL, FTL,
                        FBL, front.frontL.Width() );
        numLocalEntries += FBL.Height()*FBL.Width();

        const int numBGreens = front.BGreens.size();
        const int numBCoefficients = front.BCoefficients.size();
        for( int t=0; t<numBGreens; ++t )
            numLocalEntries += front.BGreens[t].MemorySize();
        for( int t=0; t<numBCoefficients; ++t )
            numLocalEntries += front.BCoefficients[t].MemorySize();
        // We now use the convention that four integers are equal to one 
        // double-precision complex floating point number
        numLocalEntries += 1.5*front.BValues.size();
    }
    for( int s=1; s<numDistFronts; ++s )
    {
        const DistCompressedFront<F>& front = distFronts[s];

        DistMatrix<F> FTL(grid),
                      FBL(grid);
        elem::LockedPartitionDown
        ( front.frontL, FTL,
                        FBL, front.frontL.Width() );
        numLocalEntries += FBL.LocalHeight()*FBL.LocalWidth();

        const int numBGreens = front.BGreens.size();
        const int numBCoefficients = front.BCoefficients.size();
        for( int t=0; t<numBGreens; ++t )
            numLocalEntries += front.BGreens[t].AllocatedMemory();
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
