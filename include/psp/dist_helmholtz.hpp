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
#ifndef PSP_DIST_HELMHOLTZ_HPP
#define PSP_DIST_HELMHOLTZ_HPP 1

#include "clique.hpp"

namespace psp {

template<typename F>
class DistHelmholtz
{
public:
    DistHelmholtz
    ( const FiniteDiffControl<F>& control, elemental::mpi::Comm comm );

    ~DistHelmholtz();

    // Build the sparse matrix and the preconditioner
    void Initialize( const F* localSlowness );

    // Destroy the sparse matrix and the preconditioner
    void Finalize();

    // Y := alpha A X + beta Y
    void Multiply( F alpha, const F* localX, F beta, F* localY ) const;

    // Y := approximateInv(A) Y
    void Precondition( F* localY ) const;

    // Return the number of rows of the sparse matrix that our process stores.
    int LocalSize() const;

    //void WriteParallelVtkFile( const F* vLocal, const char* basename ) const;

private:
    typedef typename elemental::RealBase<F>::type R;

    elemental::mpi::Comm comm_;
    const FiniteDiffControl<F> control_;

    // Useful constants
    const R hx_, hy_, hz_; // grid spacings
    const R bx_, by_, bz_; // (PML width)/(grid spacings)
    const int bzCeil_;     // ceil(bz)

    // Sparse matrix storage
    int localHeight_;
    std::vector<int> localToNaturalMap_;
    std::vector<int> localRowOffsets_;
    std::vector<F> localEntries_;

    // Sparse matrix communication information
    int allToAllSize_;
    std::vector<int> actualSendSizes_, actualRecvSizes_; // length p
    std::vector<int> sendIndices_, recvIndices_; // length p*allToAllSize_

    // TODO: 
    // Information for communication needed by application of A_{i+1,i} blocks.
    // This will be needed for more general stencils than SEVEN_POINT.

    bool initialized_;

    // Sparse-direct symbolic factorizations of PML-padded panels.
    // Since most of the inner panels are structurally equivalent, we can get
    // away with only a few symbolic factorizations.
    clique::symbolic::SymmFact 
        topSymbolicFact_, 
        fullInnerSymbolicFact_, 
        leftoverInnerSymbolicFact_, 
        bottomSymbolicFact_;

    // Sparse-direct numeric factorizations of PML-padded panels
    clique::numeric::SymmFrontTree<F> topFact_;
    std::vector<clique::numeric::SymmFrontTree<F>*> fullInnerFacts_;
    clique::numeric::SymmFrontTree<F> leftoverInnerFact_;
    clique::numeric::SymmFrontTree<F> bottomFact_;

    //
    // Helper routines
    //

    static void RecursiveReordering
    ( int nx, int xOffset, int xSize, int yOffset, int ySize, 
      int cutoff, int depthTilSerial, int* reordering );

    static int LocalPanelHeight
    ( int xSize, int ySize, int zSize, int cutoff, 
      unsigned commRank, unsigned log2CommSize );
    static void LocalPanelHeightRecursion
    ( int xSize, int ySize, int zSize, int cutoff, 
      unsigned commRank, unsigned log2CommSize, int& localHeight );

    static int NumLocalSupernodes
    ( int xSize, int ySize, int cutoff, 
      unsigned commRank, unsigned log2CommSize );
    static void NumLocalSupernodesRecursion
    ( int xSize, int ySize, int cutoff, 
      unsigned commRank, unsigned log2CommSize, int& numLocal );

    static int LocalZ
    ( int z, int topDepth, int innerDepth, int bottomOrigDepth,
      int bzPadded, int planesPerPanel );
    static int LocalPanelOffset
    ( int z, 
      int topDepth, int innerDepth, int planesPerPanel, int bottomOrigDepth,
      int numFullInnerPanels,
      int localTopHeight, int localInnerHeight, int localLeftoverHeight,
      int localBottomHeight );

    static void MapLocalPanelIndices
    ( int nx, int ny, int nz, int zSize, int& zOffset, int cutoff, 
      unsigned commRank, unsigned log2CommSize, 
      std::vector<int>& localToNaturalMap, std::vector<int>& localRowOffsets, 
      int& localOffset );
    static void MapLocalPanelIndicesRecursion
    ( int nx, int ny, int nz, int xSize, int ySize, int zSize,
      int xOffset, int yOffset, int zOffset, int cutoff,
      unsigned commRank, unsigned log2CommSize,
      std::vector<int>& localToNaturalMap, std::vector<int>& localRowOffsets,
      int& localOffset );

    static int OwningProcess
    ( int x, int y, int zLocal, int xSize, int ySize, unsigned log2CommSize );
    static void OwningProcessRecursion
    ( int x, int y, int zLocal, int xSize, int ySize, unsigned log2CommSize,
      int& process );
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F>
inline 
DistHelmholtz<F>::~DistHelmholtz() 
{
    const int numFullInnerPanels = fullInnerFacts_.size();
    for( int i=0; i<numFullInnerPanels; ++i )
        delete fullInnerFacts_[i];
    fullInnerFacts_.clear();
}

template<typename F>
inline int
DistHelmholtz<F>::LocalSize() const
{ return localHeight_; }

} // namespace psp

#endif // PSP_DIST_HELMHOLTZ_HPP
