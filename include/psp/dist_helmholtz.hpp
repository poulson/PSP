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

    // General panel information
    bool haveLeftover_;
    int topDepth_, innerDepth_, leftoverInnerDepth_, bottomOrigDepth_;
    int numFullInnerPanels_;
    int localTopHeight_, localFullInnerHeight_, localLeftoverInnerHeight_, 
        localBottomHeight_;

    // Sparse matrix storage
    int localHeight_;
    std::vector<int> localToNaturalMap_;
    std::vector<int> localRowOffsets_;
    std::vector<F> localEntries_;

    // Sparse matrix communication information
    int allToAllSize_;
    std::vector<int> actualSendSizes_, actualRecvSizes_; // length p
    std::vector<int> sendIndices_; // length p*allToAllSize_

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

    int LocalPanelHeight
    ( int zSize, unsigned commRank, unsigned log2CommSize ) const;
    static void LocalPanelHeightRecursion
    ( int xSize, int ySize, int zSize, int cutoff, 
      unsigned commRank, unsigned log2CommSize, int& localHeight );

    int NumLocalSupernodes( unsigned commRank, unsigned log2CommSize ) const;
    static void NumLocalSupernodesRecursion
    ( int xSize, int ySize, int cutoff, 
      unsigned commRank, unsigned log2CommSize, int& numLocal );

    int LocalZ( int z ) const;
    int LocalPanelOffset( int z ) const;

    void MapLocalPanelIndices
    ( int zSize, int& zOffset, unsigned commRank, unsigned log2CommSize, 
      int& localOffset );
    static void MapLocalPanelIndicesRecursion
    ( int nx, int ny, int nz, int xSize, int ySize, int zSize,
      int xOffset, int yOffset, int zOffset, int cutoff,
      unsigned commRank, unsigned log2CommSize,
      std::vector<int>& localToNaturalMap, std::vector<int>& localRowOffsets,
      int& localOffset );

    void MapLocalConnectionIndices
    ( int zSize, int& zOffset, unsigned commRank, unsigned log2CommSize, 
      std::vector<int>& localConnections, int& localOffset ) const;
    static void MapLocalConnectionIndicesRecursion
    ( int nx, int ny, int nz, int xSize, int ySize, int zSize,
      int xOffset, int yOffset, int zOffset, int cutoff,
      unsigned commRank, unsigned log2CommSize,
      std::vector<int>& localConnections, int& localOffset );

    int OwningProcess( int x, int y, int zLocal, unsigned log2CommSize ) const;
    static void OwningProcessRecursion
    ( int x, int y, int zLocal, int xSize, int ySize, unsigned log2CommSize,
      int& process );

    int ReorderedIndex
    ( int x, int y, int z, int zSize, int log2CommSize ) const;
    static int ReorderedIndexRecursion
    ( int x, int y, int z, int nx, int ny, int nz,
      int log2CommSize, int cutoff, int offset );

    void FillOrigPanelStruct
    ( int zSize, unsigned log2CommSize, clique::symbolic::SymmOrig& S ) const;
    void FillDistOrigPanelStruct
    ( int zSize, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
      unsigned log2CommSize, clique::symbolic::SymmOrig& S ) const;
    void FillLocalOrigPanelStruct
    ( int zSize, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
      unsigned log2CommSize, clique::symbolic::SymmOrig& S ) const;
   
    // For use in FillLocalOrigPanelStruct
    struct Box
    {
        int parentIndex, nx, ny, xOffset, yOffset;
        bool leftChild;
    };
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F>
inline 
DistHelmholtz<F>::~DistHelmholtz() 
{ }

template<typename F>
inline int
DistHelmholtz<F>::LocalSize() const
{ return localHeight_; }

} // namespace psp

#endif // PSP_DIST_HELMHOLTZ_HPP
