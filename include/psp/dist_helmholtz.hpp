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

template<typename R>
class DistHelmholtz
{
public:
    typedef std::complex<R> C;

    DistHelmholtz
    ( const FiniteDiffControl<R>& control, elemental::mpi::Comm comm );

    ~DistHelmholtz();

    // Build the sparse matrix and the preconditioner
    void Initialize( const R* localSlowness );

    // Destroy the sparse matrix and the preconditioner
    void Finalize();

    // Y := alpha A X + beta Y
    void Multiply( C alpha, const C* localX, C beta, C* localY ) const;

    // Y := approximateInv(A) Y
    void Precondition( C* localY ) const;

    // Return the number of rows of the sparse matrix that our process stores.
    int LocalSize() const;

    //void WriteParallelVtkFile( const F* vLocal, const char* basename ) const;

private:
    elemental::mpi::Comm comm_;
    const FiniteDiffControl<R> control_;

    // Frequently used extra grid constants
    const R hx_, hy_, hz_; // grid spacings
    const R bx_, by_, bz_; // (PML width)/(grid spacings)
    const int bzCeil_;     // ceil(bz)

    // Whether or not we have used slowness information to set up a preconditioner
    bool initialized_; 

    //
    // Information related to the decomposition of the domain into panels
    //

    // Information about the top panel
    int topDepth_;       // including the original PML
    int localTopHeight_; // our process's local height of the top panel

    // Information about the full inner panels
    int innerDepth_;           // total physical inner depth
    int numFullInnerPanels_;   // number of full inner panels we need
    int localFullInnerHeight_; // local height of each full inner panel

    // Information about the leftover inner panel
    bool haveLeftover_;
    int leftoverInnerDepth_;
    int localLeftoverInnerHeight_;

    // Information about the bottom panel
    int bottomOrigDepth_;
    int localBottomHeight_;

    // Symbolic factorizations of each class of panels
    clique::symbolic::SymmFact 
        topSymbolicFact_, 
        fullInnerSymbolicFact_, 
        leftoverInnerSymbolicFact_, 
        bottomSymbolicFact_;

    // Numeric factorizations of each panel
    clique::numeric::SymmFrontTree<C> topFact_;
    std::vector<clique::numeric::SymmFrontTree<C>*> fullInnerFacts_;
    clique::numeric::SymmFrontTree<C> leftoverInnerFact_;
    clique::numeric::SymmFrontTree<C> bottomFact_;

    //
    // Information related to the global sparse matrix
    //

    // Sparse matrix storage
    int localHeight_;
    std::vector<int> localToNaturalMap_;
    std::vector<int> localRowOffsets_;
    std::vector<C> localEntries_;

    // Sparse matrix communication information
    int allToAllSize_;
    std::vector<int> actualSendSizes_, actualRecvSizes_; // length p
    std::vector<int> sendIndices_; // length p*allToAllSize_

    //
    // Helper routines 
    //

    C s1Inv( int x ) const;
    C s2Inv( int y ) const;
    C s3Inv( int z ) const;
    C s3InvArtificial( int z, int zOffset, R sizeOfPML ) const;
    void FormRow
    ( R imagShift, int x, int y, int z, int zOffset, int zSize,
      int rowOffset, R alpha );

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

    // For building localToNaturalMap_, which takes our local index in the 
    // global sparse matrix and returns the original 'natural' index.
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

template<typename R>
inline 
DistHelmholtz<R>::~DistHelmholtz() 
{ }

template<typename R>
inline int
DistHelmholtz<R>::LocalSize() const
{ return localHeight_; }

} // namespace psp

#endif // PSP_DIST_HELMHOLTZ_HPP
