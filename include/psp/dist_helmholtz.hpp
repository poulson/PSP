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
    void Initialize( const GridData<R>& slowness );

    // Destroy the sparse matrix and the preconditioner
    void Finalize();

    // y := alpha A x + beta y
    void Multiply
    ( C alpha, const GridData<C>& x, C beta, GridData<C>& y ) const;

    // y := approximateInv(A) y
    void Precondition( GridData<C>& y ) const;

    //void WriteParallelVtkFile( const F* vLocal, const char* basename ) const;

private:
    elemental::mpi::Comm comm_;
    unsigned log2CommSize_;
    const FiniteDiffControl<R> control_;

    // Frequently used extra grid constants
    const R hx_, hy_, hz_; // grid spacings
    const R bx_, by_, bz_; // (PML width)/(grid spacings)
    const int bzCeil_;     // ceil(bz)

    // Whether or not we have used slowness information to set up a 
    // preconditioner
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
    // General helper routines 
    //

    C s1Inv( int x ) const;
    C s2Inv( int y ) const;
    C s3Inv( int z ) const;
    C s3InvArtificial( int z, int zOffset, R sizeOfPML ) const;

    int NumLocalSupernodes( unsigned commRank ) const;
    static void NumLocalSupernodesRecursion
    ( int xSize, int ySize, int cutoff, 
      unsigned commRank, unsigned depthTilSerial, int& numLocal );

    //
    // Global sparse helper routines
    //

    void GetGlobalSlowness
    ( const GridData<R>& slowness,
      std::vector<R>& recvSlowness,
      std::vector<int>& recvOffsets ) const;

    void FormGlobalRow( R alpha, int x, int y, int z, int row );

    int LocalPanelHeight( int zSize, int zPadding, unsigned commRank ) const;
    static void LocalPanelHeightRecursion
    ( int xSize, int ySize, int zSize, int zPadding, int cutoff, 
      unsigned commRank, unsigned depthTilSerial, int& localHeight );

    int LocalZ( int z ) const;
    int LocalPanelOffset( int z ) const;

    // For building localToNaturalMap_, which takes our local index in the 
    // global sparse matrix and returns the original 'natural' index.
    void MapLocalPanelIndices
    ( int zSize, int zPadding, int& zOffset, unsigned commRank, 
      int& localOffset );
    static void MapLocalPanelIndicesRecursion
    ( int nx, int ny, int nz, int xSize, int ySize, int zSize, int zPadding,
      int xOffset, int yOffset, int zOffset, int cutoff,
      unsigned commRank, unsigned depthTilSerial,
      std::vector<int>& localToNaturalMap, std::vector<int>& localRowOffsets,
      int& localOffset );

    void MapLocalConnectionIndices
    ( int zSize, int zPadding, int& zOffset, unsigned commRank,  
      std::vector<int>& localConnections, int& localOffset ) const;
    static void MapLocalConnectionIndicesRecursion
    ( int nx, int ny, int nz, int xSize, int ySize, int zSize, int zPadding,
      int xOffset, int yOffset, int zOffset, int cutoff,
      unsigned commRank, unsigned depthTilSerial,
      std::vector<int>& localConnections, int& localOffset );

    int OwningProcess( int x, int y, int zLocal ) const;
    static void OwningProcessRecursion
    ( int x, int y, int zLocal, int xSize, int ySize, 
      unsigned depthTilSerial, int& process );

    //
    // Helpers for the PML-padded sparse-direct portion
    //

    void GetPanelSlowness
    ( int zOffset, int zSize, 
      const clique::symbolic::SymmFact& fact,
      const GridData<R>& slowness,
      std::vector<R>& recvSlowness,
      std::vector<int>& recvOffsets,
      std::map<int,int>& panelNestedToNatural, 
      std::map<int,int>& panelNaturalToNested ) const;

    void FormLowerColumnOfSupernode
    ( R alpha, R imagShift, int x, int y, int z, int zOffset, int zSize,
      int offset, int size, int j, 
      const std::vector<int>& origLowerStruct, 
      const std::vector<int>& origLowerRelIndices,
      std::map<int,int>& panelNaturalToNested, 
      std::vector<int>& frontIndices, std::vector<C>& values ) const;

    void LocalReordering( std::map<int,int>& reordering, int zSize ) const;
    static void LocalReorderingRecursion
    ( std::map<int,int>& reordering, int offset, 
      int xOffset, int yOffset, int xSize, int ySize, int zSize, int nx, int ny,
      int depthTilSerial, int cutoff, int commRank );

    int ReorderedIndex( int x, int y, int z, int zSize ) const;
    static int ReorderedIndexRecursion
    ( int x, int y, int z, int xSize, int ySize, int zSize,
      int depthTilSerial, int cutoff, int offset );

    void FillOrigPanelStruct( int zSize, clique::symbolic::SymmOrig& S ) const;
    void FillDistOrigPanelStruct
    ( int zSize, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
      clique::symbolic::SymmOrig& S ) const;
    void FillLocalOrigPanelStruct
    ( int zSize, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
      clique::symbolic::SymmOrig& S ) const;
   
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

} // namespace psp

#endif // PSP_DIST_HELMHOLTZ_HPP
