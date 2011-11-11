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

enum Solver {
    GMRES,
    QMR
};

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

    // Solves an O(1) set of linear systems with the sweeping preconditioner
    void Solve( GridData<C>& B, Solver solver=QMR ) const;

    // Destroy the sparse matrix and the preconditioner
    void Finalize();

    // TODO: Add ability to write parallel VTK files

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
    // Solve helpers
    //

    void PullRightHandSides
    ( const GridData<C>& B, std::vector<C>& redistB ) const;
    void PushRightHandSides
    ( GridData<C>& B, const std::vector<C>& redistB ) const;

    void SolveWithGMRES( std::vector<C>& redistB ) const;
    void SolveWithQMR( std::vector<C>& redistB ) const;

    void Multiply( std::vector<C>& redistB ) const;
    void Precondition( std::vector<C>& redistB ) const;

    // B_i := T_i B_i
    void SolvePanel( std::vector<C>& B, int i ) const;
    // B_{i+1} := B_{i+1} - A_{i+1,i} B_i
    void SubdiagonalUpdate( std::vector<C>& B, int i ) const;
    // Z := B_i
    void ExtractPanel
    ( const std::vector<C>& B, int i, std::vector<C>& Z ) const;
    // B_i := -A_{i,i+1} B_{i+1}
    void MultiplySuperdiagonal( std::vector<C>& B, int i ) const;
    // B_i := B_i + Z
    void UpdatePanel( std::vector<C>& B, int i, const std::vector<C>& Z ) const;

    //
    // Information related to the decomposition of the domain into panels
    //

    // Information about the bottom panel
    int bottomDepth_;       // including the original PML
    int localBottomHeight_; // our process's local height of the bottom panel

    // Information about the full inner panels
    int innerDepth_;           // total physical inner depth
    int numFullInnerPanels_;   // number of full inner panels we need
    int localFullInnerHeight_; // local height of each full inner panel

    // Information about the leftover inner panel
    bool haveLeftover_;
    int leftoverInnerDepth_;
    int localLeftoverInnerHeight_;

    // Information about the top panel
    int topOrigDepth_;
    int localTopHeight_;

    // Symbolic factorizations of each class of panels
    clique::symbolic::SymmFact 
        bottomSymbolicFact_, 
        fullInnerSymbolicFact_, 
        leftoverInnerSymbolicFact_, 
        topSymbolicFact_;

    // Numeric factorizations of each panel
    clique::numeric::SymmFrontTree<C> bottomFact_;
    std::vector<clique::numeric::SymmFrontTree<C>*> fullInnerFacts_;
    clique::numeric::SymmFrontTree<C> leftoverInnerFact_;
    clique::numeric::SymmFrontTree<C> topFact_;

    //
    // Information related to the global sparse matrix
    //

    // Sparse matrix storage
    int localHeight_;
    std::vector<int> localToNaturalMap_;
    std::vector<int> localRowOffsets_;
    std::vector<C> localEntries_;
    std::vector<int> owningProcesses_;

    // Sparse matrix communication information
    int allToAllSize_;
    std::vector<int> actualSendSizes_, actualRecvSizes_; // length p
    std::vector<int> sendIndices_; // length p*allToAllSize_

    //
    // General helper routines 
    //

    C s1Inv( int x ) const;
    C s2Inv( int y ) const;
    C s3Inv( int v ) const;
    C s3InvArtificial( int v, int vOffset, R sizeOfPML ) const;

    int NumLocalSupernodes( unsigned commRank ) const;
    static void NumLocalSupernodesRecursion
    ( int xSize, int ySize, int cutoff, 
      unsigned commRank, unsigned depthTilSerial, int& numLocal );

    //
    // Global sparse helper routines
    //

    int PanelV( int whichPanel ) const;
    int LocalV( int v ) const;
    int LocalPanelOffset( int whichPanel ) const;
    int LocalPanelHeight( int whichPanel ) const;

    void GetGlobalSlowness
    ( const GridData<R>& slowness,
      std::vector<R>& myGlobalSlowness,
      std::vector<int>& offsets ) const;

    void FormGlobalRow( R alpha, int x, int y, int v, int row );

    int LocalPanelHeight( int vSize, int vPadding, unsigned commRank ) const;
    static void LocalPanelHeightRecursion
    ( int xSize, int ySize, int vSize, int vPadding, int cutoff, 
      unsigned commRank, unsigned depthTilSerial, int& localHeight );

    // For building localToNaturalMap_, which takes our local index in the 
    // global sparse matrix and returns the original 'natural' index.
    void MapLocalPanelIndices
    ( int vSize, int vPadding, unsigned commRank, int whichPanel );
    static void MapLocalPanelIndicesRecursion
    ( int nx, int ny, int nz, int xSize, int ySize, int vSize, int vPadding,
      int xOffset, int yOffset, int vOffset, int cutoff,
      unsigned commRank, unsigned depthTilSerial,
      std::vector<int>& localToNaturalMap, std::vector<int>& localRowOffsets,
      int& localOffset );

    void MapLocalConnectionIndices
    ( int vSize, int vPadding, unsigned commRank,  
      std::vector<int>& localConnections, int whichPanel ) const;
    static void MapLocalConnectionIndicesRecursion
    ( int nx, int ny, int nz, int xSize, int ySize, int vSize, int vPadding,
      int xOffset, int yOffset, int vOffset, int cutoff,
      unsigned commRank, unsigned depthTilSerial,
      std::vector<int>& localConnections, int& localOffset );

    int OwningProcess( int naturalIndex ) const;
    static void OwningProcessRecursion
    ( int x, int y, int vLocal, int xSize, int ySize, 
      unsigned depthTilSerial, int& process );

    //
    // Helpers for the PML-padded sparse-direct portion
    //

    void GetPanelSlowness
    ( int vOffset, int vSize, 
      const clique::symbolic::SymmFact& fact,
      const GridData<R>& slowness,
      std::vector<R>& myPanelSlowness,
      std::vector<int>& offsets,
      std::map<int,int>& panelNestedToNatural, 
      std::map<int,int>& panelNaturalToNested ) const;

    void FillPanelFronts
    ( int vOffset, int vSize,
      const clique::symbolic::SymmFact& symbFact,
            clique::numeric::SymmFrontTree<C>& fact,
      const GridData<R>& slowness,
      const std::vector<R>& myPanelSlowness,
            std::vector<int>& offsets,
            std::map<int,int>& panelNestedToNatural,
            std::map<int,int>& panelNaturalToNested ) const;

    void FormLowerColumnOfSupernode
    ( R alpha, int x, int y, int v, int vOffset, int vSize,
      int offset, int size, int j, 
      const std::vector<int>& origLowerStruct, 
      const std::vector<int>& origLowerRelIndices,
      std::map<int,int>& panelNaturalToNested, 
      std::vector<int>& frontIndices, std::vector<C>& values ) const;

    void LocalReordering( std::map<int,int>& reordering, int vSize ) const;
    static void LocalReorderingRecursion
    ( std::map<int,int>& reordering, int offset, 
      int xOffset, int yOffset, int xSize, int ySize, int vSize, int nx, int ny,
      int depthTilSerial, int cutoff, int commRank );

    int ReorderedIndex( int x, int y, int vLocal, int vSize ) const;
    static int ReorderedIndexRecursion
    ( int x, int y, int vLocal, int xSize, int ySize, int vSize,
      int depthTilSerial, int cutoff, int offset );

    void FillOrigPanelStruct( int vSize, clique::symbolic::SymmOrig& S ) const;
    void FillDistOrigPanelStruct
    ( int vSize, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
      clique::symbolic::SymmOrig& S ) const;
    void FillLocalOrigPanelStruct
    ( int vSize, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
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
