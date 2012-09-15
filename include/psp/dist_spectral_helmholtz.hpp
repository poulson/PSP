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
#ifndef PSP_DIST_SPECTRAL_HELMHOLTZ_HPP
#define PSP_DIST_SPECTRAL_HELMHOLTZ_HPP 1

namespace psp {

template<typename R>
class DistSpectralHelmholtz
{
public:
    typedef Complex<R> C;

    DistSpectralHelmholtz
    ( const SpectralDiscretization<R>& disc, mpi::Comm comm, 
      R damping=7.5, int numPlanesPerPanel=4 );

    ~DistSpectralHelmholtz();

    // Build the sparse matrix and the preconditioner
    void Initialize
    ( const DistUniformGrid<R>& velocity, 
      PanelScheme panelScheme=CLIQUE_LDL_SELINV_2D );

    // Solves an O(1) set of linear systems with the sweeping preconditioner
    void Solve
    ( DistUniformGrid<C>& B, int m=20, R relTol=1e-4, 
      bool viewIterates=false ) const;

    // Destroy the sparse matrix and the preconditioner
    void Finalize();

private:
    PanelScheme panelScheme_;
    mpi::Comm comm_;
    int distDepth_;
    const SpectralDiscretization<R> disc_;
    const R hx_, hy_, hz_; // grid spacings
    const R damping_;
    const int numPlanesPerPanel_;

    // Whether or not we have used velocity information to set up a 
    // preconditioner
    bool initialized_; 

    //
    // Solve helpers
    //

    void PullRightHandSides
    ( const DistUniformGrid<C>& gridB, Matrix<C>& B ) const;
    void PushRightHandSides
    ( DistUniformGrid<C>& gridB, const Matrix<C>& B ) const;

    void InternalSolveWithGMRES
    ( DistUniformGrid<C>& gridB, Matrix<C>& B, int m, R relTol, 
      bool viewIteratees ) const;

    bool CheckForNaN( R alpha ) const;
    bool CheckForNaN( C alpha ) const;
    bool CheckForNaN( const std::vector<R>& alpha ) const;
    bool CheckForNaN( const std::vector<C>& alpha ) const;
    bool CheckForZero( const std::vector<R>& alpha ) const;
    bool CheckForZero( const std::vector<C>& alpha ) const;

    void Norms( const Matrix<C>& xList, std::vector<R>& normList ) const;
    void InnerProducts
    ( const Matrix<C>& xList, const Matrix<C>& yList,
      std::vector<C>& alphaList ) const;

    void MultiplyColumns
    ( Matrix<C>& xList, const std::vector<C>& deltaList ) const;
    void DivideColumns
    ( Matrix<C>& xList, const std::vector<R>& deltaList ) const;
    void SubtractScaledColumns
    ( const std::vector<C>& deltaList,
      const Matrix<C>& xList, Matrix<C>& yList ) const;

    void Multiply( Matrix<C>& B ) const;
    void Precondition( Matrix<C>& B ) const;

    // B_i := T_i B_i
    void SolvePanel( Matrix<C>& B, int i ) const;

    // B_{i+1} := B_{i+1} - A_{i+1,i} B_i
    void SubdiagonalUpdate( Matrix<C>& B, int i ) const;

    // Z := B_i, B_i := 0
    void ExtractPanel( Matrix<C>& B, int i, Matrix<C>& Z ) const;

    // B_i := -A_{i,i+1} B_{i+1}
    void MultiplySuperdiagonal( Matrix<C>& B, int i ) const;

    // B_i := B_i + Z
    void UpdatePanel( Matrix<C>& B, int i, const Matrix<C>& Z ) const;

    //
    // Information related to the decomposition of the domain into panels
    //

    // Information about the bottom (and full inner) panels
    int bottomDepth_;          // including the original PML
    int localBottomHeight_;    // local height of the bottom panel
    int localFullInnerHeight_; // local height of full inner panels
    int innerDepth_;           // total physical inner depth
    int numFullInnerPanels_;   // number of full inner panels we need
    int numPanels_;            // total number of panels

    // Information about the leftover inner panel
    bool haveLeftover_;
    int leftoverInnerDepth_;
    int localLeftoverInnerHeight_;

    // Information about the top panel
    int topOrigDepth_;
    int localTopHeight_;

    // Analyses of each class of panels
    cliq::DistSymmInfo bottomInfo_, leftoverInnerInfo_, topInfo_;

    // Factorizations of each panel
    cliq::DistSymmFrontTree<C> bottomFact_;
    std::vector<cliq::DistSymmFrontTree<C>*> fullInnerFacts_;
    cliq::DistSymmFrontTree<C> leftoverInnerFact_;
    cliq::DistSymmFrontTree<C> topFact_;

    // Compressed factorizations of each panel
    DistCompressedFrontTree<C> bottomCompressedFact_;
    std::vector<DistCompressedFrontTree<C>*> fullInnerCompressedFacts_;
    DistCompressedFrontTree<C> leftoverInnerCompressedFact_;
    DistCompressedFrontTree<C> topCompressedFact_;

    //
    // Information related to the global sparse matrix
    //

    // Sparse matrix storage
    int localHeight_;
    std::vector<int> localToNaturalMap_;
    std::vector<int> localRowOffsets_;
    std::vector<int> owningProcesses_;
    std::vector<C> localEntries_;

    // Global sparse matrix communication information
    std::vector<int> globalSendCounts_, globalRecvCounts_; 
    std::vector<int> globalSendDispls_, globalRecvDispls_; 
    std::vector<int> globalSendIndices_; 

    // For redistributing pieces of B_i to form A_{i+1,i} B_i
    std::vector<int> subdiagSendCounts_, subdiagRecvCounts_;
    std::vector<int> subdiagSendDispls_, subdiagRecvDispls_;
    std::vector<int> subdiagPanelSendCounts_, subdiagPanelRecvCounts_;
    std::vector<int> subdiagPanelSendDispls_, subdiagPanelRecvDispls_;
    std::vector<int> subdiagSendIndices_;
    std::vector<int> subdiagRecvLocalIndices_, 
                     subdiagRecvLocalRows_;

    // For redistributing pieces of B_{i+1} to form A_{i,i+1} B_{i+1}
    std::vector<int> supdiagSendCounts_, supdiagRecvCounts_;
    std::vector<int> supdiagSendDispls_, supdiagRecvDispls_;
    std::vector<int> supdiagPanelSendCounts_, supdiagPanelRecvCounts_;
    std::vector<int> supdiagPanelSendDispls_, supdiagPanelRecvDispls_;
    std::vector<int> supdiagSendIndices_;
    std::vector<int> supdiagRecvLocalIndices_,
                     supdiagRecvLocalRows_;

    //
    // General helper routines 
    //

    C s1Inv( int x ) const;
    C s2Inv( int y ) const;
    C s3Inv( int v ) const;
    C s3InvArtificial( int v, int vOffset ) const;

    int NumLocalNodes( int commRank, int commSize ) const;
    static void NumLocalNodesRecursion
    ( int xSize, int ySize, int polyOrder, 
      int teamRank, int teamSize, int& numLocal );

    //
    // Global sparse helper routines
    //

    cliq::DistSymmFrontTree<C>& PanelFactorization( int whichPanel );
    DistCompressedFrontTree<C>& PanelCompressedFactorization( int whichPanel );

    const cliq::DistSymmFrontTree<C>& 
    PanelFactorization( int whichPanel ) const;
    const DistCompressedFrontTree<C>& 
    PanelCompressedFactorization( int whichPanel ) const;

    cliq::DistSymmInfo& PanelAnalysis( int whichPanel );
    const cliq::DistSymmInfo& PanelAnalysis( int whichPanel ) const;

    int PanelPadding( int whichPanel ) const;
    int PanelDepth( int whichPanel ) const;
    int WhichPanel( int v ) const;
    int PanelV( int whichPanel ) const;
    int LocalV( int v ) const;
    int LocalPanelOffset( int whichPanel ) const;
    int LocalPanelHeight( int whichPanel ) const;

    void GetGlobalVelocity
    ( const DistUniformGrid<R>& velocity,
      std::vector<R>& myGlobalVelocity,
      std::vector<int>& offsets ) const;

    void FormGlobalRow( R alpha, int x, int y, int v, int row );

    int LocalPanelHeight
    ( int vSize, int vPadding, int commRank, int commSize ) const;
    static void LocalPanelHeightRecursion
    ( int xSize, int ySize, int vSize, int vPadding, int polyOrder, 
      int teamRank, int teamSize, int& localHeight );

    // For building localToNaturalMap_, which takes our local index in the 
    // global sparse matrix and returns the original 'natural' index.
    void MapLocalPanelIndices
    ( int commRank, int commSize, int whichPanel );
    static void MapLocalPanelIndicesRecursion
    ( int nx, int ny, int nz, int xSize, int ySize, int vSize, int vPadding,
      int xOffset, int yOffset, int vOffset, int polyOrder,
      int teamRank, int teamSize,
      std::vector<int>& localToNaturalMap, std::vector<int>& localRowOffsets,
      int& localOffset );

    void MapLocalConnectionIndices
    ( int commRank, int commSize,  
      std::vector<int>& localConnections, int whichPanel ) const;
    static void MapLocalConnectionIndicesRecursion
    ( int nx, int ny, int nz, int xSize, int ySize, int vSize, int vPadding,
      int xOffset, int yOffset, int vOffset, int polyOrder,
      int teamRank, int teamSize,
      std::vector<int>& localConnections, int& localOffset );

    int OwningProcess( int naturalIndex, int commSize ) const;
    int OwningProcess( int x, int y, int v, int commSize ) const;
    static void OwningProcessRecursion
    ( int x, int y, int vLocal, int xSize, int ySize, int xOffset, int yOffset,
      int polyOrder, int teamSize, int& process );

    //
    // Helpers for the PML-padded sparse-direct portion
    //

    static int DistributedDepth( int commRank, int commSize );
    static void DistributedDepthRecursion
    ( int commRank, int commSize, int& distDepth );

    void GetPanelVelocity
    ( int vOffset, int vSize, 
      const cliq::DistSymmInfo& info,
      const DistUniformGrid<R>& velocity,
      std::vector<R>& myPanelVelocity,
      std::vector<int>& offsets,
      std::map<int,int>& panelNestedToNatural, 
      std::map<int,int>& panelNaturalToNested ) const;

    void FillPanelFronts
    ( int vOffset, int vSize,
      const cliq::DistSymmInfo& info,
            cliq::DistSymmFrontTree<C>& fact,
      const DistUniformGrid<R>& velocity,
      const std::vector<R>& myPanelVelocity,
            std::vector<int>& offsets,
            std::map<int,int>& panelNestedToNatural,
            std::map<int,int>& panelNaturalToNested ) const;
    void FillPanelFronts
    ( int vOffset, int vSize,
      const cliq::DistSymmInfo& info,
            DistCompressedFrontTree<C>& fact,
      const DistUniformGrid<R>& velocity,
      const std::vector<R>& myPanelVelocity,
            std::vector<int>& offsets,
            std::map<int,int>& panelNestedToNatural,
            std::map<int,int>& panelNaturalToNested ) const;

    void FormLowerColumnOfNode
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
      int polyOrder, int depthTilSerial, int commRank, int commSize );

    int ReorderedIndex( int x, int y, int vLocal, int vSize ) const;
    static int ReorderedIndexRecursion
    ( int x, int y, int vLocal, int xSize, int ySize, int vSize,
      int xOffset, int yOffset, int polyOrder, int depthTilSerial, int offset );

    void FillPanelElimTree( int vSize, cliq::DistSymmElimTree& eTree ) const;
    void FillPanelDistElimTree
    ( int vSize, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
      cliq::DistSymmElimTree& eTree ) const;
    void FillPanelLocalElimTree
    ( int vSize, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
      cliq::DistSymmElimTree& eTree ) const;
   
    // For use in FillPanelLocalElimTree
    struct Box
    {
        int parentIndex, nx, ny, xOffset, yOffset;
        bool leftChild;
    };
};

} // namespace psp

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

#include "./dist_spectral_helmholtz/main.hpp"
#include "./dist_spectral_helmholtz/initialize_finalize.hpp"
#include "./dist_spectral_helmholtz/solve.hpp"

#endif // PSP_DIST_SPECTRAL_HELMHOLTZ_HPP