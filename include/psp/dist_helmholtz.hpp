/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_DIST_HELMHOLTZ_HPP
#define PSP_DIST_HELMHOLTZ_HPP

namespace psp {

enum PanelScheme {
  CLIQUE_LDL_1D=0, 
  CLIQUE_LDL_SELINV_1D=1
  // TODO: Various other (e.g., intraPiv and 2D) schemes
};

template<typename Real>
class DistHelmholtz
{
public:
    typedef Complex<Real> C;

    DistHelmholtz
    ( const Discretization<Real>& disc, mpi::Comm comm, 
      Real damping=7.0, int numPlanesPerPanel=4, int cutoff=12 );

    ~DistHelmholtz();

    // Build the sparse matrix and the preconditioner
    void Initialize
    ( const DistUniformGrid<Real>& velocity, 
      PanelScheme panelScheme=CLIQUE_LDL_SELINV_1D );

    // Solves an O(1) set of linear systems with the sweeping preconditioner
    void Solve
    ( DistUniformGrid<C>& B, int m=20, Real relTol=1e-4, 
      bool viewIterates=false ) const;

    // Destroy the sparse matrix and the preconditioner
    void Finalize();

private:
    PanelScheme panelScheme_;
    mpi::Comm comm_;
    int distDepth_;
    const Discretization<Real> disc_;
    const Real hx_, hy_, hz_; // grid spacings
    const Real damping_;
    const int numPlanesPerPanel_;
    const int nestedCutoff_;

    // Whether or not we have used velocity information to set up a 
    // preconditioner
    bool initialized_; 

    // Solve helpers
    // =============

    void PullRightHandSides
    ( const DistUniformGrid<C>& gridB, Matrix<C>& B ) const;
    void PushRightHandSides
    ( DistUniformGrid<C>& gridB, const Matrix<C>& B ) const;

    void InternalSolveWithGMRES
    ( DistUniformGrid<C>& gridB, Matrix<C>& B, int m, Real relTol, 
      bool viewIteratees ) const;

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

    // Information related to the decomposition of the domain into panels
    // ==================================================================

    // Information about the bottom (and full inner) panels
    int bottomDepth_;        // including the original PML
    int localBottomSize_;    // local number of d.o.f. in bottom panel
    int localFullInnerSize_; // local number of d.o.f. in each full panel
    int innerDepth_;         // total physical inner depth
    int numFullInnerPanels_; // number of full inner panels we need
    int numPanels_;          // total number of panels

    // Information about the leftover inner panel
    bool haveLeftover_;
    int leftoverInnerDepth_;
    int localLeftoverInnerSize_;

    // Information about the top panel
    int topOrigDepth_;
    int localTopSize_;

    // Analyses of each class of panels
    cliq::DistSymmInfo bottomInfo_, leftoverInnerInfo_, topInfo_;

    // Factorizations of each panel
    cliq::DistSymmFrontTree<C> bottomFact_;
    std::vector<cliq::DistSymmFrontTree<C>*> fullInnerFacts_;
    cliq::DistSymmFrontTree<C> leftoverInnerFact_;
    cliq::DistSymmFrontTree<C> topFact_;

    // Information related to the global sparse matrix
    // ===============================================

    // Sparse matrix storage
    int localSize_;
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

    // General helper routines 
    // =======================

    C s1Inv( int x ) const;
    C s2Inv( int y ) const;
    C s3Inv( int v ) const;
    C s3InvArtificial( int v, int vOffset ) const;

    int NumLocalNodes( int commRank, int commSize ) const;
    static void NumLocalNodesRecursion
    ( int xSize, int ySize, int cutoff, 
      int teamRank, int teamSize, int& numLocal );

    static int DistributedDepth( int commRank, int commSize );
    static void DistributedDepthRecursion
    ( int commRank, int commSize, int& distDepth );

    // Reordering-related routines
    // ===========================

    int PanelPadding( int panel ) const;
    int PanelDepth( int panel ) const;
    int WhichPanel( int v ) const;
    int PanelV( int panel ) const;
    int LocalV( int v ) const;

    int LocalPanelOffset( int panel ) const;
    int LocalPanelSize( int panel ) const;

    int LocalPanelSize
    ( int vSize, int vPadding, int commRank, int commSize ) const;
    static void LocalPanelSizeRecursion
    ( int xSize, int ySize, int vSize, int vPadding, int cutoff, 
      int teamRank, int teamSize, int& localSize );

    int OwningProcess( int naturalIndex, int commSize ) const;
    int OwningProcess( int x, int y, int v, int commSize ) const;
    static void OwningProcessRecursion
    ( int x, int y, int vLocal, int xSize, int ySize, 
      int teamSize, int& process );

    int ReorderedIndex( int x, int y, int v ) const;
    int ReorderedIndex( int x, int y, int vLocal, int vSize ) const;
    static int ReorderedIndexRecursion
    ( int x, int y, int vLocal, int xSize, int ySize, int vSize,
      int depthTilSerial, int cutoff, int offset );

    int LocalReorderedIndex( int x, int y, int v ) const;
    int LocalReorderedIndex( int x, int y, int vLocal, int vSize ) const;
    static int LocalReorderedIndexRecursion
    ( int x, int y, int vLocal, int xSize, int ySize, int vSize, int cutoff, 
      int offset, int commRank, int commSize );

    void LocalReordering( std::map<int,int>& reordering, int vSize ) const;
    static void LocalReorderingRecursion
    ( std::map<int,int>& reordering, 
      int offset, int xOffset, int yOffset,
      int xSize, int ySize, int vSize, int nx, int ny,
      int cutoff, int commRank, int commSize );

    // Global sparse helper routines
    // =============================

    void GetGlobalVelocity
    ( const DistUniformGrid<Real>& velocity,
      std::vector<Real>& myGlobalVelocity, std::vector<int>& offsets ) const;

    void FormGlobalRow( Real alpha, int x, int y, int v, int row );

    // For building localToNaturalMap_, which takes our local index in the 
    // global sparse matrix and returns the original 'natural' index.
    void MapLocalPanelIndices( int commRank, int commSize, int panel );
    static void MapLocalPanelIndicesRecursion
    ( int nx, int ny, int nz, int xSize, int ySize, int vSize, int vPadding,
      int xOffset, int yOffset, int vOffset, int cutoff,
      int teamRank, int teamSize,
      std::vector<int>& localToNaturalMap, std::vector<int>& localRowOffsets,
      int& localOffset );

    void MapLocalConnectionIndices
    ( int commRank, int commSize, 
      std::vector<int>& localConnections, int panel ) const;
    static void MapLocalConnectionIndicesRecursion
    ( int nx, int ny, int nz, int xSize, int ySize, int vSize, int vPadding,
      int xOffset, int yOffset, int vOffset, int cutoff,
      int teamRank, int teamSize,
      std::vector<int>& localConnections, int& localOffset );

    // Helpers for the PML-padded sparse-direct portion
    // ================================================

          cliq::DistSymmFrontTree<C>& PanelFactorization( int panel );
    const cliq::DistSymmFrontTree<C>& PanelFactorization( int panel ) const;

    cliq::DistSymmInfo& PanelAnalysis( int panel );
    const cliq::DistSymmInfo& PanelAnalysis( int panel ) const;

    void GetPanelVelocity
    ( int vOffset, int vSize, 
      const cliq::DistSymmInfo& info,
      const DistUniformGrid<Real>& velocity,
      std::vector<Real>& myPanelVelocity,
      std::vector<int>& offsets,
      std::map<int,int>& panelNestedToNatural, 
      std::map<int,int>& panelNaturalToNested ) const;

    void FillPanelFronts
    ( int vOffset, int vSize,
      const cliq::DistSymmInfo& info,
            cliq::DistSymmFrontTree<C>& fact,
      const DistUniformGrid<Real>& velocity,
      const std::vector<Real>& myPanelVelocity,
            std::vector<int>& offsets,
            std::map<int,int>& panelNestedToNatural,
            std::map<int,int>& panelNaturalToNested ) const;

    void FormLowerColumnOfNode
    ( Real alpha, int x, int y, int v, int vOffset, int vSize,
      int offset, int size, int j, 
      const std::vector<int>& origLowerStruct, 
      const std::vector<int>& origLowerRelIndices,
      std::map<int,int>& panelNaturalToNested, 
      std::vector<int>& frontIndices, std::vector<C>& values ) const;

    void FillPanelElimTree( int vSize, cliq::DistSymmElimTree& eTree ) const;
    void FillPanelDistElimTree
    ( int vSize, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
      cliq::DistSymmElimTree& eTree ) const;
    void FillPanelLocalElimTree
    ( int vSize, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
      cliq::DistSymmElimTree& eTree ) const;
};

} // namespace psp

#include "./dist_helmholtz/main.hpp"
#include "./dist_helmholtz/initialize_finalize.hpp"
#include "./dist_helmholtz/solve.hpp"

#endif // ifndef PSP_DIST_HELMHOLTZ_HPP
