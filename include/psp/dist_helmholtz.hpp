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
    std::vector<std::vector<F> > localRows_;

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
        mainSymbolicFact_, misfitSymbolicFact_, lastSymbolicFact_;

    // Sparse-direct numeric factorizations of PML-padded panels
    std::vector<clique::numeric::SymmFrontTree<F>*> numericFacts_;

    // Helper routines
    static void RecursiveReordering
    ( int nx, int xOffset, int xSize, int yOffset, int ySize, 
      int cutoff, int depthTilSerial, int* reordering );
    static void CountLocalHeight
    ( int xSize, int ySize, int zSize, int cutoff, 
      unsigned commRank, unsigned log2CommSize, int& localHeight );
    static void CountLocalSupernodes
    ( int xSize, int ySize, int cutoff, 
      unsigned commRank, unsigned log2CommSize, int& numLocal );
    static void ConvertCoordsToProcess
    ( int x, int y, int zLocal, int xSize, int ySize, unsigned log2CommSize, 
      int& process );

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
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F>
inline 
DistHelmholtz<F>::~DistHelmholtz() 
{
    const int numPanels = numericFacts_.size();
    for( int i=0; i<numPanels; ++i )
        delete numericFacts_[i];
    numericFacts_.clear();
}

template<typename F>
inline int
DistHelmholtz<F>::LocalSize() const
{ return localHeight_; }

} // namespace psp

#endif // PSP_DIST_HELMHOLTZ_HPP
