/*
   Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and 
   The University of Texas at Austin
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
namespace psp {

template<typename R>
DistHelmholtz<R>::DistHelmholtz
( const Discretization<R>& disc, mpi::Comm comm,
  R damping, int numPlanesPerPanel, int cutoff )
: comm_(comm), disc_(disc),
  hx_(disc.wx/(disc.nx+1)),
  hy_(disc.wy/(disc.ny+1)),
  hz_(disc.wz/(disc.nz+1)),
  damping_(damping),
  numPlanesPerPanel_(numPlanesPerPanel), 
  nestedCutoff_(cutoff),
  initialized_(false)
{
    // Pull out some information about our communicator
    const int commRank = mpi::CommRank( comm );
    const int commSize = mpi::CommSize( comm );
    distDepth_ = DistributedDepth( commRank, commSize );

    // Decide if the domain is sufficiently deep to warrant sweeping
    const int nx = disc.nx;
    const int ny = disc.ny;
    const int nz = disc.nz;
    const int bz = disc.bz;
    const bool topHasPML = (disc.topBC == PML);
    bottomDepth_ = bz+numPlanesPerPanel_;
    topOrigDepth_ = (topHasPML ? bz+numPlanesPerPanel_ : numPlanesPerPanel_ );
    if( nz <= bottomDepth_+topOrigDepth_ )
        throw std::logic_error
        ("The domain is very shallow. Please run a sparse-direct factorization "
         "instead.");

    // Compute the depths of each interior panel class and the number of 
    // full inner panels.
    //
    //    -----------   sweep dir
    //   | Top       |     /\ 
    //   | Leftover? |     ||
    //       ...           ||
    //   | Inner     |     ||
    //   | Bottom    |     ||
    //    -----------
    innerDepth_ = nz-(bottomDepth_+topOrigDepth_);
    leftoverInnerDepth_ = innerDepth_ % numPlanesPerPanel_;
    haveLeftover_ = ( leftoverInnerDepth_ != 0 );
    numFullInnerPanels_ = innerDepth_ / numPlanesPerPanel_;
    numPanels_ = 1 + numFullInnerPanels_ + haveLeftover_ + 1;

#ifndef RELEASE
    if( commRank == 0 )
    {
        const int bx = disc.bx;
        const int by = disc.by;
        std::cout << "nx=" << nx << ", ny=" << ny << ", nz=" << nz << "\n"
                  << "bx=" << bx << ", by=" << by << ", bz=" << bz << "\n"
                  << "# of planes/panel = " << numPlanesPerPanel_ << "\n"
                  << "\n"
                  << "bottom depth          = " << bottomDepth_ << "\n"
                  << "top orig depth        = " << topOrigDepth_ << "\n"
                  << "inner depth           = " << innerDepth_ << "\n"
                  << "leftover inner depth  = " << leftoverInnerDepth_ << "\n"
                  << "\n"
                  << "# of full inner panels = " << numFullInnerPanels_ << "\n"
                  << "# of panels            = " << numPanels_ << "\n"
                  << std::endl;
    }
#endif

    //
    // Set up the symbolic description of the global sparse matrix
    //

    // Compute the number of rows we own of the sparse distributed matrix
    localBottomSize_ = 
        LocalPanelSize( bottomDepth_, 0, commRank, commSize );
    localFullInnerSize_ = 
        LocalPanelSize( numPlanesPerPanel_, bz, commRank, commSize );
    localLeftoverInnerSize_ = 
        LocalPanelSize( leftoverInnerDepth_, bz, commRank, commSize );
    localTopSize_ = LocalPanelSize( topOrigDepth_, bz, commRank, commSize );
    localSize_ = localBottomSize_ +
                   numFullInnerPanels_*localFullInnerSize_ +
                   localLeftoverInnerSize_ + 
                   localTopSize_;

    // Compute the natural indices of the rows of the global sparse matrix 
    // that are owned by our process. Also, set up the offsets for the 
    // (soon-to-be-computed) packed storage of the connectivity of our local 
    // rows.
    localToNaturalMap_.resize( localSize_ );
    localRowOffsets_.resize( localSize_+1 );
    localRowOffsets_[0] = 0;
    for( int panel=0; panel<numPanels_; ++panel )
        MapLocalPanelIndices( commRank, commSize, panel );

    // Fill in the natural indices of the connections to our local rows of the
    // global sparse matrix.
    std::vector<int> localConnections( localRowOffsets_.back() );
    for( int panel=0; panel<numPanels_; ++panel )
        MapLocalConnectionIndices
        ( commRank, commSize, localConnections, panel );

    // Count the number of indices that we need to recv from each process 
    // during the global sparse matrix multiplication.
    globalRecvCounts_.resize( commSize, 0 );
    owningProcesses_.resize( localRowOffsets_.back() );
    for( int iLocal=0; iLocal<localSize_; ++iLocal )
    {
        // Handle the diagonal value
        const int rowOffset = localRowOffsets_[iLocal];
        owningProcesses_[rowOffset] = commRank;

        // Handle the off-diagonal values
        const int rowSize = localRowOffsets_[iLocal+1]-rowOffset;
        for( int jLocal=1; jLocal<rowSize; ++jLocal )
        {
            const int naturalCol = localConnections[rowOffset+jLocal];
            const int proc = OwningProcess( naturalCol, commSize );
            owningProcesses_[rowOffset+jLocal] = proc;
            ++globalRecvCounts_[proc];
        }
    }
    globalSendCounts_.resize( commSize );
    mpi::AllToAll
    ( &globalRecvCounts_[0], 1,
      &globalSendCounts_[0], 1, comm );

    // Compute the send and recv offsets and total sizes
    int totalSendCount=0, totalRecvCount=0;
    globalSendDispls_.resize( commSize );
    globalRecvDispls_.resize( commSize );
    for( int proc=0; proc<commSize; ++proc )
    {
        globalSendDispls_[proc] = totalSendCount;
        globalRecvDispls_[proc] = totalRecvCount;
        totalSendCount += globalSendCounts_[proc];
        totalRecvCount += globalRecvCounts_[proc];
    }

    // Pack and send the list of indices that we will need from each process
    std::vector<int> offsets = globalRecvDispls_;
    std::vector<int> recvIndices( totalRecvCount );
    for( int iLocal=0; iLocal<localSize_; ++iLocal )
    {
        const int rowOffset = localRowOffsets_[iLocal];
        const int rowSize = localRowOffsets_[iLocal+1]-rowOffset;
        // skip the diagonal value...
        for( int jLocal=1; jLocal<rowSize; ++jLocal )
        {
            const int naturalCol = localConnections[rowOffset+jLocal];
            const int proc = OwningProcess( naturalCol, commSize );
            recvIndices[offsets[proc]++] = naturalCol;
        }
    }
    offsets.clear();
    globalSendIndices_.resize( totalSendCount );
    mpi::AllToAll
    ( &recvIndices[0],        &globalRecvCounts_[0], &globalRecvDispls_[0],
      &globalSendIndices_[0], &globalSendCounts_[0], &globalSendDispls_[0], 
      comm );
    recvIndices.clear();

    // Invert the local to natural map
    std::map<int,int> naturalToLocalMap;
    for( int iLocal=0; iLocal<localSize_; ++iLocal )
        naturalToLocalMap[localToNaturalMap_[iLocal]] = iLocal;

    // Convert the natural indices to their local indices in place
    for( int i=0; i<totalSendCount; ++i )
        globalSendIndices_[i] = naturalToLocalMap[globalSendIndices_[i]];

    // Count the number of indices that we need to recv from each process 
    // during the subdiagonal and superdiagonal block sparse matrix 
    // multiplications.
    //
    // TODO: Simplify by splitting into three different classes of panels.
    std::vector<int> subdiagRecvCountsPerm( (numPanels_-1)*commSize, 0 );
    std::vector<int> supdiagRecvCountsPerm( (numPanels_-1)*commSize, 0 );
    for( int iLocal=0; iLocal<localSize_; ++iLocal )
    {
        // Get our natural coordinates
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int x = naturalIndex % nx;
        const int y = (naturalIndex/nx) % ny;
        const int z = naturalIndex/(nx*ny);
        const int v = (nz-1) - z;

        // If there is a v connection which spans panels, count it here.
        const int localV = LocalV( v );
        const int panel = WhichPanel( v );
        const int panelDepth = PanelDepth( panel );
        const int panelPadding = PanelPadding( panel );
        if( localV == panelPadding && panel != 0 )
        {
            // Handle connection to previous panel, which is relevant to the
            // computation of A_{i+1,i} B_i
            const int proc = OwningProcess( x, y, v-1, commSize );
            ++subdiagRecvCountsPerm[proc*(numPanels_-1)+(panel-1)];
        }
        if( localV == panelPadding+panelDepth-1 && panel != numPanels_-1 )
        {
            // Handle connection to next panel, which is relevant to the 
            // computation of A_{i,i+1} B_{i+1}
            const int proc = OwningProcess( x, y, v+1, commSize );
            ++supdiagRecvCountsPerm[proc*(numPanels_-1)+panel];
        }
    }
    std::vector<int> subdiagSendCountsPerm( (numPanels_-1)*commSize );
    std::vector<int> supdiagSendCountsPerm( (numPanels_-1)*commSize );
    mpi::AllToAll
    ( &subdiagRecvCountsPerm[0], numPanels_-1,
      &subdiagSendCountsPerm[0], numPanels_-1, comm );
    mpi::AllToAll
    ( &supdiagRecvCountsPerm[0], numPanels_-1,
      &supdiagSendCountsPerm[0], numPanels_-1, comm );

    // Reorganize the send and recv counts into an appropriately ordered buffer
    // for per-panel communication
    subdiagSendCounts_.resize( (numPanels_-1)*commSize );
    subdiagRecvCounts_.resize( (numPanels_-1)*commSize );
    supdiagSendCounts_.resize( (numPanels_-1)*commSize );
    supdiagRecvCounts_.resize( (numPanels_-1)*commSize );
    for( int proc=0; proc<commSize; ++proc )
    {
        for( int i=0; i<numPanels_-1; ++i )
        {
            const int oldIndex = proc*(numPanels_-1) + i;
            const int newIndex = i*commSize + proc;
            subdiagSendCounts_[newIndex] = subdiagSendCountsPerm[oldIndex];
            subdiagRecvCounts_[newIndex] = subdiagRecvCountsPerm[oldIndex];
            supdiagSendCounts_[newIndex] = supdiagSendCountsPerm[oldIndex];
            supdiagRecvCounts_[newIndex] = supdiagRecvCountsPerm[oldIndex];
        }
    }
    subdiagSendCountsPerm.clear();
    subdiagRecvCountsPerm.clear();
    supdiagSendCountsPerm.clear();
    supdiagRecvCountsPerm.clear();

    // Compute the send and recv offsets and total sizes
    int subdiagTotalSendCount=0, subdiagTotalRecvCount=0,
        supdiagTotalSendCount=0, supdiagTotalRecvCount=0;
    subdiagSendDispls_.resize( (numPanels_-1)*commSize );
    subdiagRecvDispls_.resize( (numPanels_-1)*commSize );
    supdiagSendDispls_.resize( (numPanels_-1)*commSize );
    supdiagRecvDispls_.resize( (numPanels_-1)*commSize );
    subdiagPanelSendCounts_.resize( numPanels_-1, 0 );
    supdiagPanelSendCounts_.resize( numPanels_-1, 0 );
    subdiagPanelRecvCounts_.resize( numPanels_-1, 0 );
    supdiagPanelRecvCounts_.resize( numPanels_-1, 0 );
    subdiagPanelSendDispls_.resize( numPanels_-1 );
    supdiagPanelSendDispls_.resize( numPanels_-1 );
    subdiagPanelRecvDispls_.resize( numPanels_-1 );
    supdiagPanelRecvDispls_.resize( numPanels_-1 );
    for( int i=0; i<numPanels_-1; ++i )
    {
        for( int proc=0; proc<commSize; ++proc )
        {
            const int index = i*commSize + proc;
            subdiagSendDispls_[index] = subdiagPanelSendCounts_[i];
            supdiagSendDispls_[index] = supdiagPanelSendCounts_[i];
            subdiagRecvDispls_[index] = subdiagPanelRecvCounts_[i];
            supdiagRecvDispls_[index] = supdiagPanelRecvCounts_[i];
            subdiagPanelSendCounts_[i] += subdiagSendCounts_[index];
            supdiagPanelSendCounts_[i] += supdiagSendCounts_[index];
            subdiagPanelRecvCounts_[i] += subdiagRecvCounts_[index];
            supdiagPanelRecvCounts_[i] += supdiagRecvCounts_[index];
        }
        subdiagPanelSendDispls_[i] = subdiagTotalSendCount;
        supdiagPanelSendDispls_[i] = supdiagTotalSendCount;
        subdiagPanelRecvDispls_[i] = subdiagTotalRecvCount;
        supdiagPanelRecvDispls_[i] = supdiagTotalRecvCount;
        subdiagTotalSendCount += subdiagPanelSendCounts_[i];
        supdiagTotalSendCount += supdiagPanelSendCounts_[i];
        subdiagTotalRecvCount += subdiagPanelRecvCounts_[i];
        supdiagTotalRecvCount += supdiagPanelRecvCounts_[i];
    }

    // Pack and send the lists of indices that we will need from each 
    // process.
    std::vector<int> subdiagOffsets( (numPanels_-1)*commSize ),
                     supdiagOffsets( (numPanels_-1)*commSize );
    for( int i=0; i<numPanels_-1; ++i )
    {
        const int subdiagOffset = subdiagPanelRecvDispls_[i];
        const int supdiagOffset = supdiagPanelRecvDispls_[i];
        for( int proc=0; proc<commSize; ++proc ) 
        {
            const int index = i*commSize + proc;
            subdiagOffsets[index] = subdiagRecvDispls_[index]+subdiagOffset;
            supdiagOffsets[index] = supdiagRecvDispls_[index]+supdiagOffset;
        }
    }
    std::vector<int> subdiagRecvIndices( subdiagTotalRecvCount ),
                     supdiagRecvIndices( supdiagTotalRecvCount );
    subdiagRecvLocalIndices_.resize( subdiagTotalRecvCount );
    supdiagRecvLocalIndices_.resize( supdiagTotalRecvCount );
    subdiagRecvLocalRows_.resize( subdiagTotalRecvCount );
    supdiagRecvLocalRows_.resize( supdiagTotalRecvCount );
    for( int iLocal=0; iLocal<localSize_; ++iLocal )
    {
        // Get our natural coordinates
        const int rowOffset = localRowOffsets_[iLocal];
        const int rowSize = localRowOffsets_[iLocal+1] - rowOffset;
        const int naturalRow = localToNaturalMap_[iLocal];
        const int x = naturalRow % nx;
        const int y = (naturalRow/nx) % ny;
        const int z = naturalRow/(nx*ny);
        const int v = (nz-1) - z;

        // If there is a v connection which spans panels, count it here.
        const int localV = LocalV( v );
        const int panel = WhichPanel( v );
        const int panelDepth = PanelDepth( panel );
        const int panelPadding = PanelPadding( panel );
        if( localV == panelPadding && panel != 0 )
        {
            // Handle connection to previous panel, which is relevant to the
            // computation of A_{i+1,i} B_i

            // Search for the appropriate index in this row
            int localIndex = -1;
            const int naturalCol = x + y*nx + (z+1)*nx*ny;
            for( int jLocal=1; jLocal<rowSize; ++jLocal )
            {
                if( localConnections[rowOffset+jLocal] == naturalCol )
                {
                    localIndex = rowOffset+jLocal;
                    break;
                }
            }
#ifndef RELEASE
            if( localIndex == -1 )
                throw std::logic_error("Did not find subdiag connection");
#endif

            const int proc = OwningProcess( x, y, v-1, commSize );
            const int index = (panel-1)*commSize + proc;
            const int offset = subdiagOffsets[index];
            subdiagRecvIndices[offset] = naturalCol;
            subdiagRecvLocalIndices_[offset] = localIndex;
            subdiagRecvLocalRows_[offset] = iLocal;

            ++subdiagOffsets[index];
        }
        if( localV == panelPadding+panelDepth-1 && panel != numPanels_-1 )
        {
            // Handle connection to next panel, which is relevant to the 
            // computation of A_{i,i+1} B_{i+1}

            // Search for the appropriate index in this row
            int localIndex = -1;
            const int naturalCol = x + y*nx + (z-1)*nx*ny;
            for( int jLocal=1; jLocal<rowSize; ++jLocal )
            {
                if( localConnections[rowOffset+jLocal] == naturalCol )
                {
                    localIndex = rowOffset+jLocal;
                    break;
                }
            }
#ifndef RELEASE
            if( localIndex == -1 )
                throw std::logic_error("Did not find supdiag connection");
#endif

            const int proc = OwningProcess( x, y, v+1, commSize );
            const int index = panel*commSize + proc;
            const int offset = supdiagOffsets[index];
            supdiagRecvIndices[offset] = naturalCol;
            supdiagRecvLocalIndices_[offset] = localIndex;
            supdiagRecvLocalRows_[offset] = iLocal;

            ++supdiagOffsets[index];
        }
    }
    subdiagOffsets.clear();
    supdiagOffsets.clear();
    subdiagSendIndices_.resize( subdiagTotalSendCount );
    supdiagSendIndices_.resize( supdiagTotalSendCount );
    // TODO: Think about reducing to a single AllToAll?
    for( int i=0; i<numPanels_-1; ++i )
    {
        mpi::AllToAll
        ( &subdiagRecvIndices[subdiagPanelRecvDispls_[i]], 
          &subdiagRecvCounts_[i*commSize],
          &subdiagRecvDispls_[i*commSize],
          &subdiagSendIndices_[subdiagPanelSendDispls_[i]],
          &subdiagSendCounts_[i*commSize], 
          &subdiagSendDispls_[i*commSize],
          comm );
        mpi::AllToAll
        ( &supdiagRecvIndices[supdiagPanelRecvDispls_[i]], 
          &supdiagRecvCounts_[i*commSize], 
          &supdiagRecvDispls_[i*commSize],
          &supdiagSendIndices_[supdiagPanelSendDispls_[i]],
          &supdiagSendCounts_[i*commSize], 
          &supdiagSendDispls_[i*commSize],
          comm );
    }
    subdiagRecvIndices.clear();
    supdiagRecvIndices.clear();

    // Convert the natural indices to their local indices in place
    for( int i=0; i<subdiagTotalSendCount; ++i )
        subdiagSendIndices_[i] = naturalToLocalMap[subdiagSendIndices_[i]];
    for( int i=0; i<supdiagTotalSendCount; ++i )
        supdiagSendIndices_[i] = naturalToLocalMap[supdiagSendIndices_[i]];

    //
    // Form the symbolic factorizations of the three prototypical panels
    //

    // Fill the original structures (in the nested-dissection ordering)
    cliq::DistSymmElimTree bottomETree, leftoverInnerETree, topETree;
    FillPanelElimTree( bottomDepth_, bottomETree );
    if( haveLeftover_ )
        FillPanelElimTree( leftoverInnerDepth_+bz, leftoverInnerETree );
    FillPanelElimTree( topOrigDepth_+bz, topETree );

    // Perform the parallel analyses
    cliq::SymmetricAnalysis( bottomETree, bottomInfo_, true ); 
    if( haveLeftover_ )
        cliq::SymmetricAnalysis( leftoverInnerETree, leftoverInnerInfo_, true );
    cliq::SymmetricAnalysis( topETree, topInfo_, true );
}

template<typename R>
inline
DistHelmholtz<R>::~DistHelmholtz()
{ }

template<typename R>
int
DistHelmholtz<R>::LocalPanelSize
( int vSize, int vPadding, int commRank, int commSize ) const
{
    int localSize = 0;
    LocalPanelSizeRecursion
    ( disc_.nx, disc_.ny, vSize, vPadding, nestedCutoff_, 
      commRank, commSize, localSize );
    return localSize;
}

template<typename R>
void
DistHelmholtz<R>::LocalPanelSizeRecursion
( int xSize, int ySize, int vSize, int vPadding, int cutoff, 
  int teamRank, int teamSize, int& localSize ) 
{
    if( teamSize == 1 && xSize*ySize <= cutoff )
    {
        // Add the leaf
        localSize += xSize*ySize*vSize;
    }
    else if( xSize >= ySize )
    {
        //
        // Cut the x dimension
        //

        // Add our local portion of the partition
        const int alignment = (ySize*vPadding) % teamSize;
        const int colShift = Shift<int>( teamRank, alignment, teamSize );
        localSize += Length<int>( ySize*vSize, colShift, teamSize );

        // Add the left and/or right sides
        const int xLeftSize = (xSize-1) / 2;
        if( teamSize == 1 )
        {
            // Add the left side
            LocalPanelSizeRecursion
            ( xLeftSize, ySize, vSize, vPadding, cutoff, 0, 1, localSize );
            // Add the right side
            LocalPanelSizeRecursion
            ( xSize-(xLeftSize+1), ySize, vSize, vPadding, cutoff, 0, 1, 
              localSize );
        }
        else
        {
            const int leftTeamSize = teamSize/2;
            const int rightTeamSize = teamSize - leftTeamSize;

            const bool onLeft = ( teamRank < leftTeamSize );
            const int newTeamRank = 
                ( onLeft ? teamRank : teamRank-leftTeamSize );

            if( onLeft )
            {
                // Add the left side
                LocalPanelSizeRecursion
                ( xLeftSize, ySize, vSize, vPadding, cutoff,
                  newTeamRank, leftTeamSize, localSize );
            }
            else
            {
                // Add the right side
                LocalPanelSizeRecursion
                ( xSize-(xLeftSize+1), ySize, vSize, vPadding, cutoff,
                  newTeamRank, rightTeamSize, localSize );
            }
        }
    }
    else
    {
        //
        // Cut the y dimension 
        //

        // Add our local portion of the partition
        const int alignment = (xSize*vPadding) % teamSize;
        const int colShift = Shift<int>( teamRank, alignment, teamSize );
        localSize += Length<int>( xSize*vSize, colShift, teamSize );

        // Add the left and/or right sides
        const int yLeftSize = (ySize-1) / 2;
        if( teamSize == 1 )
        {
            // Add the left side
            LocalPanelSizeRecursion
            ( xSize, yLeftSize, vSize, vPadding, cutoff, 0, 1, localSize );
            // Add the right side
            LocalPanelSizeRecursion
            ( xSize, ySize-(yLeftSize+1), vSize, vPadding, cutoff, 0, 1, 
              localSize );
        }
        else
        {
            const int leftTeamSize = teamSize/2;
            const int rightTeamSize = teamSize - leftTeamSize;

            const bool onLeft = ( teamRank < leftTeamSize );
            const int newTeamRank = 
                ( onLeft ? teamRank : teamRank-leftTeamSize );

            if( onLeft )
            {
                // Add the left side
                LocalPanelSizeRecursion
                ( xSize, yLeftSize, vSize, vPadding, cutoff,
                  newTeamRank, leftTeamSize, localSize );
            }
            else
            {
                // Add the right side
                LocalPanelSizeRecursion
                ( xSize, ySize-(yLeftSize+1), vSize, vPadding, cutoff, 
                  newTeamRank, rightTeamSize, localSize );
            }
        }
    }
}

template<typename R>
int
DistHelmholtz<R>::NumLocalNodes( int commRank, int commSize ) const
{
    int numLocalNodes = 0;
    NumLocalNodesRecursion
    ( disc_.nx, disc_.ny, nestedCutoff_, commRank, commSize, 
      numLocalNodes );
    return numLocalNodes;
}

template<typename R>
void
DistHelmholtz<R>::NumLocalNodesRecursion
( int xSize, int ySize, int cutoff, int teamRank, int teamSize, int& numLocal )
{
    if( teamSize == 1 && xSize*ySize <= cutoff )
    {
        ++numLocal;
    }
    else if( xSize >= ySize )
    {
        //
        // Cut the x dimension
        //
        const int xLeftSize = (xSize-1) / 2;

        // Add our local portion of the partition
        if( teamSize == 1 )
        {
            // Add the separator
            ++numLocal;

            // Add the left side
            NumLocalNodesRecursion
            ( xLeftSize, ySize, cutoff, 0, 1, numLocal );

            // Add the right side
            NumLocalNodesRecursion
            ( xSize-(xLeftSize+1), ySize, cutoff, 0, 1, numLocal );
        }
        else
        {
            const int leftTeamSize = teamSize/2;
            const int rightTeamSize = teamSize - leftTeamSize;

            const bool onLeft = ( teamRank < leftTeamSize );
            const int newTeamRank = 
                ( onLeft ? teamRank : teamRank-leftTeamSize );

            if( onLeft )
            {
                // Add the left side
                NumLocalNodesRecursion
                ( xLeftSize, ySize, cutoff, newTeamRank, leftTeamSize, 
                  numLocal );
            }
            else
            {
                // Add the right side
                NumLocalNodesRecursion
                ( xSize-(xLeftSize+1), ySize, cutoff, newTeamRank, 
                  rightTeamSize, numLocal );
            }
        }
    }
    else
    {
        //
        // Cut the y dimension 
        //
        const int yLeftSize = (ySize-1) / 2;

        // Add our local portion of the partition
        if( teamSize == 1 )
        {
            // Add the separator
            ++numLocal;

            // Add the left side
            NumLocalNodesRecursion
            ( xSize, yLeftSize, cutoff, 0, 1, numLocal );
            // Add the right side
            NumLocalNodesRecursion
            ( xSize, ySize-(yLeftSize+1), cutoff, 0, 1, numLocal );
        }
        else
        {
            const int leftTeamSize = teamSize/2;
            const int rightTeamSize = teamSize - leftTeamSize;

            const bool onLeft = ( teamRank < leftTeamSize );
            const int newTeamRank =
                ( onLeft ? teamRank : teamRank-leftTeamSize );

            if( onLeft )
            {
                // Add the left side
                NumLocalNodesRecursion
                ( xSize, yLeftSize, cutoff, newTeamRank, leftTeamSize, 
                  numLocal );
            }
            else
            {
                // Add the right side
                NumLocalNodesRecursion
                ( xSize, ySize-(yLeftSize+1), cutoff, newTeamRank, 
                  rightTeamSize, numLocal );
            }
        }
    }
}

template<typename R>
cliq::DistSymmFrontTree<Complex<R> >&
DistHelmholtz<R>::PanelFactorization( int panel )
{
    if( panel == 0 )
        return bottomFact_;
    else if( panel < 1 + numFullInnerPanels_ )
        return *fullInnerFacts_[panel-1];
    else if( haveLeftover_ && panel == 1 + numFullInnerPanels_ )
        return leftoverInnerFact_;
    else
        return topFact_;
}

template<typename R>
DistCompressedFrontTree<Complex<R> >&
DistHelmholtz<R>::PanelCompressedFactorization( int panel )
{
    if( panel == 0 )
        return bottomCompressedFact_;
    else if( panel < 1 + numFullInnerPanels_ )
        return *fullInnerCompressedFacts_[panel-1];
    else if( haveLeftover_ && panel == 1 + numFullInnerPanels_ )
        return leftoverInnerCompressedFact_;
    else
        return topCompressedFact_;
}

template<typename R>
const cliq::DistSymmFrontTree<Complex<R> >&
DistHelmholtz<R>::PanelFactorization( int panel ) const
{
    if( panel == 0 )
        return bottomFact_;
    else if( panel < 1 + numFullInnerPanels_ )
        return *fullInnerFacts_[panel-1];
    else if( haveLeftover_ && panel == 1 + numFullInnerPanels_ )
        return leftoverInnerFact_;
    else
        return topFact_;
}

template<typename R>
const DistCompressedFrontTree<Complex<R> >&
DistHelmholtz<R>::PanelCompressedFactorization( int panel ) const
{
    if( panel == 0 )
        return bottomCompressedFact_;
    else if( panel < 1 + numFullInnerPanels_ )
        return *fullInnerCompressedFacts_[panel-1];
    else if( haveLeftover_ && panel == 1 + numFullInnerPanels_ )
        return leftoverInnerCompressedFact_;
    else
        return topCompressedFact_;
}

template<typename R>
cliq::DistSymmInfo& DistHelmholtz<R>::PanelAnalysis( int panel )
{
    if( panel < 1 + numFullInnerPanels_ )
        return bottomInfo_;
    else if( haveLeftover_ && panel == 1 + numFullInnerPanels_ )
        return leftoverInnerInfo_;
    else
        return topInfo_;
}

template<typename R>
const cliq::DistSymmInfo&
DistHelmholtz<R>::PanelAnalysis( int panel ) const
{
    if( panel < 1 + numFullInnerPanels_ )
        return bottomInfo_;
    else if( haveLeftover_ && panel == 1 + numFullInnerPanels_ )
        return leftoverInnerInfo_;
    else
        return topInfo_;
}

template<typename R>
int
DistHelmholtz<R>::PanelPadding( int panel ) const
{
    if( panel == 0 )
        return 0;
    else
        return disc_.bz;
}

template<typename R>
int
DistHelmholtz<R>::PanelDepth( int panel ) const
{
    if( panel == 0 )
        return bottomDepth_;
    else if( panel < 1 + numFullInnerPanels_ )
        return numPlanesPerPanel_;
    else if( haveLeftover_ && panel == 1 + numFullInnerPanels_ )
        return leftoverInnerDepth_;
    else
        return topOrigDepth_;
}

template<typename R>
int
DistHelmholtz<R>::WhichPanel( int v ) const
{
    if( v < bottomDepth_ )
        return 0;
    else if( v < bottomDepth_ + innerDepth_ )
        return (v-bottomDepth_)/numPlanesPerPanel_ + 1;
    else
        return numPanels_ - 1;
}

template<typename R>
int 
DistHelmholtz<R>::LocalV( int v ) const
{
    if( v < bottomDepth_ )
        return v;
    else if( v < bottomDepth_ + innerDepth_ )
        return ((v-bottomDepth_) % numPlanesPerPanel_) + disc_.bz;
    else // v in [topDepth+innerDepth,topDepth+innerDepth+bottomOrigDepth)
        return (v - (bottomDepth_+innerDepth_)) + disc_.bz;
}

// Return the lowest v index of the specified panel
template<typename R>
int
DistHelmholtz<R>::PanelV( int panel ) const
{
    if( panel == 0 )
        return 0;
    else if( panel < numFullInnerPanels_+1 )
        return bottomDepth_ + numPlanesPerPanel_*(panel-1);
    else if( haveLeftover_ && panel == numFullInnerPanels_+1 )
        return bottomDepth_ + numPlanesPerPanel_*numFullInnerPanels_;
    else
        return bottomDepth_ + numPlanesPerPanel_*numFullInnerPanels_ + 
               leftoverInnerDepth_;
}

// Return the local offset into the global sparse matrix for the specified 
// panel, numbered from the bottom up
template<typename R>
int 
DistHelmholtz<R>::LocalPanelOffset( int panel ) const
{
    if( panel == 0 )
        return 0;
    else if( panel < numFullInnerPanels_+1 )
        return localBottomSize_ + localFullInnerSize_*(panel-1);
    else if( haveLeftover_ && panel == numFullInnerPanels_+1 )
        return localBottomSize_ + localFullInnerSize_*numFullInnerPanels_;
    else
        return localBottomSize_ + localFullInnerSize_*numFullInnerPanels_ +
               localLeftoverInnerSize_;
}

template<typename R>
int
DistHelmholtz<R>::LocalPanelSize( int panel ) const
{
    if( panel == 0 )
        return localBottomSize_;
    else if( panel < numFullInnerPanels_+1 )
        return localFullInnerSize_;
    else if( haveLeftover_ && panel == numFullInnerPanels_+1 )
        return localLeftoverInnerSize_;
    else 
        return localTopSize_;
}

template<typename R>
void
DistHelmholtz<R>::MapLocalPanelIndices( int commRank, int commSize, int panel ) 
{
    const int vSize = PanelDepth( panel );
    const int vPadding = PanelPadding( panel );
    const int vOffset = PanelV( panel );
    int localOffset = LocalPanelOffset( panel );
    MapLocalPanelIndicesRecursion
    ( disc_.nx, disc_.ny, disc_.nz, disc_.nx, disc_.ny, vSize, 
      vPadding, 0, 0, vOffset, nestedCutoff_, commRank, commSize, 
      localToNaturalMap_, localRowOffsets_, localOffset );
}

template<typename R>
void
DistHelmholtz<R>::MapLocalPanelIndicesRecursion
( int nx, int ny, int nz, int xSize, int ySize, int vSize, int vPadding,
  int xOffset, int yOffset, int vOffset, int cutoff, 
  int teamRank, int teamSize,
  std::vector<int>& localToNaturalMap, std::vector<int>& localRowOffsets,
  int& localOffset )
{
    if( teamSize == 1 && xSize*ySize <= cutoff )
    {
        // Add the leaf
        for( int vDelta=0; vDelta<vSize; ++vDelta )
        {
            const int v = vOffset + vDelta;
            const int z = (nz-1) - v;
            for( int yDelta=0; yDelta<ySize; ++yDelta )
            {
                const int y = yOffset + yDelta;
                for( int xDelta=0; xDelta<xSize; ++xDelta )
                {
                    const int x = xOffset + xDelta;

                    // Map this local entry to the global natural index
                    localToNaturalMap[localOffset] = x + y*nx + z*nx*ny;

                    // Compute the number of connections from this row
                    int numConnections = 1; // always count diagonal
                    if( x > 0 )
                        ++numConnections;
                    if( x+1 < nx ) 
                        ++numConnections;
                    if( y > 0 )
                        ++numConnections;
                    if( y+1 < ny )
                        ++numConnections;
                    if( v > 0 )
                        ++numConnections;
                    if( v+1 < nz )
                        ++numConnections;
                    localRowOffsets[localOffset+1] = 
                        localRowOffsets[localOffset] + numConnections;

                    ++localOffset;
                }
            }
        }
    }
    else if( xSize >= ySize )
    {
        //
        // Cut the x dimension
        //

        // Add the left and/or right sides
        const int xLeftSize = (xSize-1) / 2;
        if( teamSize == 1 )
        {
            // Add the left side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xLeftSize, ySize, vSize, vPadding, 
              xOffset, yOffset, vOffset, cutoff, 0, 1, localToNaturalMap, 
              localRowOffsets, localOffset );
            // Add the right side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize-(xLeftSize+1), ySize, vSize, vPadding,
              xOffset+(xLeftSize+1), yOffset, vOffset, cutoff, 
              0, 1, localToNaturalMap, localRowOffsets, localOffset );
        }
        else
        {
            const int leftTeamSize = teamSize/2;
            const int rightTeamSize = teamSize - leftTeamSize;

            const bool onLeft = ( teamRank < leftTeamSize );
            const int newTeamRank =
                ( onLeft ? teamRank : teamRank-leftTeamSize );

            if( onLeft )
            {
                // Add the left side
                MapLocalPanelIndicesRecursion
                ( nx, ny, nz, xLeftSize, ySize, vSize, vPadding, 
                  xOffset, yOffset, vOffset, cutoff, newTeamRank,
                  leftTeamSize, localToNaturalMap, localRowOffsets, 
                  localOffset );
            }
            else
            {
                // Add the right side
                MapLocalPanelIndicesRecursion
                ( nx, ny, nz, xSize-(xLeftSize+1), ySize, vSize, vPadding,
                  xOffset+(xLeftSize+1), yOffset, vOffset, cutoff,
                  newTeamRank, rightTeamSize, localToNaturalMap, 
                  localRowOffsets, localOffset );
            }
        }
        
        // Add our local portion of the partition
        const int alignment = (ySize*vPadding) % teamSize;
        const int colShift = Shift<int>( teamRank, alignment, teamSize );
        const int localSize = Length<int>( ySize*vSize, colShift, teamSize );
        for( int iLocal=0; iLocal<localSize; ++iLocal )
        {
            const int i = colShift + iLocal*teamSize;
            const int yDelta = i % ySize;
            const int vDelta = i / ySize;
            const int x = xOffset + xLeftSize;
            const int y = yOffset + yDelta;
            const int v = vOffset + vDelta;
            const int z = (nz-1) - v;

            // Map this local entry to the global natrual index
            localToNaturalMap[localOffset] = x + y*nx + z*nx*ny;

            // Compute the number of connections from this row
            int numConnections = 1; // always count diagonal
            if( x > 0 )
                ++numConnections;
            if( x+1 < nx )
                ++numConnections;
            if( y > 0 )
                ++numConnections;
            if( y+1 < ny )
                ++numConnections;
            if( v > 0 )
                ++numConnections;
            if( v+1 < nz )
                ++numConnections;
            localRowOffsets[localOffset+1] = 
                localRowOffsets[localOffset] + numConnections;

            ++localOffset;
        }
    }
    else
    {
        //
        // Cut the y dimension
        //

        // Add the left and/or right sides
        const int yLeftSize = (ySize-1) / 2;
        if( teamSize == 1 )
        {
            // Add the left side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize, yLeftSize, vSize, vPadding, 
              xOffset, yOffset, vOffset, cutoff, 0, 1, localToNaturalMap, 
              localRowOffsets, localOffset );
            // Add the right side
            MapLocalPanelIndicesRecursion
            ( nx, ny, nz, xSize, ySize-(yLeftSize+1), vSize, vPadding,
              xOffset, yOffset+(yLeftSize+1), vOffset, cutoff, 
              0, 1, localToNaturalMap, localRowOffsets, localOffset );
        }
        else
        {
            const int leftTeamSize = teamSize/2;
            const int rightTeamSize = teamSize - leftTeamSize;

            const bool onLeft = ( teamRank < leftTeamSize );
            const int newTeamRank =
                ( onLeft ? teamRank : teamRank-leftTeamSize );

            if( onLeft )
            {
                // Add the left side
                MapLocalPanelIndicesRecursion
                ( nx, ny, nz, xSize, yLeftSize, vSize, vPadding, 
                  xOffset, yOffset, vOffset, cutoff, newTeamRank, 
                  leftTeamSize, localToNaturalMap, localRowOffsets, 
                  localOffset );
            }
            else
            {
                // Add the right side
                MapLocalPanelIndicesRecursion
                ( nx, ny, nz, xSize, ySize-(yLeftSize+1), vSize, vPadding,
                  xOffset, yOffset+(yLeftSize+1), vOffset, cutoff,
                  newTeamRank, rightTeamSize, localToNaturalMap, 
                  localRowOffsets, localOffset );
            }
        }
        
        // Add our local portion of the partition
        const int alignment = (xSize*vPadding) % teamSize;
        const int colShift = Shift<int>( teamRank, alignment, teamSize );
        const int localSize = Length<int>( xSize*vSize, colShift, teamSize );
        for( int iLocal=0; iLocal<localSize; ++iLocal )
        {
            const int i = colShift + iLocal*teamSize;
            const int xDelta = i % xSize;
            const int vDelta = i / xSize;
            const int x = xOffset + xDelta;
            const int y = yOffset + yLeftSize;
            const int v = vOffset + vDelta;
            const int z = (nz-1) - v;

            // Map this local entry to the global natrual index
            localToNaturalMap[localOffset] = x + y*nx + z*nx*ny;

            // Compute the number of connections from this row
            int numConnections = 1; // always count diagonal
            if( x > 0 )
                ++numConnections;
            if( x+1 < nx )
                ++numConnections;
            if( y > 0 )
                ++numConnections;
            if( y+1 < ny )
                ++numConnections;
            if( v > 0 )
                ++numConnections;
            if( v+1 < nz )
                ++numConnections;
            localRowOffsets[localOffset+1] = 
                localRowOffsets[localOffset] + numConnections;

            ++localOffset;
        }
    }
}

template<typename R>
void
DistHelmholtz<R>::MapLocalConnectionIndices
( int commRank, int commSize, 
  std::vector<int>& localConnections, int panel ) const
{
    const int vSize = PanelDepth( panel );
    const int vPadding = PanelPadding( panel );
    const int vOffset = PanelV( panel );
    int panelOffset = LocalPanelOffset( panel );
    int localOffset = localRowOffsets_[panelOffset];
    MapLocalConnectionIndicesRecursion
    ( disc_.nx, disc_.ny, disc_.nz, disc_.nx, disc_.ny, vSize, 
      vPadding, 0, 0, vOffset, nestedCutoff_, commRank, commSize, 
      localConnections, localOffset );
}

template<typename R>
void
DistHelmholtz<R>::MapLocalConnectionIndicesRecursion
( int nx, int ny, int nz, int xSize, int ySize, int vSize, int vPadding,
  int xOffset, int yOffset, int vOffset, int cutoff, 
  int teamRank, int teamSize,
  std::vector<int>& localConnections, int& localOffset )
{
    if( teamSize == 1 && xSize*ySize <= cutoff )
    {
        // Add the leaf
        for( int vDelta=0; vDelta<vSize; ++vDelta )
        {
            const int v = vOffset + vDelta;
            const int z = (nz-1) - v;
            for( int yDelta=0; yDelta<ySize; ++yDelta )
            {
                const int y = yOffset + yDelta;
                for( int xDelta=0; xDelta<xSize; ++xDelta )
                {
                    const int x = xOffset + xDelta;

                    localConnections[localOffset++] = x + y*nx + z*nx*ny;
                    if( x > 0 )
                        localConnections[localOffset++] = (x-1)+y*nx+z*nx*ny;
                    if( x+1 < nx ) 
                        localConnections[localOffset++] = (x+1)+y*nx+z*nx*ny;
                    if( y > 0 )
                        localConnections[localOffset++] = x+(y-1)*nx+z*nx*ny;
                    if( y+1 < ny )
                        localConnections[localOffset++] = x+(y+1)*nx+z*nx*ny;
                    if( v > 0 )
                        localConnections[localOffset++] = x+y*nx+(z+1)*nx*ny;
                    if( v+1 < nz )
                        localConnections[localOffset++] = x+y*nx+(z-1)*nx*ny;
                }
            }
        }
    }
    else if( xSize >= ySize )
    {
        //
        // Cut the x dimension
        //

        // Add the left and/or right sides
        const int xLeftSize = (xSize-1) / 2;
        if( teamSize == 1 )
        {
            // Add the left side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xLeftSize, ySize, vSize, vPadding, 
              xOffset, yOffset, vOffset, cutoff, 0, 1, localConnections, 
              localOffset );
            // Add the right side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize-(xLeftSize+1), ySize, vSize, vPadding,
              xOffset+(xLeftSize+1), yOffset, vOffset, cutoff, 
              0, 1, localConnections, localOffset );
        }
        else
        {
            const int leftTeamSize = teamSize/2;
            const int rightTeamSize = teamSize - leftTeamSize;

            const bool onLeft = ( teamRank < leftTeamSize );
            const int newTeamRank =
                ( onLeft ? teamRank : teamRank-leftTeamSize );

            if( onLeft )
            {
                // Add the left side
                MapLocalConnectionIndicesRecursion
                ( nx, ny, nz, xLeftSize, ySize, vSize, vPadding, 
                  xOffset, yOffset, vOffset, cutoff, newTeamRank, 
                  leftTeamSize, localConnections, localOffset );
            }
            else
            {
                // Add the right side
                MapLocalConnectionIndicesRecursion
                ( nx, ny, nz, xSize-(xLeftSize+1), ySize, vSize, vPadding,
                  xOffset+(xLeftSize+1), yOffset, vOffset, cutoff,
                  newTeamRank, rightTeamSize, localConnections, 
                  localOffset );
            }
        }
        
        // Add our local portion of the partition
        const int alignment = (ySize*vPadding) % teamSize;
        const int colShift = Shift<int>( teamRank, alignment, teamSize );
        const int localSize = Length<int>( ySize*vSize, colShift, teamSize );
        for( int iLocal=0; iLocal<localSize; ++iLocal )
        {
            const int i = colShift + iLocal*teamSize;
            const int yDelta = i % ySize;
            const int vDelta = i / ySize;
            const int x = xOffset + xLeftSize;
            const int y = yOffset + yDelta;
            const int v = vOffset + vDelta;
            const int z = (nz-1) - v;

            localConnections[localOffset++] = x+y*nx+z*nx*ny;
            if( x > 0 )
                localConnections[localOffset++] = (x-1) + y*nx + z*nx*ny;
            if( x+1 < nx )
                localConnections[localOffset++] = (x+1) + y*nx + z*nx*ny;
            if( y > 0 )
                localConnections[localOffset++] = x + (y-1)*nx + z*nx*ny;
            if( y+1 < ny )
                localConnections[localOffset++] = x + (y+1)*nx + z*nx*ny;
            if( v > 0 )
                localConnections[localOffset++] = x + y*nx + (z+1)*nx*ny;
            if( v+1 < nz )
                localConnections[localOffset++] = x + y*nx + (z-1)*nx*ny;
        }
    }
    else
    {
        //
        // Cut the y dimension
        //

        // Add the left and/or right sides
        const int yLeftSize = (ySize-1) / 2;
        if( teamSize == 1 )
        {
            // Add the left side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize, yLeftSize, vSize, vPadding, 
              xOffset, yOffset, vOffset, cutoff, 0, 1, localConnections, 
              localOffset );
            // Add the right side
            MapLocalConnectionIndicesRecursion
            ( nx, ny, nz, xSize, ySize-(yLeftSize+1), vSize, vPadding,
              xOffset, yOffset+(yLeftSize+1), vOffset, cutoff, 
              0, 1, localConnections, localOffset );
        }
        else
        {
            const int leftTeamSize = teamSize/2;
            const int rightTeamSize = teamSize - leftTeamSize;

            const bool onLeft = ( teamRank < leftTeamSize );
            const int newTeamRank =
                ( onLeft ? teamRank : teamRank-leftTeamSize );

            if( onLeft )
            {
                // Add the left side
                MapLocalConnectionIndicesRecursion
                ( nx, ny, nz, xSize, yLeftSize, vSize, vPadding, 
                  xOffset, yOffset, vOffset, cutoff, newTeamRank, 
                  leftTeamSize, localConnections, localOffset );
            }
            else
            {
                // Add the right side
                MapLocalConnectionIndicesRecursion
                ( nx, ny, nz, xSize, ySize-(yLeftSize+1), vSize, vPadding,
                  xOffset, yOffset+(yLeftSize+1), vOffset, cutoff,
                  newTeamRank, rightTeamSize, localConnections, 
                  localOffset );
            }
        }
        
        // Add our local portion of the partition
        const int alignment = (xSize*vPadding) % teamSize;
        const int colShift = Shift<int>( teamRank, alignment, teamSize );
        const int localSize = Length<int>( xSize*vSize, colShift, teamSize );
        for( int iLocal=0; iLocal<localSize; ++iLocal )
        {
            const int i = colShift + iLocal*teamSize;
            const int xDelta = i % xSize;
            const int vDelta = i / xSize;
            const int x = xOffset + xDelta;
            const int y = yOffset + yLeftSize;
            const int v = vOffset + vDelta;
            const int z = (nz-1) - v;

            localConnections[localOffset++] = x + y*nx + z*nx*ny;
            if( x > 0 )
                localConnections[localOffset++] = (x-1) + y*nx + z*nx*ny;
            if( x+1 < nx )
                localConnections[localOffset++] = (x+1) + y*nx + z*nx*ny;
            if( y > 0 )
                localConnections[localOffset++] = x + (y-1)*nx + z*nx*ny;
            if( y+1 < ny )
                localConnections[localOffset++] = x + (y+1)*nx + z*nx*ny;
            if( v > 0 )
                localConnections[localOffset++] = x + y*nx + (z+1)*nx*ny;
            if( v+1 < nz )
                localConnections[localOffset++] = x + y*nx + (z-1)*nx*ny;
        }
    }
}

template<typename R>
int
DistHelmholtz<R>::OwningProcess( int naturalIndex, int commSize ) const
{
    const int nx = disc_.nx;
    const int ny = disc_.ny;
    const int nz = disc_.nz;

    const int x = naturalIndex % nx;
    const int y = (naturalIndex/nx) % ny;
    const int z = naturalIndex/(nx*ny);
    const int v = (nz-1) - z;
    const int vLocal = LocalV( v );

    int proc = 0;
    OwningProcessRecursion( x, y, vLocal, nx, ny, commSize, proc );
    return proc;
}

template<typename R>
int
DistHelmholtz<R>::OwningProcess( int x, int y, int v, int commSize ) const
{
    const int nx = disc_.nx;
    const int ny = disc_.ny;
    const int vLocal = LocalV( v );

    int proc = 0;
    OwningProcessRecursion( x, y, vLocal, nx, ny, commSize, proc );
    return proc;
}

template<typename R>
void
DistHelmholtz<R>::OwningProcessRecursion
( int x, int y, int vLocal, int xSize, int ySize, int teamSize, int& proc )
{
    if( teamSize == 1 )
        return;

    const int leftTeamSize = teamSize/2;
    const int rightTeamSize = teamSize - leftTeamSize;

    if( xSize >= ySize )
    {
        //
        // Cut the x dimension
        //
        const int xLeftSize = (xSize-1) / 2;
        if( x == xLeftSize )
        {
            proc += (y+vLocal*ySize) % teamSize;
        }
        else if( x > xLeftSize )
        { 
            // Continue down the right side
            proc += leftTeamSize;
            OwningProcessRecursion
            ( x-(xLeftSize+1), y, vLocal, xSize-(xLeftSize+1), ySize, 
              rightTeamSize, proc );
        }
        else // x < leftSize
        {
            // Continue down the left side
            OwningProcessRecursion
            ( x, y, vLocal, xLeftSize, ySize, leftTeamSize, proc );
        }
    }
    else
    {
        //
        // Cut the y dimension 
        //
        const int yLeftSize = (ySize-1) / 2;
        if( y == yLeftSize )
        {
            proc += (x+vLocal*xSize) % teamSize;
        }
        else if( y > yLeftSize )
        { 
            // Continue down the right side
            proc += leftTeamSize;
            OwningProcessRecursion
            ( x, y-(yLeftSize+1), vLocal, xSize, ySize-(yLeftSize+1), 
              rightTeamSize, proc );
        }
        else // x < leftSize
        {
            // Continue down the left side
            OwningProcessRecursion
            ( x, y, vLocal, xSize, yLeftSize, leftTeamSize, proc );
        }
    }
}

template<typename R>
int
DistHelmholtz<R>::DistributedDepth( int commRank, int commSize ) 
{
    int distDepth = 0;
    DistributedDepthRecursion( commRank, commSize, distDepth );
    return distDepth;
}

template<typename R>
void
DistHelmholtz<R>::DistributedDepthRecursion
( int commRank, int commSize, int& distDepth )
{
    if( commSize == 1 )
        return;

    ++distDepth;
    const int leftTeamSize = commSize/2;
    const int rightTeamSize = commSize - leftTeamSize;
    if( commRank < leftTeamSize )
        DistributedDepthRecursion
        ( commRank, leftTeamSize, distDepth );
    else
        DistributedDepthRecursion
        ( commRank-leftTeamSize, rightTeamSize, distDepth );
}

template<typename R>
int
DistHelmholtz<R>::ReorderedIndex( int x, int y, int v ) const
{
    const int panel = WhichPanel( v );
    const int padding = PanelPadding( panel );
    const int panelDepth = PanelDepth( panel );
    const int vLocal = LocalV( v ) - padding;

    const int vPanel = PanelV( panel );
    const int panelIndex = ReorderedIndex( x, y, vLocal, panelDepth );
    return panelIndex + disc_.nx*disc_.ny*vPanel;
}

template<typename R>
int
DistHelmholtz<R>::ReorderedIndex( int x, int y, int vLocal, int vSize ) const
{
    int index = 
        ReorderedIndexRecursion
        ( x, y, vLocal, disc_.nx, disc_.ny, vSize, distDepth_, 
          nestedCutoff_, 0 );
    return index;
}

template<typename R>
int
DistHelmholtz<R>::ReorderedIndexRecursion
( int x, int y, int vLocal, int xSize, int ySize, int vSize,
  int depthTilSerial, int cutoff, int offset )
{
    const int nextDepthTilSerial = std::max(depthTilSerial-1,0);
    if( depthTilSerial == 0 && xSize*ySize <= cutoff )
    {
        // We have satisfied the nested dissection constraints
        return offset + (x+y*xSize+vLocal*xSize*ySize);
    }
    else if( xSize >= ySize )
    {
        // Partition the X dimension
        const int middle = (xSize-1)/2;
        if( x < middle )
        {
            return ReorderedIndexRecursion
            ( x, y, vLocal, middle, ySize, vSize, nextDepthTilSerial, cutoff,
              offset );
        }
        else if( x == middle )
        {
            return offset + std::max(xSize-1,0)*ySize*vSize + (y+vLocal*ySize);
        }
        else // x > middle
        {
            return ReorderedIndexRecursion
            ( x-middle-1, y, vLocal, std::max(xSize-middle-1,0), ySize, vSize,
              nextDepthTilSerial, cutoff, offset+middle*ySize*vSize );
        }
    }
    else
    {
        // Partition the Y dimension
        const int middle = (ySize-1)/2;
        if( y < middle )
        {
            return ReorderedIndexRecursion
            ( x, y, vLocal, xSize, middle, vSize, nextDepthTilSerial, cutoff, 
              offset );
        }
        else if( y == middle )
        {
            return offset + xSize*std::max(ySize-1,0)*vSize + (x+vLocal*xSize);
        }
        else // y > middle 
        {
            return ReorderedIndexRecursion
            ( x, y-middle-1, vLocal, xSize, std::max(ySize-middle-1,0), vSize,
              nextDepthTilSerial, cutoff, offset+xSize*middle*vSize );
        }
    }
}

template<typename R>
void
DistHelmholtz<R>::FillPanelElimTree
( int vSize, cliq::DistSymmElimTree& eTree ) const
{
    int nxSub=disc_.nx, nySub=disc_.ny, xOffset=0, yOffset=0;    
    FillPanelDistElimTree( vSize, nxSub, nySub, xOffset, yOffset, eTree );
    FillPanelLocalElimTree( vSize, nxSub, nySub, xOffset, yOffset, eTree );
}

template<typename R>
void
DistHelmholtz<R>::FillPanelDistElimTree
( int vSize, int& nxSub, int& nySub, int& xOffset, int& yOffset,
  cliq::DistSymmElimTree& eTree ) const
{
    const int numDistNodes = distDepth_+1;

    eTree.distNodes.resize( numDistNodes );
    mpi::CommDup( comm_, eTree.distNodes.back().comm );

    const int nx = disc_.nx;
    const int ny = disc_.ny;
    const int cutoff = nestedCutoff_;
    mpi::CommDup( comm_, eTree.distNodes.back().comm );

    // Fill the distributed nodes
    for( int s=numDistNodes-1; s>0; --s )
    {
        cliq::DistSymmNode& node = eTree.distNodes[s];
        cliq::DistSymmNode& childNode = eTree.distNodes[s-1];

        const int nodeCommRank = mpi::CommRank( node.comm );
        const int nodeCommSize = mpi::CommSize( node.comm );
        const int leftTeamSize = nodeCommSize/2;

        const bool onLeft = ( nodeCommRank < leftTeamSize );
        const int childNodeCommRank = 
            ( onLeft ? nodeCommRank : nodeCommRank-leftTeamSize );
        mpi::CommSplit( node.comm, onLeft, childNodeCommRank, childNode.comm );
        childNode.onLeft = onLeft;

        if( nxSub >= nySub )
        {
            // Form the structure of a partition of the X dimension
            const int middle = (nxSub-1)/2;
            node.size = nySub*vSize;
            node.offset = ReorderedIndex( xOffset+middle, yOffset, 0, vSize );

            // Allocate space for the lower structure
            int numJoins = 0;
            if( yOffset > 0 )
                ++numJoins;
            if( yOffset+nySub < ny )
                ++numJoins;
            node.lowerStruct.resize( numJoins*vSize );

            // Fill the (unsorted) lower structure
            int joinOffset = 0;
            if( yOffset > 0 )
                for( int i=0; i<vSize; ++i )
                    node.lowerStruct[joinOffset++] = ReorderedIndex
                    ( xOffset+middle, yOffset-1, i, vSize );
            if( yOffset+nySub < ny )
                for( int i=0; i<vSize; ++i )
                    node.lowerStruct[joinOffset++] = ReorderedIndex
                    ( xOffset+middle, yOffset+nySub, i, vSize );

            // Sort the lower structure
            std::sort( node.lowerStruct.begin(), node.lowerStruct.end() );

            // Pick the new offsets and sizes based upon our rank
            if( onLeft )
            {
                xOffset = xOffset;
                nxSub = middle;
            }
            else
            {
                xOffset = xOffset+middle+1;
                nxSub = std::max(nxSub-middle-1,0);
            }
        }
        else
        {
            // Form the structure of a partition of the Y dimension
            const int middle = (nySub-1)/2;
            node.size = nxSub*vSize;
            node.offset = ReorderedIndex( xOffset, yOffset+middle, 0, vSize );

            // Allocate space for the lower structure
            int numJoins = 0;
            if( xOffset > 0 )
                ++numJoins;
            if( xOffset+nxSub < nx )
                ++numJoins;
            node.lowerStruct.resize( numJoins*vSize );

            // Fill the (unsorted) lower structure
            int joinOffset = 0;
            if( xOffset > 0 )
                for( int i=0; i<vSize; ++i )
                    node.lowerStruct[joinOffset++] = ReorderedIndex
                    ( xOffset-1, yOffset+middle, i, vSize );
            if( xOffset+nxSub < nx )
                for( int i=0; i<vSize; ++i )
                    node.lowerStruct[joinOffset++] = ReorderedIndex
                    ( xOffset+nxSub, yOffset+middle, i, vSize );

            // Sort the lower structure
            std::sort( node.lowerStruct.begin(), node.lowerStruct.end() );

            // Pick the new offsets and sizes based upon our rank
            if( onLeft )
            {
                yOffset = yOffset;
                nySub = middle;
            }
            else
            {
                yOffset = yOffset+middle+1;
                nySub = std::max(nySub-middle-1,0);
            }
        }
    }

    // Fill the bottom node, which is only owned by a single process
    cliq::DistSymmNode& node = eTree.distNodes[0];
    if( nxSub*nySub <= cutoff )
    {
        node.size = nxSub*nySub*vSize;
        node.offset = ReorderedIndex( xOffset, yOffset, 0, vSize );

        // Count, allocate, and fill the lower struct
        int joinSize = 0;
        if( xOffset > 0 )
            joinSize += nySub;
        if( xOffset+nxSub < nx )
            joinSize += nySub;
        if( yOffset > 0 )
            joinSize += nxSub;
        if( yOffset+nySub < ny )
            joinSize += nxSub;
        node.lowerStruct.resize( joinSize*vSize );

        int joinOffset = 0;
        if( xOffset > 0 )
            for( int i=0; i<vSize; ++i )
                for( int j=0; j<nySub; ++j )
                    node.lowerStruct[joinOffset++] = ReorderedIndex
                    ( xOffset-1, yOffset+j, i, vSize );
        if( xOffset+nxSub < nx )
            for( int i=0; i<vSize; ++i )
                for( int j=0; j<nySub; ++j )
                    node.lowerStruct[joinOffset++] = ReorderedIndex
                    ( xOffset+nxSub, yOffset+j, i, vSize );
        if( yOffset > 0 )
            for( int i=0; i<vSize; ++i )
                for( int j=0; j<nxSub; ++j )
                    node.lowerStruct[joinOffset++] = ReorderedIndex
                    ( xOffset+j, yOffset-1, i, vSize );
        if( yOffset+nySub < ny )
            for( int i=0; i<vSize; ++i )
                for( int j=0; j<nxSub; ++j )
                    node.lowerStruct[joinOffset++] = ReorderedIndex
                    ( xOffset+j, yOffset+nySub, i, vSize );

        // Sort the lower structure
        std::sort( node.lowerStruct.begin(), node.lowerStruct.end() );
    }
    else if( nxSub >= nySub )
    {
        // Form the structure of a partition of the X dimension
        const int middle = (nxSub-1)/2;
        node.size = nySub*vSize;
        node.offset = ReorderedIndex( xOffset+middle, yOffset, 0, vSize );

        // Allocate space for the lower structure
        int numJoins = 0;
        if( yOffset > 0 )
            ++numJoins;
        if( yOffset+nySub < ny )
            ++numJoins;
        node.lowerStruct.resize( numJoins*vSize );

        // Fill the (unsorted) lower structure
        int joinOffset = 0;
        if( yOffset > 0 )
            for( int i=0; i<vSize; ++i )
                node.lowerStruct[joinOffset++] = ReorderedIndex
                ( xOffset+middle, yOffset-1, i, vSize );
        if( yOffset+nySub < ny )
            for( int i=0; i<vSize; ++i )
                node.lowerStruct[joinOffset++] = ReorderedIndex
                ( xOffset+middle, yOffset+nySub, i, vSize );

        // Sort the lower structure
        std::sort( node.lowerStruct.begin(), node.lowerStruct.end() );
    }
    else
    {
        // Form the structure of a partition of the Y dimension
        const int middle = (nySub-1)/2;
        node.size = nxSub*vSize;
        node.offset = ReorderedIndex( xOffset, yOffset+middle, 0, vSize );

        // Allocate space for the lower structure
        int numJoins = 0;
        if( xOffset > 0 )
            ++numJoins;
        if( xOffset+nxSub < nx )
            ++numJoins;
        node.lowerStruct.resize( numJoins*vSize );

        // Fill the (unsorted) lower structure
        int joinOffset = 0;
        if( xOffset > 0 )
            for( int i=0; i<vSize; ++i )
                node.lowerStruct[joinOffset++] = ReorderedIndex
                ( xOffset-1, yOffset+middle, i, vSize );
        if( xOffset+nxSub < nx )
            for( int i=0; i<vSize; ++i )
                node.lowerStruct[joinOffset++] = ReorderedIndex
                ( xOffset+nxSub, yOffset+middle, i, vSize );

        // Sort the lower structure
        std::sort( node.lowerStruct.begin(), node.lowerStruct.end() );
    }
}

template<typename R>
void
DistHelmholtz<R>::FillPanelLocalElimTree
( int vSize, int& nxSub, int& nySub, int& xOffset, int& yOffset, 
  cliq::DistSymmElimTree& eTree ) const
{
    const int commRank = mpi::CommRank( comm_ );
    const int commSize = mpi::CommSize( comm_ );
    const int numLocalNodes = NumLocalNodes( commRank, commSize );
    eTree.localNodes.resize( numLocalNodes );

    const int cutoff = nestedCutoff_;
    const int nx = disc_.nx;
    const int ny = disc_.ny;

    // Initialize with the local root's box
    std::stack<Box> boxStack;
    {
        Box box;
        box.parentIndex = -1;
        box.nx = nxSub;
        box.ny = nySub;
        box.xOffset = xOffset;
        box.yOffset = yOffset;
        box.leftChild = false;
        boxStack.push(box);
    }

    // Fill the local tree
    for( int s=numLocalNodes-1; s>=0; --s )
    {
        Box box = boxStack.top();
        boxStack.pop();

        eTree.localNodes[s] = new cliq::SymmNode;
        cliq::SymmNode& node = *eTree.localNodes[s];
        node.parent = box.parentIndex;
        if( node.parent != -1 )
        {
            if( box.leftChild )
                eTree.localNodes[node.parent]->children[0] = s;
            else
                eTree.localNodes[node.parent]->children[1] = s;
        }

        if( box.nx*box.ny <= cutoff )
        {
            node.size = box.nx*box.ny*vSize;
            node.offset = ReorderedIndex( box.xOffset, box.yOffset, 0, vSize );
            node.children.clear();

            // Count, allocate, and fill the lower struct
            int joinSize = 0;
            if( box.xOffset > 0 )
                joinSize += box.ny;
            if( box.xOffset+box.nx < nx )
                joinSize += box.ny;
            if( box.yOffset > 0 )
                joinSize += box.nx;
            if( box.yOffset+box.ny < ny )
                joinSize += box.nx;
            node.lowerStruct.resize( joinSize*vSize );

            int joinOffset = 0;
            if( box.xOffset > 0 )
                for( int i=0; i<vSize; ++i )
                    for( int j=0; j<box.ny; ++j )
                        node.lowerStruct[joinOffset++] = ReorderedIndex
                        ( box.xOffset-1, box.yOffset+j, i, vSize );
            if( box.xOffset+box.nx < nx )
                for( int i=0; i<vSize; ++i )
                    for( int j=0; j<box.ny; ++j )
                        node.lowerStruct[joinOffset++] = ReorderedIndex
                        ( box.xOffset+box.nx, box.yOffset+j, i, vSize );
            if( box.yOffset > 0 )
                for( int i=0; i<vSize; ++i )
                    for( int j=0; j<box.nx; ++j )
                        node.lowerStruct[joinOffset++] = ReorderedIndex
                        ( box.xOffset+j, box.yOffset-1, i, vSize );
            if( box.yOffset+box.ny < ny )
                for( int i=0; i<vSize; ++i )
                    for( int j=0; j<box.nx; ++j )
                        node.lowerStruct[joinOffset++] = ReorderedIndex
                        ( box.xOffset+j, box.yOffset+box.ny, i, vSize );

            // Sort the lower structure
            std::sort( node.lowerStruct.begin(), node.lowerStruct.end() );
        }
        else
        {
            node.children.resize(2);
            if( box.nx >= box.ny )
            {
                // Partition the X dimension (this is the separator)
                const int middle = (box.nx-1)/2;
                node.size = box.ny*vSize;
                node.offset = ReorderedIndex
                    ( box.xOffset+middle, box.yOffset, 0, vSize );

                // Count, allocate, and fill the lower struct
                int numJoins = 0;
                if( box.yOffset > 0 )
                    ++numJoins;
                if( box.yOffset+box.ny < ny )
                    ++numJoins;
                node.lowerStruct.resize( numJoins*vSize );

                int joinOffset = 0;
                if( box.yOffset > 0 )
                    for( int i=0; i<vSize; ++i )
                        node.lowerStruct[joinOffset++] = ReorderedIndex
                        ( box.xOffset+middle, box.yOffset-1, i, vSize );
                if( box.yOffset+box.ny < ny )
                    for( int i=0; i<vSize; ++i )
                        node.lowerStruct[joinOffset++] = ReorderedIndex
                        ( box.xOffset+middle, box.yOffset+box.ny, i, vSize );

                // Sort the lower structure
                std::sort( node.lowerStruct.begin(), node.lowerStruct.end() );

                // Push the left child box onto the stack
                Box leftBox;
                leftBox.parentIndex = s;
                leftBox.nx = middle;
                leftBox.ny = box.ny;
                leftBox.xOffset = box.xOffset;
                leftBox.yOffset = box.yOffset;
                leftBox.leftChild = true;
                boxStack.push( leftBox );

                // Push the right child box onto the stack
                Box rightBox;
                rightBox.parentIndex = s;
                rightBox.nx = std::max(box.nx-middle-1,0);
                rightBox.ny = box.ny;
                rightBox.xOffset = box.xOffset+middle+1;
                rightBox.yOffset = box.yOffset;
                rightBox.leftChild = false;
                boxStack.push( rightBox );
            }
            else
            {
                // Partition the Y dimension (this is the separator)
                const int middle = (box.ny-1)/2;
                node.size = box.nx*vSize;
                node.offset = ReorderedIndex
                    ( box.xOffset, box.yOffset+middle, 0, vSize );

                // Count, allocate, and fill the lower struct
                int numJoins = 0;
                if( box.xOffset > 0 )
                    ++numJoins;
                if( box.xOffset+box.nx < nx )
                    ++numJoins;
                node.lowerStruct.resize( numJoins*vSize );

                int joinOffset = 0;
                if( box.xOffset > 0 )
                    for( int i=0; i<vSize; ++i )
                        node.lowerStruct[joinOffset++] = ReorderedIndex
                        ( box.xOffset-1, box.yOffset+middle, i, vSize );
                if( box.xOffset+box.nx < nx )
                    for( int i=0; i<vSize; ++i )
                        node.lowerStruct[joinOffset++] = ReorderedIndex
                        ( box.xOffset+box.nx, box.yOffset+middle, i, vSize );

                // Sort the lower structure
                std::sort( node.lowerStruct.begin(), node.lowerStruct.end() );

                // Push the left child box onto the stack
                Box leftBox;
                leftBox.parentIndex = s;
                leftBox.nx = box.nx;
                leftBox.ny = middle;
                leftBox.xOffset = box.xOffset;
                leftBox.yOffset = box.yOffset;
                leftBox.leftChild = true;
                boxStack.push( leftBox );

                // Push the right child box onto the stack
                Box rightBox;
                rightBox.parentIndex = s;
                rightBox.nx = box.nx;
                rightBox.ny = std::max(box.ny-middle-1,0);
                rightBox.xOffset = box.xOffset;
                rightBox.yOffset = box.yOffset+middle+1;
                rightBox.leftChild = false;
                boxStack.push( rightBox );
            }
        }
    }
}

} // namespace psp
