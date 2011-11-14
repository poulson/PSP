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

template<typename R>
void
psp::DistHelmholtz<R>::Solve( GridData<C>& B, Solver solver ) const
{
    if( !elemental::mpi::CongruentComms( comm_, B.Comm() ) )
        throw std::logic_error("B does not have a congruent comm");

    // Convert B into custom nested-dissection based ordering
    std::vector<C> redistB;
    PullRightHandSides( B, redistB );

    // Solve the systems of equations
    switch( solver )
    {
    case GMRES: SolveWithGMRES( redistB ); break;
    case QMR:   SolveWithQMR( redistB );   break;
    }

    // Restore the solutions back into the GridData form
    PushRightHandSides( B, redistB );
}

template<typename R>
void
psp::DistHelmholtz<R>::PullRightHandSides
( const GridData<C>& B, std::vector<C>& redistB ) const
{
    const int commSize = mpi::CommSize( comm_ );

    // Pack and send the amount of data that we will need to recv
    std::vector<int> recvCounts( commSize, 0 );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = B.OwningProcess( naturalIndex );
        ++recvCounts[proc];
    }
    std::vector<int> sendCounts( commSize );
    mpi::AllToAll
    ( &recvCounts[0], 1,
      &sendCounts[0], 1, comm_ );

    // Compute the send and recv offsets and total sizes
    int totalSendCount=0, totalRecvCount=0;
    std::vector<int> sendDispls( commSize ), recvDispls( commSize );
    for( int proc=0; proc<commSize; ++proc )
    {
        sendDispls[proc] = totalSendCount;
        recvDispls[proc] = totalRecvCount;
        totalSendCount += sendCounts[proc];
        totalRecvCount += recvCounts[proc];
    }

    // Pack and send the indices that we will need to recv from each process.
    std::vector<int> offsets = recvDispls;
    std::vector<int> recvIndices( totalRecvCount );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = B.OwningProcess( naturalIndex );
        recvIndices[++offsets[proc]] = naturalIndex;
    }
    std::vector<int> sendIndices( totalSendCount );
    mpi::AllToAll
    ( &recvIndices[0], &recvCounts[0], &recvDispls[0],
      &sendIndices[0], &sendCounts[0], &sendDispls[0], comm_ );
    recvIndices.clear();

    // Scale the send and recv counts to make up for their being several RHS
    const int numRhs = B.NumScalars();
    for( int proc=0; proc<commSize; ++proc )
    {
        sendCounts[proc] *= numRhs;
        recvCounts[proc] *= numRhs;
        sendDispls[proc] *= numRhs;
        recvDispls[proc] *= numRhs;
    }
    totalSendCount *= numRhs;
    totalRecvCount *= numRhs;

    // Pack and send our right-hand side data.
    std::vector<C> sendB( totalSendCount );
    const C* BBuffer = B.LockedLocalBuffer();
    for( int proc=0; proc<commSize; ++proc )
    {
        C* procB = &sendB[sendDispls[proc]];
        const int* procIndices = &sendIndices[sendDispls[proc]/numRhs];
        const int numLocalIndices = sendCounts[proc]/numRhs;
        for( int s=0; s<numLocalIndices; ++s )
        {
            const int naturalIndex = procIndices[s];
            const int localIndex = B.LocalIndex( naturalIndex );
            for( int k=0; k<numRhs; ++k )
                procB[s*numRhs+k] = BBuffer[localIndex+k];
        }
    }
    sendIndices.clear();
    std::vector<C> recvB( totalRecvCount );
    mpi::AllToAll
    ( &sendB[0], &sendCounts[0], &sendDispls[0], 
      &recvB[0], &recvCounts[0], &recvDispls[0], comm_ );
    sendB.clear();

    // Unpack the received right-hand side data
    offsets = recvDispls;
    redistB.resize( localHeight_*numRhs );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = B.OwningProcess( naturalIndex );
        for( int k=0; k<numRhs; ++k )
            redistB[iLocal+k*localHeight_] = recvB[offsets[proc]+k];
        offsets[proc] += numRhs;
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::PushRightHandSides
( GridData<C>& B, const std::vector<C>& redistB ) const
{
    const int numRhs = B.NumScalars();
    const int commSize = mpi::CommSize( comm_ );

    // Pack and send the amount of data that we will need to send.
    std::vector<int> sendCounts( commSize, 0 );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = B.OwningProcess( naturalIndex );
        ++sendCounts[proc];
    }
    std::vector<int> recvCounts( commSize );
    mpi::AllToAll
    ( &sendCounts[0], 1, 
      &recvCounts[0], 1, comm_ );

    // Compute the send and recv offsets and total sizes
    int totalSendCount=0, totalRecvCount=0;
    std::vector<int> sendDispls( commSize ), recvDispls( commSize );
    for( int proc=0; proc<commSize; ++proc )
    {
        sendDispls[proc] = totalSendCount;
        recvDispls[proc] = totalRecvCount;
        totalSendCount += sendCounts[proc];
        totalRecvCount += recvCounts[proc];
    }

    // Pack and send the particular indices that we will need to send to 
    // each process.
    std::vector<int> offsets = sendDispls;
    std::vector<int> sendIndices( totalSendCount );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = B.OwningProcess( naturalIndex );
        sendIndices[++offsets[proc]] = naturalIndex;
    }
    std::vector<int> recvIndices( totalRecvCount );
    mpi::AllToAll
    ( &sendIndices[0], &sendCounts[0], &sendDispls[0], 
      &recvIndices[0], &recvCounts[0], &recvDispls[0], comm_ );
    sendIndices.clear();

    // Scale the counts and offsets by the number of right-hand sides
    totalSendCount *= numRhs;
    totalRecvCount *= numRhs;
    for( int proc=0; proc<commSize; ++proc )
    {
        sendCounts[proc] *= numRhs;
        recvCounts[proc] *= numRhs;
        sendDispls[proc] *= numRhs;
        recvDispls[proc] *= numRhs;
    }

    // Pack and send the right-hand side data
    offsets = sendDispls;
    std::vector<C> sendB( totalSendCount );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal]; 
        const int proc = B.OwningProcess( naturalIndex );
        for( int k=0; k<numRhs; ++k )
            sendB[offsets[proc]+k] = redistB[iLocal+k*localHeight_];
        offsets[proc] += numRhs;
    }
    std::vector<C> recvB( totalRecvCount );
    mpi::AllToAll
    ( &sendB[0], &sendCounts[0], &sendDispls[0],
      &recvB[0], &recvCounts[0], &recvDispls[0], comm_ );
    sendB.clear();

    // Unpack the right-hand side data
    C* BBuffer = B.LocalBuffer();
    for( int proc=0; proc<commSize; ++proc )
    {
        const C* procB = &recvB[recvDispls[proc]];
        const int* procIndices = &recvIndices[recvDispls[proc]/numRhs];
        const int numLocalIndices = recvCounts[proc]/numRhs;
        for( int s=0; s<numLocalIndices; ++s )
        {
            const int naturalIndex = procIndices[s];
            const int localIndex = B.LocalIndex( naturalIndex );
            for( int k=0; k<numRhs; ++k )
                BBuffer[localIndex+k] = procB[s*numRhs+k];
        }
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::SolveWithGMRES( std::vector<C>& redistB ) const
{
    throw std::logic_error("GMRES not yet implemented");
}

template<typename R>
void
psp::DistHelmholtz<R>::SolveWithQMR( std::vector<C>& redistB ) const
{
    // TODO: Implement QMR in terms of Multiply and Precondition
}

// B := A B
template<typename R>
void
psp::DistHelmholtz<R>::Multiply( std::vector<C>& redistB ) const
{
    const int numRhs = redistB.size() / localHeight_;
    const int commSize = mpi::CommSize( comm_ );

    // Modify the basic send/recv information for the number of right-hand sides
    std::vector<int> sendCounts = globalSendCounts_;
    std::vector<int> sendDispls = globalSendDispls_;
    std::vector<int> recvCounts = globalRecvCounts_;
    std::vector<int> recvDispls = globalRecvDispls_;
    for( int proc=0; proc<commSize; ++proc )
    {
        sendCounts[proc] *= numRhs;
        sendDispls[proc] *= numRhs;
        recvCounts[proc] *= numRhs;
        recvDispls[proc] *= numRhs;
    }
    const int totalSendCount = sendDispls.back() + sendCounts.back();
    const int totalRecvCount = recvDispls.back() + recvCounts.back();

    // Pack and scatter our portion of the right-hand sides
    std::vector<C> sendRhs( totalSendCount );
    for( int proc=0; proc<commSize; ++proc )
    {
        const int sendSize = globalSendCounts_[proc];
        C* procRhs = &sendRhs[sendDispls[proc]];
        const int* procIndices = &globalSendIndices_[globalSendDispls_[proc]];
        for( int s=0; s<sendSize; ++s )
        {
            const int iLocal = procIndices[s];
            for( int k=0; k<numRhs; ++k )
                procRhs[s*numRhs+k] = redistB[iLocal+k*localHeight_];
        }
    }
    std::vector<C> recvRhs( totalRecvCount );
    mpi::AllToAll
    ( &sendRhs[0], &sendCounts[0], &sendDispls[0], 
      &recvRhs[0], &recvCounts[0], &recvDispls[0], comm_ );
    sendRhs.clear();

    // Run the local multiplies to form the result
    std::vector<int> offsets = recvDispls;
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        // Multiply by the diagonal value
        const int rowOffset = localRowOffsets_[iLocal];
        const C diagVal = localEntries_[rowOffset];
        for( int k=0; k<numRhs; ++k )
            redistB[iLocal+k*localHeight_] *= diagVal;

        // Multiply by the off-diagonal values
        const int rowSize = localRowOffsets_[iLocal+1]-rowOffset;
        for( int jLocal=1; jLocal<rowSize; ++jLocal )
        {
            const int proc = owningProcesses_[rowOffset+jLocal];
            const C offDiagVal = localEntries_[rowOffset+jLocal];
            for( int k=0; k<numRhs; ++k )
                redistB[iLocal+k*localHeight_] += 
                    offDiagVal*recvRhs[offsets[proc]+k];
            offsets[proc] += numRhs;
        }
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::Precondition( std::vector<C>& B ) const
{
    // Apply the sweeping preconditioner
    //
    // Simple algorithm:
    //   // Solve against L
    //   for i=1,...,m-1
    //     B_{i+1} := B_{i+1} - A_{i+1,i} T_i B_i
    //   // Solve against D 
    //   for i=1,...,m
    //     B_i := T_i B_i
    //   // Solve against L^T
    //   for i=m-1,...,1
    //     B_i := B_i - T_i A_{i,i+1} B_{i+1}
    //
    // Practical algorithm:
    //   // Solve against L D
    //   for i=1,...,m-1
    //     B_i := T_i B_i
    //     B_{i+1} := B_{i+1} - A_{i+1,i} B_i
    //   B_m := T_m B_m
    //   // Solve against L^T
    //   for i=m-1,...,1
    //     Z := B_i
    //     B_i := -A_{i,i+1} B_{i+1}
    //     B_i := T_i B_i
    //     B_i := B_i + Z
    //

    // Solve against L D
    for( int i=0; i<numPanels_-1; ++i )
    {
        SolvePanel( B, i );
        SubdiagonalUpdate( B, i );
    }
    SolvePanel( B, numPanels_-1 );

    // Solve against L^T
    std::vector<C> Z;
    const int numRhs = B.size() / localHeight_;
    for( int i=numPanels_-2; i>=0; --i )
    {
        ExtractPanel( B, i, Z );
        MultiplySuperdiagonal( B, i );    
        SolvePanel( B, i );
        UpdatePanel( B, i, Z );
    }
}

// B_i := T_i B_i
template<typename R>
void
psp::DistHelmholtz<R>::SolvePanel( std::vector<C>& B, int i ) const
{
    const int numRhs = B.size() / localHeight_;
    const int panelPadding = PanelPadding( i );
    const int panelDepth = PanelDepth( i );
    const int localPanelOffset = LocalPanelOffset( i );
    const int localPanelHeight = LocalPanelHeight( i );
    const clique::symbolic::SymmFact& symbFact = 
        PanelSymbolicFactorization( i );

    elemental::Matrix<C> localPanelB( localHeight_, numRhs );
    localPanelB.SetToZero();

    // For each supernode, pull in each right-hand side with a memcpy
    int BOffset = 0;
    const int numLocalSupernodes = symbFact.local.supernodes.size();
    for( int t=0; t<numLocalSupernodes; ++t )
    {
        const clique::symbolic::LocalSymmFactSupernode& sn = 
            symbFact.local.supernodes[t];
        const int size = sn.size;
        const int myOffset = sn.myOffset;

#ifndef RELEASE
        if( size % (panelPadding+panelDepth) != 0 )
            throw std::logic_error("Local supernode size problem");
#endif
        const int xySize = size/(panelPadding+panelDepth);
        const int paddingSize = xySize*panelPadding;
        const int remainingSize = size - paddingSize;

        for( int k=0; k<numRhs; ++k )
            std::memcpy
            ( localPanelB.Buffer(myOffset+paddingSize,k), 
              &B[BOffset+k*localHeight_], 
              remainingSize*sizeof(C) );
        BOffset += remainingSize;
    }
    const int numDistSupernodes = symbFact.dist.supernodes.size();
    for( int t=0; t<numDistSupernodes; ++t )
    {
        const clique::symbolic::DistSymmFactSupernode& sn = 
            symbFact.dist.supernodes[t];
        const int size = sn.size;
        const int localOffset1d = sn.localOffset1d;
        const int localSize1d = sn.localSize1d;

        const elemental::Grid& grid = *sn.grid;
        const int gridSize = grid.Size();
        const int gridRank = grid.VCRank();

#ifndef RELEASE
        if( size % (panelPadding+panelDepth) != 0 )
            throw std::logic_error("Dist supernode size problem");
#endif
        const int xySize = size/(panelPadding+panelDepth);
        const int paddingSize = xySize*panelPadding;
        const int localPaddingSize = 
            elemental::LocalLength( paddingSize, gridRank, gridSize );
        const int localRemainingSize = localSize1d - localPaddingSize;

        for( int k=0; k<numRhs; ++k )
            std::memcpy
            ( localPanelB.Buffer(localOffset1d+localPaddingSize,k),
              &B[BOffset+k*localHeight_],
              localRemainingSize*sizeof(C) );
        BOffset += localRemainingSize;
    }

    // Solve against the panel
    const clique::numeric::SymmFrontTree<C>& fact = 
        PanelNumericFactorization( i );
    clique::numeric::LDLSolve( TRANSPOSE, symbFact, fact, localPanelB, true );

    // For each supernode, extract each right-hand side with memcpy
    BOffset = 0;
    for( int t=0; t<numLocalSupernodes; ++t )
    {
        const clique::symbolic::LocalSymmFactSupernode& sn = 
            symbFact.local.supernodes[t];
        const int size = sn.size;
        const int myOffset = sn.myOffset;

#ifndef RELEASE
        if( size % (panelPadding+panelDepth) != 0 )
            throw std::logic_error("Local supernode size problem");
#endif
        const int xySize = size/(panelPadding+panelDepth);
        const int paddingSize = xySize*panelPadding;
        const int remainingSize = size - paddingSize;

        for( int k=0; k<numRhs; ++k )
            std::memcpy
            ( &B[BOffset+k*localHeight_],
              localPanelB.LockedBuffer(myOffset+paddingSize,k), 
              remainingSize*sizeof(C) );
        BOffset += remainingSize;
    }
    for( int t=0; t<numDistSupernodes; ++t )
    {
        const clique::symbolic::DistSymmFactSupernode& sn = 
            symbFact.dist.supernodes[t];
        const int size = sn.size;
        const int localOffset1d = sn.localOffset1d;
        const int localSize1d = sn.localSize1d;

        const elemental::Grid& grid = *sn.grid;
        const int gridSize = grid.Size();
        const int gridRank = grid.VCRank();

#ifndef RELEASE
        if( size % (panelPadding+panelDepth) != 0 )
            throw std::logic_error("Dist supernode size problem");
#endif
        const int xySize = size/(panelPadding+panelDepth);
        const int paddingSize = xySize*panelPadding;
        const int localPaddingSize = 
            elemental::LocalLength( paddingSize, gridRank, gridSize );
        const int localRemainingSize = localSize1d - localPaddingSize;

        for( int k=0; k<numRhs; ++k )
            std::memcpy
            ( &B[BOffset+k*localHeight_],
              localPanelB.LockedBuffer(localOffset1d+localPaddingSize,k),
              localRemainingSize*sizeof(C) );
        BOffset += localRemainingSize;
    }
}

// B_{i+1} := B_{i+1} - A_{i+1,i} B_i
template<typename R>
void
psp::DistHelmholtz<R>::SubdiagonalUpdate( std::vector<C>& B, int i ) const
{
    // TODO:
    //   1) Gather the necessary pieces of B_i
    //   2) Perform the local multiply
}

// Z := B_i
template<typename R>
void
psp::DistHelmholtz<R>::ExtractPanel
( const std::vector<C>& B, int i, std::vector<C>& Z ) const
{
    const int localPanelOffset = LocalPanelOffset( i );
    const int localPanelHeight = LocalPanelHeight( i );
    const int numRhs = B.size() / localHeight_;
    Z.resize( localPanelHeight*numRhs );

    for( int k=0; k<numRhs; ++k )
        std::memcpy
        ( &Z[k*localPanelHeight], &B[localPanelOffset+k*localHeight_],
          localPanelHeight*sizeof(C) );
}

// B_i := -A_{i,i+1} B_{i+1}
template<typename R>
void
psp::DistHelmholtz<R>::MultiplySuperdiagonal( std::vector<C>& B, int i ) const
{
    // TODO:
    //   1) Gather the necessary pieces of B_{i+1}
    //   2) Perform the local multiply
}

// B_i := B_i + Z
template<typename R>
void
psp::DistHelmholtz<R>::UpdatePanel
( std::vector<C>& B, int i, const std::vector<C>& Z ) const
{
    const int localPanelOffset = LocalPanelOffset( i );
    const int localPanelHeight = LocalPanelHeight( i );
    const int numRhs = Z.size() / localPanelHeight;
    for( int k=0; k<numRhs; ++k )
        for( int s=0; s<localPanelHeight; ++s )
            B[localPanelOffset+s+k*localHeight_] += Z[s+k*localPanelHeight];
}

