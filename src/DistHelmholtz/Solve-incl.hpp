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
    // Pack and send the amount of data that we will need to recv from each
    // process.
    const int commSize = mpi::CommSize( comm_ );
    std::vector<int> recvPairs( 2*commSize, 0 );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = B.OwningProcess( naturalIndex );
        ++recvPairs[2*proc];
    }
    int maxSize = 0;
    for( int proc=0; proc<commSize; ++proc )
        maxSize = std::max(recvPairs[2*proc],maxSize);
    for( int proc=0; proc<commSize; ++proc )
        recvPairs[2*proc+1] = maxSize;
    std::vector<int> sendPairs( 2*commSize );
    mpi::AllToAll( &recvPairs[0], 2, &sendPairs[0], 2, comm_ );
    recvPairs.clear();

    // Pack and send the indices that we will need to recv from each process.
    for( int proc=0; proc<commSize; ++proc )
        maxSize = std::max(sendPairs[2*proc+1],maxSize);
    std::vector<int> actualSendSizes( commSize );
    for( int proc=0; proc<commSize; ++proc )
        actualSendSizes[proc] = sendPairs[2*proc];
    sendPairs.clear();
    std::vector<int> recvIndices( maxSize*commSize );
    std::vector<int> offsets( commSize );
    for( int proc=0; proc<commSize; ++proc )
        offsets[proc] = maxSize*proc;
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = B.OwningProcess( naturalIndex );
        recvIndices[++offsets[proc]] = naturalIndex;
    }
    std::vector<int> sendIndices( maxSize*commSize );
    mpi::AllToAll( &recvIndices[0], maxSize, &sendIndices[0], maxSize, comm_ );
    recvIndices.clear();

    // Pack and send our right-hand side data.
    const int numRhs = B.NumScalars();
    std::vector<C> sendB( maxSize*commSize*numRhs );
    const C* BBuffer = B.LockedLocalBuffer();
    for( int proc=0; proc<commSize; ++proc )
    {
        C* send = &sendB[proc*maxSize*numRhs];
        for( int iLocal=0; iLocal<actualSendSizes[proc]; ++iLocal )
        {
            const int naturalIndex = sendIndices[iLocal];
            const int localIndex = B.LocalIndex( naturalIndex );
            for( int k=0; k<numRhs; ++k )
                send[iLocal*numRhs+k] = BBuffer[localIndex+k];
        }
    }
    sendIndices.clear();
    std::vector<C> recvB( maxSize*commSize*numRhs );
    mpi::AllToAll
    ( &sendB[0], maxSize*numRhs, &recvB[0], maxSize*numRhs, comm_ );
    sendB.clear();

    // Unpack the received right-hand side data
    redistB.resize( localHeight_*numRhs );
    for( int proc=0; proc<commSize; ++proc )
        offsets[proc] = maxSize*proc;
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = B.OwningProcess( naturalIndex );
        for( int k=0; k<numRhs; ++k )
            redistB[iLocal+k*localHeight_] = recvB[numRhs*offsets[proc]+k];
        ++offsets[proc];
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::PushRightHandSides
( GridData<C>& B, const std::vector<C>& redistB ) const
{
    // Pack and send the amount of data that we will need to send to 
    // each process.
    const int numRhs = B.NumScalars();
    const int commSize = mpi::CommSize( comm_ );
    std::vector<int> sendPairs( 2*commSize, 0 );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = B.OwningProcess( naturalIndex );
        ++sendPairs[2*proc];
    }
    int maxSize = 0;
    for( int proc=0; proc<commSize; ++proc )
        maxSize = std::max(sendPairs[2*proc],maxSize);
    for( int proc=0; proc<commSize; ++proc )
        sendPairs[2*proc+1] = maxSize;
    std::vector<int> recvPairs( 2*commSize );
    mpi::AllToAll( &sendPairs[0], 2, &recvPairs[0], 2, comm_ );
    sendPairs.clear();

    // Pack and send the particular indices that we will need to send to 
    // each process.
    for( int proc=0; proc<commSize; ++proc )
        maxSize = std::max(recvPairs[2*proc+1],maxSize);
    std::vector<int> actualRecvSizes( commSize );
    for( int proc=0; proc<commSize; ++proc )
        actualRecvSizes[proc] = recvPairs[2*proc];
    recvPairs.clear();
    std::vector<int> sendIndices( maxSize*commSize );
    std::vector<int> offsets( commSize );
    for( int proc=0; proc<commSize; ++proc )
        offsets[proc] = maxSize*proc;
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = B.OwningProcess( naturalIndex );
        sendIndices[++offsets[proc]] = naturalIndex;
    }
    std::vector<int> recvIndices( maxSize*commSize );
    mpi::AllToAll( &sendIndices[0], maxSize, &recvIndices[0], maxSize, comm_ );
    sendIndices.clear();

    // Pack and send the right-hand side data
    for( int proc=0; proc<commSize; ++proc )
        offsets[proc] = maxSize*proc;
    std::vector<C> sendB( maxSize*commSize*numRhs );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal]; 
        const int proc = B.OwningProcess( naturalIndex );
        for( int k=0; k<numRhs; ++k )
            sendB[offsets[proc]*numRhs+k] = redistB[iLocal+k*localHeight_];
        ++offsets[proc];
    }
    std::vector<C> recvB( maxSize*commSize*numRhs );
    mpi::AllToAll
    ( &sendB[0], maxSize*numRhs, &recvB[0], maxSize*numRhs, comm_ );
    sendB.clear();

    // Unpack the right-hand side data
    C* BBuffer = B.LocalBuffer();
    for( int proc=0; proc<commSize; ++proc )
    {
        const C* recv = &recvB[proc*maxSize*numRhs];
        for( int iLocal=0; iLocal<actualRecvSizes[proc]; ++iLocal )
        {
            const int naturalIndex = recvIndices[iLocal];
            const int localIndex = B.LocalIndex( naturalIndex );
            for( int k=0; k<numRhs; ++k )
                BBuffer[localIndex+k] = recv[iLocal*numRhs+k];
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
    // Pack and scatter our portion of the right-hand sides
    const int numRhs = redistB.size() / localHeight_;
    const int commSize = mpi::CommSize( comm_ );
    std::vector<C> sendRhs( commSize*allToAllSize_*numRhs );
    for( int proc=0; proc<commSize; ++proc )
    {
        C* send = &sendRhs[proc*allToAllSize_*numRhs];
        for( int iLocal=0; iLocal<actualSendSizes_[proc]; ++iLocal )
            for( int k=0; k<numRhs; ++k )
                send[iLocal*numRhs+k] = redistB[iLocal+k*localHeight_];
    }
    std::vector<C> recvRhs( commSize*allToAllSize_*numRhs );
    mpi::AllToAll
    ( &sendRhs[0], allToAllSize_*numRhs, 
      &recvRhs[0], allToAllSize_*numRhs, comm_ );
    sendRhs.clear();

    // Run the local multiplies to form the result
    std::vector<int> offsets( commSize );
    for( int proc=0; proc<commSize; ++proc )
        offsets[proc] = proc*allToAllSize_;
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
                    offDiagVal*recvRhs[offsets[proc]*numRhs+k];
            ++offsets[proc];
        }
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::Precondition( std::vector<C>& redistB ) const
{
    const int numRhs = redistB.size() / localHeight_;

    // TODO: 
    // Apply the sweeping preconditioner
    //
    // 1) Solve against L
    // 2) Solve against D
    // 3) Solve against L^T
    //
    // Each panel solve will require a careful memcpy for each supernode, 
    // and the application of the block Gauss transforms will also require 
    // multiplication with a portion of the global sparse matrix. Perhaps 
    // it should be its own routine?
}

