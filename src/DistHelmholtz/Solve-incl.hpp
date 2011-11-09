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
    const int numRhs = B.NumScalars();
    const int commSize = mpi::CommSize( comm_ );

    redistB.resize( localHeight_*numRhs );

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
    for( int proc=0; proc<commSize; ++proc )
        maxSize = std::max(sendPairs[2*proc+1],maxSize);
    std::vector<int> actualSendSizes( commSize );
    for( int proc=0; proc<commSize; ++proc )
        actualSendSizes[proc] = sendPairs[2*proc];
    sendPairs.clear();
    std::vector<int> recvIndices( maxSize*commSize );
    std::vector<int> recvOffsets( commSize );
    for( int proc=0; proc<commSize; ++proc )
        recvOffsets[proc] = maxSize*proc;
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = B.OwningProcess( naturalIndex );
        recvIndices[++recvOffsets[proc]] = naturalIndex;
    }
    std::vector<int> sendIndices( maxSize*commSize );
    mpi::AllToAll( &recvIndices[0], maxSize, &sendIndices[0], maxSize, comm_ );
    recvIndices.clear();
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

    // Reinitialize the recv offsets for unpacking the right-hand side data
    for( int proc=0; proc<commSize; ++proc )
        recvOffsets[proc] = maxSize*proc;

    // Unpack the received right-hand side data
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = B.OwningProcess( naturalIndex );
        for( int k=0; k<numRhs; ++k )
            redistB[iLocal+k*localHeight_] = recvB[numRhs*recvOffsets[proc]+k];
        ++recvOffsets[proc];
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::PushRightHandSides
( GridData<C>& B, const std::vector<C>& redistB ) const
{
    // TODO
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
    // TODO
}
