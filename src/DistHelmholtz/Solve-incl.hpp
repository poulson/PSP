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
psp::DistHelmholtz<R>::Solve
( GridData<C>& gridB, Solver solver, int maxIterations ) const
{
    if( !elemental::mpi::CongruentComms( comm_, gridB.Comm() ) )
        throw std::logic_error("B does not have a congruent comm");

    // Convert B into custom nested-dissection based ordering
    elemental::Matrix<C> B;
#ifndef RELEASE
    const int commRank = elemental::mpi::CommRank( comm_ );
    if( commRank == 0 )
    {
        std::cout << "Pulling right-hand sides...";
        std::cout.flush();
    }
#endif
    PullRightHandSides( gridB, B );
#ifndef RELEASE
    if( commRank == 0 )
        std::cout << "done" << std::endl;
    elemental::Matrix<C> origB = B;
#endif

    // Solve the systems of equations
    switch( solver )
    {
    case GMRES: SolveWithGMRES( B, maxIterations ); break;
    case QMR:   SolveWithQMR( B, maxIterations );   break;
    }

#ifndef RELEASE
    if( commRank == 0 )
    {
        std::cout << "Checking error...";
        std::cout.flush();
    }
    elemental::Matrix<C> Y = B;
    Multiply( Y );
    elemental::basic::Axpy( (C)-1, origB, Y );
    const R norm = elemental::advanced::Norm( Y );
    if( commRank == 0 )
        std::cout << "||AB - origB||_F = " << norm << std::endl;
#endif

    // Restore the solutions back into the GridData form
#ifndef RELEASE
    if( commRank == 0 )
    {
        std::cout << "Pushing right-hand sides...";
        std::cout.flush();
    }
#endif
    PushRightHandSides( gridB, B );
#ifndef RELEASE
    if( commRank == 0 )
        std::cout << "done" << std::endl;
#endif
}

template<typename R>
void
psp::DistHelmholtz<R>::PullRightHandSides
( const GridData<C>& gridB, elemental::Matrix<C>& B ) const
{
    const int commSize = mpi::CommSize( comm_ );

    // Pack and send the amount of data that we will need to recv
    std::vector<int> recvCounts( commSize, 0 );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = gridB.OwningProcess( naturalIndex );
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
        const int proc = gridB.OwningProcess( naturalIndex );
        recvIndices[++offsets[proc]] = naturalIndex;
    }
    std::vector<int> sendIndices( totalSendCount );
    mpi::AllToAll
    ( &recvIndices[0], &recvCounts[0], &recvDispls[0],
      &sendIndices[0], &sendCounts[0], &sendDispls[0], comm_ );
    recvIndices.clear();

    // Scale the send and recv counts to make up for their being several RHS
    const int numRhs = gridB.NumScalars();
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
    const C* gridBBuffer = gridB.LockedLocalBuffer();
    for( int proc=0; proc<commSize; ++proc )
    {
        C* procB = &sendB[sendDispls[proc]];
        const int* procIndices = &sendIndices[sendDispls[proc]/numRhs];
        const int numLocalIndices = sendCounts[proc]/numRhs;
        for( int s=0; s<numLocalIndices; ++s )
        {
            const int naturalIndex = procIndices[s];
            const int localIndex = gridB.LocalIndex( naturalIndex );
            for( int k=0; k<numRhs; ++k )
                procB[s*numRhs+k] = gridBBuffer[localIndex+k];
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
    B.ResizeTo( localHeight_, numRhs, localHeight_ );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = gridB.OwningProcess( naturalIndex );
        for( int k=0; k<numRhs; ++k )
            B.Set( iLocal, k, recvB[offsets[proc]+k] );
        offsets[proc] += numRhs;
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::PushRightHandSides
( GridData<C>& gridB, const elemental::Matrix<C>& B ) const
{
    const int numRhs = gridB.NumScalars();
    const int commSize = mpi::CommSize( comm_ );

    // Pack and send the amount of data that we will need to send.
    std::vector<int> sendCounts( commSize, 0 );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = gridB.OwningProcess( naturalIndex );
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
        const int proc = gridB.OwningProcess( naturalIndex );
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
        const int proc = gridB.OwningProcess( naturalIndex );
        for( int k=0; k<numRhs; ++k )
            sendB[offsets[proc]+k] = B.Get( iLocal, k );
        offsets[proc] += numRhs;
    }
    std::vector<C> recvB( totalRecvCount );
    mpi::AllToAll
    ( &sendB[0], &sendCounts[0], &sendDispls[0],
      &recvB[0], &recvCounts[0], &recvDispls[0], comm_ );
    sendB.clear();

    // Unpack the right-hand side data
    C* gridBBuffer = gridB.LocalBuffer();
    for( int proc=0; proc<commSize; ++proc )
    {
        const C* procB = &recvB[recvDispls[proc]];
        const int* procIndices = &recvIndices[recvDispls[proc]/numRhs];
        const int numLocalIndices = recvCounts[proc]/numRhs;
        for( int s=0; s<numLocalIndices; ++s )
        {
            const int naturalIndex = procIndices[s];
            const int localIndex = gridB.LocalIndex( naturalIndex );
            for( int k=0; k<numRhs; ++k )
                gridBBuffer[localIndex+k] = procB[s*numRhs+k];
        }
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::SolveWithGMRES
( elemental::Matrix<C>& B, int maxIterations ) const
{
    throw std::logic_error("GMRES not yet implemented");
}

// Based on Algorithm 8.1 from the paper 
// "An implementation of the QMR method based on coupled two-term recurrences",
// by R. Freund and N. Nachtigal.
template<typename R>
void
psp::DistHelmholtz<R>::SolveWithQMR
( elemental::Matrix<C>& B, int maxIterations ) const
{
    // TODO: Make this an easily changed parameter
    const R exitRatio = 1e-16;

    const int commSize = mpi::CommSize( comm_ );
    const int commRank = mpi::CommRank( comm_ );
    const int localHeight = B.Height();
    const int numRhs = B.Width();

    // Set aside the memory necessary for QMR
    elemental::Matrix<C> Z( localHeight, numRhs );
    elemental::Matrix<C> V( localHeight, numRhs );
    elemental::Matrix<C> D( localHeight, numRhs );
    elemental::Matrix<C> P( localHeight, numRhs );
    std::vector<R> origRho( numRhs );
    std::vector<R> rhoLast( numRhs ), cLast( numRhs ), thetaLast( numRhs ),
                   rho( numRhs ), c( numRhs ), theta( numRhs );
    std::vector<C> epsilonLast( numRhs ), etaLast( numRhs ),
                   delta( numRhs ), epsilon( numRhs ), negativeBeta( numRhs ),
                   eta( numRhs ), tempScalar( numRhs );

    // Bootstrap QMR
    Z = B;
    Precondition( Z );
    Norms( Z, origRho );
    DivideColumns( Z, origRho );
    V = B;
    DivideColumns( V, origRho );
    B.SetToZero();
    D.SetToZero();
    P.SetToZero();
    R maxOrigRho = 0;
    if( commRank == 0 )
    {
        for( int j=0; j<numRhs; ++j )
        {
            maxOrigRho = std::max(maxOrigRho,origRho[j]);
            D.Set( 0, j, origRho[j] );
            P.Set( 0, j, origRho[j] );
        }
    }
#ifndef RELEASE
    if( commRank == 0 )
        std::cout << "Max orig rho: " << maxOrigRho << std::endl;
#endif
    rhoLast = origRho;
    for( int j=0; j<numRhs; ++j )
    {
        cLast[j] = 1;
        thetaLast[j] = 0;
        epsilonLast[j] = 1;
        etaLast[j] = -1;
    }

    // Loop until convergence (or too many iterations)
    for( int it=0; it<maxIterations; ++it )
    {
#ifndef RELEASE
        if( commRank == 0 )
            std::cout << "  Iteration " << it << " of QMR." << std::endl;
#endif
        //
        // Step 1 (we already have each inv(M) v_n sitting in Z)
        //

        // Ensure that QMR hasn't broken down
        bool zeroEpsilonLast = false;
        for( int j=0; j<numRhs; ++j )
        {
            if( epsilonLast[j] == (C)0 )
            {
                if( commRank == 0 )
                    std::cerr << "epsilonLast[" << j << "] was zero on "
                              << "iteration " << it << std::endl;
                zeroEpsilonLast = true;
            }
        }
        if( zeroEpsilonLast )
            break;

        // Compute delta_j = v_j^T z_j
        PseudoInnerProducts( V, Z, delta );

        // Ensure that QMR hasn't broken down
        bool zeroDelta = false;
        for( int j=0; j<numRhs; ++j )
        {
            if( delta[j] == (C)0 ) 
            {
                if( commRank == 0 )
                    std::cerr << "delta[" << j << "] was zero on "
                              << "iteration " << it << std::endl;
                zeroDelta = true;
            }
        }
        if( zeroDelta )
            break;

        //
        // Step 2
        //

        for( int j=0; j<numRhs; ++j )
            tempScalar[j] = -rhoLast[j]*delta[j]/epsilonLast[j];
        ScaleColumns( P, tempScalar );
        elemental::basic::Axpy( (C)1, Z, P );

        //
        // Step 3
        //

        Z = P;
        Multiply( Z );
        PseudoInnerProducts( P, Z, epsilon );
        for( int j=0; j<numRhs; ++j )
            negativeBeta[j] = -epsilon[j]/delta[j];
        ScaleColumns( V, negativeBeta );
        elemental::basic::Axpy( (C)1, Z, V );
        Z = V;
        Precondition( Z );
        Norms( Z, rho );

        //
        // Step 4
        //

        const R one = 1;
        for( int j=0; j<numRhs; ++j )
        {
            theta[j] = rho[j]/(cLast[j]*elemental::Abs(negativeBeta[j]));
            c[j] = one/sqrt(1+theta[j]*theta[j]);
            eta[j] = -etaLast[j]*rhoLast[j]*c[j]*c[j]/
                      (-negativeBeta[j]*cLast[j]*cLast[j]);
            const R temp = thetaLast[j]*c[j];
            tempScalar[j] = temp*temp;
        }
        ScaleColumns( D, tempScalar );
        AddScaledColumns( eta, P, D );
        elemental::basic::Axpy( (C)1, D, B );

        //
        // Step 5 (modified to add a success condition)
        //

        // See if we have a sufficient solution
        R maxRelRho = 0;
        for( int j=0; j<numRhs; ++j )
            maxRelRho = std::max(maxRelRho,rho[j]/origRho[j]);
        if( maxRelRho < exitRatio )
            break;
#ifndef RELEASE
        if( commRank == 0 )
            std::cout << "  maxRelRho=" << maxRelRho << std::endl;
#endif

        // Ensure that QMR hasn't broken down
        bool zeroRho = false;
        for( int j=0; j<numRhs; ++j )
        {
            if( rho[j] == (R)0 )
            {
                if( commRank == 0 )
                    std::cerr << "rho[" << j << "] was zero on "
                              << "iteration " << it << std::endl;
                zeroRho = true;
            }
        }
        if( zeroRho )
            break;

        // Normalize Z and V with the rho's
        DivideColumns( Z, rho );
        DivideColumns( V, rho );

        //
        // Set up for the next iteration (if there is one)
        //
        if( it == maxIterations-1 )
        {
            if( commRank == 0 )
                std::cerr << "QMR did not converge in " << maxIterations 
                          << " iterations" << std::endl;
            break;
        }
        rhoLast = rho;
        cLast = c;
        thetaLast = theta;
        epsilonLast = epsilon;
        etaLast = eta;
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::Norms
( const elemental::Matrix<C>& X, std::vector<R>& norms ) const
{
    const int localHeight = X.Height();
    const int numCols = X.Width();
    const int commSize = mpi::CommSize( comm_ );
    std::vector<R> localNorms( numCols );
    for( int j=0; j<numCols; ++j )
        localNorms[j] = 
            elemental::blas::Nrm2( localHeight, X.LockedBuffer(0,j), 1 );
    std::vector<R> allLocalNorms( numCols*commSize );
    mpi::AllGather
    ( &localNorms[0], numCols, &allLocalNorms[0], numCols, comm_ );
    norms.resize( numCols );
    for( int j=0; j<numCols; ++j )
        norms[j] = 
            elemental::blas::Nrm2( commSize, &allLocalNorms[j], numCols );
}

template<typename R>
void
psp::DistHelmholtz<R>::PseudoInnerProducts
( const elemental::Matrix<C>& X, const elemental::Matrix<C>& Y,
  std::vector<C>& alphas ) const
{
    const int localHeight = X.Height();
    const int numCols = X.Width();
    std::vector<C> localAlphas( numCols );
    for( int j=0; j<numCols; ++j )
        localAlphas[j] = 
            elemental::blas::Dotu
            ( localHeight, X.LockedBuffer(0,j), 1,
                           Y.LockedBuffer(0,j), 1 );
    alphas.resize( numCols );
    mpi::AllReduce( &localAlphas[0], &alphas[0], numCols, MPI_SUM, comm_ );
}

template<typename R>
void
psp::DistHelmholtz<R>::DivideColumns
( elemental::Matrix<C>& X, const std::vector<R>& d ) const
{
    const R one = 1;
    const int localHeight = X.Height();
    const int numCols = X.Width();
    for( int j=0; j<numCols; ++j )
    {
        const R invDelta = one/d[j];
        C* XCol = X.Buffer(0,j);
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
            XCol[iLocal] *= invDelta;
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::ScaleColumns
( elemental::Matrix<C>& X, const std::vector<C>& d ) const
{
    const int localHeight = X.Height();
    const int numCols = X.Width();
    for( int j=0;j<numCols; ++j )
    {
        const C delta = d[j];
        C* XCol = X.Buffer(0,j);
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
            XCol[iLocal] *= delta;
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::AddScaledColumns
( const std::vector<C>& d, 
  const elemental::Matrix<C>& X, elemental::Matrix<C>& Y ) const
{
    const int localHeight = X.Height();
    const int numCols = X.Width();
    for( int j=0; j<numCols; ++j )
    {
        const C delta = d[j];
        C* YCol = Y.Buffer(0,j);
        const C* XCol = X.LockedBuffer(0,j);
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
            YCol[j] += delta*XCol[j];
    }
}

// B := A B
template<typename R>
void
psp::DistHelmholtz<R>::Multiply( elemental::Matrix<C>& B ) const
{
    const int numRhs = B.Width();
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
                procRhs[s*numRhs+k] = B.Get( iLocal, k );
        }
    }
    std::vector<C> recvRhs( totalRecvCount );
    mpi::AllToAll
    ( &sendRhs[0], &sendCounts[0], &sendDispls[0], 
      &recvRhs[0], &recvCounts[0], &recvDispls[0], comm_ );
    sendRhs.clear();

    // Run the local multiplies to form the result
    std::vector<int> offsets = recvDispls;
    C* BBuffer = B.Buffer();
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        // Multiply by the diagonal value
        const int rowOffset = localRowOffsets_[iLocal];
        const C diagVal = localEntries_[rowOffset];
        for( int k=0; k<numRhs; ++k )
            BBuffer[iLocal+k*localHeight_] *= diagVal;

        // Multiply by the off-diagonal values
        const int rowSize = localRowOffsets_[iLocal+1]-rowOffset;
        for( int jLocal=1; jLocal<rowSize; ++jLocal )
        {
            const int proc = owningProcesses_[rowOffset+jLocal];
            const C offDiagVal = localEntries_[rowOffset+jLocal];
            for( int k=0; k<numRhs; ++k )
                BBuffer[iLocal+k*localHeight_] +=
                    offDiagVal*recvRhs[offsets[proc]+k];
            offsets[proc] += numRhs;
        }
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::Precondition( elemental::Matrix<C>& B ) const
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
    elemental::Matrix<C> Z;
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
psp::DistHelmholtz<R>::SolvePanel( elemental::Matrix<C>& B, int i ) const
{
    const int numRhs = B.Width();
    const int panelPadding = PanelPadding( i );
    const int panelDepth = PanelDepth( i );
    const clique::symbolic::SymmFact& symbFact = 
        PanelSymbolicFactorization( i );

    elemental::Matrix<C> localPanelB( localHeight_, numRhs );
    localPanelB.SetToZero();

    // For each supernode, pull in each right-hand side with a memcpy
    int BOffset = LocalPanelOffset( i );
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
        const int remainingSize = xySize*panelDepth;

        for( int k=0; k<numRhs; ++k )
            std::memcpy
            ( localPanelB.Buffer(myOffset+paddingSize,k), 
              B.LockedBuffer(BOffset,k),
              remainingSize*sizeof(C) );
        BOffset += remainingSize;
    }
    const int numDistSupernodes = symbFact.dist.supernodes.size();
    for( int t=1; t<numDistSupernodes; ++t )
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
              B.LockedBuffer(BOffset,k),
              localRemainingSize*sizeof(C) );
        BOffset += localRemainingSize;
    }
#ifndef RELEASE
    if( BOffset != LocalPanelOffset(i)+LocalPanelHeight(i) )
    {
        std::cout << "BOffset=" << BOffset << "\n"
                  << "LocalPanelOffset(i)+LocalPanelHeight(i)="
                  << LocalPanelOffset(i)+LocalPanelHeight(i) << std::endl;
        throw std::logic_error("Invalid BOffset usage in pull");
    }
#endif

    // Solve against the panel
    const clique::numeric::SymmFrontTree<C>& fact = 
        PanelNumericFactorization( i );
    clique::numeric::LDLSolve( TRANSPOSE, symbFact, fact, localPanelB, true );

    // For each supernode, extract each right-hand side with memcpy
    BOffset = LocalPanelOffset( i );
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
            ( B.Buffer(BOffset,k),
              localPanelB.LockedBuffer(myOffset+paddingSize,k), 
              remainingSize*sizeof(C) );
        BOffset += remainingSize;
    }
    for( int t=1; t<numDistSupernodes; ++t )
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
            ( B.Buffer(BOffset,k),
              localPanelB.LockedBuffer(localOffset1d+localPaddingSize,k),
              localRemainingSize*sizeof(C) );
        BOffset += localRemainingSize;
    }
#ifndef RELEASE
    if( BOffset != LocalPanelOffset(i)+LocalPanelHeight(i) )
        throw std::logic_error("Invalid BOffset usage in push");
#endif
}

// B_{i+1} := B_{i+1} - A_{i+1,i} B_i
template<typename R>
void
psp::DistHelmholtz<R>::SubdiagonalUpdate( elemental::Matrix<C>& B, int i ) const
{
    const int commSize = mpi::CommSize( comm_ );
    const int numRhs = B.Width();
    const int panelSendCount = subdiagPanelSendCounts_[i];
    const int panelRecvCount = subdiagPanelRecvCounts_[i];
    const int panelSendOffset = subdiagPanelSendDispls_[i];
    const int panelRecvOffset = subdiagPanelRecvDispls_[i];

    // Pack and alltoall the local d.o.f. at the back of B_i
    std::vector<C> sendBuffer( panelSendCount*numRhs );
    for( int s=0; s<panelSendCount; ++s )
    {
        const int iLocal = subdiagSendIndices_[panelSendOffset+s];
        for( int k=0; k<numRhs; ++k ) 
            sendBuffer[s*numRhs+k] = B.Get( iLocal, k );
    }

    // Prepare the send and recv information
    std::vector<int> sendCounts( commSize ), sendDispls( commSize ),
                     recvCounts( commSize ), recvDispls( commSize );
    for( int proc=0; proc<commSize; ++proc )
    {
        const int index = i*commSize + proc;
        sendCounts[proc] = subdiagSendCounts_[index]*numRhs;
        sendDispls[proc] = subdiagSendDispls_[index]*numRhs;
        recvCounts[proc] = subdiagRecvCounts_[index]*numRhs;
        recvDispls[proc] = subdiagRecvDispls_[index]*numRhs;
    }

    std::vector<C> recvBuffer( panelRecvCount*numRhs );
    mpi::AllToAll
    ( &sendBuffer[0], &sendCounts[0], &sendDispls[0],
      &recvBuffer[0], &recvCounts[0], &recvDispls[0], comm_ );
    sendBuffer.clear();
    sendCounts.clear();
    sendDispls.clear();

    // Perform the local update
    const int* recvLocalRows = &subdiagRecvLocalRows_[panelRecvOffset];
    const int* recvLocalIndices = &subdiagRecvLocalIndices_[panelRecvOffset];
    for( int proc=0; proc<commSize; ++proc )
    {
        const int procSize = recvCounts[proc]/numRhs;
        const int procOffset = recvDispls[proc]/numRhs;

        const C* procValues = &recvBuffer[recvDispls[proc]];
        const int* procLocalRows = &recvLocalRows[procOffset];
        const int* procLocalIndices = &recvLocalIndices[procOffset];
        for( int s=0; s<procSize; ++s )
        {
            const int iLocal = procLocalRows[s];
            const int localIndex = procLocalIndices[s];
            const C alpha = localEntries_[localIndex];
            for( int k=0; k<numRhs; ++k )
            {
                const C beta = procValues[s*numRhs+k];
                B.Update( iLocal, k, -alpha*beta );
            }
        }
    }
}

// Z := B_i
template<typename R>
void
psp::DistHelmholtz<R>::ExtractPanel
( const elemental::Matrix<C>& B, int i, elemental::Matrix<C>& Z ) const
{
    const int localPanelOffset = LocalPanelOffset( i );
    const int localPanelHeight = LocalPanelHeight( i );
    const int numRhs = B.Width();
    Z.ResizeTo( localPanelHeight, numRhs );

    for( int k=0; k<numRhs; ++k )
        std::memcpy
        ( Z.Buffer(0,k), B.LockedBuffer(localPanelOffset,k),
          localPanelHeight*sizeof(C) );
}

// B_i := -A_{i,i+1} B_{i+1}
template<typename R>
void
psp::DistHelmholtz<R>::MultiplySuperdiagonal
( elemental::Matrix<C>& B, int i ) const
{
    const int commSize = mpi::CommSize( comm_ );
    const int numRhs = B.Width();
    const int panelSendCount = supdiagPanelSendCounts_[i];
    const int panelRecvCount = supdiagPanelRecvCounts_[i];
    const int panelSendOffset = supdiagPanelSendDispls_[i];
    const int panelRecvOffset = supdiagPanelRecvDispls_[i];

    // Pack and alltoall the local d.o.f. at the front of B_{i+1}
    std::vector<C> sendBuffer( panelSendCount*numRhs );
    for( int s=0; s<panelSendCount; ++s )
    {
        const int iLocal = supdiagSendIndices_[panelSendOffset+s];
        for( int k=0; k<numRhs; ++k ) 
            sendBuffer[s*numRhs+k] = B.Get( iLocal, k );
    }

    // Prepare the send and recv information
    std::vector<int> sendCounts( commSize ), sendDispls( commSize ),
                     recvCounts( commSize ), recvDispls( commSize );
    for( int proc=0; proc<commSize; ++proc )
    {
        const int index = i*commSize + proc;
        sendCounts[proc] = supdiagSendCounts_[index]*numRhs;
        sendDispls[proc] = supdiagSendDispls_[index]*numRhs;
        recvCounts[proc] = supdiagRecvCounts_[index]*numRhs;
        recvDispls[proc] = supdiagRecvDispls_[index]*numRhs;
    }

    std::vector<C> recvBuffer( panelRecvCount*numRhs );
    mpi::AllToAll
    ( &sendBuffer[0], &sendCounts[0], &sendDispls[0],
      &recvBuffer[0], &recvCounts[0], &recvDispls[0], comm_ );
    sendBuffer.clear();
    sendCounts.clear();
    sendDispls.clear();

    // Perform the local multiply
    const int* recvLocalRows = &supdiagRecvLocalRows_[panelRecvOffset];
    const int* recvLocalIndices = &supdiagRecvLocalIndices_[panelRecvOffset];
    for( int proc=0; proc<commSize; ++proc )
    {
        const int procSize = recvCounts[proc]/numRhs;
        const int procOffset = recvDispls[proc]/numRhs;

        const C* procValues = &recvBuffer[recvDispls[proc]];
        const int* procLocalRows = &recvLocalRows[procOffset];
        const int* procLocalIndices = &recvLocalIndices[procOffset];
        for( int s=0; s<procSize; ++s )
        {
            const int iLocal = procLocalRows[s];
            const int localIndex = procLocalIndices[s];
            const C alpha = localEntries_[localIndex];
            for( int k=0; k<numRhs; ++k )
            {
                const C beta = procValues[s*numRhs+k];
                B.Set( iLocal, k, -alpha*beta );
            }
        }
    }
}

// B_i := B_i + Z
template<typename R>
void
psp::DistHelmholtz<R>::UpdatePanel
( elemental::Matrix<C>& B, int i, const elemental::Matrix<C>& Z ) const
{
    const int localPanelOffset = LocalPanelOffset( i );
    const int localPanelHeight = LocalPanelHeight( i );
    const int numRhs = Z.Width();
    for( int k=0; k<numRhs; ++k )
        for( int s=0; s<localPanelHeight; ++s )
            B.Update( localPanelOffset+s, k, Z.Get(s,k) );
}

