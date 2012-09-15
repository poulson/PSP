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

namespace psp {

template<typename R>
void
DistHelmholtz<R>::Solve
( DistUniformGrid<C>& gridB, int m, R relTol, bool viewIterates ) const
{
    if( !mpi::CongruentComms( comm_, gridB.Comm() ) )
        throw std::logic_error("B does not have a congruent comm");
    const int commRank = mpi::CommRank( comm_ );

    // Convert B into custom nested-dissection based ordering
    Matrix<C> B;
    {
        if( commRank == 0 )
        {
            std::cout << "  pulling right-hand sides...";
            std::cout.flush();
        }
        const double startTime = mpi::Time();

        PullRightHandSides( gridB, B );

        const double stopTime = mpi::Time();
        if( commRank == 0 )
            std::cout << stopTime-startTime << " secs" << std::endl;
    }

    // Solve the systems of equations
    InternalSolveWithGMRES( gridB, B, m, relTol, viewIterates );

    // Restore the solutions back into the original form
    {
        if( commRank == 0 )
        {
            std::cout << "  pushing right-hand sides...";
            std::cout.flush();
        }
        const double startTime = mpi::Time();

        PushRightHandSides( gridB, B );

        const double stopTime = mpi::Time();
        if( commRank == 0 )
            std::cout << stopTime-startTime << " secs" << std::endl;
    }
}

template<typename R>
void
DistHelmholtz<R>::PullRightHandSides
( const DistUniformGrid<C>& gridB, Matrix<C>& B ) const
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
        recvIndices[offsets[proc]++] = naturalIndex;
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
DistHelmholtz<R>::PushRightHandSides
( DistUniformGrid<C>& gridB, const Matrix<C>& B ) const
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
        sendIndices[offsets[proc]++] = naturalIndex;
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
DistHelmholtz<R>::InternalSolveWithGMRES
( DistUniformGrid<C>& gridB, Matrix<C>& bList, int m, R relTol,
  bool viewIterates ) const
{
    const int numRhs = bList.Width();
    const int localHeight = bList.Height();
    const int commRank = mpi::CommRank( comm_ );

    Matrix<C> VInter( localHeight, numRhs*m ), // interwoven
              x0List( localHeight, numRhs   ), // contiguous
              xList(  localHeight, numRhs   ), // contiguous
              wList(  localHeight, numRhs   ), // contiguous
              zList(  m+1,         numRhs   ), // contiguous
              HList(  m,           m*numRhs ); // contiguous
#ifdef PRINT_RITZ_VALUES
    Matrix<C> HListCopy( m, m*numRhs );
    elem::MakeZeros( HListCopy );
    bool firstBatch=true;
#endif

    // For storing Givens rotations
    Matrix<R> csList( m, numRhs );
    Matrix<C> snList( m, numRhs );

    // Various scalars
    std::vector<C> alphaList( numRhs );
    std::vector<R> betaList( numRhs ), deltaList( numRhs );
    std::vector<R> origResidNormList( numRhs ), residNormList( numRhs ), 
                   relResidNormList( numRhs );

    // x := 0
    elem::MakeZeros( xList );

    // w := b (= b - A x_0)
    // origResidNorm := ||w||_2
    wList = bList;
    Norms( wList, origResidNormList );
    const bool origResidHasNaN = CheckForNaN( origResidNormList );
    if( origResidHasNaN )
        throw std::runtime_error("Original residual norms had a NaN");

    int it=0;
    bool converged=false;
    while( !converged )
    {
        if( commRank == 0 )
            std::cout << "  starting iteration " << it << "..." << std::endl;

        // x0 := x
        x0List = xList;

        // w := inv(M) w
        // beta := ||w||_2
        {
#ifndef RELEASE
            mpi::Barrier( comm_ );
#endif
            if( commRank == 0 )
            {
                std::cout << "  startup preconditioner application...";
                std::cout.flush();
            }
            const double startTime = mpi::Time();
            Precondition( wList );
#ifndef RELEASE
            mpi::Barrier( comm_ );
#endif
            const double stopTime = mpi::Time();
            if( commRank == 0 )
                std::cout << stopTime-startTime << " secs" << std::endl;
        }
        Norms( wList, betaList );
        const bool betaListHasNaN = CheckForNaN( betaList );
        if( betaListHasNaN )
            throw std::runtime_error("beta list had a NaN");

        // v0 := w / beta
        Matrix<C> v0List;
        v0List.View( VInter, 0, 0, localHeight, numRhs );
        v0List = wList;
        DivideColumns( v0List, betaList );

        // z := beta e_0
        elem::MakeZeros( zList );
        for( int k=0; k<numRhs; ++k )
            zList.Set(0,k,betaList[k]);

        for( int j=0; j<m; ++j )
        {
            // w := A v_j
            Matrix<C> vjList;
            vjList.LockedView( VInter, 0, j*numRhs, localHeight, numRhs );
            wList = vjList;
            {
#ifndef RELEASE
                mpi::Barrier( comm_ );
#endif
                if( commRank == 0 )
                {
                    std::cout << "    multiplying...";
                    std::cout.flush();
                }
                const double startTime = mpi::Time();
                Multiply( wList );
#ifndef RELEASE
                mpi::Barrier( comm_ );
#endif
                const double stopTime = mpi::Time();
                if( commRank == 0 )
                    std::cout << stopTime-startTime << " secs" << std::endl;
            }
#ifndef RELEASE
            Norms( wList, deltaList );
            const bool multiplyHasNaN = CheckForNaN( deltaList );
            if( multiplyHasNaN )
                throw std::runtime_error("multiply had a NaN");
#endif

            // w := inv(M) w
            {
                if( commRank == 0 )
                {
                    std::cout << "    preconditioning...";
                    std::cout.flush();
                }
                const double startTime = mpi::Time();
                Precondition( wList );
#ifndef RELEASE
                mpi::Barrier( comm_ );
#endif
                const double stopTime = mpi::Time();
                if( commRank == 0 )
                    std::cout << stopTime-startTime << " secs" << std::endl;
            }
#ifndef RELEASE
            Norms( wList, deltaList );
            const bool preconditionHasNaN = CheckForNaN( deltaList );
            if( preconditionHasNaN )
                throw std::runtime_error("precondition had a NaN");
#endif

            // Run the j'th step of Arnoldi
            {
                if( commRank == 0 )
                {
                    std::cout << "    Arnoldi step...";
                    std::cout.flush();
                }
                const double startTime = mpi::Time();
                for( int i=0; i<=j; ++i )
                {
                    // H(i,j) := v_i' w
                    Matrix<C> viList;
                    viList.LockedView
                    ( VInter, 0, i*numRhs, localHeight, numRhs );
                    InnerProducts( viList, wList, alphaList );
                    for( int k=0; k<numRhs; ++k )
                        HList.Set(i,j+k*m,alphaList[k]);
#ifdef PRINT_RITZ_VALUES
                    if( firstBatch )
                        for( int k=0; k<numRhs; ++k )
                            HListCopy.Set(i,j+k*m,alphaList[k]);
#endif

                    // w := w - H(i,j) v_i
                    SubtractScaledColumns( alphaList, viList, wList );
                }
                Norms( wList, deltaList );
                const bool deltaListHasNaN = CheckForNaN( deltaList );
                if( deltaListHasNaN )
                    throw std::runtime_error("delta list had a NaN");
                // TODO: Handle "lucky breakdown" much more carefully
                const bool zeroDelta = CheckForZero( deltaList );
                if( zeroDelta )
                {
                    if( commRank == 0 ) 
                        std::cout 
                            << "GMRES halted due to a (usually) lucky "
                               "breakdown, but this is trickier for multiple "
                               "right-hand sides." << std::endl;
                    return;
                }
                if( j+1 != m )
                {
                    Matrix<C> vjp1List;
                    vjp1List.View
                    ( VInter, 0, (j+1)*numRhs, localHeight, numRhs );
                    vjp1List = wList;
                    DivideColumns( vjp1List, deltaList );
#ifdef PRINT_RITZ_VALUES
                    if( firstBatch )
                        for( int k=0; k<numRhs; ++k )
                            HListCopy.Set(j+1,j+k*m,deltaList[k]);
#endif
                }
#ifndef RELEASE
                mpi::Barrier( comm_ );
#endif
                const double stopTime = mpi::Time();
                if( commRank == 0 )
                    std::cout << stopTime-startTime << " secs" << std::endl;
            }

            // Apply the previous rotations to the new column of each H
            {
                if( commRank == 0 )
                {
                    std::cout << "    applying previous rotations...";
                    std::cout.flush();
                }
                const double startTime = mpi::Time();
                for( int k=0; k<numRhs; ++k )
                {
                    Matrix<C> H;
                    H.View( HList, 0, k*m, j+1, j+1 );
                    for( int i=0; i<j; ++i )
                    {
                        const R c = csList.Get(i,k);
                        const C s = snList.Get(i,k);
                        const C sConj = Conj(s);
                        const C eta_i_j = H.Get(i,j);
                        const C eta_ip1_j = H.Get(i+1,j);
                        H.Set( i,   j,  c    *eta_i_j + s*eta_ip1_j );
                        H.Set( i+1, j, -sConj*eta_i_j + c*eta_ip1_j );
                    }
                }
#ifndef RELEASE
                mpi::Barrier( comm_ );
#endif
                const double stopTime = mpi::Time();
                if( commRank == 0 )
                    std::cout << stopTime-startTime << " secs" << std::endl;
            }

            // Generate the new rotation and apply it to our current column
            // and to z, the rotated beta*e_0, then solve for the residual 
            // minimizer
            {
                if( commRank == 0 )
                {
                    std::cout << "    rotating and minimizing residual...";
                    std::cout.flush();
                }
                const double startTime = mpi::Time();
                for( int k=0; k<numRhs; ++k )
                {
                    // Apply the rotation to the new column of H
                    Matrix<C> H;
                    H.View( HList, 0, k*m, j+1, j+1 );
                    const C eta_j_j = H.Get(j,j);
                    const C eta_jp1_j = deltaList[k];
                    if( CheckForNaN(eta_j_j) )
                        throw std::runtime_error("H(j,j) was NaN");
                    if( CheckForNaN(eta_jp1_j) )
                        throw std::runtime_error("H(j+1,j) was NaN");
                    R c;
                    C s, rho;
                    lapack::ComputeGivens
                    ( eta_j_j, eta_jp1_j, &c, &s, &rho );
                    if( CheckForNaN(c) )
                        throw std::runtime_error("c in Givens was NaN");
                    if( CheckForNaN(s) )
                        throw std::runtime_error("s in Givens was NaN");
                    if( CheckForNaN(rho) )
                        throw std::runtime_error("rho in Givens was NaN");
                    H.Set(j,j,rho);
                    csList.Set(j,k,c);
                    snList.Set(j,k,s);

                    // Apply the rotation to z
                    const C sConj = Conj(s);
                    const C zeta_j = zList.Get(j,k);
                    const C zeta_jp1 = zList.Get(j+1,k);
                    zList.Set( j,   k,  c    *zeta_j + s*zeta_jp1 );
                    zList.Set( j+1, k, -sConj*zeta_j + c*zeta_jp1 );

                    // Minimize the residual
                    Matrix<C> y, z;
                    z.LockedView( zList, 0, k, j+1, 1 );
                    y = z;
                    elem::Trsv( UPPER, NORMAL, NON_UNIT, H, y );

                    // x := x0 + Vj y
                    Matrix<C> x, x0, vi;
                    x.View(         xList, 0, k, localHeight, 1 );
                    x0.LockedView( x0List, 0, k, localHeight, 1 );
                    x = x0;
                    for( int i=0; i<=j; ++i )
                    {
                        const C eta_i = y.Get(i,0);
                        vi.LockedView( VInter, 0, i*numRhs+k, localHeight, 1 );
                        elem::Axpy( eta_i, vi, x );
                    }
                }
#ifndef RELEASE
                mpi::Barrier( comm_ );
#endif
                const double stopTime = mpi::Time();
                if( commRank == 0 )
                    std::cout << stopTime-startTime << " secs" << std::endl;
            }

            // w := b - A x
            wList = xList; 
            elem::Scal( (C)-1, wList );
            {
#ifndef RELEASE
                mpi::Barrier( comm_ );
#endif
                if( commRank == 0 )
                {
                    std::cout << "    residual multiply...";
                    std::cout.flush();
                }
                const double startTime = mpi::Time();
                Multiply( wList );
#ifndef RELEASE
                mpi::Barrier( comm_ );
#endif
                const double stopTime = mpi::Time();
                if( commRank == 0 )
                    std::cout << stopTime-startTime << " secs" << std::endl;
            }
            elem::Axpy( (C)1, bList, wList );

            // Residual checks
            Norms( wList, residNormList );
            const bool residNormListHasNaN = CheckForNaN( residNormList );
            if( residNormListHasNaN )
                throw std::runtime_error("resid norm list has NaN");
            for( int k=0; k<numRhs; ++k )
                relResidNormList[k] = residNormList[k]/origResidNormList[k];
            R maxRelResidNorm = 0;
            for( int k=0; k<numRhs; ++k )
                maxRelResidNorm = std::max(maxRelResidNorm,relResidNormList[k]);
            if( maxRelResidNorm < relTol )
            {
                if( commRank == 0 )
                    std::cout << "  converged with relative tolerance: " 
                              << maxRelResidNorm << std::endl;
                converged = true;
                ++it;
                break;
            }
            else
            {
                if( commRank == 0 )
                {
                    std::cout << "  finished iteration " << it << " with "
                              << "maxRelResidNorm=" << maxRelResidNorm << "\n";
                    for( int k=0; k<numRhs; ++k )
                        std::cout << "    rel. residual " << k << ": " 
                                  << relResidNormList[k] << "\n";
                    std::cout.flush();
                }
            }
            if( viewIterates )
            {
                PushRightHandSides( gridB, xList );
                std::ostringstream os;
                os << "iterates-" << it;
                gridB.WriteVolume( os.str() );
            }
            ++it;
        }

#ifdef PRINT_RITZ_VALUES
        if( firstBatch && commRank == 0 )
        {
            std::vector<C> ritzVals( it );
            for( int k=0; k<numRhs; ++k )
            {
                Matrix<C> H;
                H.View( HListCopy, 0, k*m, it, it );
                elem::lapack::HessenbergEig
                ( it, H.Buffer(), H.LDim(), &ritzVals[0] );

                std::cout << "Ritz values for " << k 
                          << "'th right-hand side before first restart:\n";
                for( int s=0; s<it; ++s ) 
                    std::cout << ritzVals[s] << " ";
                std::cout << std::endl;
            }
        }
        firstBatch = false;
#endif
    }
    bList = xList;
}

template<typename R>
bool
DistHelmholtz<R>::CheckForNaN( R alpha ) const
{
    return alpha != alpha; // hopefully this is not optimized away
}

template<typename R>
bool
DistHelmholtz<R>::CheckForNaN( C alpha ) const
{
    return alpha != alpha; // hopefully this is not optimized away
}

template<typename R>
bool
DistHelmholtz<R>::CheckForNaN( const std::vector<R>& alphaList ) const
{
    bool foundNaN = false;
    for( unsigned k=0; k<alphaList.size(); ++k )
        if( CheckForNaN(alphaList[k]) )
            foundNaN = true;
    return foundNaN;
}

template<typename R>
bool
DistHelmholtz<R>::CheckForNaN( const std::vector<C>& alphaList ) const
{
    bool foundNaN = false;
    for( unsigned k=0; k<alphaList.size(); ++k )
        if( CheckForNaN(alphaList[k]) )
            foundNaN = true;
    return foundNaN;
}

template<typename R>
bool
DistHelmholtz<R>::CheckForZero( const std::vector<R>& alphaList ) const
{
    bool foundZero = false;
    for( unsigned k=0; k<alphaList.size(); ++k )
        if( alphaList[k] == (R)0 ) // think about using a tolerance instead
            foundZero = true;
    return foundZero;
}

template<typename R>
bool
DistHelmholtz<R>::CheckForZero( const std::vector<C>& alphaList ) const
{
    bool foundZero = false;
    for( unsigned k=0; k<alphaList.size(); ++k )
        if( alphaList[k] == (C)0 ) // think about using a tolerance instead
            foundZero = true;
    return foundZero;
}

template<typename R>
void
DistHelmholtz<R>::Norms
( const Matrix<C>& xList, std::vector<R>& normList ) const
{
    const int numCols = xList.Width();
    const int localHeight = xList.Height();
    const int commSize = mpi::CommSize( comm_ );
    std::vector<R> localNorms( numCols );
    for( int j=0; j<numCols; ++j )
        localNorms[j] = blas::Nrm2( localHeight, xList.LockedBuffer(0,j), 1 );
    std::vector<R> allLocalNorms( numCols*commSize );
    mpi::AllGather
    ( &localNorms[0], numCols, &allLocalNorms[0], numCols, comm_ );
    normList.resize( numCols );
    for( int j=0; j<numCols; ++j )
        normList[j] = blas::Nrm2( commSize, &allLocalNorms[j], numCols );
}

template<typename R>
void
DistHelmholtz<R>::InnerProducts
( const Matrix<C>& xList, const Matrix<C>& yList,
  std::vector<C>& alphaList ) const
{
    const int numCols = xList.Width();
    const int localHeight = xList.Height();
    std::vector<C> localAlphaList( numCols );
    for( int j=0; j<numCols; ++j )
        localAlphaList[j] = 
            blas::Dot
            ( localHeight, xList.LockedBuffer(0,j), 1,
                           yList.LockedBuffer(0,j), 1 );
    alphaList.resize( numCols );
    mpi::AllReduce
    ( &localAlphaList[0], &alphaList[0], numCols, MPI_SUM, comm_ );
}

template<typename R>
void
DistHelmholtz<R>::DivideColumns
( Matrix<C>& xList, const std::vector<R>& deltaList ) const
{
    const R one = 1;
    const int numCols = xList.Width();
    const int localHeight = xList.Height();
    for( int j=0; j<numCols; ++j )
    {
        const R invDelta = one/deltaList[j];
        C* x = xList.Buffer(0,j);
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
            x[iLocal] *= invDelta;
    }
}

template<typename R>
void
DistHelmholtz<R>::MultiplyColumns
( Matrix<C>& xList, const std::vector<C>& deltaList ) const
{
    const int numCols = xList.Width();
    const int localHeight = xList.Height();
    for( int j=0;j<numCols; ++j )
    {
        const C delta = deltaList[j];
        C* x = xList.Buffer(0,j);
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
            x[iLocal] *= delta;
    }
}

template<typename R>
void
DistHelmholtz<R>::SubtractScaledColumns
( const std::vector<C>& deltaList, 
  const Matrix<C>& xList, Matrix<C>& yList ) const
{
    const int numCols = xList.Width();
    const int localHeight = xList.Height();
    for( int j=0; j<numCols; ++j )
    {
        const C delta = deltaList[j];
        const C* x = xList.LockedBuffer(0,j);
        C* y = yList.Buffer(0,j);
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
            y[iLocal] -= delta*x[iLocal];
    }
}

// B := A B
template<typename R>
void
DistHelmholtz<R>::Multiply( Matrix<C>& B ) const
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
    cliq::SparseAllToAll
    ( sendRhs, sendCounts, sendDispls,
      recvRhs, recvCounts, recvDispls, comm_ );
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
DistHelmholtz<R>::Precondition( Matrix<C>& B ) const
{
    // Apply the sweeping preconditioner
    //
    // Simple algorithm:
    //   // Solve against L
    //   for i=0,...,m-2
    //     B_{i+1} := B_{i+1} - A_{i+1,i} T_i B_i
    //   end
    //   // Solve against D 
    //   for i=0,...,m-1
    //     B_i := T_i B_i
    //   end
    //   // Solve against L^T
    //   for i=m-2,...,0
    //     B_i := B_i - T_i A_{i,i+1} B_{i+1}
    //   end
    //
    // Practical algorithm:
    //   // Solve against L D
    //   for i=0,...,m-2
    //     B_i := T_i B_i
    //     B_{i+1} := B_{i+1} - A_{i+1,i} B_i
    //   end
    //   B_{m-1} := T_{m-1} B_{m-1}
    //   // Solve against L^T
    //   for i=m-2,...,0
    //     Z := B_i
    //     B_i := -A_{i,i+1} B_{i+1}
    //     B_i := T_i B_i
    //     B_i := B_i + Z
    //   end
    //

    // Solve against L D
    for( int i=0; i<numPanels_-1; ++i )
    {
        SolvePanel( B, i );
        SubdiagonalUpdate( B, i );
    }
    SolvePanel( B, numPanels_-1 );

    // Solve against L^T
    Matrix<C> Z;
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
DistHelmholtz<R>::SolvePanel( Matrix<C>& B, int i ) const
{
    const cliq::DistSymmInfo& info = PanelAnalysis( i );
    const int numRhs = B.Width();
    const int panelPadding = PanelPadding( i );
    const int panelDepth = PanelDepth( i );
    const int localHeight1d = 
        info.distNodes.back().localOffset1d + 
        info.distNodes.back().localSize1d;

    Matrix<C> localPanelB;
    elem::Zeros( localHeight1d, numRhs, localPanelB );

    // For each node, pull in each right-hand side with a memcpy
    int BOffset = LocalPanelOffset( i );
    const int numLocalNodes = info.localNodes.size();
    for( int t=0; t<numLocalNodes; ++t )
    {
        const cliq::LocalSymmNodeInfo& node = info.localNodes[t];
        const int size = node.size;
        const int myOffset = node.myOffset;

#ifndef RELEASE
        if( size % (panelPadding+panelDepth) != 0 )
            throw std::logic_error("Local node size problem");
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
    const int numDistNodes = info.distNodes.size();
    for( int t=1; t<numDistNodes; ++t )
    {
        const cliq::DistSymmNodeInfo& node = info.distNodes[t];
        const int size = node.size;
        const int localOffset1d = node.localOffset1d;
        const int localSize1d = node.localSize1d;

        const Grid& grid = *node.grid;
        const int gridSize = grid.Size();
        const int gridRank = grid.VCRank();

#ifndef RELEASE
        if( size % (panelPadding+panelDepth) != 0 )
            throw std::logic_error("Dist node size problem");
#endif
        const int xySize = size/(panelPadding+panelDepth);
        const int paddingSize = xySize*panelPadding;
        const int localPaddingSize = 
            LocalLength( paddingSize, gridRank, gridSize );
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
        throw std::logic_error("Invalid BOffset usage in pull");
#endif

    // Solve against the panel
    if( panelScheme_ == COMPRESSED_BLOCK_LDL_2D )
        CompressedBlockLDLSolve
        ( info, PanelCompressedFactorization( i ), localPanelB );
    else
        cliq::Solve( info, PanelFactorization( i ), localPanelB );

    // For each node, extract each right-hand side with memcpy
    BOffset = LocalPanelOffset( i );
    for( int t=0; t<numLocalNodes; ++t )
    {
        const cliq::LocalSymmNodeInfo& node = info.localNodes[t];
        const int size = node.size;
        const int myOffset = node.myOffset;

#ifndef RELEASE
        if( size % (panelPadding+panelDepth) != 0 )
            throw std::logic_error("Local node size problem");
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
    for( int t=1; t<numDistNodes; ++t )
    {
        const cliq::DistSymmNodeInfo& node = info.distNodes[t];
        const int size = node.size;
        const int localOffset1d = node.localOffset1d;
        const int localSize1d = node.localSize1d;

        const Grid& grid = *node.grid;
        const int gridSize = grid.Size();
        const int gridRank = grid.VCRank();

#ifndef RELEASE
        if( size % (panelPadding+panelDepth) != 0 )
            throw std::logic_error("Dist node size problem");
#endif
        const int xySize = size/(panelPadding+panelDepth);
        const int paddingSize = xySize*panelPadding;
        const int localPaddingSize = 
            LocalLength( paddingSize, gridRank, gridSize );
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
DistHelmholtz<R>::SubdiagonalUpdate( Matrix<C>& B, int i ) const
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
    cliq::SparseAllToAll
    ( sendBuffer, sendCounts, sendDispls,
      recvBuffer, recvCounts, recvDispls, comm_ );
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
// B_i := 0
template<typename R>
void
DistHelmholtz<R>::ExtractPanel( Matrix<C>& B, int i, Matrix<C>& Z ) const
{
    const int localPanelOffset = LocalPanelOffset( i );
    const int localPanelHeight = LocalPanelHeight( i );
    const int numRhs = B.Width();
    Z.ResizeTo( localPanelHeight, numRhs );

    for( int k=0; k<numRhs; ++k )
    {
        elem::MemCopy
        ( Z.Buffer(0,k), B.LockedBuffer(localPanelOffset,k), localPanelHeight );
        elem::MemZero( B.Buffer(localPanelOffset,k), localPanelHeight );
    }
}

// B_i := -A_{i,i+1} B_{i+1}
template<typename R>
void
DistHelmholtz<R>::MultiplySuperdiagonal( Matrix<C>& B, int i ) const
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
#ifndef RELEASE
        if( iLocal < LocalPanelOffset( i+1 ) )
        {
            std::cout << "s=" << s << "\n"
                      << "offset i+1=" << LocalPanelOffset(i+1) << ", \n"
                      << "iLocal=" << iLocal << std::endl;
            throw std::logic_error("Send index was too small");
        }
        if( iLocal >= LocalPanelOffset(i+1)+LocalPanelHeight(i+1) )
        {
            std::cout << "s=" << s << "\n"
                      << "offset i+1=" << LocalPanelOffset(i+1) << ", \n"
                      << "height i+1=" << LocalPanelHeight(i+1) << ", \n"
                      << "iLocal    =" << iLocal << std::endl;
            throw std::logic_error("Send index was too big");
        }
#endif
        for( int k=0; k<numRhs; ++k ) 
            sendBuffer[s*numRhs+k] = B.Get( iLocal, k );
    }
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
    cliq::SparseAllToAll
    ( sendBuffer, sendCounts, sendDispls,
      recvBuffer, recvCounts, recvDispls, comm_ );
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
DistHelmholtz<R>::UpdatePanel
( Matrix<C>& B, int i, const Matrix<C>& Z ) const
{
    const int localPanelOffset = LocalPanelOffset( i );
    const int localPanelHeight = LocalPanelHeight( i );
    const int numRhs = Z.Width();
    for( int k=0; k<numRhs; ++k )
        for( int s=0; s<localPanelHeight; ++s )
            B.Update( localPanelOffset+s, k, Z.Get(s,k) );
}

} // namespace psp