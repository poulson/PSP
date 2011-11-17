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
psp::DistHelmholtz<R>::Initialize( const GridData<R>& slowness )
{
    if( !elemental::mpi::CongruentComms( comm_, slowness.Comm() ) )
        throw std::logic_error("Slowness does not have a congruent comm");
    if( slowness.NumScalars() != 1 )
        throw std::logic_error("Slowness grid should have one entry per point");

    //
    // Initialize and factor the top panel (first, since it is the largest)
    //
#ifndef RELEASE
    const int commRank = elemental::mpi::CommRank( comm_ );
    if( commRank == 0 )
    {
        std::cout << "Initializing top panel...";
        std::cout.flush();
    }
#endif
    {
        // Retrieve the slowness for this panel
        const int vOffset = bottomDepth_ + innerDepth_ - bzCeil_;
        const int vSize = topOrigDepth_ + bzCeil_;
        std::vector<R> myPanelSlowness;
        std::vector<int> offsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelSlowness
        ( vOffset, vSize, topSymbolicFact_, slowness,
          myPanelSlowness, offsets, 
          panelNestedToNatural, panelNaturalToNested );

        // Initialize the fronts with the original sparse matrix
        FillPanelFronts
        ( vOffset, vSize, topSymbolicFact_, topFact_,
          slowness, myPanelSlowness, offsets, 
          panelNestedToNatural, panelNaturalToNested );

        // Compute the sparse-direct LDL^T factorization
        clique::numeric::LDL( clique::TRANSPOSE, topSymbolicFact_, topFact_ );

        // Redistribute the LDL^T factorization for faster solves
        clique::numeric::SetSolveMode( topFact_, clique::FEW_RHS );
    }
#ifndef RELEASE
    if( commRank == 0 )
        std::cout << "done" << std::endl;
#endif

    //
    // Initialize and factor the bottom panel
    //
#ifndef RELEASE
    if( commRank == 0 )
    {
        std::cout << "Initializing bottom panel...";
        std::cout.flush();
    }
#endif
    {
        // Retrieve the slowness for this panel
        const int vOffset = 0;
        const int vSize = bottomDepth_;
        std::vector<R> myPanelSlowness;
        std::vector<int> offsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelSlowness
        ( vOffset, vSize, bottomSymbolicFact_, slowness,
          myPanelSlowness, offsets,
          panelNestedToNatural, panelNaturalToNested );

        // Initialize the fronts with the original sparse matrix
        FillPanelFronts
        ( vOffset, vSize, bottomSymbolicFact_, bottomFact_,
          slowness, myPanelSlowness, offsets,
          panelNestedToNatural, panelNaturalToNested );

        // Compute the sparse-direct LDL^T factorization
        clique::numeric::LDL
        ( clique::TRANSPOSE, bottomSymbolicFact_, bottomFact_ );

        // Redistribute the LDL^T factorization for faster solves
        clique::numeric::SetSolveMode( bottomFact_, clique::FEW_RHS );
    }
#ifndef RELEASE
    if( commRank == 0 )
        std::cout << "done" << std::endl;
#endif

    //
    // Initialize and factor the full inner panels
    //
    fullInnerFacts_.resize( numFullInnerPanels_ );
    for( int k=0; k<numFullInnerPanels_; ++k )
    {
#ifndef RELEASE
        if( commRank == 0 )
        {
            std::cout << "Initializing inner panel " << k << " of " 
                      << numFullInnerPanels_ << "...";
            std::cout.flush();
        }
#endif

        // Retrieve the slowness for this panel
        const int numPlanesPerPanel = control_.numPlanesPerPanel;
        const int vOffset = bottomDepth_ + k*numPlanesPerPanel - bzCeil_;
        const int vSize = numPlanesPerPanel + bzCeil_;
        std::vector<R> myPanelSlowness;
        std::vector<int> offsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelSlowness
        ( vOffset, vSize, bottomSymbolicFact_, slowness,
          myPanelSlowness, offsets, 
          panelNestedToNatural, panelNaturalToNested );

        // Initialize the fronts with the original sparse matrix
        fullInnerFacts_[k] = new clique::numeric::SymmFrontTree<C>;
        clique::numeric::SymmFrontTree<C>& fullInnerFact = *fullInnerFacts_[k];
        FillPanelFronts
        ( vOffset, vSize, bottomSymbolicFact_, fullInnerFact,
          slowness, myPanelSlowness, offsets,
          panelNestedToNatural, panelNaturalToNested );

        // Compute the sparse-direct LDL^T factorization of the k'th inner panel
        clique::numeric::LDL
        ( clique::TRANSPOSE, bottomSymbolicFact_, fullInnerFact );

        // Redistribute the LDL^T factorization for faster solves
        clique::numeric::SetSolveMode( fullInnerFact, clique::FEW_RHS );

#ifndef RELEASE
        if( commRank == 0 )
            std::cout << "done" << std::endl;
#endif
    }

    //
    // Initialize and factor the leftover inner panel (if it exists)
    //
    if( haveLeftover_ )
    {        
#ifndef RELEASE
        if( commRank == 0 )
        {
            std::cout << "Initializing the leftover panel...";
            std::cout.flush();
        }
#endif

        // Retrieve the slowness for this panel
        const int vOffset = bottomDepth_ + innerDepth_ - 
                            leftoverInnerDepth_ - bzCeil_;
        const int vSize = leftoverInnerDepth_ + bzCeil_;
        std::vector<R> myPanelSlowness;
        std::vector<int> offsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelSlowness
        ( vOffset, vSize, leftoverInnerSymbolicFact_, slowness,
          myPanelSlowness, offsets, 
          panelNestedToNatural, panelNaturalToNested );

        // Initialize the fronts with the original sparse matrix
        FillPanelFronts
        ( vOffset, vSize, leftoverInnerSymbolicFact_, leftoverInnerFact_,
          slowness, myPanelSlowness, offsets,
          panelNestedToNatural, panelNaturalToNested );

        // Compute the sparse-direct LDL^T factorization of the leftover panel
        clique::numeric::LDL
        ( clique::TRANSPOSE, leftoverInnerSymbolicFact_, leftoverInnerFact_ );

        // Redistribute the LDL^T factorization for faster solves
        clique::numeric::SetSolveMode( leftoverInnerFact_, clique::FEW_RHS );

#ifndef RELEASE
        if( commRank == 0 )
            std::cout << "done" << std::endl;
#endif
    }
    
    //
    // Initialize the global sparse matrix
    //

#ifndef RELEASE
    if( commRank == 0 )
    {
        std::cout << "Initializing global sparse matrix...";
        std::cout.flush();
    }
#endif

    // Gather the slowness for the global sparse matrix
    std::vector<R> myGlobalSlowness;
    std::vector<int> offsets;
    GetGlobalSlowness( slowness, myGlobalSlowness, offsets );

    // Now make use of the redistributed slowness data to form the global 
    // sparse matrix
    const int nx = control_.nx;
    const int ny = control_.ny;
    const int nz = control_.nz;
    localEntries_.resize( localRowOffsets_.back() );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int x = naturalIndex % nx;
        const int y = (naturalIndex/nx) % ny;
        const int z = naturalIndex/(nx*ny);
        const int proc = slowness.OwningProcess( x, y, z );

        const R alpha = myGlobalSlowness[offsets[proc]++];
        const int rowOffset = localRowOffsets_[iLocal];
        const int v = (nz-1) - z;
        FormGlobalRow( alpha, x, y, v, rowOffset );
    }

#ifndef RELEASE
    if( commRank == 0 )
        std::cout << "done" << std::endl;
#endif
}

template<typename R>
void
psp::DistHelmholtz<R>::Finalize()
{
    // Release the global sparse matrix memory
    localEntries_.clear();

    // Release the padded panel memory
    bottomFact_.local.fronts.clear();
    bottomFact_.dist.fronts.clear();
    for( int k=0; k<numFullInnerPanels_; ++k )
        delete fullInnerFacts_[k];
    fullInnerFacts_.clear();
    leftoverInnerFact_.local.fronts.clear();
    leftoverInnerFact_.dist.fronts.clear();
    topFact_.local.fronts.clear();
    topFact_.dist.fronts.clear();
}

template<typename R>
std::complex<R>
psp::DistHelmholtz<R>::s1Inv( int x ) const
{
    if( x+1 < bx_ && control_.frontBC==PML )
    {
        const R delta = bx_ - (x+1);
        const R realPart = 1;
        const R imagPart = 
            control_.Cx*delta*delta/(bx_*bx_*bx_*hx_*control_.omega);
        return C(realPart,imagPart);
    }
    else if( x > (control_.nx-bx_) && control_.backBC==PML )
    {
        const R delta = x-(control_.nx-bx_);
        const R realPart = 1;
        const R imagPart =
            control_.Cx*delta*delta/(bx_*bx_*bx_*hx_*control_.omega);
        return C(realPart,imagPart);
    }
    else
        return 1;
}

template<typename R>
std::complex<R>
psp::DistHelmholtz<R>::s2Inv( int y ) const
{
    if( y+1 < by_ && control_.leftBC==PML )
    {
        const R delta = by_ - (y+1);
        const R realPart = 1;
        const R imagPart = 
            control_.Cy*delta*delta/(by_*by_*by_*hy_*control_.omega);
        return C(realPart,imagPart);
    }
    else if( y > (control_.ny-by_) && control_.rightBC==PML )
    {
        const R delta = y-(control_.ny-by_);
        const R realPart = 1;
        const R imagPart =
            control_.Cy*delta*delta/(by_*by_*by_*hy_*control_.omega);
        return C(realPart,imagPart);
    }
    else
        return 1;
}

template<typename R>
std::complex<R>
psp::DistHelmholtz<R>::s3Inv( int v ) const
{
    if( v+1 < bz_ )
    {
        const R delta = bz_ - (v+1);
        const R realPart = 1;
        const R imagPart = 
            control_.Cz*delta*delta/(bz_*bz_*bz_*hz_*control_.omega);
        return C(realPart,imagPart);
    }
    else if( v > (control_.nz-bz_) && control_.topBC==PML )
    {
        const R delta = v - (control_.nz-bz_);
        const R realPart = 1;
        const R imagPart = 
            control_.Cz*delta*delta/(bz_*bz_*bz_*hz_*control_.omega);
        return C(realPart,imagPart);
    }
    else
        return 1;
}

template<typename R>
std::complex<R>
psp::DistHelmholtz<R>::s3InvArtificial( int v, int vOffset, R sizeOfPML ) const
{
    if( v+1 < vOffset+sizeOfPML )
    {
        const R delta = vOffset + sizeOfPML - (v+1);
        const R realPart = 1;
        const R imagPart = 
            control_.Cz*delta*delta/
            (sizeOfPML*sizeOfPML*sizeOfPML*hz_*control_.omega);
        return C(realPart,imagPart);
    }
    else if( v > (control_.nz-bz_) && control_.topBC==PML )
    {
        const R delta = v - (control_.nz-bz_);
        const R realPart = 1;
        const R imagPart = 
            control_.Cz*delta*delta/(bz_*bz_*bz_*hz_*control_.omega);
        return C(realPart,imagPart);
    }
    else
        return 1;
}

template<typename R>
void
psp::DistHelmholtz<R>::FormGlobalRow
( R alpha, int x, int y, int v, int row )
{
    // Evaluate all of the inverse s functions
    const C s1InvL = s1Inv( x-1 );
    const C s1InvM = s1Inv( x   );
    const C s1InvR = s1Inv( x+1 );
    const C s2InvL = s2Inv( y-1 );
    const C s2InvM = s2Inv( y   );
    const C s2InvR = s2Inv( y+1 );
    const C s3InvL = s3Inv( v-1 );
    const C s3InvM = s3Inv( v   );
    const C s3InvR = s3Inv( v+1 );

    // Compute all of the x-shifted terms
    const C xTempL = s2InvM*s3InvM/s1InvL;
    const C xTempM = s2InvM*s3InvM/s1InvM;
    const C xTempR = s2InvM*s3InvM/s1InvR;
    const C xTermL = (xTempL+xTempM) / (2*hx_*hx_);
    const C xTermR = (xTempR+xTempM) / (2*hx_*hx_);

    // Compute all of the y-shifted terms
    const C yTempL = s1InvM*s3InvM/s2InvL;
    const C yTempM = s1InvM*s3InvM/s2InvM;
    const C yTempR = s1InvM*s3InvM/s2InvR;
    const C yTermL = (yTempL+yTempM) / (2*hy_*hy_);
    const C yTermR = (yTempR+yTempM) / (2*hy_*hy_);

    // Compute all of the v-shifted terms
    const C vTempL = s1InvM*s2InvM/s3InvL;
    const C vTempM = s1InvM*s2InvM/s3InvM;
    const C vTempR = s1InvM*s2InvM/s3InvR;
    const C vTermL = (vTempL+vTempM) / (2*hz_*hz_);
    const C vTermR = (vTempR+vTempM) / (2*hz_*hz_);

    // Compute the center term
    const C centerTerm = -(xTermL+xTermR+yTermL+yTermR+vTermL+vTermR) + 
        (control_.omega*alpha)*(control_.omega*alpha)*s1InvM*s2InvM*s3InvM;

    // Fill in the center term
    int offset = row;
    localEntries_[offset++] = centerTerm;

    // Fill the rest of the terms
    if( x > 0 )
        localEntries_[offset++] = xTermL;
    if( x+1 < control_.nx )
        localEntries_[offset++] = xTermR;
    if( y > 0 )
        localEntries_[offset++] = yTermL;
    if( y+1 < control_.ny )
        localEntries_[offset++] = yTermR;
    if( v > 0 )
        localEntries_[offset++] = vTermL;
    if( v+1 < control_.nz )
        localEntries_[offset++] = vTermR;
}

template<typename R>
void
psp::DistHelmholtz<R>::FormLowerColumnOfSupernode
( R alpha, int x, int y, int v, int vOffset, int vSize, 
  int offset, int size, int j,
  const std::vector<int>& origLowerStruct, 
  const std::vector<int>& origLowerRelIndices,
        std::map<int,int>& panelNaturalToNested, 
        std::vector<int>& frontIndices, 
        std::vector<C>& values ) const
{
    const R pmlSize = bzCeil_;

    // Evaluate all of the inverse s functions
    const C s1InvL = s1Inv( x-1 );
    const C s1InvM = s1Inv( x   );
    const C s1InvR = s1Inv( x+1 );
    const C s2InvL = s2Inv( y-1 );
    const C s2InvM = s2Inv( y   );
    const C s2InvR = s2Inv( y+1 );
    const C s3InvL = s3InvArtificial( v-1, vOffset, pmlSize );
    const C s3InvM = s3InvArtificial( v,   vOffset, pmlSize );
    const C s3InvR = s3InvArtificial( v+1, vOffset, pmlSize );

    // Compute all of the x-shifted terms
    const C xTempL = s2InvM*s3InvM/s1InvL;
    const C xTempM = s2InvM*s3InvM/s1InvM;
    const C xTempR = s2InvM*s3InvM/s1InvR;
    const C xTermL = (xTempL+xTempM) / (2*hx_*hx_);
    const C xTermR = (xTempR+xTempM) / (2*hx_*hx_);

    // Compute all of the y-shifted terms
    const C yTempL = s1InvM*s3InvM/s2InvL;
    const C yTempM = s1InvM*s3InvM/s2InvM;
    const C yTempR = s1InvM*s3InvM/s2InvR;
    const C yTermL = (yTempL+yTempM) / (2*hy_*hy_);
    const C yTermR = (yTempR+yTempM) / (2*hy_*hy_);

    // Compute all of the v-shifted terms
    const C vTempL = s1InvM*s2InvM/s3InvL;
    const C vTempM = s1InvM*s2InvM/s3InvM;
    const C vTempR = s1InvM*s2InvM/s3InvR;
    const C vTermL = (vTempL+vTempM) / (2*hz_*hz_);
    const C vTermR = (vTempR+vTempM) / (2*hz_*hz_);

    // Compute the center term
    const C centerTerm = -(xTermL+xTermR+yTermL+yTermR+vTermL+vTermR) + 
        (control_.omega*alpha)*(control_.omega*alpha)*s1InvM*s2InvM*s3InvM + 
        C(0,control_.imagShift);
    const int vLocal = v - vOffset;
    const int nx = control_.nx;
    const int ny = control_.ny;

    // Set up the memory
    std::vector<int>::const_iterator first;
    frontIndices.reserve( 7 );
    frontIndices.resize( 1 );
    values.reserve( 7 );
    values.resize( 1 );

    // Center term
    frontIndices[0] = j;
    values[0] = centerTerm;

    // Left connection
    if( x > 0 )
    {
        const int nestedIndex = ReorderedIndex( x-1, y, vLocal, vSize );
        if( nestedIndex > offset+j )
        {
            if( nestedIndex < offset+size )
            {
                frontIndices.push_back( nestedIndex-offset ); 
            }
            else
            {
                first = std::lower_bound
                    ( origLowerStruct.begin(), origLowerStruct.end(), 
                      nestedIndex ); 
#ifndef RELEASE
                if( first == origLowerStruct.end() )
                    throw std::logic_error("Did not find original connection");
#endif
                const int whichLower = int(first-origLowerStruct.begin());
                frontIndices.push_back( origLowerRelIndices[whichLower] );
            }
            values.push_back( xTermL );
        }
    }

    // Right connection
    if( x+1 < nx )
    {
        const int nestedIndex = ReorderedIndex( x-1, y, vLocal, vSize );
        if( nestedIndex > offset+j )
        {
            if( nestedIndex < offset+size )
            {
                frontIndices.push_back( nestedIndex-offset );
            }
            else
            {
                first = std::lower_bound
                    ( origLowerStruct.begin(), origLowerStruct.end(), 
                      nestedIndex ); 
#ifndef RELEASE
                if( first == origLowerStruct.end() )
                    throw std::logic_error("Did not find original connection");
#endif
                const int whichLower = int(first-origLowerStruct.begin());
                frontIndices.push_back( origLowerRelIndices[whichLower] );
            }
            values.push_back( xTermR );
        }
    }

    // Front connection
    if( y > 0 )
    {
        const int nestedIndex = ReorderedIndex( x-1, y, vLocal, vSize );
        if( nestedIndex > offset+j )
        {
            if( nestedIndex < offset+size )
            {
                frontIndices.push_back( nestedIndex-offset );
            }
            else
            {
                first = std::lower_bound
                    ( origLowerStruct.begin(), origLowerStruct.end(), 
                      nestedIndex ); 
#ifndef RELEASE
                if( first == origLowerStruct.end() )
                    throw std::logic_error("Did not find original connection");
#endif
                const int whichLower = int(first-origLowerStruct.begin());
                frontIndices.push_back( origLowerRelIndices[whichLower] );
            }
            values.push_back( yTermL );
        }
    }

    // Back connection
    if( y+1 < ny )
    {
        const int nestedIndex = ReorderedIndex( x-1, y, vLocal, vSize );
        if( nestedIndex > offset+j )
        {
            if( nestedIndex < offset+size )
            {
                frontIndices.push_back( nestedIndex-offset );
            }
            else
            {
                first = std::lower_bound
                    ( origLowerStruct.begin(), origLowerStruct.end(), 
                      nestedIndex ); 
#ifndef RELEASE
                if( first == origLowerStruct.end() )
                    throw std::logic_error("Did not find original connection");
#endif
                const int whichLower = int(first-origLowerStruct.begin());
                frontIndices.push_back( origLowerRelIndices[whichLower] );
            }
            values.push_back( yTermR );
        }
    }

    // Bottom connection
    if( vLocal > 0 )
    {
        const int nestedIndex = ReorderedIndex( x-1, y, vLocal, vSize );
        if( nestedIndex > offset+j )
        {
            if( nestedIndex < offset+size )
            {
                frontIndices.push_back( nestedIndex-offset );
            }
            else
            {
                first = std::lower_bound
                    ( origLowerStruct.begin(), origLowerStruct.end(), 
                      nestedIndex ); 
#ifndef RELEASE
                if( first == origLowerStruct.end() )
                    throw std::logic_error("Did not find original connection");
#endif
                const int whichLower = int(first-origLowerStruct.begin());
                frontIndices.push_back( origLowerRelIndices[whichLower] );
            }
            values.push_back( vTermL );
        }
    }

    // Top connection
    if( vLocal+1 < vSize )
    {
        const int nestedIndex = ReorderedIndex( x-1, y, vLocal, vSize );
        if( nestedIndex > offset+j )
        {
            if( nestedIndex < offset+size )
            {
                frontIndices.push_back( nestedIndex-offset );
            }
            else
            {
                first = std::lower_bound
                    ( origLowerStruct.begin(), origLowerStruct.end(), 
                      nestedIndex ); 
#ifndef RELEASE
                if( first == origLowerStruct.end() )
                    throw std::logic_error("Did not find original connection");
#endif
                const int whichLower = int(first-origLowerStruct.begin());
                frontIndices.push_back( origLowerRelIndices[whichLower] );
            }
            values.push_back( vTermR );
        }
    }
}
        
template<typename R>
void
psp::DistHelmholtz<R>::GetGlobalSlowness
( const GridData<R>& slowness,
        std::vector<R>& myGlobalSlowness,
        std::vector<int>& offsets ) const
{
    const int commSize = mpi::CommSize( comm_ );

    // Pack and send the amount of data that we need to recv from each process.
    std::vector<int> recvCounts( commSize, 0 );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = slowness.OwningProcess( naturalIndex );
        ++recvCounts[proc];
    }
    std::vector<int> sendCounts( commSize );
    mpi::AllToAll
    ( &recvCounts[0], 1, 
      &sendCounts[0], 1, comm_ );

    // Compute the send and recv displacement vectors, as well as the total
    // send and recv counts
    int totalSendCount=0, totalRecvCount=0;
    std::vector<int> sendDispls( commSize ), recvDispls( commSize );
    for( int proc=0; proc<commSize; ++proc )
    {
        sendDispls[proc] = totalSendCount;
        recvDispls[proc] = totalRecvCount;
        totalSendCount += sendCounts[proc];
        totalRecvCount += recvCounts[proc];
    }

    // Pack and send the indices that we need to recv from each process.
    std::vector<int> recvIndices( totalRecvCount );
    offsets = recvDispls;
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = slowness.OwningProcess( naturalIndex );
        recvIndices[offsets[proc]++] = naturalIndex;
    }
    std::vector<int> sendIndices( totalSendCount );
    mpi::AllToAll
    ( &recvIndices[0], &recvCounts[0], &recvDispls[0], 
      &sendIndices[0], &sendCounts[0], &sendDispls[0], comm_ );
    recvIndices.clear();

    // Pack and send our slowness data.
    std::vector<R> sendSlowness( totalSendCount );
    const R* localSlowness = slowness.LockedLocalBuffer();
    for( int proc=0; proc<commSize; ++proc )
    {
        R* procSlowness = &sendSlowness[sendDispls[proc]];
        const int* procIndices = &sendIndices[sendDispls[proc]];
        for( int iLocal=0; iLocal<sendCounts[proc]; ++iLocal )
        {
            const int naturalIndex = procIndices[iLocal];
            const int localIndex = slowness.LocalIndex( naturalIndex );
            procSlowness[iLocal] = localSlowness[localIndex];
        }
    }
    sendIndices.clear();

    myGlobalSlowness.resize( totalRecvCount );
    mpi::AllToAll
    ( &sendSlowness[0],     &sendCounts[0], &sendDispls[0],
      &myGlobalSlowness[0], &recvCounts[0], &recvDispls[0], comm_ );

    // Reset the offsets
    offsets = recvDispls;
}

template<typename R>
void
psp::DistHelmholtz<R>::GetPanelSlowness
( int vOffset, int vSize, 
  const clique::symbolic::SymmFact& fact,
  const GridData<R>& slowness,
        std::vector<R>& myPanelSlowness,
        std::vector<int>& offsets,
        std::map<int,int>& panelNestedToNatural,
        std::map<int,int>& panelNaturalToNested ) const
{
    const int nx = control_.nx;
    const int ny = control_.ny;
    const int nz = control_.nz;
    const int commSize = mpi::CommSize( comm_ );

    // Compute the reorderings for the indices in the supernodes in our 
    // local tree
    panelNestedToNatural.clear();
    panelNaturalToNested.clear();
    LocalReordering( panelNestedToNatural, vSize );
    std::map<int,int>::const_iterator it;
    for( it=panelNestedToNatural.begin(); 
         it!=panelNestedToNatural.end(); ++it )
        panelNaturalToNested[it->second] = it->first;

    //
    // Gather the slowness data using three AllToAlls
    //

    // Send the amount of data that we need to recv from each process.
    std::vector<int> recvCounts( commSize, 0 );
    const int numLocalSupernodes = fact.local.supernodes.size();
    for( int t=0; t<numLocalSupernodes; ++t )
    {
        const clique::symbolic::LocalSymmFactSupernode& sn = 
            fact.local.supernodes[t];
        const int size = sn.size;
        const int offset = sn.offset;
        for( int j=0; j<size; ++j )
        {
            const int naturalIndex = panelNestedToNatural[offset+j];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int v = vOffset + naturalIndex/(nx*ny);
            const int z = (nz-1) - v;
            const int proc = slowness.OwningProcess( x, y, z );
            ++recvCounts[proc];
        }
    }
    const int numDistSupernodes = fact.dist.supernodes.size();
    for( int t=1; t<numDistSupernodes; ++t )
    {
        const clique::symbolic::DistSymmFactSupernode& sn = 
            fact.dist.supernodes[t];
        const clique::Grid& grid = *sn.grid;
        const int gridCol = grid.MRRank();
        const int gridWidth = grid.Width();

        const int size = sn.size;
        const int offset = sn.offset;
        const int localWidth = 
            elemental::LocalLength( size, gridCol, gridWidth );
        for( int jLocal=0; jLocal<localWidth; ++jLocal )
        {
            const int j = gridCol + jLocal*gridWidth;
            const int naturalIndex = panelNestedToNatural[offset+j];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int v = vOffset + naturalIndex/(nx*ny);
            const int z = (nz-1) - v;
            const int proc = slowness.OwningProcess( x, y, z );
            ++recvCounts[proc];
        }
    }
    std::vector<int> sendCounts( commSize );
    mpi::AllToAll
    ( &recvCounts[0], 1,
      &sendCounts[0], 1, comm_ );

    // Build the send and recv displacements and count the totals send and
    // recv sizes.
    int totalSendCount=0, totalRecvCount=0;
    std::vector<int> sendDispls( commSize ), recvDispls( commSize );
    for( int proc=0; proc<commSize; ++proc )
    {
        sendDispls[proc] = totalSendCount;
        recvDispls[proc] = totalRecvCount;
        totalSendCount += sendCounts[proc];
        totalRecvCount += recvCounts[proc];
    }

    // Send the indices that we need to recv from each process.
    offsets = recvDispls;
    std::vector<int> recvIndices( totalRecvCount );
    for( int t=0; t<numLocalSupernodes; ++t )
    {
        const clique::symbolic::LocalSymmFactSupernode& sn = 
            fact.local.supernodes[t];
        const int size = sn.size;
        const int offset = sn.offset;
        for( int j=0; j<size; ++j )
        {
            const int naturalIndex = panelNestedToNatural[offset+j];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int v = vOffset + naturalIndex/(nx*ny);
            const int z = (nz-1) - v;
            const int proc = slowness.OwningProcess( x, y, z );
            recvIndices[offsets[proc]++] = naturalIndex;
        }
    }
    for( int t=1; t<numDistSupernodes; ++t )
    {
        const clique::symbolic::DistSymmFactSupernode& sn = 
            fact.dist.supernodes[t];
        const clique::Grid& grid = *sn.grid;
        const int gridCol = grid.MRRank();
        const int gridWidth = grid.Width();

        const int size = sn.size;
        const int offset = sn.offset;
        const int localWidth = 
            elemental::LocalLength( size, gridCol, gridWidth );
        for( int jLocal=0; jLocal<localWidth; ++jLocal )
        {
            const int j = gridCol + jLocal*gridWidth;
            const int naturalIndex = panelNestedToNatural[offset+j];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int v = vOffset + naturalIndex/(nx*ny);
            const int z = (nz-1) - v;
            const int proc = slowness.OwningProcess( x, y, z );
            recvIndices[offsets[proc]++] = naturalIndex;
        }
    }
    std::vector<int> sendIndices( totalSendCount );
    mpi::AllToAll
    ( &recvIndices[0], &recvCounts[0], &recvDispls[0],
      &sendIndices[0], &sendCounts[0], &sendDispls[0], comm_ );
    recvIndices.clear();

    // Pack and send our slowness data.
    std::vector<R> sendSlowness( totalSendCount );
    const R* localSlowness = slowness.LockedLocalBuffer();
    for( int proc=0; proc<commSize; ++proc )
    {
        R* procSlowness = &sendSlowness[sendDispls[proc]];
        const int* procIndices = &sendIndices[sendDispls[proc]];
        for( int iLocal=0; iLocal<sendCounts[proc]; ++iLocal )
        {
            const int naturalIndex = procIndices[iLocal];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int v = vOffset + naturalIndex/(nx*ny);
            const int z = (nz-1) - v;
            const int localIndex = slowness.LocalIndex( x, y, z );
            procSlowness[iLocal] = localSlowness[localIndex];
        }
    }
    sendIndices.clear();
    myPanelSlowness.resize( totalRecvCount );
    mpi::AllToAll
    ( &sendSlowness[0],    &sendCounts[0], &sendDispls[0],
      &myPanelSlowness[0], &recvCounts[0], &recvDispls[0], comm_ );
    sendSlowness.clear();

    // Reset the offsets
    offsets = recvDispls;
}

template<typename R>
void psp::DistHelmholtz<R>::LocalReordering
( std::map<int,int>& reordering, int vSize ) const
{
    int offset = 0;
    LocalReorderingRecursion
    ( reordering, offset,
      0, 0, control_.nx, control_.ny, vSize, control_.nx, control_.ny,
      log2CommSize_, control_.cutoff, mpi::CommRank(comm_) );
}

template<typename R>
void psp::DistHelmholtz<R>::LocalReorderingRecursion
( std::map<int,int>& reordering, int offset,
  int xOffset, int yOffset, int xSize, int ySize, int vSize, int nx, int ny,
  int depthTilSerial, int cutoff, int commRank )
{
    const int nextDepthTilSerial = std::max(depthTilSerial-1,0);
    if( depthTilSerial == 0 && xSize*ySize <= cutoff )
    {
        for( int vDelta=0; vDelta<vSize; ++vDelta )
        {
            const int v = vDelta;
            for( int yDelta=0; yDelta<ySize; ++yDelta )
            {
                const int y = yOffset + yDelta;
                for( int xDelta=0; xDelta<xSize; ++xDelta )
                {
                    const int x = xOffset + xDelta;
                    const int index = x + y*nx + v*nx*ny;
                    reordering[offset++] = index;
                }
            }
        }
    }
    else if( xSize >= ySize )
    {
        //
        // Partition the X dimension
        //
        const int middle = (xSize-1)/2;
        const bool onLeft = 
            ( depthTilSerial==0 ? 
              true :
              (commRank&(1u<<(depthTilSerial-1)))==0 );
        const bool onRight =
            ( depthTilSerial==0 ?
              true :
              (commRank&(1u<<(depthTilSerial-1)))!=0 );

        // Recurse on the left side
        if( onLeft )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset, yOffset, middle, ySize, vSize, nx, ny,
              nextDepthTilSerial, cutoff, commRank/2 );
        offset += middle*ySize*vSize;

        // Recurse on the right side
        if( onRight )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset+middle+1, yOffset,
              std::max(xSize-middle-1,0), ySize, vSize, nx, ny,
              nextDepthTilSerial, cutoff, commRank/2 );
        offset += std::max(xSize-middle-1,0)*ySize*vSize;

        // Store the separator
        const int x = xOffset + middle;
        for( int vDelta=0; vDelta<vSize; ++vDelta )
        {
            const int v = vDelta;
            for( int yDelta=0; yDelta<ySize; ++yDelta )
            {
                const int y = yOffset + yDelta;
                const int index = x + y*nx + v*nx*ny;
                reordering[offset++] = index;
            }
        }
    }
    else
    {
        //
        // Partition the Y dimension
        //
        const int middle = (ySize-1)/2; 
        const bool onLeft = 
            ( depthTilSerial==0 ? 
              true :
              (commRank&(1u<<(depthTilSerial-1)))==0 );
        const bool onRight =
            ( depthTilSerial==0 ?
              true :
              (commRank&(1u<<(depthTilSerial-1)))!=0 );

        // Recurse on the left side
        if( onLeft )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset, yOffset, xSize, middle, vSize, nx, ny,
              nextDepthTilSerial, cutoff, commRank/2 );
        offset += xSize*middle*vSize;

        // Recurse on the right side
        if( onRight )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset, yOffset+middle+1,
              xSize, std::max(ySize-middle-1,0), vSize, nx, ny,
              nextDepthTilSerial, cutoff, commRank/2 );
        offset += xSize*std::max(ySize-middle-1,0)*vSize;

        // Store the separator
        const int y = yOffset + middle;
        for( int vDelta=0; vDelta<vSize; ++vDelta )
        {
            const int v = vDelta;
            for( int xDelta=0; xDelta<xSize; ++xDelta )
            {
                const int x = xOffset + xDelta;
                const int index = x + y*nx + v*nx*ny;
                reordering[offset++] = index;
            }
        }
    }
}

template<typename R>        
void
psp::DistHelmholtz<R>::FillPanelFronts
( int vOffset, int vSize, 
  const clique::symbolic::SymmFact& symbFact,
        clique::numeric::SymmFrontTree<C>& fact,
  const GridData<R>& slowness,
  const std::vector<R>& myPanelSlowness,
        std::vector<int>& offsets,
        std::map<int,int>& panelNestedToNatural,
        std::map<int,int>& panelNaturalToNested ) const
{
    const int nx = control_.nx;
    const int ny = control_.ny;
    const int nz = control_.nz;

    // Initialize the local portion of the panel
    std::vector<int> frontIndices;
    std::vector<C> values;
    const int numLocalSupernodes = symbFact.local.supernodes.size();
    fact.local.fronts.resize( numLocalSupernodes );
    for( int t=0; t<numLocalSupernodes; ++t )
    {
        clique::numeric::LocalSymmFront<C>& front = fact.local.fronts[t];
        const clique::symbolic::LocalSymmFactSupernode& symbSN = 
            symbFact.local.supernodes[t];

        // Initialize this front
        const int offset = symbSN.offset;
        const int size = symbSN.size;
        const int updateSize = symbSN.lowerStruct.size();
        const int frontSize = size + updateSize;
        front.frontL.ResizeTo( frontSize, size );
        front.frontR.ResizeTo( frontSize, updateSize );
        front.frontL.SetToZero();
        front.frontR.SetToZero();
        for( int j=0; j<size; ++j )
        {
            // Extract the slowness from the recv buffer
            const int panelNaturalIndex = panelNestedToNatural[j+offset];
            const int x = panelNaturalIndex % nx;
            const int y = (panelNaturalIndex/nx) % ny;
            const int vPanel = panelNaturalIndex/(nx*ny);
            const int v = vOffset + vPanel;
            const int z = (nz-1) - v;
            const int proc = slowness.OwningProcess( x, y, z );
            const R alpha = myPanelSlowness[offsets[proc]++];

            // Form the j'th lower column of this supernode
            FormLowerColumnOfSupernode
            ( alpha, x, y, v, vOffset, vSize, offset, size, j,
              symbSN.origLowerStruct, symbSN.origLowerRelIndices, 
              panelNaturalToNested, frontIndices, values );
            const int numMatches = frontIndices.size();
            for( int k=0; k<numMatches; ++k )
                front.frontL.Set( frontIndices[k], j, values[k] );
        }
    }

    // Initialize the distributed part of the panel
    const int numDistSupernodes = symbFact.dist.supernodes.size();
    fact.dist.fronts.resize( numDistSupernodes );
    clique::numeric::InitializeDistLeaf( symbFact, fact );
    for( int t=1; t<numDistSupernodes; ++t )
    {
        clique::numeric::DistSymmFront<C>& front = fact.dist.fronts[t];
        const clique::symbolic::DistSymmFactSupernode& symbSN = 
            symbFact.dist.supernodes[t];

        // Initialize this front
        Grid& grid = *symbSN.grid;
        const int gridHeight = grid.Height();
        const int gridWidth = grid.Width();
        const int gridRow = grid.MCRank();
        const int gridCol = grid.MRRank();
        const int offset = symbSN.offset;
        const int size = symbSN.size;
        const int updateSize = symbSN.lowerStruct.size();
        const int frontSize = size + updateSize;
        front.front2dL.SetGrid( grid );
        front.front2dR.SetGrid( grid );
        front.front2dL.ResizeTo( frontSize, size );
        front.front2dR.ResizeTo( frontSize, updateSize );
        front.front2dL.SetToZero();
        front.front2dR.SetToZero();
        const int localSize = front.front2dL.LocalWidth();
        for( int jLocal=0; jLocal<localSize; ++jLocal )
        {
            const int j = gridCol + jLocal*gridWidth;

            // Extract the slowness from the recv buffer
            const int panelNaturalIndex = panelNestedToNatural[j+offset];
            const int x = panelNaturalIndex % nx;
            const int y = (panelNaturalIndex/nx) % ny;
            const int vPanel = panelNaturalIndex/(nx*ny);
            const int v = vOffset + vPanel;
            const int z = (nz-1) - v;
            const int proc = slowness.OwningProcess( x, y, z );
            const R alpha = myPanelSlowness[offsets[proc]++];

            // Form the j'th lower column of this supernode
            FormLowerColumnOfSupernode
            ( alpha, x, y, v, vOffset, vSize, offset, size, j,
              symbSN.origLowerStruct, symbSN.origLowerRelIndices, 
              panelNaturalToNested, frontIndices, values );
            const int numMatches = frontIndices.size();
            for( int k=0; k<numMatches; ++k )
            {
                const int i = frontIndices[k];
                if( i % gridHeight == gridRow )
                {
                    const int iLocal = (i-gridRow) / gridHeight;
                    front.front2dL.SetLocalEntry
                    ( iLocal, jLocal, values[k] );
                }
            }
        }
    }
}

