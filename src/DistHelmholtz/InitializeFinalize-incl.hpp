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

// The localSlowness is assumed to correspond to the local degrees of freedom 
// of the panels, with the front panel first.
template<typename R>
void
psp::DistHelmholtz<R>::Initialize( const GridData<R>& slowness )
{
    const int nx = control_.nx;
    const int ny = control_.ny;
    const int nz = control_.nz;
    const int xStride = slowness.XStride();
    const int yStride = slowness.YStride();
    const int zStride = slowness.ZStride();
    const int commSize = mpi::CommSize( comm_ );
    elemental::mpi::Comm slownessComm = slowness.Comm();
    if( !elemental::mpi::CongruentComms( comm_, slownessComm ) )
        throw std::logic_error("Slowness does not have a congruent comm");

    //
    // Initialize and factor the top panel (first, since it is the largest)
    //
    {
        // Retrieve the slowness for this panel
        const int vOffset = bottomDepth_ + innerDepth_ - bzCeil_;
        const int vSize = topOrigDepth_ + bzCeil_;
        std::vector<R> recvSlowness;
        std::vector<int> recvOffsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelSlowness
        ( vOffset, vSize, topSymbolicFact_, slowness,
          recvSlowness, recvOffsets, 
          panelNestedToNatural, panelNaturalToNested );

        // Initialize the fronts with the original sparse matrix
        FillPanelFronts
        ( vOffset, vSize, topSymbolicFact_, topFact_,
          slowness, recvSlowness, recvOffsets, 
          panelNestedToNatural, panelNaturalToNested );

        // Compute the sparse-direct LDL^T factorization
        clique::numeric::LDL( clique::TRANSPOSE, topSymbolicFact_, topFact_ );

        // Redistribute the LDL^T factorization for faster solves
        clique::numeric::SetSolveMode( topFact_, clique::FEW_RHS );
    }

    //
    // Initialize and factor the bottom panel
    //
    {
        // Retrieve the slowness for this panel
        const int vOffset = 0;
        const int vSize = bottomDepth_;
        std::vector<R> recvSlowness;
        std::vector<int> recvOffsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelSlowness
        ( vOffset, vSize, bottomSymbolicFact_, slowness,
          recvSlowness, recvOffsets,
          panelNestedToNatural, panelNaturalToNested );

        // Initialize the fronts with the original sparse matrix
        FillPanelFronts
        ( vOffset, vSize, bottomSymbolicFact_, bottomFact_,
          slowness, recvSlowness, recvOffsets,
          panelNestedToNatural, panelNaturalToNested );

        // Compute the sparse-direct LDL^T factorization
        clique::numeric::LDL
        ( clique::TRANSPOSE, bottomSymbolicFact_, bottomFact_ );

        // Redistribute the LDL^T factorization for faster solves
        clique::numeric::SetSolveMode( bottomFact_, clique::FEW_RHS );
    }

    //
    // Initialize and factor the full inner panels
    //
    for( int k=0; k<numFullInnerPanels_; ++k )
    {
        // Retrieve the slowness for this panel
        const int numPlanesPerPanel = control_.numPlanesPerPanel;
        const int vOffset = bottomDepth_ + k*numPlanesPerPanel - bzCeil_;
        const int vSize = numPlanesPerPanel + bzCeil_;
        std::vector<R> recvSlowness;
        std::vector<int> recvOffsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelSlowness
        ( vOffset, vSize, fullInnerSymbolicFact_, slowness,
          recvSlowness, recvOffsets, 
          panelNestedToNatural, panelNaturalToNested );

        // Initialize the fronts with the original sparse matrix
        clique::numeric::SymmFrontTree<C>& fullInnerFact = *fullInnerFacts_[k];
        FillPanelFronts
        ( vOffset, vSize, fullInnerSymbolicFact_, fullInnerFact,
          slowness, recvSlowness, recvOffsets,
          panelNestedToNatural, panelNaturalToNested );

        // Compute the sparse-direct LDL^T factorization of the k'th inner panel
        clique::numeric::LDL
        ( clique::TRANSPOSE, fullInnerSymbolicFact_, fullInnerFact );

        // Redistribute the LDL^T factorization for faster solves
        clique::numeric::SetSolveMode( fullInnerFact, clique::FEW_RHS );
    }

    //
    // Initialize and factor the leftover inner panel (if it exists)
    //
    if( haveLeftover_ )
    {        
        // Retrieve the slowness for this panel
        const int vOffset = bottomDepth_ + innerDepth_ - 
                            leftoverInnerDepth_ - bzCeil_;
        const int vSize = leftoverInnerDepth_ + bzCeil_;
        std::vector<R> recvSlowness;
        std::vector<int> recvOffsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelSlowness
        ( vOffset, vSize, leftoverInnerSymbolicFact_, slowness,
          recvSlowness, recvOffsets, 
          panelNestedToNatural, panelNaturalToNested );

        // Initialize the fronts with the original sparse matrix
        FillPanelFronts
        ( vOffset, vSize, leftoverInnerSymbolicFact_, leftoverInnerFact_,
          slowness, recvSlowness, recvOffsets,
          panelNestedToNatural, panelNaturalToNested );

        // Compute the sparse-direct LDL^T factorization of the leftover panel
        clique::numeric::LDL
        ( clique::TRANSPOSE, leftoverInnerSymbolicFact_, leftoverInnerFact_ );

        // Redistribute the LDL^T factorization for faster solves
        clique::numeric::SetSolveMode( leftoverInnerFact_, clique::FEW_RHS );
    }
    
    //
    // Initialize the global sparse matrix
    //

    // Gather the slowness for the global sparse matrix
    std::vector<R> recvSlowness;
    std::vector<int> recvOffsets;
    GetGlobalSlowness( slowness, recvSlowness, recvOffsets );

    // Now make use of the redistributed slowness data to form the global 
    // sparse matrix
    localEntries_.resize( localRowOffsets_.back() );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int x = naturalIndex % nx;
        const int y = (naturalIndex/nx) % ny;
        const int z = naturalIndex/(nx*ny);
        // TODO: Replace with GridData call which allows for different orderings
        const int xProc = x % xStride;
        const int yProc = y % yStride;
        const int zProc = z % zStride;
        const int proc = xProc + yProc*xStride + zProc*xStride*yStride;

        const R alpha = recvSlowness[++recvOffsets[proc]];
        const int rowOffset = localRowOffsets_[iLocal];
        const int v = (nz-1) - z;
        FormGlobalRow( alpha, x, y, v, rowOffset );
    }
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

    // Fill in the connections
    std::vector<int>::const_iterator first;
    frontIndices.resize( 1 );
    values.resize( 1 );
    // Center term
    frontIndices[0] = j;
    values[0] = centerTerm;
    // Left connection
    if( x > 0 )
    {
        const int naturalIndex = (x-1) + y*nx + vLocal*nx*ny;
        if( panelNaturalToNested.count(naturalIndex) )
        {
            const int nestedIndex = panelNaturalToNested[naturalIndex];
            if( nestedIndex > offset+j && nestedIndex < offset+size )
            {
                frontIndices.push_back( nestedIndex-offset ); 
                values.push_back( xTermL );
            }
            else
            {
                first = std::lower_bound
                    ( origLowerStruct.begin(), origLowerStruct.end(), 
                      nestedIndex ); 
                if( first!=origLowerStruct.end() && !(nestedIndex<*first) )
                {
                    const int whichLower = int(first-origLowerStruct.begin());
                    frontIndices.push_back( origLowerRelIndices[whichLower] );
                    values.push_back( xTermL );
                }
            }
        }
    }
    if( x+1 < nx )
    {
        const int naturalIndex = (x+1) + y*nx + vLocal*nx*ny;
        if( panelNaturalToNested.count(naturalIndex) )
        {
            const int nestedIndex = panelNaturalToNested[naturalIndex];
            if( nestedIndex > offset+j && nestedIndex < offset+size )
            {
                frontIndices.push_back( nestedIndex-offset );
                values.push_back( xTermR );
            }
            else
            {
                first = std::lower_bound
                    ( origLowerStruct.begin(), origLowerStruct.end(), 
                      nestedIndex ); 
                if( first!=origLowerStruct.end() && !(nestedIndex<*first) )
                {
                    const int whichLower = int(first-origLowerStruct.begin());
                    frontIndices.push_back( origLowerRelIndices[whichLower] );
                    values.push_back( xTermR );
                }
            }
        }
    }
    if( y > 0 )
    {
        const int naturalIndex = x + (y-1)*nx + vLocal*nx*ny;
        if( panelNaturalToNested.count(naturalIndex) )
        {
            const int nestedIndex = panelNaturalToNested[naturalIndex];
            if( nestedIndex > offset+j && nestedIndex < offset+size )
            {
                frontIndices.push_back( nestedIndex-offset );
                values.push_back( yTermL );
            }
            else
            {
                first = std::lower_bound
                    ( origLowerStruct.begin(), origLowerStruct.end(), 
                      nestedIndex ); 
                if( first!=origLowerStruct.end() && !(nestedIndex<*first) )
                {
                    const int whichLower = int(first-origLowerStruct.begin());
                    frontIndices.push_back( origLowerRelIndices[whichLower] );
                    values.push_back( yTermL );
                }
            }
        }
    }
    if( y+1 < ny )
    {
        const int naturalIndex = x + (y+1)*nx + vLocal*nx*ny;
        if( panelNaturalToNested.count(naturalIndex) )
        {
            const int nestedIndex = panelNaturalToNested[naturalIndex];
            if( nestedIndex > offset+j && nestedIndex < offset+size )
            {
                frontIndices.push_back( nestedIndex-offset );
                values.push_back( yTermR );
            }
            else
            {
                first = std::lower_bound
                    ( origLowerStruct.begin(), origLowerStruct.end(), 
                      nestedIndex ); 
                if( first!=origLowerStruct.end() && !(nestedIndex<*first) )
                {
                    const int whichLower = int(first-origLowerStruct.begin());
                    frontIndices.push_back( origLowerRelIndices[whichLower] );
                    values.push_back( yTermR );
                }
            }
        }
    }
    if( vLocal > 0 )
    {
        const int naturalIndex = x + y*nx + (vLocal-1)*nx*ny;
        if( panelNaturalToNested.count(naturalIndex) )
        {
            const int nestedIndex = panelNaturalToNested[naturalIndex];
            if( nestedIndex > offset+j && nestedIndex < offset+size )
            {
                frontIndices.push_back( nestedIndex-offset );
                values.push_back( vTermL );
            }
            else
            {
                first = std::lower_bound
                    ( origLowerStruct.begin(), origLowerStruct.end(), 
                      nestedIndex ); 
                if( first!=origLowerStruct.end() && !(nestedIndex<*first) )
                {
                    const int whichLower = int(first-origLowerStruct.begin());
                    frontIndices.push_back( origLowerRelIndices[whichLower] );
                    values.push_back( vTermL );
                }
            }
        }
    }
    if( vLocal+1 < vSize )
    {
        const int naturalIndex = x + y*nx + (vLocal+1)*nx*ny;
        if( panelNaturalToNested.count(naturalIndex) )
        {
            const int nestedIndex = panelNaturalToNested[naturalIndex];
            if( nestedIndex > offset+j && nestedIndex < offset+size )
            {
                frontIndices.push_back( nestedIndex-offset );
                values.push_back( vTermR );
            }
            else
            {
                first = std::lower_bound
                    ( origLowerStruct.begin(), origLowerStruct.end(), 
                      nestedIndex ); 
                if( first!=origLowerStruct.end() && !(nestedIndex<*first) )
                {
                    const int whichLower = int(first-origLowerStruct.begin());
                    frontIndices.push_back( origLowerRelIndices[whichLower] );
                    values.push_back( vTermR );
                }
            }
        }
    }
}
        
template<typename R>
void
psp::DistHelmholtz<R>::GetGlobalSlowness
( const GridData<R>& slowness,
        std::vector<R>& recvSlowness,
        std::vector<int>& recvOffsets ) const
{
    const int nx = control_.nx;
    const int ny = control_.ny;
    const int xShift = slowness.XShift();
    const int yShift = slowness.YShift();
    const int zShift = slowness.ZShift();
    const int xStride = slowness.XStride();
    const int yStride = slowness.YStride();
    const int zStride = slowness.ZStride();
    const int xLocalSize = slowness.XLocalSize();
    const int yLocalSize = slowness.YLocalSize();
    const R* localSlowness = slowness.LockedLocalBuffer();
    const int commSize = mpi::CommSize( comm_ );

    std::vector<int> recvPairs( 2*commSize, 0 );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int x = naturalIndex % nx;
        const int y = (naturalIndex/nx) % ny;
        const int z = naturalIndex/(nx*ny);
        const int xProc = x % xStride;
        const int yProc = y % yStride;
        const int zProc = z % zStride;
        const int proc = xProc + yProc*xStride + zProc*xStride*yStride;
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
    recvOffsets.resize( commSize );
    for( int proc=0; proc<commSize; ++proc )
        recvOffsets[proc] = maxSize*proc;
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int x = naturalIndex % nx;
        const int y = (naturalIndex/nx) % ny;
        const int z = naturalIndex/(nx*ny);
        const int xProc = x % xStride;
        const int yProc = y % yStride;
        const int zProc = z % zStride;
        const int proc = xProc + yProc*xStride + zProc*xStride*yStride;
        recvIndices[++recvOffsets[proc]] = naturalIndex;
    }
    std::vector<int> sendIndices( maxSize*commSize );
    mpi::AllToAll( &recvIndices[0], maxSize, &sendIndices[0], maxSize, comm_ );
    recvIndices.clear();
    std::vector<R> sendSlowness( maxSize*commSize );
    for( int proc=0; proc<commSize; ++proc )
    {
        R* send = &sendSlowness[proc*maxSize];
        for( int iLocal=0; iLocal<actualSendSizes[proc]; ++iLocal )
        {
            const int naturalIndex = sendIndices[iLocal];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int z = naturalIndex/(nx*ny);
            const int xLocal = (x-xShift) / xStride;
            const int yLocal = (y-yShift) / yStride;
            const int zLocal = (z-zShift) / zStride;
            const int localIndex = 
                xLocal + yLocal*xLocalSize + zLocal*xLocalSize*yLocalSize;
            send[iLocal] = localSlowness[localIndex];
        }
    }
    sendIndices.clear();
    recvSlowness.resize( maxSize*commSize );
    mpi::AllToAll
    ( &sendSlowness[0], maxSize, &recvSlowness[0], maxSize, comm_ );

    // Reset the recv offsets
    for( int proc=0; proc<commSize; ++proc )
        recvOffsets[proc] = maxSize*proc;
}

template<typename R>
void
psp::DistHelmholtz<R>::GetPanelSlowness
( int vOffset, int vSize, 
  const clique::symbolic::SymmFact& fact,
  const GridData<R>& slowness,
        std::vector<R>& recvSlowness,
        std::vector<int>& recvOffsets,
        std::map<int,int>& panelNestedToNatural,
        std::map<int,int>& panelNaturalToNested ) const
{
    const int nx = control_.nx;
    const int ny = control_.ny;
    const int nz = control_.nz;
    const int xShift = slowness.XShift();
    const int yShift = slowness.YShift();
    const int zShift = slowness.ZShift();
    const int xStride = slowness.XStride();
    const int yStride = slowness.YStride();
    const int zStride = slowness.ZStride();
    const int xLocalSize = slowness.XLocalSize();
    const int yLocalSize = slowness.YLocalSize();
    const R* localSlowness = slowness.LockedLocalBuffer();
    const int commSize = mpi::CommSize( comm_ );

    const clique::symbolic::LocalSymmFact& localFact = fact.local;
    const clique::symbolic::DistSymmFact& distFact = fact.dist;
    const int numLocalSupernodes = fact.local.supernodes.size();
    const int numDistSupernodes = fact.dist.supernodes.size();

    // Compute the reorderings for the indices in the supernodes in our 
    // local tree
    panelNestedToNatural.clear();
    panelNaturalToNested.clear();
    LocalReordering( panelNestedToNatural, vSize );
    std::map<int,int>::const_iterator it;
    for( it=panelNestedToNatural.begin(); 
         it!=panelNestedToNatural.end(); ++it )
        panelNaturalToNested[it->second] = it->first;

    // Gather the slowness data using three AllToAlls
    std::vector<int> recvPairs( 2*commSize, 0 );
    for( int t=0; t<numLocalSupernodes; ++t )
    {
        const clique::symbolic::LocalSymmFactSupernode& sn = 
            localFact.supernodes[t];
        const int size = sn.size;
        const int offset = sn.offset;
        for( int j=0; j<size; ++j )
        {
            const int naturalIndex = panelNestedToNatural[offset+j];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int v = vOffset + naturalIndex/(nx*ny);
            const int z = (nz-1) - v;
            const int xProc = x % xStride;
            const int yProc = y % yStride;
            const int zProc = z % zStride;
            const int proc = xProc + yProc*xStride + zProc*xStride*yStride;
            ++recvPairs[2*proc];
        }
    }
    for( int t=0; t<numDistSupernodes; ++t )
    {
        const clique::symbolic::DistSymmFactSupernode& sn = 
            distFact.supernodes[t];
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
            const int xProc = x % xStride;
            const int yProc = y % yStride;
            const int zProc = z % zStride;
            const int proc = xProc + yProc*xStride + zProc*xStride*yStride;
            ++recvPairs[2*proc];
        }
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
    recvOffsets.resize( commSize );
    for( int proc=0; proc<commSize; ++proc )
        recvOffsets[proc] = maxSize*proc;
    for( int t=0; t<numLocalSupernodes; ++t )
    {
        const clique::symbolic::LocalSymmFactSupernode& sn = 
            localFact.supernodes[t];
        const int size = sn.size;
        const int offset = sn.offset;
        for( int j=0; j<size; ++j )
        {
            const int naturalIndex = panelNestedToNatural[offset+j];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int v = vOffset + naturalIndex/(nx*ny);
            const int z = (nz-1) - v;
            const int xProc = x % xStride;
            const int yProc = y % yStride;
            const int zProc = z % zStride;
            const int proc = xProc + yProc*xStride + zProc*xStride*yStride;
            recvIndices[++recvOffsets[proc]] = naturalIndex;
        }
    }
    for( int t=0; t<numDistSupernodes; ++t )
    {
        const clique::symbolic::DistSymmFactSupernode& sn = 
            distFact.supernodes[t];
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
            const int xProc = x % xStride;
            const int yProc = y % yStride;
            const int zProc = z % zStride;
            const int proc = xProc + yProc*xStride + zProc*xStride*yStride;
            recvIndices[++recvOffsets[proc]] = naturalIndex;
        }
    }
    std::vector<int> sendIndices( maxSize*commSize );
    mpi::AllToAll
    ( &recvIndices[0], maxSize, &sendIndices[0], maxSize, comm_ );
    recvIndices.clear();
    std::vector<R> sendSlowness( maxSize*commSize );
    for( int proc=0; proc<commSize; ++proc )
    {
        R* send = &sendSlowness[proc*maxSize];
        for( int iLocal=0; iLocal<actualSendSizes[proc]; ++iLocal )
        {
            const int naturalIndex = sendIndices[iLocal];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int v = vOffset + naturalIndex/(nx*ny);
            const int z = (nz-1) - v;
            const int xLocal = (x-xShift) / xStride;
            const int yLocal = (y-yShift) / yStride;
            const int zLocal = (z-zShift) / zStride;
            const int localIndex =
                xLocal + yLocal*xLocalSize + zLocal*xLocalSize*yLocalSize;
            send[iLocal] = localSlowness[localIndex];
        }
    }
    sendIndices.clear();
    recvSlowness.resize( maxSize*commSize );
    mpi::AllToAll
    ( &sendSlowness[0], maxSize, &recvSlowness[0], maxSize, comm_ );
    sendSlowness.clear();

    // Reset the recv offsets
    for( int proc=0; proc<commSize; ++proc )
        recvOffsets[proc] = maxSize*proc;
}

template<typename R>        
void
psp::DistHelmholtz<R>::FillPanelFronts
( int vOffset, int vSize, 
  const clique::symbolic::SymmFact& symbFact,
        clique::numeric::SymmFrontTree<C>& fact,
  const GridData<R>& slowness,
  const std::vector<R>& recvSlowness,
        std::vector<int>& recvOffsets,
        std::map<int,int>& panelNestedToNatural,
        std::map<int,int>& panelNaturalToNested ) const
{
    const int nx = control_.nx;
    const int ny = control_.ny;
    const int nz = control_.nz;
    const int xStride = slowness.XStride();
    const int yStride = slowness.YStride();
    const int zStride = slowness.ZStride();

    // Grab a few convenience variables
    const clique::symbolic::LocalSymmFact& localSymbFact = symbFact.local;
    const clique::symbolic::DistSymmFact& distSymbFact = symbFact.dist;
    const int numLocalSupernodes = localSymbFact.supernodes.size();
    const int numDistSupernodes = distSymbFact.supernodes.size();

    // Initialize the local part of the bottom panel
    std::vector<int> frontIndices;
    std::vector<C> values;
    clique::numeric::LocalSymmFrontTree<C>& localFact = fact.local;
    localFact.fronts.resize( numLocalSupernodes );
    for( int t=0; t<numLocalSupernodes; ++t )
    {
        clique::numeric::LocalSymmFront<C>& front = localFact.fronts[t];
        const clique::symbolic::LocalSymmFactSupernode& symbSN = 
            localSymbFact.supernodes[t];

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
            const int xProc = x % xStride;
            const int yProc = y % yStride;
            const int zProc = z % zStride;
            const int proc = xProc + yProc*xStride + zProc*xStride*yStride;
            const R alpha = recvSlowness[++recvOffsets[proc]];

            // Form the j'th lower column of this supernode
            FormLowerColumnOfSupernode
            ( alpha, x, y, v, vOffset, vSize, size, offset, j,
              symbSN.origLowerStruct, symbSN.origLowerRelIndices, 
              panelNaturalToNested, frontIndices, values );
            const int numMatches = frontIndices.size();
            for( int k=0; k<numMatches; ++k )
                front.frontL.Set( frontIndices[k], j, values[k] );
        }
    }

    // Initialize the distributed part of the bottom panel
    clique::numeric::DistSymmFrontTree<C>& distFact = fact.dist;
    distFact.fronts.resize( numDistSupernodes );
    for( int t=0; t<numDistSupernodes; ++t )
    {
        clique::numeric::DistSymmFront<C>& front = distFact.fronts[t];
        const clique::symbolic::DistSymmFactSupernode& symbSN = 
            distSymbFact.supernodes[t];

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
            const int xProc = x % xStride;
            const int yProc = y % yStride;
            const int zProc = z % zStride;
            const int proc = xProc + yProc*xStride + zProc*xStride*yStride;
            const R alpha = recvSlowness[++recvOffsets[proc]];

            // Form the j'th lower column of this supernode
            FormLowerColumnOfSupernode
            ( alpha, x, y, v, vOffset, vSize, size, offset, j,
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

