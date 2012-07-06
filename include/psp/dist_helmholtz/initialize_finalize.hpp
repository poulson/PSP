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
DistHelmholtz<R>::Initialize
( const GridData<R>& velocity, PanelScheme panelScheme )
{
    if( panelScheme < 0 || panelScheme > 2 )
    {
        std::ostringstream msg;
        msg << "Invalid panelScheme value: " << panelScheme;
        throw std::logic_error( msg.str().c_str() );
    }
    panelScheme_ = panelScheme;
    if( disc_.nx != velocity.XSize() ||
        disc_.ny != velocity.YSize() ||
        disc_.nz != velocity.ZSize() )
        throw std::logic_error("Velocity grid is incorrect");
    if( !mpi::CongruentComms( comm_, velocity.Comm() ) )
        throw std::logic_error("Velocity does not have a congruent comm");
    if( velocity.NumScalars() != 1 )
        throw std::logic_error("Velocity grid should have one entry per point");
    const int commRank = mpi::CommRank( comm_ );
    const R omega = disc_.omega;
    const R wx = disc_.wx;
    const R wy = disc_.wy;
    const R wz = disc_.wz;
    const int nx = disc_.nx;
    const int ny = disc_.ny;
    const int nz = disc_.nz;
    const int bx = disc_.bx;
    const int by = disc_.by;
    const int bz = disc_.bz;

    // Compute the minimum and maximum velocities, then the characteristic 
    // wavelength and the maximum number of wavelengths in an x/y/z direction.
    R maxLocalVelocity=-1;
    const int xLocalSize = velocity.XLocalSize();
    const int yLocalSize = velocity.YLocalSize();
    const int zLocalSize = velocity.ZLocalSize();
    const int localSize = xLocalSize*yLocalSize*zLocalSize;
    const R* localVelocity = velocity.LockedLocalBuffer();
    for( int i=0; i<localSize; ++i )
        maxLocalVelocity=std::max(maxLocalVelocity,localVelocity[i]);
    R maxVelocity;
    mpi::AllReduce( &maxLocalVelocity, &maxVelocity, 1, mpi::MAX, comm_ );
    R minLocalVelocity=maxVelocity;
    for( int i=0; i<localSize; ++i )
        minLocalVelocity=std::min(minLocalVelocity,localVelocity[i]);
    R minVelocity;
    mpi::AllReduce( &minLocalVelocity, &minVelocity, 1, mpi::MIN, comm_ );
    const R medianVelocity = (minVelocity+maxVelocity) / 2;
    const R wavelength = 2.0*M_PI*medianVelocity/omega;
    const R maxDimension = std::max(std::max(wx,wy),wz);
    const R numWavelengths = maxDimension / wavelength;
    const R xSizeInWavelengths = wx / wavelength;
    const R ySizeInWavelengths = wy / wavelength;
    const R zSizeInWavelengths = wz / wavelength;
    const R xPPW = nx / xSizeInWavelengths;
    const R yPPW = ny / ySizeInWavelengths;
    const R zPPW = nz / zSizeInWavelengths;
    const R minPPW = std::min(std::min(xPPW,yPPW),zPPW);
    const int minPML = std::min(bx,std::min(by,bz));
    const R xPMLPerWave = hx_*bx / wavelength;
    const R yPMLPerWave = hy_*by / wavelength;
    const R zPMLPerWave = hz_*bz / wavelength;
    const R minPMLPerWave = 
        std::min(std::min(xPMLPerWave,yPMLPerWave),zPMLPerWave);
    if( commRank == 0 )
    {
        if( minPPW < 7 || minPML < 5 )
        {
            std::cout << "nx:                        " << nx << "\n"
                      << "ny:                        " << ny << "\n" 
                      << "nz:                        " << nz << "\n"
                      << "hx:                        " << hx_ << "\n" 
                      << "hy:                        " << hy_ << "\n"
                      << "hz:                        " << hz_ << "\n"
                      << "bx:                        " << bx << "\n" 
                      << "by:                        " << by << "\n"
                      << "bz:                        " << bz << "\n"
                      << "Min velocity:              " << minVelocity << "\n"
                      << "Max velocity:              " << maxVelocity << "\n"
                      << "Median velocity:           " << medianVelocity << "\n"
                      << "Characteristic wavelength: " << wavelength << "\n"
                      << "Max dimension:             " << maxDimension << "\n"
                      << "# of wavelengths:          " << numWavelengths << "\n"
                      << "Min points/wavelength:     " << minPPW << "\n"
                      << "Min PML-size               " << minPML << "\n"
                      << "Min PML-size/wavelength:   " << minPMLPerWave << "\n"
                      << std::endl;
            if( minPPW < 7 )
                std::cerr << "WARNING: minimum points/wavelength is very small"
                          << std::endl;
            if( minPML < 5 )
                std::cerr << "WARNING: minimum PML size is very small" 
                          << std::endl;
        }
    }

    //
    // Initialize and factor the top panel (first, since it is the largest)
    //
    {
        if( commRank == 0 )
            std::cout << "  initializing top panel..." << std::endl;
        const double startTime = mpi::Time();

        // Retrieve the velocity for this panel
        const double gatherStartTime = startTime;
        const int vOffset = bottomDepth_ + innerDepth_ - bz;
        const int vSize = topOrigDepth_ + bz;
        std::vector<R> myPanelVelocity;
        std::vector<int> offsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelVelocity
        ( vOffset, vSize, topSymbolicFact_, velocity,
          myPanelVelocity, offsets, 
          panelNestedToNatural, panelNaturalToNested );
        const double gatherStopTime = mpi::Time();

        // Initialize the fronts with the original sparse matrix
        const double fillStartTime = gatherStopTime;
        if( panelScheme == COMPRESSED_2D_BLOCK_LDL )
            FillPanelFronts
            ( vOffset, vSize, topSymbolicFact_, topCompressedFact_,
              velocity, myPanelVelocity, offsets, 
              panelNestedToNatural, panelNaturalToNested );
        else
            FillPanelFronts
            ( vOffset, vSize, topSymbolicFact_, topFact_,
              velocity, myPanelVelocity, offsets, 
              panelNestedToNatural, panelNaturalToNested );
        const double fillStopTime = mpi::Time();

        // Compute the sparse-direct LDL^T factorization
        const double ldlStartTime = fillStopTime;
        if( panelScheme == COMPRESSED_2D_BLOCK_LDL )
            CompressedBlockLDL
            ( TRANSPOSE, topSymbolicFact_, topCompressedFact_, vSize );
        else
            cliq::numeric::LDL( TRANSPOSE, topSymbolicFact_, topFact_ );
        const double ldlStopTime = mpi::Time();

        // Redistribute the LDL^T factorization for faster solves
        const double redistStartTime = ldlStopTime;
        if( panelScheme == CLIQUE_FAST_2D_LDL )
            cliq::numeric::SetSolveMode( topFact_, cliq::FAST_2D_LDL );
        else if( panelScheme == CLIQUE_NORMAL_1D )
            cliq::numeric::SetSolveMode( topFact_, cliq::NORMAL_1D );
        const double redistStopTime = mpi::Time();

        const double stopTime = redistStopTime;
        if( commRank == 0 )
        {
            std::cout << "    gather: " << gatherStopTime-gatherStartTime 
                      << " secs\n"
                      << "    fill:   " << fillStopTime-fillStartTime 
                      << " secs\n"
                      << "    ldl:    " << ldlStopTime-ldlStartTime 
                      << " secs\n"
                      << "    redist: " << redistStopTime-redistStartTime
                      << " secs\n"
                      << "    total:  " << stopTime-startTime 
                      << " secs\n" 
                      << std::endl;
        }
    }

    //
    // Initialize and factor the bottom panel
    //
    {
        if( commRank == 0 )
            std::cout << "  initializing bottom panel..." << std::endl;
        const double startTime = mpi::Time();

        // Retrieve the velocity for this panel
        const double gatherStartTime = startTime;
        const int vOffset = 0;
        const int vSize = bottomDepth_;
        std::vector<R> myPanelVelocity;
        std::vector<int> offsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelVelocity
        ( vOffset, vSize, bottomSymbolicFact_, velocity,
          myPanelVelocity, offsets,
          panelNestedToNatural, panelNaturalToNested );
        const double gatherStopTime = mpi::Time();

        // Initialize the fronts with the original sparse matrix
        const double fillStartTime = gatherStopTime;
        if( panelScheme == COMPRESSED_2D_BLOCK_LDL )
            FillPanelFronts
            ( vOffset, vSize, bottomSymbolicFact_, bottomCompressedFact_,
              velocity, myPanelVelocity, offsets,
              panelNestedToNatural, panelNaturalToNested );
        else
            FillPanelFronts
            ( vOffset, vSize, bottomSymbolicFact_, bottomFact_,
              velocity, myPanelVelocity, offsets,
              panelNestedToNatural, panelNaturalToNested );
        const double fillStopTime = mpi::Time();

        // Compute the sparse-direct LDL^T factorization
        const double ldlStartTime = fillStopTime;
        if( panelScheme == COMPRESSED_2D_BLOCK_LDL )
            CompressedBlockLDL
            ( TRANSPOSE, bottomSymbolicFact_, bottomCompressedFact_, vSize );
        else
            cliq::numeric::LDL( TRANSPOSE, bottomSymbolicFact_, bottomFact_ );
        const double ldlStopTime = mpi::Time();

        // Redistribute and/or compress the LDL^T factorization
        const double redistStartTime = ldlStopTime;
        if( panelScheme == CLIQUE_FAST_2D_LDL )
            cliq::numeric::SetSolveMode( bottomFact_, cliq::FAST_2D_LDL );
        else if( panelScheme == CLIQUE_NORMAL_1D )
            cliq::numeric::SetSolveMode( bottomFact_, cliq::NORMAL_1D );
        const double redistStopTime = mpi::Time();

        const double stopTime = redistStopTime;
        if( commRank == 0 )
        {
            std::cout << "    gather: " << gatherStopTime-gatherStartTime     
                      << " secs\n"
                      << "    fill:   " << fillStopTime-fillStartTime                       << " secs\n"
                      << "    ldl:    " << ldlStopTime-ldlStartTime
                      << " secs\n"
                      << "    redist: " << redistStopTime-redistStartTime
                      << " secs\n"
                      << "    total:  " << stopTime-startTime 
                      << " secs\n"
                      << std::endl;
        }
    }

    //
    // Initialize and factor the full inner panels
    //
    if( panelScheme == COMPRESSED_2D_BLOCK_LDL )
        fullInnerCompressedFacts_.resize( numFullInnerPanels_ );
    else
        fullInnerFacts_.resize( numFullInnerPanels_ );
    for( int k=0; k<numFullInnerPanels_; ++k )
    {
        if( commRank == 0 )
            std::cout << "  initializing inner panel " << k << " of " 
                      << numFullInnerPanels_ << "..." << std::endl;
        const double startTime = mpi::Time();

        // Retrieve the velocity for this panel
        const double gatherStartTime = startTime;
        const int vOffset = bottomDepth_ + k*numPlanesPerPanel_ - bz;
        const int vSize = numPlanesPerPanel_ + bz;
        std::vector<R> myPanelVelocity;
        std::vector<int> offsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelVelocity
        ( vOffset, vSize, bottomSymbolicFact_, velocity,
          myPanelVelocity, offsets, 
          panelNestedToNatural, panelNaturalToNested );
        const double gatherStopTime = mpi::Time();

        // Initialize the fronts with the original sparse matrix
        const double fillStartTime = gatherStopTime;
        if( panelScheme == COMPRESSED_2D_BLOCK_LDL )
        {
            fullInnerCompressedFacts_[k] = new CompressedFrontTree<C>;
            FillPanelFronts
            ( vOffset, vSize, bottomSymbolicFact_, 
              *fullInnerCompressedFacts_[k],
              velocity, myPanelVelocity, offsets,
              panelNestedToNatural, panelNaturalToNested );
        }
        else 
        {
            fullInnerFacts_[k] = new cliq::numeric::SymmFrontTree<C>;
            FillPanelFronts
            ( vOffset, vSize, bottomSymbolicFact_, *fullInnerFacts_[k],
              velocity, myPanelVelocity, offsets,
              panelNestedToNatural, panelNaturalToNested );
        }
        const double fillStopTime = mpi::Time();

        // Compute the sparse-direct LDL^T factorization of the k'th inner panel
        const double ldlStartTime = fillStopTime;
        if( panelScheme == COMPRESSED_2D_BLOCK_LDL )
            CompressedBlockLDL
            ( TRANSPOSE, bottomSymbolicFact_, 
              *fullInnerCompressedFacts_[k], vSize );
        else
            cliq::numeric::LDL
            ( TRANSPOSE, bottomSymbolicFact_, *fullInnerFacts_[k] );
        const double ldlStopTime = mpi::Time();

        // Redistribute and/or compress the LDL^T factorization
        const double redistStartTime = ldlStopTime;
        if( panelScheme == CLIQUE_FAST_2D_LDL )
            cliq::numeric::SetSolveMode
            ( *fullInnerFacts_[k], cliq::FAST_2D_LDL );
        else if( panelScheme == CLIQUE_NORMAL_1D )
            cliq::numeric::SetSolveMode
            ( *fullInnerFacts_[k], cliq::NORMAL_1D );
        const double redistStopTime = mpi::Time();

        const double stopTime = redistStopTime;
        if( commRank == 0 )
        {
            std::cout << "    gather: " << gatherStopTime-gatherStartTime     
                      << " secs\n"
                      << "    fill:   " << fillStopTime-fillStartTime 
                      << " secs\n"
                      << "    ldl:    " << ldlStopTime-ldlStartTime
                      << " secs\n"
                      << "    redist: " << redistStopTime-redistStartTime
                      << " secs\n"
                      << "    total:  " << stopTime-startTime 
                      << " secs\n"
                      << std::endl;
        }
    }

    //
    // Initialize and factor the leftover inner panel (if it exists)
    //
    if( haveLeftover_ )
    {        
        if( commRank == 0 )
            std::cout << "  initializing the leftover panel..." << std::endl;
        const double startTime = mpi::Time();

        // Retrieve the velocity for this panel
        const double gatherStartTime = startTime;
        const int vOffset = bottomDepth_ + innerDepth_ - 
                            leftoverInnerDepth_ - bz;
        const int vSize = leftoverInnerDepth_ + bz;
        std::vector<R> myPanelVelocity;
        std::vector<int> offsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelVelocity
        ( vOffset, vSize, leftoverInnerSymbolicFact_, velocity,
          myPanelVelocity, offsets, 
          panelNestedToNatural, panelNaturalToNested );
        const double gatherStopTime = mpi::Time();

        // Initialize the fronts with the original sparse matrix
        const double fillStartTime = gatherStopTime;
        if( panelScheme == COMPRESSED_2D_BLOCK_LDL )
            FillPanelFronts
            ( vOffset, vSize, leftoverInnerSymbolicFact_, 
              leftoverInnerCompressedFact_,
              velocity, myPanelVelocity, offsets,
              panelNestedToNatural, panelNaturalToNested );
        else
            FillPanelFronts
            ( vOffset, vSize, leftoverInnerSymbolicFact_, leftoverInnerFact_,
              velocity, myPanelVelocity, offsets,
              panelNestedToNatural, panelNaturalToNested );
        const double fillStopTime = mpi::Time();

        // Compute the sparse-direct LDL^T factorization of the leftover panel
        const double ldlStartTime = fillStopTime;
        if( panelScheme == COMPRESSED_2D_BLOCK_LDL )
            CompressedBlockLDL
            ( TRANSPOSE,
              leftoverInnerSymbolicFact_, leftoverInnerCompressedFact_, vSize );
        else
            cliq::numeric::LDL
            ( TRANSPOSE, leftoverInnerSymbolicFact_, leftoverInnerFact_ );
        const double ldlStopTime = mpi::Time();

        // Redistribute and/or compress the LDL^T factorization
        const double redistStartTime = ldlStopTime;
        if( panelScheme == CLIQUE_FAST_2D_LDL )
            cliq::numeric::SetSolveMode
            ( leftoverInnerFact_, cliq::FAST_2D_LDL );
        else if( panelScheme == CLIQUE_NORMAL_1D )
            cliq::numeric::SetSolveMode
            ( leftoverInnerFact_, cliq::NORMAL_1D );
        const double redistStopTime = mpi::Time();

        const double stopTime = redistStopTime;
        if( commRank == 0 )
        {
            std::cout << "    gather: " << gatherStopTime-gatherStartTime     
                      << " secs\n"
                      << "    fill:   " << fillStopTime-fillStartTime                       << " secs\n"
                      << "    ldl:    " << ldlStopTime-ldlStartTime
                      << " secs\n"
                      << "    redist: " << redistStopTime-redistStartTime
                      << " secs\n"
                      << "    total:  " << stopTime-startTime 
                      << " secs\n"
                      << std::endl;
        }
    }
    
    //
    // Initialize the global sparse matrix
    //
    {
        if( commRank == 0 )
        {
            std::cout << "  initializing global sparse matrix...";
            std::cout.flush();
        }
        const double startTime = mpi::Time();

        // Gather the velocity for the global sparse matrix
        std::vector<R> myGlobalVelocity;
        std::vector<int> offsets;
        GetGlobalVelocity( velocity, myGlobalVelocity, offsets );

        // Now make use of the redistributed velocity data to form the global 
        // sparse matrix
        localEntries_.resize( localRowOffsets_.back() );
        for( int iLocal=0; iLocal<localHeight_; ++iLocal )
        {
            const int naturalIndex = localToNaturalMap_[iLocal];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int z = naturalIndex/(nx*ny);
            const int proc = velocity.OwningProcess( x, y, z );

            const R alpha = myGlobalVelocity[offsets[proc]++];
            const int rowOffset = localRowOffsets_[iLocal];
            const int v = (nz-1) - z;
            FormGlobalRow( alpha, x, y, v, rowOffset );
        }

        const double stopTime = mpi::Time();
        if( commRank == 0 )
            std::cout << stopTime-startTime << " secs" << std::endl;
    }
}

template<typename R>
void
DistHelmholtz<R>::Finalize()
{
    // Release the global sparse matrix memory
    localEntries_.clear();

    // Release the padded panel memory
    if( panelScheme_ == COMPRESSED_2D_BLOCK_LDL )
    {
        bottomCompressedFact_.local.fronts.clear();
        bottomCompressedFact_.dist.fronts.clear();
        for( int k=0; k<numFullInnerPanels_; ++k )
            delete fullInnerCompressedFacts_[k];
        fullInnerCompressedFacts_.clear();
        leftoverInnerCompressedFact_.local.fronts.clear();
        leftoverInnerCompressedFact_.dist.fronts.clear();
        topCompressedFact_.local.fronts.clear();
        topCompressedFact_.dist.fronts.clear();
    }
    else
    {
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
}

template<typename R>
Complex<R>
DistHelmholtz<R>::s1Inv( int x ) const
{
    const int bx = disc_.bx;
    const R etax = bx*hx_;
    if( x+1 < bx && disc_.frontBC==PML )
    {
        const R delta = bx - (x+1);
        const R realPart = 1;
        const R imagPart = 
            (disc_.Cx/etax)*(delta/bx)*(delta/bx)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else if( x > (disc_.nx-bx) && disc_.backBC==PML )
    {
        const R delta = x-(disc_.nx-bx);
        const R realPart = 1;
        const R imagPart =
            (disc_.Cx/etax)*(delta/bx)*(delta/bx)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else
        return 1;
}

template<typename R>
Complex<R>
DistHelmholtz<R>::s2Inv( int y ) const
{
    const int by = disc_.by;
    const R etay = by*hy_;
    if( y+1 < by && disc_.leftBC==PML )
    {
        const R delta = by - (y+1);
        const R realPart = 1;
        const R imagPart = 
            (disc_.Cy/etay)*(delta/by)*(delta/by)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else if( y > (disc_.ny-by) && disc_.rightBC==PML )
    {
        const R delta = y-(disc_.ny-by);
        const R realPart = 1;
        const R imagPart =
            (disc_.Cy/etay)*(delta/by)*(delta/by)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else
        return 1;
}

template<typename R>
Complex<R>
DistHelmholtz<R>::s3Inv( int v ) const
{
    const int bz = disc_.bz;
    const R etaz = bz*hz_;
    if( v+1 < bz )
    {
        const R delta = bz - (v+1);
        const R realPart = 1;
        const R imagPart = 
            (disc_.Cz/etaz)*(delta/bz)*(delta/bz)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else if( v > (disc_.nz-bz) && disc_.topBC==PML )
    {
        const R delta = v - (disc_.nz-bz);
        const R realPart = 1;
        const R imagPart = 
            (disc_.Cz/etaz)*(delta/bz)*(delta/bz)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else
        return 1;
}

template<typename R>
Complex<R>
DistHelmholtz<R>::s3InvArtificial( int v, int vOffset ) const
{
    const int bz = disc_.bz;
    const R etaz = bz*hz_;
    if( v+1 < vOffset+bz )
    {
        const R delta = vOffset + bz - (v+1);
        const R realPart = 1;
        const R imagPart = 
            (disc_.Cz/etaz)*(delta/bz)*(delta/bz)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else if( v > (disc_.nz-bz) && disc_.topBC==PML )
    {
        const R delta = v - (disc_.nz-bz);
        const R realPart = 1;
        const R imagPart = 
            (disc_.Cz/etaz)*(delta/bz)*(delta/bz)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else
        return 1;
}

template<typename R>
void
DistHelmholtz<R>::FormGlobalRow
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
        (disc_.omega/alpha)*(disc_.omega/alpha)*s1InvM*s2InvM*s3InvM;

    // Fill in the center term
    int offset = row;
    localEntries_[offset++] = centerTerm;

    // Fill the rest of the terms
    if( x > 0 )
        localEntries_[offset++] = xTermL;
    if( x+1 < disc_.nx )
        localEntries_[offset++] = xTermR;
    if( y > 0 )
        localEntries_[offset++] = yTermL;
    if( y+1 < disc_.ny )
        localEntries_[offset++] = yTermR;
    if( v > 0 )
        localEntries_[offset++] = vTermL;
    if( v+1 < disc_.nz )
        localEntries_[offset++] = vTermR;
}

template<typename R>
void
DistHelmholtz<R>::FormLowerColumnOfSupernode
( R alpha, int x, int y, int v, int vOffset, int vSize, 
  int offset, int size, int j,
  const std::vector<int>& origLowerStruct, 
  const std::vector<int>& origLowerRelIndices,
        std::map<int,int>& panelNaturalToNested, 
        std::vector<int>& frontIndices, 
        std::vector<C>& values ) const
{
    // Evaluate all of the inverse s functions
    const C s1InvL = s1Inv( x-1 );
    const C s1InvM = s1Inv( x   );
    const C s1InvR = s1Inv( x+1 );
    const C s2InvL = s2Inv( y-1 );
    const C s2InvM = s2Inv( y   );
    const C s2InvR = s2Inv( y+1 );
    const C s3InvL = s3InvArtificial( v-1, vOffset );
    const C s3InvM = s3InvArtificial( v,   vOffset );
    const C s3InvR = s3InvArtificial( v+1, vOffset );

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
    const C shiftedOmega = C(disc_.omega,damping_);
    const C centerTerm = -(xTermL+xTermR+yTermL+yTermR+vTermL+vTermR) + 
        (shiftedOmega/alpha)*(shiftedOmega/alpha)*s1InvM*s2InvM*s3InvM;
    const int vLocal = v - vOffset;
    const int nx = disc_.nx;
    const int ny = disc_.ny;

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
        const int nestedIndex = ReorderedIndex( x+1, y, vLocal, vSize );
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
        const int nestedIndex = ReorderedIndex( x, y-1, vLocal, vSize );
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
        const int nestedIndex = ReorderedIndex( x, y+1, vLocal, vSize );
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
        const int nestedIndex = ReorderedIndex( x, y, vLocal-1, vSize );
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
        const int nestedIndex = ReorderedIndex( x, y, vLocal+1, vSize );
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
DistHelmholtz<R>::GetGlobalVelocity
( const GridData<R>& velocity,
        std::vector<R>& myGlobalVelocity,
        std::vector<int>& offsets ) const
{
    const int commSize = mpi::CommSize( comm_ );

    // Pack and send the amount of data that we need to recv from each process.
    std::vector<int> recvCounts( commSize, 0 );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = velocity.OwningProcess( naturalIndex );
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
        const int proc = velocity.OwningProcess( naturalIndex );
        recvIndices[offsets[proc]++] = naturalIndex;
    }
    std::vector<int> sendIndices( totalSendCount );
    mpi::AllToAll
    ( &recvIndices[0], &recvCounts[0], &recvDispls[0], 
      &sendIndices[0], &sendCounts[0], &sendDispls[0], comm_ );
    recvIndices.clear();

    // Pack and send our velocity data.
    std::vector<R> sendVelocity( totalSendCount );
    const R* localVelocity = velocity.LockedLocalBuffer();
    for( int proc=0; proc<commSize; ++proc )
    {
        R* procVelocity = &sendVelocity[sendDispls[proc]];
        const int* procIndices = &sendIndices[sendDispls[proc]];
        for( int iLocal=0; iLocal<sendCounts[proc]; ++iLocal )
        {
            const int naturalIndex = procIndices[iLocal];
            const int localIndex = velocity.LocalIndex( naturalIndex );
            procVelocity[iLocal] = localVelocity[localIndex];
        }
    }
    sendIndices.clear();

    myGlobalVelocity.resize( totalRecvCount );
    mpi::AllToAll
    ( &sendVelocity[0],     &sendCounts[0], &sendDispls[0],
      &myGlobalVelocity[0], &recvCounts[0], &recvDispls[0], comm_ );

    // Reset the offsets
    offsets = recvDispls;
}

template<typename R>
void
DistHelmholtz<R>::GetPanelVelocity
( int vOffset, int vSize, 
  const cliq::symbolic::SymmFact& fact,
  const GridData<R>& velocity,
        std::vector<R>& myPanelVelocity,
        std::vector<int>& offsets,
        std::map<int,int>& panelNestedToNatural,
        std::map<int,int>& panelNaturalToNested ) const
{
    const int nx = disc_.nx;
    const int ny = disc_.ny;
    const int nz = disc_.nz;
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
    // Gather the velocity data using three AllToAlls
    //

    // Send the amount of data that we need to recv from each process.
    std::vector<int> recvCounts( commSize, 0 );
    const int numLocalSupernodes = fact.local.supernodes.size();
    for( int t=0; t<numLocalSupernodes; ++t )
    {
        const cliq::symbolic::LocalSymmFactSupernode& sn = 
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
            const int proc = velocity.OwningProcess( x, y, z );
            ++recvCounts[proc];
        }
    }
    const int numDistSupernodes = fact.dist.supernodes.size();
    for( int t=1; t<numDistSupernodes; ++t )
    {
        const cliq::symbolic::DistSymmFactSupernode& sn = 
            fact.dist.supernodes[t];
        const cliq::Grid& grid = *sn.grid;
        const int gridCol = grid.MRRank();
        const int gridWidth = grid.Width();

        const int size = sn.size;
        const int offset = sn.offset;
        const int localWidth = LocalLength( size, gridCol, gridWidth );
        for( int jLocal=0; jLocal<localWidth; ++jLocal )
        {
            const int j = gridCol + jLocal*gridWidth;
            const int naturalIndex = panelNestedToNatural[offset+j];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int v = vOffset + naturalIndex/(nx*ny);
            const int z = (nz-1) - v;
            const int proc = velocity.OwningProcess( x, y, z );
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
        const cliq::symbolic::LocalSymmFactSupernode& sn = 
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
            const int proc = velocity.OwningProcess( x, y, z );
            recvIndices[offsets[proc]++] = naturalIndex;
        }
    }
    for( int t=1; t<numDistSupernodes; ++t )
    {
        const cliq::symbolic::DistSymmFactSupernode& sn = 
            fact.dist.supernodes[t];
        const cliq::Grid& grid = *sn.grid;
        const int gridCol = grid.MRRank();
        const int gridWidth = grid.Width();

        const int size = sn.size;
        const int offset = sn.offset;
        const int localWidth = LocalLength( size, gridCol, gridWidth );
        for( int jLocal=0; jLocal<localWidth; ++jLocal )
        {
            const int j = gridCol + jLocal*gridWidth;
            const int naturalIndex = panelNestedToNatural[offset+j];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int v = vOffset + naturalIndex/(nx*ny);
            const int z = (nz-1) - v;
            const int proc = velocity.OwningProcess( x, y, z );
            recvIndices[offsets[proc]++] = naturalIndex;
        }
    }
    std::vector<int> sendIndices( totalSendCount );
    mpi::AllToAll
    ( &recvIndices[0], &recvCounts[0], &recvDispls[0],
      &sendIndices[0], &sendCounts[0], &sendDispls[0], comm_ );
    recvIndices.clear();

    // Pack and send our velocity data.
    std::vector<R> sendVelocity( totalSendCount );
    const R* localVelocity = velocity.LockedLocalBuffer();
    for( int proc=0; proc<commSize; ++proc )
    {
        R* procVelocity = &sendVelocity[sendDispls[proc]];
        const int* procIndices = &sendIndices[sendDispls[proc]];
        for( int iLocal=0; iLocal<sendCounts[proc]; ++iLocal )
        {
            const int naturalIndex = procIndices[iLocal];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int v = vOffset + naturalIndex/(nx*ny);
            const int z = (nz-1) - v;
            const int localIndex = velocity.LocalIndex( x, y, z );
            procVelocity[iLocal] = localVelocity[localIndex];
        }
    }
    sendIndices.clear();
    myPanelVelocity.resize( totalRecvCount );
    mpi::AllToAll
    ( &sendVelocity[0],    &sendCounts[0], &sendDispls[0],
      &myPanelVelocity[0], &recvCounts[0], &recvDispls[0], comm_ );
    sendVelocity.clear();

    // Reset the offsets
    offsets = recvDispls;
}

template<typename R>
void DistHelmholtz<R>::LocalReordering
( std::map<int,int>& reordering, int vSize ) const
{
    int offset = 0;
    LocalReorderingRecursion
    ( reordering, offset,
      0, 0, disc_.nx, disc_.ny, vSize, disc_.nx, disc_.ny,
      log2CommSize_, nestedCutoff_, mpi::CommRank(comm_) );
}

template<typename R>
void DistHelmholtz<R>::LocalReorderingRecursion
( std::map<int,int>& reordering, int offset,
  int xOffset, int yOffset, int xSize, int ySize, int vSize, int nx, int ny,
  int depthTilSerial, int cutoff, unsigned commRank )
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
        const int xLeftSize = (xSize-1)/2;
        const int xRightSize = std::max(xSize-xLeftSize-1,0);
        const unsigned powerOfTwo = 1u << (depthTilSerial-1);
        const bool onLeft = 
            ( depthTilSerial==0 ? true : (commRank&powerOfTwo)==0 );
        const bool onRight =
            ( depthTilSerial==0 ? true : (commRank&powerOfTwo)!=0 );

        // Recurse on the left side
        if( onLeft )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset, yOffset, xLeftSize, ySize, vSize, nx, ny,
              nextDepthTilSerial, cutoff, commRank );
        offset += xLeftSize*ySize*vSize;

        // Recurse on the right side
        if( onRight )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset+xLeftSize+1, yOffset, xRightSize, ySize, vSize, nx, ny,
              nextDepthTilSerial, cutoff, commRank );
        offset += xRightSize*ySize*vSize;

        // Store the separator
        const int x = xOffset + xLeftSize;
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
        const int yLeftSize = (ySize-1)/2;
        const int yRightSize = std::max(ySize-yLeftSize-1,0);
        const unsigned powerOfTwo = 1u << (depthTilSerial-1);
        const bool onLeft = 
            ( depthTilSerial==0 ? true : (commRank&powerOfTwo)==0 );
        const bool onRight =
            ( depthTilSerial==0 ? true : (commRank&powerOfTwo)!=0 );

        // Recurse on the left side
        if( onLeft )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset, yOffset, xSize, yLeftSize, vSize, nx, ny,
              nextDepthTilSerial, cutoff, commRank );
        offset += xSize*yLeftSize*vSize;

        // Recurse on the right side
        if( onRight )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset, yOffset+yLeftSize+1, xSize, yRightSize, vSize, nx, ny,
              nextDepthTilSerial, cutoff, commRank );
        offset += xSize*yRightSize*vSize;

        // Store the separator
        const int y = yOffset + yLeftSize;
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
DistHelmholtz<R>::FillPanelFronts
( int vOffset, int vSize, 
  const cliq::symbolic::SymmFact& symbFact,
        cliq::numeric::SymmFrontTree<C>& fact,
  const GridData<R>& velocity,
  const std::vector<R>& myPanelVelocity,
        std::vector<int>& offsets,
        std::map<int,int>& panelNestedToNatural,
        std::map<int,int>& panelNaturalToNested ) const
{
    const int nx = disc_.nx;
    const int ny = disc_.ny;
    const int nz = disc_.nz;

    // Initialize the local portion of the panel
    std::vector<int> frontIndices;
    std::vector<C> values;
    const int numLocalSupernodes = symbFact.local.supernodes.size();
    fact.local.fronts.resize( numLocalSupernodes );
    for( int t=0; t<numLocalSupernodes; ++t )
    {
        cliq::numeric::LocalSymmFront<C>& front = fact.local.fronts[t];
        const cliq::symbolic::LocalSymmFactSupernode& symbSN = 
            symbFact.local.supernodes[t];

        // Initialize this front
        const int offset = symbSN.offset;
        const int size = symbSN.size;
        const int updateSize = symbSN.lowerStruct.size();
        const int frontSize = size + updateSize;
        elem::Zeros( frontSize, size, front.frontL );
        for( int j=0; j<size; ++j )
        {
            // Extract the velocity from the recv buffer
            const int panelNaturalIndex = panelNestedToNatural[j+offset];
            const int x = panelNaturalIndex % nx;
            const int y = (panelNaturalIndex/nx) % ny;
            const int vPanel = panelNaturalIndex/(nx*ny);
            const int v = vOffset + vPanel;
            const int z = (nz-1) - v;
            const int proc = velocity.OwningProcess( x, y, z );
            const R alpha = myPanelVelocity[offsets[proc]++];

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
    fact.dist.mode = cliq::NORMAL_2D;
    fact.dist.fronts.resize( numDistSupernodes );
    cliq::numeric::InitializeDistLeaf( symbFact, fact );
    for( int t=1; t<numDistSupernodes; ++t )
    {
        cliq::numeric::DistSymmFront<C>& front = fact.dist.fronts[t];
        const cliq::symbolic::DistSymmFactSupernode& symbSN = 
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
        elem::Zeros( frontSize, size, front.front2dL );
        const int localSize = front.front2dL.LocalWidth();
        for( int jLocal=0; jLocal<localSize; ++jLocal )
        {
            const int j = gridCol + jLocal*gridWidth;

            // Extract the velocity from the recv buffer
            const int panelNaturalIndex = panelNestedToNatural[j+offset];
            const int x = panelNaturalIndex % nx;
            const int y = (panelNaturalIndex/nx) % ny;
            const int vPanel = panelNaturalIndex/(nx*ny);
            const int v = vOffset + vPanel;
            const int z = (nz-1) - v;
            const int proc = velocity.OwningProcess( x, y, z );
            const R alpha = myPanelVelocity[offsets[proc]++];

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
                    front.front2dL.SetLocal( iLocal, jLocal, values[k] );
                }
            }
        }
    }
}

template<typename R>        
void
DistHelmholtz<R>::FillPanelFronts
( int vOffset, int vSize, 
  const cliq::symbolic::SymmFact& symbFact,
        CompressedFrontTree<C>& fact,
  const GridData<R>& velocity,
  const std::vector<R>& myPanelVelocity,
        std::vector<int>& offsets,
        std::map<int,int>& panelNestedToNatural,
        std::map<int,int>& panelNaturalToNested ) const
{
    const int nx = disc_.nx;
    const int ny = disc_.ny;
    const int nz = disc_.nz;

    // Initialize the local portion of the panel
    std::vector<int> frontIndices;
    std::vector<C> values;
    const int numLocalSupernodes = symbFact.local.supernodes.size();
    fact.local.fronts.resize( numLocalSupernodes );
    for( int t=0; t<numLocalSupernodes; ++t )
    {
        LocalCompressedFront<C>& front = fact.local.fronts[t];
        const cliq::symbolic::LocalSymmFactSupernode& symbSN = 
            symbFact.local.supernodes[t];

        // Initialize this front
        const int offset = symbSN.offset;
        const int size = symbSN.size;
        const int updateSize = symbSN.lowerStruct.size();
        const int frontSize = size + updateSize;
        elem::Zeros( frontSize, size, front.frontL );
        for( int j=0; j<size; ++j )
        {
            // Extract the velocity from the recv buffer
            const int panelNaturalIndex = panelNestedToNatural[j+offset];
            const int x = panelNaturalIndex % nx;
            const int y = (panelNaturalIndex/nx) % ny;
            const int vPanel = panelNaturalIndex/(nx*ny);
            const int v = vOffset + vPanel;
            const int z = (nz-1) - v;
            const int proc = velocity.OwningProcess( x, y, z );
            const R alpha = myPanelVelocity[offsets[proc]++];

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
    // Replacement for cliq::numeric::InitializeDistLeaf
    {
        const cliq::symbolic::DistSymmFactSupernode& sn = 
            symbFact.dist.supernodes[0];
        Matrix<C>& topLocalFrontL = fact.local.fronts.back().frontL;
        DistMatrix<C>& frontL = fact.dist.fronts[0].frontL;
        frontL.LockedView
        ( topLocalFrontL.Height(), topLocalFrontL.Width(), 0, 0,
          topLocalFrontL.LockedBuffer(), topLocalFrontL.LDim(),
          *sn.grid );
    }
    for( int t=1; t<numDistSupernodes; ++t )
    {
        DistCompressedFront<C>& front = fact.dist.fronts[t];
        const cliq::symbolic::DistSymmFactSupernode& symbSN = 
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
        front.frontL.SetGrid( grid );
        elem::Zeros( frontSize, size, front.frontL );
        const int localSize = front.frontL.LocalWidth();
        for( int jLocal=0; jLocal<localSize; ++jLocal )
        {
            const int j = gridCol + jLocal*gridWidth;

            // Extract the velocity from the recv buffer
            const int panelNaturalIndex = panelNestedToNatural[j+offset];
            const int x = panelNaturalIndex % nx;
            const int y = (panelNaturalIndex/nx) % ny;
            const int vPanel = panelNaturalIndex/(nx*ny);
            const int v = vOffset + vPanel;
            const int z = (nz-1) - v;
            const int proc = velocity.OwningProcess( x, y, z );
            const R alpha = myPanelVelocity[offsets[proc]++];

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
                    front.frontL.SetLocal( iLocal, jLocal, values[k] );
                }
            }
        }
    }
}

} // namespace psp

