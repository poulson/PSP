/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/

namespace psp {

template<typename Real>
void
DistHelmholtz<Real>::Initialize
( const DistUniformGrid<Real>& velocity, PanelScheme panelScheme )
{
    if( disc_.omega == Real(0) )
        LogicError("PML does not work at zero frequency");
    if( panelScheme < 0 || panelScheme > 2 )
        LogicError("Invalid panelScheme value: ",panelScheme);
    panelScheme_ = panelScheme;
    if( disc_.nx != velocity.XSize() ||
        disc_.ny != velocity.YSize() ||
        disc_.nz != velocity.ZSize() )
        LogicError("Velocity grid is incorrect");
    if( !mpi::CongruentComms( comm_, velocity.Comm() ) )
        LogicError("Velocity does not have a congruent comm");
    if( velocity.NumScalars() != 1 )
        LogicError("Velocity grid should have one entry per point");
    const int commRank = mpi::CommRank( comm_ );
    const Real omega = disc_.omega;
    const Real wx = disc_.wx;
    const Real wy = disc_.wy;
    const Real wz = disc_.wz;
    const int nx = disc_.nx;
    const int ny = disc_.ny;
    const int nz = disc_.nz;
    const int bx = disc_.bx;
    const int by = disc_.by;
    const int bz = disc_.bz;

    // Compute the minimum and maximum velocities, then the characteristic 
    // wavelength and the maximum number of wavelengths in an x/y/z direction.
    Real maxLocalVelocity=-1;
    const int xLocalSize = velocity.XLocalSize();
    const int yLocalSize = velocity.YLocalSize();
    const int zLocalSize = velocity.ZLocalSize();
    const int localSize = xLocalSize*yLocalSize*zLocalSize;
    const Real* localVelocity = velocity.LockedBuffer();
    for( int i=0; i<localSize; ++i )
        maxLocalVelocity=std::max(maxLocalVelocity,localVelocity[i]);
    Real maxVelocity;
    mpi::AllReduce( &maxLocalVelocity, &maxVelocity, 1, mpi::MAX, comm_ );
    Real minLocalVelocity=maxVelocity;
    for( int i=0; i<localSize; ++i )
        minLocalVelocity=std::min(minLocalVelocity,localVelocity[i]);
    Real minVelocity;
    mpi::AllReduce( &minLocalVelocity, &minVelocity, 1, mpi::MIN, comm_ );
    const Real medianVelocity = (minVelocity+maxVelocity) / 2;
    const Real wavelength = 2.0*M_PI*minVelocity/omega;
    const Real maxDimension = std::max(std::max(wx,wy),wz);
    const Real numWavelengths = maxDimension / wavelength;
    const Real xSizeInWavelengths = wx / wavelength;
    const Real ySizeInWavelengths = wy / wavelength;
    const Real zSizeInWavelengths = wz / wavelength;
    const Real xPPW = nx / xSizeInWavelengths;
    const Real yPPW = ny / ySizeInWavelengths;
    const Real zPPW = nz / zSizeInWavelengths;
    const Real minPPW = std::min(std::min(xPPW,yPPW),zPPW);
    const Real xPMLMagRule = 12.*maxDimension/minPPW;
    const Real yPMLMagRule = 12.*maxDimension/minPPW;
    const Real zPMLMagRule = 12.*maxDimension/minPPW;
    const int minPML = std::min(bx,std::min(by,bz));
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
                      << "Minimum wavelength:        " << wavelength << "\n"
                      << "Max dimension:             " << maxDimension << "\n"
                      << "# of wavelengths:          " << numWavelengths << "\n"
                      << "Min points/wavelength:     " << minPPW << "\n"
                      << "Min PML-size               " << minPML << "\n"
                      << std::endl;
            if( minPPW < 7 )
                std::cerr << "WARNING: minimum points/wavelength is very small"
                          << std::endl;
            if( minPML < 5 )
                std::cerr << "WARNING: minimum PML size is very small" 
                          << std::endl;
        }
        std::cout 
          << "Ratio of coordinate-stretching magnitudes from rule of thumb:\n"
          << "  sigmax ratio: " << disc_.sigmax/xPMLMagRule << " = " 
          << disc_.sigmax << "/" << xPMLMagRule << "\n"
          << "  sigmay ratio: " << disc_.sigmay/yPMLMagRule << " = " 
          << disc_.sigmay << "/" << yPMLMagRule << "\n"
          << "  sigmaz ratio: " << disc_.sigmaz/zPMLMagRule << " = " 
          << disc_.sigmaz << "/" << zPMLMagRule << "\n" << std::endl;
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
        std::vector<Real> myPanelVelocity;
        std::vector<int> offsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelVelocity
        ( vOffset, vSize, topInfo_, velocity,
          myPanelVelocity, offsets, 
          panelNestedToNatural, panelNaturalToNested );
        const double gatherStopTime = mpi::Time();

        // Initialize the fronts with the original sparse matrix
        const double fillStartTime = gatherStopTime;
        FillPanelFronts
        ( vOffset, vSize, topInfo_, topFact_,
          velocity, myPanelVelocity, offsets, 
          panelNestedToNatural, panelNaturalToNested );
        const double fillStopTime = mpi::Time();

        // Compute the sparse-direct LDL^T factorization 
        // (and possibly invert and/or redistribute the fronts)
        const double ldlStartTime = fillStopTime;
        if( panelScheme == CLIQUE_LDL_SELINV_1D )
            cliq::LDL( topInfo_, topFact_, cliq::LDL_SELINV_1D );
        else if( panelScheme == CLIQUE_LDL_1D )
            cliq::LDL( topInfo_, topFact_, cliq::LDL_1D );
        else
            LogicError("Unhandled factorization case");
        const double ldlStopTime = mpi::Time();

        const double stopTime = mpi::Time();
        if( commRank == 0 )
        {
            std::cout << "    gather: " << gatherStopTime-gatherStartTime 
                      << " secs\n"
                      << "    fill:   " << fillStopTime-fillStartTime 
                      << " secs\n"
                      << "    ldl:    " << ldlStopTime-ldlStartTime 
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
        std::vector<Real> myPanelVelocity;
        std::vector<int> offsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelVelocity
        ( vOffset, vSize, bottomInfo_, velocity, myPanelVelocity, offsets,
          panelNestedToNatural, panelNaturalToNested );
        const double gatherStopTime = mpi::Time();

        // Initialize the fronts with the original sparse matrix
        const double fillStartTime = gatherStopTime;
        FillPanelFronts
        ( vOffset, vSize, bottomInfo_, bottomFact_,
          velocity, myPanelVelocity, offsets,
          panelNestedToNatural, panelNaturalToNested );
        const double fillStopTime = mpi::Time();

        // Compute the sparse-direct LDL^T factorization
        // (and possibly invert and/or redistribute the fronts)
        const double ldlStartTime = fillStopTime;
        if( panelScheme == CLIQUE_LDL_SELINV_1D )
            cliq::LDL( bottomInfo_, bottomFact_, cliq::LDL_SELINV_1D );
        else if( panelScheme == CLIQUE_LDL_1D )
            cliq::LDL( bottomInfo_, bottomFact_, cliq::LDL_1D );
        else
            LogicError("Unhandled factorization case");
        const double ldlStopTime = mpi::Time();

        const double stopTime = mpi::Time();
        if( commRank == 0 )
        {
            std::cout << "    gather: " << gatherStopTime-gatherStartTime     
                      << " secs\n"
                      << "    fill:   " << fillStopTime-fillStartTime           
                      << " secs\n"
                      << "    ldl:    " << ldlStopTime-ldlStartTime
                      << " secs\n"
                      << "    total:  " << stopTime-startTime 
                      << " secs\n"
                      << std::endl;
        }
    }

    //
    // Initialize and factor the full inner panels
    //
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
        std::vector<Real> myPanelVelocity;
        std::vector<int> offsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelVelocity
        ( vOffset, vSize, bottomInfo_, velocity,
          myPanelVelocity, offsets, 
          panelNestedToNatural, panelNaturalToNested );
        const double gatherStopTime = mpi::Time();

        // Initialize the fronts with the original sparse matrix
        const double fillStartTime = gatherStopTime;
        fullInnerFacts_[k] = new cliq::DistSymmFrontTree<C>;
        FillPanelFronts
        ( vOffset, vSize, bottomInfo_, *fullInnerFacts_[k],
          velocity, myPanelVelocity, offsets,
          panelNestedToNatural, panelNaturalToNested );
        const double fillStopTime = mpi::Time();

        // Compute the sparse-direct LDL^T factorization of the k'th inner panel
        // (and possibly invert and/or redistribute the fronts)
        const double ldlStartTime = fillStopTime;
        if( panelScheme == CLIQUE_LDL_SELINV_1D )
            cliq::LDL( bottomInfo_, *fullInnerFacts_[k], cliq::LDL_SELINV_1D );
        else if( panelScheme == CLIQUE_LDL_1D )
            cliq::LDL( bottomInfo_, *fullInnerFacts_[k], cliq::LDL_1D );
        else
            LogicError("Unhandled factorization case");
        const double ldlStopTime = mpi::Time();

        const double stopTime = mpi::Time();
        if( commRank == 0 )
        {
            std::cout << "    gather: " << gatherStopTime-gatherStartTime     
                      << " secs\n"
                      << "    fill:   " << fillStopTime-fillStartTime 
                      << " secs\n"
                      << "    ldl:    " << ldlStopTime-ldlStartTime
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
        std::vector<Real> myPanelVelocity;
        std::vector<int> offsets;
        std::map<int,int> panelNestedToNatural, panelNaturalToNested;
        GetPanelVelocity
        ( vOffset, vSize, leftoverInnerInfo_, velocity,
          myPanelVelocity, offsets, 
          panelNestedToNatural, panelNaturalToNested );
        const double gatherStopTime = mpi::Time();

        // Initialize the fronts with the original sparse matrix
        const double fillStartTime = gatherStopTime;
        FillPanelFronts
        ( vOffset, vSize, leftoverInnerInfo_, leftoverInnerFact_,
          velocity, myPanelVelocity, offsets,
          panelNestedToNatural, panelNaturalToNested );
        const double fillStopTime = mpi::Time();

        // Compute the sparse-direct LDL^T factorization of the leftover panel
        // (and possibly invert and/or redistribute the fronts)
        const double ldlStartTime = fillStopTime;
        if( panelScheme == CLIQUE_LDL_SELINV_1D )
            cliq::LDL
            ( leftoverInnerInfo_, leftoverInnerFact_, cliq::LDL_SELINV_1D );
        else if( panelScheme == CLIQUE_LDL_1D )
            cliq::LDL
            ( leftoverInnerInfo_, leftoverInnerFact_, cliq::LDL_1D );
        else
            LogicError("Unhandled factorization case");
        const double ldlStopTime = mpi::Time();

        const double stopTime = mpi::Time();
        if( commRank == 0 )
        {
            std::cout << "    gather: " << gatherStopTime-gatherStartTime     
                      << " secs\n"
                      << "    fill:   " << fillStopTime-fillStartTime 
                      << " secs\n"
                      << "    ldl:    " << ldlStopTime-ldlStartTime
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
        std::vector<Real> myGlobalVelocity;
        std::vector<int> offsets;
        GetGlobalVelocity( velocity, myGlobalVelocity, offsets );

        // Now make use of the redistributed velocity data to form the global 
        // sparse matrix
        localEntries_.resize( localRowOffsets_.back() );
        for( int iLocal=0; iLocal<localSize_; ++iLocal )
        {
            const int naturalIndex = localToNaturalMap_[iLocal];
            const int x = naturalIndex % nx;
            const int y = (naturalIndex/nx) % ny;
            const int z = naturalIndex/(nx*ny);
            const int proc = velocity.OwningProcess( x, y, z );

            const Real alpha = myGlobalVelocity[offsets[proc]++];
            const int rowOffset = localRowOffsets_[iLocal];
            const int v = (nz-1) - z;
            FormGlobalRow( alpha, x, y, v, rowOffset );
        }

        const double stopTime = mpi::Time();
        if( commRank == 0 )
            std::cout << stopTime-startTime << " secs" << std::endl;
    }
}

template<typename Real>
void
DistHelmholtz<Real>::Finalize()
{
    // Release the global sparse matrix memory
    SwapClear( localEntries_ );

    // Release the padded panel memory
    SwapClear( bottomFact_.localFronts );
    SwapClear( bottomFact_.distFronts );
    for( int k=0; k<numFullInnerPanels_; ++k )
        delete fullInnerFacts_[k];
    SwapClear( fullInnerFacts_ );
    SwapClear( leftoverInnerFact_.localFronts );
    SwapClear( leftoverInnerFact_.distFronts );
    SwapClear( topFact_.localFronts );
    SwapClear( topFact_.distFronts );
}

//
// With zero dirichlet boundary conditions, the right-most grid point is 
// implicitly zero. With PML, the takeoff function starts at this implicit
// grid point (NOT at the last physical grid point) and reaches all the way 
// to the last (implicit) grid point. Thus, the diagrams for the two situations
// look like this:
//
//    ... x o
//
//    ... x p p p o
//
// where the former uses 'x' to mark a physical grid point, and the 'o' 
// denotes the implicit zero-Dirichlet node. In the second case, we have 
// three PML grid points added, and the take-off function only begins as the 
// left-most 'p' grid point, where it is zero.
//

template<typename Real>
Complex<Real>
DistHelmholtz<Real>::s1Inv( int x ) const
{
    const int bx = disc_.bx;
    const Real etax = bx*hx_;
    if( x < (bx-1) && disc_.frontBC==PML )
    {
        const Real delta = (bx-1) - x;
        const Real realPart = 1;
        const Real imagPart = 
            (disc_.sigmax/etax)*(delta/bx)*(delta/bx)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else if( x > (disc_.nx-bx) && disc_.backBC==PML )
    {
        const Real delta = x-(disc_.nx-bx);
        const Real realPart = 1;
        const Real imagPart =
            (disc_.sigmax/etax)*(delta/bx)*(delta/bx)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else
        return 1;
}

template<typename Real>
Complex<Real>
DistHelmholtz<Real>::s2Inv( int y ) const
{
    const int by = disc_.by;
    const Real etay = by*hy_;
    if( y < (by-1) && disc_.leftBC==PML )
    {
        const Real delta = (by-1) - y;
        const Real realPart = 1;
        const Real imagPart = 
            (disc_.sigmay/etay)*(delta/by)*(delta/by)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else if( y > (disc_.ny-by) && disc_.rightBC==PML )
    {
        const Real delta = y-(disc_.ny-by);
        const Real realPart = 1;
        const Real imagPart =
            (disc_.sigmay/etay)*(delta/by)*(delta/by)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else
        return 1;
}

template<typename Real>
Complex<Real>
DistHelmholtz<Real>::s3Inv( int v ) const
{
    const int bz = disc_.bz;
    const Real etaz = bz*hz_;
    if( v < bz-1 )
    {
        const Real delta = (bz-1) - v;
        const Real realPart = 1;
        const Real imagPart = 
            (disc_.sigmaz/etaz)*(delta/bz)*(delta/bz)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else if( v > (disc_.nz-bz) && disc_.topBC==PML )
    {
        const Real delta = v - (disc_.nz-bz);
        const Real realPart = 1;
        const Real imagPart = 
            (disc_.sigmaz/etaz)*(delta/bz)*(delta/bz)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else
        return 1;
}

template<typename Real>
Complex<Real>
DistHelmholtz<Real>::s3InvArtificial( int v, int vOffset ) const
{
    const int bz = disc_.bz;
    const Real etaz = bz*hz_;
    if( v < vOffset+bz-1 )
    {
        const Real delta = (vOffset+bz-1) - v;
        const Real realPart = 1;
        const Real imagPart = 
            (disc_.sigmaz/etaz)*(delta/bz)*(delta/bz)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else if( v > (disc_.nz-bz) && disc_.topBC==PML )
    {
        const Real delta = v - (disc_.nz-bz);
        const Real realPart = 1;
        const Real imagPart = 
            (disc_.sigmaz/etaz)*(delta/bz)*(delta/bz)*
            (2*M_PI/disc_.omega);
        return C(realPart,imagPart);
    }
    else
        return 1;
}

template<typename Real>
void
DistHelmholtz<Real>::FormGlobalRow( Real alpha, int x, int y, int v, int row )
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
    const C xTop = s2InvM*s3InvM;
    const C xTempL = xTop/s1InvL;
    const C xTempM = xTop/s1InvM;
    const C xTempR = xTop/s1InvR;
    const C xTermL = (xTempL+xTempM) / (2*hx_*hx_);
    const C xTermR = (xTempR+xTempM) / (2*hx_*hx_);

    // Compute all of the y-shifted terms
    const C yTop = s1InvM*s3InvM;
    const C yTempL = yTop/s2InvL;
    const C yTempM = yTop/s2InvM;
    const C yTempR = yTop/s2InvR;
    const C yTermL = (yTempL+yTempM) / (2*hy_*hy_);
    const C yTermR = (yTempR+yTempM) / (2*hy_*hy_);

    // Compute all of the v-shifted terms
    const C zTop = s1InvM*s2InvM;
    const C vTempL = zTop/s3InvL;
    const C vTempM = zTop/s3InvM;
    const C vTempR = zTop/s3InvR;
    const C vTermL = (vTempL+vTempM) / (2*hz_*hz_);
    const C vTermR = (vTempR+vTempM) / (2*hz_*hz_);

    // Compute the center term
    const C centerTerm = (xTermL+xTermR+yTermL+yTermR+vTermL+vTermR) -
        (disc_.omega/alpha)*(disc_.omega/alpha)*s1InvM*s2InvM*s3InvM;

    // Fill in the center term
    int offset = row;
    localEntries_[offset++] = centerTerm;

    // Fill the rest of the terms
    if( x > 0 )
        localEntries_[offset++] = -xTermL;
    if( x+1 < disc_.nx )
        localEntries_[offset++] = -xTermR;
    if( y > 0 )
        localEntries_[offset++] = -yTermL;
    if( y+1 < disc_.ny )
        localEntries_[offset++] = -yTermR;
    if( v > 0 )
        localEntries_[offset++] = -vTermL;
    if( v+1 < disc_.nz )
        localEntries_[offset++] = -vTermR;
}

template<typename Real>
void
DistHelmholtz<Real>::FormLowerColumnOfNode
( Real alpha, int x, int y, int v, int vOffset, int vSize, 
  int offset, int size, int j,
  const std::vector<int>& origLowerStruct, 
  const std::vector<int>& origLowerRelInds,
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
    const C centerTerm = (xTermL+xTermR+yTermL+yTermR+vTermL+vTermR) -
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
                DEBUG_ONLY(
                    if( first == origLowerStruct.end() )
                        LogicError("Did not find original connection");
                )
                const int whichLower = int(first-origLowerStruct.begin());
                frontIndices.push_back( origLowerRelInds[whichLower] );
            }
            values.push_back( -xTermL );
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
                DEBUG_ONLY(
                    if( first == origLowerStruct.end() )
                        LogicError("Did not find original connection");
                )
                const int whichLower = int(first-origLowerStruct.begin());
                frontIndices.push_back( origLowerRelInds[whichLower] );
            }
            values.push_back( -xTermR );
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
                DEBUG_ONLY(
                    if( first == origLowerStruct.end() )
                        LogicError("Did not find original connection");
                )
                const int whichLower = int(first-origLowerStruct.begin());
                frontIndices.push_back( origLowerRelInds[whichLower] );
            }
            values.push_back( -yTermL );
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
                DEBUG_ONLY(
                    if( first == origLowerStruct.end() )
                        LogicError("Did not find original connection");
                )
                const int whichLower = int(first-origLowerStruct.begin());
                frontIndices.push_back( origLowerRelInds[whichLower] );
            }
            values.push_back( -yTermR );
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
                DEBUG_ONLY(
                    if( first == origLowerStruct.end() )
                        LogicError("Did not find original connection");
                )
                const int whichLower = int(first-origLowerStruct.begin());
                frontIndices.push_back( origLowerRelInds[whichLower] );
            }
            values.push_back( -vTermL );
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
                DEBUG_ONLY(
                    if( first == origLowerStruct.end() )
                        LogicError("Did not find original connection");
                )
                const int whichLower = int(first-origLowerStruct.begin());
                frontIndices.push_back( origLowerRelInds[whichLower] );
            }
            values.push_back( -vTermR );
        }
    }
}
        
template<typename Real>
void
DistHelmholtz<Real>::GetGlobalVelocity
( const DistUniformGrid<Real>& velocity,
        std::vector<Real>& myGlobalVelocity,
        std::vector<int>& offsets ) const
{
    const int commSize = mpi::CommSize( comm_ );

    // Pack and send the amount of data that we need to recv from each process.
    std::vector<int> recvCounts( commSize, 0 );
    for( int iLocal=0; iLocal<localSize_; ++iLocal )
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
    for( int iLocal=0; iLocal<localSize_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int proc = velocity.OwningProcess( naturalIndex );
        recvIndices[offsets[proc]++] = naturalIndex;
    }
    std::vector<int> sendIndices( totalSendCount );
    mpi::AllToAll
    ( &recvIndices[0], &recvCounts[0], &recvDispls[0], 
      &sendIndices[0], &sendCounts[0], &sendDispls[0], comm_ );
    SwapClear( recvIndices );

    // Pack and send our velocity data.
    std::vector<Real> sendVelocity( totalSendCount );
    const Real* localVelocity = velocity.LockedBuffer();
    for( int proc=0; proc<commSize; ++proc )
    {
        Real* procVelocity = &sendVelocity[sendDispls[proc]];
        const int* procIndices = &sendIndices[sendDispls[proc]];
        for( int iLocal=0; iLocal<sendCounts[proc]; ++iLocal )
        {
            const int naturalIndex = procIndices[iLocal];
            const int localIndex = velocity.LocalIndex( naturalIndex );
            procVelocity[iLocal] = localVelocity[localIndex];
        }
    }
    SwapClear( sendIndices );

    myGlobalVelocity.resize( totalRecvCount );
    mpi::AllToAll
    ( &sendVelocity[0],     &sendCounts[0], &sendDispls[0],
      &myGlobalVelocity[0], &recvCounts[0], &recvDispls[0], comm_ );

    // Reset the offsets
    offsets = recvDispls;
}

template<typename Real>
void
DistHelmholtz<Real>::GetPanelVelocity
( int vOffset, int vSize, 
  const cliq::DistSymmInfo& info,
  const DistUniformGrid<Real>& velocity,
        std::vector<Real>& myPanelVelocity,
        std::vector<int>& offsets,
        std::map<int,int>& panelNestedToNatural,
        std::map<int,int>& panelNaturalToNested ) const
{
    const int nx = disc_.nx;
    const int ny = disc_.ny;
    const int nz = disc_.nz;
    const int commSize = mpi::CommSize( comm_ );

    // Compute the reorderings for the indices in the nodes in our local tree
    SwapClear( panelNestedToNatural );
    SwapClear( panelNaturalToNested );
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
    const int numLocalNodes = info.localNodes.size();
    for( int t=0; t<numLocalNodes; ++t )
    {
        const cliq::SymmNodeInfo& node = info.localNodes[t];
        const int size = node.size;
        const int offset = node.off;
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
    const int numDistNodes = info.distNodes.size();
    for( int t=1; t<numDistNodes; ++t )
    {
        const cliq::DistSymmNodeInfo& node = info.distNodes[t];
        const Grid& grid = *node.grid;
        const int gridCol = grid.Col();
        const int gridWidth = grid.Width();

        const int size = node.size;
        const int offset = node.off;
        const int localWidth = Length( size, gridCol, gridWidth );
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
    for( int t=0; t<numLocalNodes; ++t )
    {
        const cliq::SymmNodeInfo& node = info.localNodes[t];
        const int size = node.size;
        const int offset = node.off;
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
    for( int t=1; t<numDistNodes; ++t )
    {
        const cliq::DistSymmNodeInfo& node = info.distNodes[t];
        const Grid& grid = *node.grid;
        const int gridCol = grid.Col();
        const int gridWidth = grid.Width();

        const int size = node.size;
        const int offset = node.off;
        const int localWidth = Length( size, gridCol, gridWidth );
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
    SwapClear( recvIndices );

    // Pack and send our velocity data.
    std::vector<Real> sendVelocity( totalSendCount );
    const Real* localVelocity = velocity.LockedBuffer();
    for( int proc=0; proc<commSize; ++proc )
    {
        Real* procVelocity = &sendVelocity[sendDispls[proc]];
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
    SwapClear( sendIndices );
    myPanelVelocity.resize( totalRecvCount );
    mpi::AllToAll
    ( &sendVelocity[0],    &sendCounts[0], &sendDispls[0],
      &myPanelVelocity[0], &recvCounts[0], &recvDispls[0], comm_ );
    SwapClear( sendVelocity );

    // Reset the offsets
    offsets = recvDispls;
}

template<typename Real>
int DistHelmholtz<Real>::LocalReorderedIndex( int x, int y, int v ) const
{
    const int panel = WhichPanel( v );

    const int panelDepth = PanelDepth( panel );
    const int panelPadding = PanelPadding( panel );
    const int vLocal = LocalV( v ) - panelPadding;

    const int localPanelOffset = LocalPanelOffset( panel );
    const int localPanelIndex = LocalReorderedIndex( x, y, vLocal, panelDepth );

    return localPanelOffset + localPanelIndex;
}

template<typename Real>
int DistHelmholtz<Real>::LocalReorderedIndex
( int x, int y, int vLocal, int vSize ) const
{
    const int commRank = mpi::CommRank( comm_ );
    const int commSize = mpi::CommSize( comm_ );
    int offset = 0;
    return LocalReorderedIndexRecursion
    ( x, y, vLocal, disc_.nx, disc_.ny, vSize, nestedCutoff_, offset, 
      commRank, commSize );
}

template<typename Real>
int DistHelmholtz<Real>::LocalReorderedIndexRecursion
( int x, int y, int vLocal, int xSize, int ySize, int vSize, 
  int cutoff, int offset, int commRank, int commSize )
{
    if( commSize == 0 && xSize*ySize <= cutoff )
    {
        return offset + x + y*xSize + vLocal*xSize*ySize;
    }
    else if( xSize >= ySize )
    {
        //
        // Partition the X dimension
        //
        const int xLeftSize = (xSize-1)/2;
        const int xRightSize = std::max(xSize-xLeftSize-1,0);

        const int leftTeamSize = ( commSize==1 ? 1 : commSize/2 );
        const int rightTeamSize = ( commSize==1 ? 1 : commSize - leftTeamSize );
        const bool onLeft = ( commRank < leftTeamSize );
        const bool onRight = ( commSize==1 || !onLeft );
        const int childCommRank = ( onLeft ? commRank : commRank-leftTeamSize );

        // Recurse on the left side
        if( x < xLeftSize )
            return LocalReorderedIndexRecursion
            ( x, y, vLocal, xLeftSize, ySize, vSize, cutoff, offset,
              childCommRank, leftTeamSize );
        else if( onLeft )
            LocalPanelSizeRecursion
            ( xLeftSize, ySize, vSize, 0, cutoff, childCommRank, leftTeamSize, 
              offset );

        // Recurse on the right side
        if( x > xLeftSize )
            return LocalReorderedIndexRecursion
            ( x-(xLeftSize+1), y, vLocal, xRightSize, ySize, vSize, cutoff, 
              offset, childCommRank, rightTeamSize );
        else if( onRight )
            LocalPanelSizeRecursion
            ( xRightSize, ySize, vSize, 0, cutoff, childCommRank, rightTeamSize,
              offset );

        // Store the separator
        if( x == xLeftSize )
            return offset + y + vLocal*ySize;
    }
    else
    {
        //
        // Partition the Y dimension
        //
        const int yLeftSize = (ySize-1)/2;
        const int yRightSize = std::max(ySize-yLeftSize-1,0);

        const int leftTeamSize = ( commSize==1 ? 1 : commSize/2 );
        const int rightTeamSize = ( commSize==1 ? 1 : commSize - leftTeamSize );
        const bool onLeft = ( commRank < leftTeamSize );
        const bool onRight = ( commSize==1 || !onLeft );
        const int childCommRank = ( onLeft ? commRank : commRank-leftTeamSize );

        // Recurse on the left side
        if( y < yLeftSize )
            return LocalReorderedIndexRecursion
            ( x, y, vLocal, xSize, yLeftSize, vSize, cutoff, offset,
              childCommRank, leftTeamSize );
        else if( onLeft )
            LocalPanelSizeRecursion
            ( xSize, yLeftSize, vSize, 0, cutoff, childCommRank, leftTeamSize, 
              offset );

        // Recurse on the right side
        if( y > yLeftSize )
            return LocalReorderedIndexRecursion
            ( x, y-(yLeftSize+1), vLocal, xSize, yRightSize, vSize, cutoff, 
              offset, childCommRank, rightTeamSize );
        else if( onRight )
            LocalPanelSizeRecursion
            ( xSize, yRightSize, vSize, 0, cutoff, childCommRank, rightTeamSize,
              offset );

        // Store the separator
        if( y == yLeftSize )
            return offset + x + vLocal*xSize;
    }
}

template<typename Real>
void DistHelmholtz<Real>::LocalReordering
( std::map<int,int>& reordering, int vSize ) const
{
    const int commRank = mpi::CommRank( comm_ );
    const int commSize = mpi::CommSize( comm_ );
    int offset = 0;
    LocalReorderingRecursion
    ( reordering, offset,
      0, 0, disc_.nx, disc_.ny, vSize, disc_.nx, disc_.ny,
      nestedCutoff_, commRank, commSize );
}

template<typename Real>
void DistHelmholtz<Real>::LocalReorderingRecursion
( std::map<int,int>& reordering, int offset,
  int xOffset, int yOffset, int xSize, int ySize, int vSize, int nx, int ny,
  int cutoff, int commRank, int commSize )
{
    if( commSize == 1 && xSize*ySize <= cutoff )
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

        const int leftTeamSize = ( commSize==1 ? 1 : commSize/2 );
        const int rightTeamSize = ( commSize==1 ? 1 : commSize - leftTeamSize );
        const bool onLeft = ( commRank < leftTeamSize );
        const bool onRight = ( commSize==1 || !onLeft );
        const int childCommRank = ( onLeft ? commRank : commRank-leftTeamSize );

        // Recurse on the left side
        if( onLeft )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset, yOffset, xLeftSize, ySize, vSize, nx, ny,
              cutoff, childCommRank, leftTeamSize );
        offset += xLeftSize*ySize*vSize;

        // Recurse on the right side
        if( onRight )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset+xLeftSize+1, yOffset, xRightSize, ySize, vSize, nx, ny,
              cutoff, childCommRank, rightTeamSize );
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

        const int leftTeamSize = ( commSize==1 ? 1 : commSize/2 );
        const int rightTeamSize = ( commSize==1 ? 1 : commSize - leftTeamSize );
        const bool onLeft = ( commRank < leftTeamSize );
        const bool onRight = ( commSize==1 || !onLeft );
        const int childCommRank = ( onLeft ? commRank : commRank-leftTeamSize );

        // Recurse on the left side
        if( onLeft )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset, yOffset, xSize, yLeftSize, vSize, nx, ny,
              cutoff, childCommRank, leftTeamSize );
        offset += xSize*yLeftSize*vSize;

        // Recurse on the right side
        if( onRight )
            LocalReorderingRecursion
            ( reordering, offset,
              xOffset, yOffset+yLeftSize+1, xSize, yRightSize, vSize, nx, ny,
              cutoff, childCommRank, rightTeamSize );
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

template<typename Real>        
void
DistHelmholtz<Real>::FillPanelFronts
( int vOffset, int vSize, 
  const cliq::DistSymmInfo& info,
        cliq::DistSymmFrontTree<C>& fact,
  const DistUniformGrid<Real>& velocity,
  const std::vector<Real>& myPanelVelocity,
        std::vector<int>& offsets,
        std::map<int,int>& panelNestedToNatural,
        std::map<int,int>& panelNaturalToNested ) const
{
    const int nx = disc_.nx;
    const int ny = disc_.ny;
    const int nz = disc_.nz;
    fact.isHermitian = false;

    // Initialize the local portion of the panel
    std::vector<int> frontIndices;
    std::vector<C> values;
    const int numLocalNodes = info.localNodes.size();
    fact.localFronts.resize( numLocalNodes );
    for( int t=0; t<numLocalNodes; ++t )
    {
        cliq::SymmFront<C>& front = fact.localFronts[t];
        const cliq::SymmNodeInfo& node = info.localNodes[t];

        // Initialize this front
        const int offset = node.off;
        const int size = node.size;
        const int updateSize = node.lowerStruct.size();
        const int frontSize = size + updateSize;
        elem::Zeros( front.frontL, frontSize, size );
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
            const Real alpha = myPanelVelocity[offsets[proc]++];

            // Form the j'th lower column of this node
            FormLowerColumnOfNode
            ( alpha, x, y, v, vOffset, vSize, offset, size, j,
              node.origLowerStruct, node.origLowerRelInds, 
              panelNaturalToNested, frontIndices, values );
            const int numMatches = frontIndices.size();
            for( int k=0; k<numMatches; ++k )
                front.frontL.Set( frontIndices[k], j, values[k] );
        }
    }

    // Initialize the distributed part of the panel
    const int numDistNodes = info.distNodes.size();
    fact.frontType = cliq::SYMM_2D;
    fact.distFronts.resize( numDistNodes );
    cliq::InitializeDistLeaf( info, fact );
    for( int t=1; t<numDistNodes; ++t )
    {
        cliq::DistSymmFront<C>& front = fact.distFronts[t];
        const cliq::DistSymmNodeInfo& node = info.distNodes[t];

        // Initialize this front
        Grid& grid = *node.grid;
        const int gridHeight = grid.Height();
        const int gridWidth = grid.Width();
        const int gridRow = grid.Row();
        const int gridCol = grid.Col();
        const int offset = node.off;
        const int size = node.size;
        const int updateSize = node.lowerStruct.size();
        const int frontSize = size + updateSize;
        front.front2dL.SetGrid( grid );
        elem::Zeros( front.front2dL, frontSize, size );
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
            const Real alpha = myPanelVelocity[offsets[proc]++];

            // Form the j'th lower column of this node
            FormLowerColumnOfNode
            ( alpha, x, y, v, vOffset, vSize, offset, size, j,
              node.origLowerStruct, node.origLowerRelInds, 
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

template<typename Real>        
void
DistHelmholtz<Real>::FillPanelFronts
( int vOffset, int vSize, 
  const cliq::DistSymmInfo& info,
        DistCompressedFrontTree<C>& fact,
  const DistUniformGrid<Real>& velocity,
  const std::vector<Real>& myPanelVelocity,
        std::vector<int>& offsets,
        std::map<int,int>& panelNestedToNatural,
        std::map<int,int>& panelNaturalToNested ) const
{
    const int nx = disc_.nx;
    const int ny = disc_.ny;
    const int nz = disc_.nz;
    fact.isHermitian = false;

    // Initialize the local portion of the panel
    std::vector<int> frontIndices;
    std::vector<C> values;
    const int numLocalNodes = info.localNodes.size();
    fact.localFronts.resize( numLocalNodes );
    for( int t=0; t<numLocalNodes; ++t )
    {
        CompressedFront<C>& front = fact.localFronts[t];
        const cliq::SymmNodeInfo& node = info.localNodes[t];

        // Initialize this front
        const int offset = node.off;
        const int size = node.size;
        const int updateSize = node.lowerStruct.size();
        const int frontSize = size + updateSize;
        elem::Zeros( front.frontL, frontSize, size );
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
            const Real alpha = myPanelVelocity[offsets[proc]++];

            // Form the j'th lower column of this node
            FormLowerColumnOfNode
            ( alpha, x, y, v, vOffset, vSize, offset, size, j,
              node.origLowerStruct, node.origLowerRelInds, 
              panelNaturalToNested, frontIndices, values );
            const int numMatches = frontIndices.size();
            for( int k=0; k<numMatches; ++k )
                front.frontL.Set( frontIndices[k], j, values[k] );
        }
    }

    // Initialize the distributed part of the panel
    const int numDistNodes = info.distNodes.size();
    fact.distFronts.resize( numDistNodes );
    // Replacement for cliq::InitializeDistLeaf
    {
        const cliq::DistSymmNodeInfo& node = info.distNodes[0];
        Matrix<C>& topLocalFrontL = fact.localFronts.back().frontL;
        fact.distFronts[0].grid = node.grid;
        DistMatrix<C>& frontL = fact.distFronts[0].frontL;
        frontL.LockedAttach
        ( topLocalFrontL.Height(), topLocalFrontL.Width(), 0, 0,
          topLocalFrontL.LockedBuffer(), topLocalFrontL.LDim(),
          *node.grid );
    }
    for( int t=1; t<numDistNodes; ++t )
    {
        DistCompressedFront<C>& front = fact.distFronts[t];
        const cliq::DistSymmNodeInfo& node = info.distNodes[t];

        // Initialize this front
        Grid& grid = *node.grid;
        front.grid = node.grid;
        const int gridHeight = grid.Height();
        const int gridWidth = grid.Width();
        const int gridRow = grid.Row();
        const int gridCol = grid.Col();
        const int offset = node.off;
        const int size = node.size;
        const int updateSize = node.lowerStruct.size();
        const int frontSize = size + updateSize;
        front.frontL.SetGrid( grid );
        elem::Zeros( front.frontL, frontSize, size );
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
            const Real alpha = myPanelVelocity[offsets[proc]++];

            // Form the j'th lower column of this node
            FormLowerColumnOfNode
            ( alpha, x, y, v, vOffset, vSize, offset, size, j,
              node.origLowerStruct, node.origLowerRelInds, 
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
