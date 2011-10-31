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

    // Gather the necessary slowness data using two AllToAll's
    std::vector<R> localSlowness;
    {
        // TODO
    }

    // Initialize the global sparse matrix first
    localEntries_.resize( localRowOffsets_.back() );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int x = naturalIndex % nx;
        const int y = (naturalIndex/nx) % ny;
        const int z = naturalIndex/(nx*ny);

        const R alpha = localSlowness[iLocal];
        const int rowOffset = localRowOffsets_[iLocal];
        FormRow( 0, x, y, z, 0, nz, rowOffset, alpha );
    }

    //
    // Initialize and factor the bottom panel (first, since it is the largest)
    //
    {
        // Perform two AllToAll's in order to gather the slowness data
        // TODO

        // Initialize the local part of the bottom panel
        clique::numeric::LocalSymmFrontTree<C>& localFact = bottomFact_.local;
        const clique::symbolic::LocalSymmFact& localSymbFact = 
            bottomSymbolicFact_.local;
        const int numLocalSupernodes = localSymbFact.supernodes.size();
        for( int t=0; t<numLocalSupernodes; ++t )
        {
            clique::numeric::LocalSymmFront<C>& front = localFact.fronts[t];
            const clique::symbolic::LocalSymmFactSupernode& symbSN = 
                localSymbFact.supernodes[t];

            // Initialize this front
            const int size = symbSN.size;
            const int updateSize = symbSN.lowerStruct.size();
            const int frontSize = size + updateSize;
            front.frontL.ResizeTo( frontSize, size );
            front.frontR.ResizeTo( frontSize, updateSize );
            front.frontL.SetToZero();
            front.frontR.SetToZero();
            for( int j=0; j<size; ++j )
            {
                // Fill in the j'th column of frontL
                // TODO
            }
        }

        // Initialize the distributed part of the bottom panel
        clique::numeric::DistSymmFrontTree<C>& distFact = bottomFact_.dist;
        const clique::symbolic::DistSymmFact& distSymbFact = 
            bottomSymbolicFact_.dist;
        const int numDistSupernodes = distSymbFact.supernodes.size();
        for( int t=0; t<numDistSupernodes; ++t )
        {
            clique::numeric::DistSymmFront<C>& front = distFact.fronts[t];
            const clique::symbolic::DistSymmFactSupernode& symbSN = 
                distSymbFact.supernodes[t];

            // Initialize this front
            Grid& grid = *symbSN.grid;
            const int gridWidth = grid.Width();
            const int gridCol = grid.MRRank();
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

                // Fill in the j'th column of frontL
                // TODO
            }
        }

        // Compute the sparse-direct LDL^T factorization of the bottom panel
        clique::numeric::LDL
        ( clique::TRANSPOSE, bottomSymbolicFact_, bottomFact_ );

        // Redistribute the LDL^T factorization for faster solves
        clique::numeric::SetSolveMode( bottomFact_, clique::FEW_RHS );
    }

    //
    // Initialize and factor the top panel
    //
    {
        // Initialize the local part of the top panel
        clique::numeric::LocalSymmFrontTree<C>& localFact = topFact_.local;
        const clique::symbolic::LocalSymmFact& localSymbFact = 
            topSymbolicFact_.local;
        const int numLocalSupernodes = localSymbFact.supernodes.size();
        for( int t=0; t<numLocalSupernodes; ++t )
        {
            clique::numeric::LocalSymmFront<C>& front = localFact.fronts[t];
            const clique::symbolic::LocalSymmFactSupernode& symbSN = 
                localSymbFact.supernodes[t];

            // TODO: Initialize this front
        }

        // Initialize the distributed part of the top panel
        clique::numeric::DistSymmFrontTree<C>& distFact = topFact_.dist;
        const clique::symbolic::DistSymmFact& distSymbFact = 
            topSymbolicFact_.dist;
        const int numDistSupernodes = distSymbFact.supernodes.size();
        for( int t=0; t<numDistSupernodes; ++t )
        {
            clique::numeric::DistSymmFront<C>& front = distFact.fronts[t];
            const clique::symbolic::DistSymmFactSupernode& symbSN = 
                distSymbFact.supernodes[t];

            // TODO: Initialize this front
        }

        // Compute the sparse-direct LDL^T factorization of the top panel
        clique::numeric::LDL
        ( clique::TRANSPOSE, topSymbolicFact_, topFact_ );

        // Redistribute the LDL^T factorization for faster solves
        clique::numeric::SetSolveMode( topFact_, clique::FEW_RHS );
    }

    //
    // Initialize and factor the full inner panels
    //
    for( int k=0; k<numFullInnerPanels_; ++k )
    {
        clique::numeric::SymmFrontTree<C>& fullInnerFact = *fullInnerFacts_[k];

        // Initialize the local part of the k'th full inner panel
        clique::numeric::LocalSymmFrontTree<C>& localFact = fullInnerFact.local;
        const clique::symbolic::LocalSymmFact& localSymbFact = 
            fullInnerSymbolicFact_.local;
        const int numLocalSupernodes = localSymbFact.supernodes.size();
        for( int t=0; t<numLocalSupernodes; ++t )
        {
            clique::numeric::LocalSymmFront<C>& front = localFact.fronts[t];
            const clique::symbolic::LocalSymmFactSupernode& symbSN = 
                localSymbFact.supernodes[t];

            // TODO: Initialize this front
        }

        // Initialize the distributed part of the k'th full inner panel
        clique::numeric::DistSymmFrontTree<C>& distFact = fullInnerFact.dist;
        const clique::symbolic::DistSymmFact& distSymbFact = 
            fullInnerSymbolicFact_.dist;
        const int numDistSupernodes = distSymbFact.supernodes.size();
        for( int t=0; t<numDistSupernodes; ++t )
        {
            clique::numeric::DistSymmFront<C>& front = distFact.fronts[t];
            const clique::symbolic::DistSymmFactSupernode& symbSN = 
                distSymbFact.supernodes[t];

            // TODO: Initialize this front
        }

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
        // Initialize the local portion of the leftover panel
        clique::numeric::LocalSymmFrontTree<C>& localFact = 
            leftoverInnerFact_.local;
        const clique::symbolic::LocalSymmFact& localSymbFact = 
            leftoverInnerSymbolicFact_.local;
        const int numLocalSupernodes = localSymbFact.supernodes.size();
        for( int t=0; t<numLocalSupernodes; ++t )
        {
            clique::numeric::LocalSymmFront<C>& front = localFact.fronts[t];
            const clique::symbolic::LocalSymmFactSupernode& symbSN = 
                localSymbFact.supernodes[t];

            // TODO: Initialize this front
        }

        // Initialize the distributed portion of the leftover panel
        clique::numeric::DistSymmFrontTree<C>& distFact = 
            leftoverInnerFact_.dist;
        const clique::symbolic::DistSymmFact& distSymbFact = 
            leftoverInnerSymbolicFact_.dist;
        const int numDistSupernodes = distSymbFact.supernodes.size();
        for( int t=0; t<numDistSupernodes; ++t )
        {
            clique::numeric::DistSymmFront<C>& front = distFact.fronts[t];
            const clique::symbolic::DistSymmFactSupernode& symbSN = 
                distSymbFact.supernodes[t];

            // TODO: Initialize this front
        }

        // Compute the sparse-direct LDL^T factorization of the leftover panel
        clique::numeric::LDL
        ( clique::TRANSPOSE, leftoverInnerSymbolicFact_, leftoverInnerFact_ );

        // Redistribute the LDL^T factorization for faster solves
        clique::numeric::SetSolveMode( leftoverInnerFact_, clique::FEW_RHS );
    }
}

template<typename R>
void
psp::DistHelmholtz<R>::Finalize()
{
    // Release the global sparse matrix memory
    localEntries_.clear();

    // Release the padded panel memory
    topFact_.local.fronts.clear();
    topFact_.dist.fronts.clear();
    for( int k=0; k<numFullInnerPanels_; ++k )
        delete fullInnerFacts_[k];
    fullInnerFacts_.clear();
    leftoverInnerFact_.local.fronts.clear();
    leftoverInnerFact_.dist.fronts.clear();
    bottomFact_.local.fronts.clear();
    bottomFact_.dist.fronts.clear();
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
psp::DistHelmholtz<R>::s3Inv( int z ) const
{
    if( z+1 < bz_ )
    {
        const R delta = bz_ - (z+1);
        const R realPart = 1;
        const R imagPart = 
            control_.Cz*delta*delta/(bz_*bz_*bz_*hz_*control_.omega);
        return C(realPart,imagPart);
    }
    else if( z > (control_.nz-bz_) && control_.bottomBC==PML )
    {
        const R delta = z - (control_.nz-bz_);
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
psp::DistHelmholtz<R>::s3InvArtificial( int z, int zOffset, R sizeOfPML ) const
{
    if( z+1 < zOffset+sizeOfPML )
    {
        const R delta = zOffset + sizeOfPML - (z+1);
        const R realPart = 1;
        const R imagPart = 
            control_.Cz*delta*delta/
            (sizeOfPML*sizeOfPML*sizeOfPML*hz_*control_.omega);
        return C(realPart,imagPart);
    }
    else if( z > (control_.nz-bz_) && control_.bottomBC==PML )
    {
        const R delta = z - (control_.nz-bz_);
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
psp::DistHelmholtz<R>::FormRow
( R imagShift, int x, int y, int z, int zOffset, int zSize, int rowOffset, 
  R alpha )
{
    const R pmlSize = bzCeil_;

    // Evaluate all of the inverse s functions
    const C s1InvL = s1Inv( x-1 );
    const C s1InvM = s1Inv( x   );
    const C s1InvR = s1Inv( x+1 );
    const C s2InvL = s2Inv( y-1 );
    const C s2InvM = s2Inv( y   );
    const C s2InvR = s2Inv( y+1 );
    const C s3InvL = s3InvArtificial( z-1, zOffset, pmlSize );
    const C s3InvM = s3InvArtificial( z,   zOffset, pmlSize );
    const C s3InvR = s3InvArtificial( z+1, zOffset, pmlSize );

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

    // Compute all of the z-shifted terms
    const C zTempL = s1InvM*s2InvM/s3InvL;
    const C zTempM = s1InvM*s2InvM/s3InvM;
    const C zTempR = s1InvM*s2InvM/s3InvR;
    const C zTermL = (zTempL+zTempM) / (2*hz_*hz_);
    const C zTermR = (zTempR+zTempM) / (2*hz_*hz_);

    // Compute the center term
    const C centerTerm = -(xTermL+xTermR+yTermL+yTermR+zTermL+zTermR) + 
        (control_.omega*alpha)*(control_.omega*alpha)*s1InvM*s2InvM*s3InvM + 
        C(0,imagShift);

    // Fill in the center term
    int offset = rowOffset;
    localEntries_[offset++] = centerTerm;

    // Fill the rest of the terms
    const int zLocal = z - zOffset;
    if( x > 0 )
        localEntries_[offset++] = xTermL;
    if( x+1 < control_.nx )
        localEntries_[offset++] = xTermR;
    if( y > 0 )
        localEntries_[offset++] = yTermL;
    if( y+1 < control_.ny )
        localEntries_[offset++] = yTermR;
    if( zLocal > 0 )
        localEntries_[offset++] = zTermL;
    if( zLocal+1 < zSize )
        localEntries_[offset++] = zTermR;
}

