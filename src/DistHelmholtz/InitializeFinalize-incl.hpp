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
psp::DistHelmholtz<R>::Initialize( const R* localSlowness )
{
    const int nx = control_.nx;
    const int ny = control_.ny;
    const int nz = control_.nz;

    // Initialize the global sparse matrix first
    localEntries_.resize( localRowOffsets_.back() );
    for( int iLocal=0; iLocal<localHeight_; ++iLocal )
    {
        const int naturalIndex = localToNaturalMap_[iLocal];
        const int x = naturalIndex % nx;
        const int y = (naturalIndex/nx) % ny;
        const int z = naturalIndex/(nx*ny);

        // NOTE: Here we are exploiting the fact that the structure is 
        //       consistent (i.e., a 7-point stencil).
        int offset = localRowOffsets_[iLocal];
        if( x > 0 )
        {
            // Fill left connection 
            // localEntries_[offset++] = ...
        }
        if( x+1 < nx )
        {
            // Fill right connection
            // localEntries_[offset++] = ...
        }
        if( y > 0 )
        {
            // Fill bottom connection
            // localEntries_[offset++] = ...
        }
        if( y+1 < ny )
        {
            // Fill top connection
            // localEntries_[offset++] = ...
        }
        if( z > 0 )
        {
            // Fill front connection
            // localEntries_[offset++] = ...
        }
        if( z+1 < nz )
        {
            // Fill back connection
            // localEntries_[offset++] = ...
        }
    }

    // Initialize the front panel
    // TODO

    // Initialize the full inner panels
    for( int k=0; k<numFullInnerPanels_; ++k )
    {
        // TODO
    }

    if( haveLeftover_ )
    {
        // Initialize the leftover panel
        // TODO
    }

    // Initialize the back panel
    // TODO
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

// HERE
