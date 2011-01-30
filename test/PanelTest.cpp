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

   Additional permissions under GNU GPL version 3 section 7

   If you modify this Program, or any covered work, by linking or combining it
   with MUMPS and/or ParMetis (or modified versions of those libraries),
   containing parts covered by the terms of the respective licenses of MUMPS
   and ParMetis, the licensors of this Program grant you additional permission
   to convey the resulting work. {Corresponding Source for a non-source form of
   such a combination shall include the source code for the parts of MUMPS and
   ParMetis used as well as that of the covered work.}
*/
#include "psp.hpp"
#include <vector>

int
main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    MPI_Comm_rank( comm, &rank );

    // This is just an exercise in filling the control structure
    psp::FiniteDifferenceControl control;    
    control.stencil = psp::SEVEN_POINT;
    control.nx = 1000;
    control.ny = 1000;
    control.nz = 10;
    control.wx = 100.;
    control.wy = 100.;
    control.wz = 10.;
    control.omega = 0.75;
    control.C = 1.;
    control.b = 5;
    control.d = 5;
    control.frontBC = psp::PML;
    control.rightBC = psp::PML;
    control.backBC = psp::PML;
    control.leftBC = psp::PML;
    control.topBC = psp::PML;

    // Go ahead and define the number of degrees of freedom
    const int n = control.nx * control.ny * control.nz;

    // Create a handle for an instance of MUMPS
    psp::ZMumpsHandle handle;
    psp::ZMumpsInit( handle );

    // Perform the analysis step
    if( rank == 0 )
    {
        // Compute total number of nonzeros in the lower triangle.
        // There is certainly a faster way to compute this, but this will do
        // for now.
        int nz = 0;
        for( int vtx=0; vtx<nvtxs; ++vtx )
        {
            const int i = vtx % control.nx;
            const int j = (vtx/control.nx) % control.ny;
            const int k = vtx/(control.nx*control.ny);

            ++nz;              // count connection to self
            if( i != 0 ) ++nz; // count connection to (i-1,j,k)
            if( j != 0 ) ++nz; // count connection to (i,j-1,k)
            if( k != 0 ) ++nz; // count connection to (i,j,k-1)
        }

        // Allocate vectors for storing the row and column indices
        std::vector<int> rowIndices(nz);
        std::vector<int> colIndices(nz);

        // Fill in row and column indices each vertex at a time
        int z = 0;
        const int nvtxs = control.nx * control.ny * control.nz;
        for( int vtx=0; vtx<nvtxs; ++vtx )
        {
            const int i = vtx % control.nx;
            const int j = (vtx/control.nx) % control.ny;
            const int k = vtx/(control.nx*control.ny); // no need for a modulo

            // Add the connection to ourself
            rowIndices[z] = vtx;
            colIndices[z] = vtx;
            ++z;

            // Handle the possible connection in the x direction
            if( i != 0 )
            {
                // Add the connection to (i-1,j,k)
                rowIndices[z] = vtx;
                colIndices[z] = vtx-1;
                ++z;
            }

            // Handle the possible connection in the y direction
            if( j != 0 )
            {
                // Add the connection to (i,j-1,k)
                rowIndices[z] = vtx;
                colIndices[z] = vtx-control.nx;
                ++z;
            }

            // Handle the possible connection in the z direction
            if( k != 0 )
            {
                // Add the connection to (i,j,k-1)
                rowIndices[z] = vtx;
                colIndices[z] = vtx-(control.nx*control.ny);
                ++z;
            }
        }

        // Call the analysis phase of MUMPS (use ParMetis for now)
        psp::ZMumpsHostAnalysisWithMetisOrdering
        ( handle, &rowIndices[0], &colIndices[0] );
    }
    else
    {
        // Call the slave analysis routine
        psp::ZMumpsSlaveAnalysisWithMetisOrdering( handle );
    }

    // Create local portion of distributed matrix and then factor it
    int numLocalPivots;
    {
        // int numLocalEntries = ...
        std::vector<int> localRowIndices(numLocalEntries);
        std::vector<int> localColIndices(numLocalEntries);
        std::vector<DComplex> localA(numLocalEntries);
        // TODO: Fill local row indices...
        // TODO: Fill local col indices...

        // Factor
        numLocalPivots = 
            psp::ZMumpsFactorization
            ( handle, numLocalEntries, &localRowIndices[0], &localColIndices[0],
              &localA[0] );
    }
    
    // Solve with various numbers of RHS
    for( int numRhs = 1; numRhs <= 100; numRhs *= 10 )
    {
        // Allocate space for the solution
        std::vector<DComplex> localSolutions(numRhs*numLocalPivots);
        std::vector<int> localIntegers(numLocalPivots);

        if( rank == 0 )
        {
            // Allocate and fill the 10 RHS
            std::vector<DComplex> rhs(numRhs*n);
            for( int i=0; i<numRhs*n; ++i )
            {
                // Just set each RHS to be e1
                if( i % n == 0 )
                {
                    rhs[i].r = 1;
                    rhs[i].i = 0;
                }
                else
                {
                    rhs[i].r = 0;
                    rhs[i].i = 0;
                }
            }

            psp::ZMumpsHostSolve
            ( numRhs, rhsBuffer, n, localSolutionBuffer, numLocalPivots,
              localIntegerBuffer );
        }
        else
        {
            psp::ZMumpsSlaveSolve( localSolutionBuffer, numLocalPivots );
        }
    }

    // Finalize our MUMPS instance
    psp::ZMumpsFinalize( handle );

    MPI_Finalize();
    return 0;
}

