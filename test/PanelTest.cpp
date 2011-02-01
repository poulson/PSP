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
#include <iostream>

#include <cmath>
#include <cstdlib>
#include <ctime>

DComplex SampleUnitBall()
{
    // Grab a uniform sample from [0,1]
    double r = static_cast<double>(rand())/static_cast<double>(RAND_MAX);
    // Grab a uniform sample from [0,2*pi]
    double theta = 
        2*M_PI*static_cast<double>(rand())/static_cast<double>(RAND_MAX);
    DComplex sample;
    sample.r = r*cos(theta);
    sample.i = r*sin(theta);
    return sample;
}

int
main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    double startTime = MPI_Wtime();

    int rank, numProcesses;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses );

    // Seed the random number generator
    srand( time(NULL)+rank );

    // This is just an exercise in filling the control structure
    psp::FiniteDiffControl control;    
    control.stencil = psp::SEVEN_POINT;
    //control.nx = 200;
    //control.ny = 200;
    control.nx = 10;
    control.ny = 10;
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
    const int numVertices = control.nx * control.ny * control.nz;
    if( rank == 0 )
        std::cout << "numVertices = " << numVertices << std::endl;

    psp::ZMumpsHandle handle;
    try 
    {
        // Create a handle for an instance of MUMPS
        MPI_Barrier( comm );
        if( rank == 0 )
        {
            double t = MPI_Wtime() - startTime;
            std::cout << "Starting ZMumpsInit at t=" << t << std::endl;
        }
        psp::ZMumpsInit( handle );
        MPI_Barrier( comm );
        if( rank == 0 )
        {
            double t = MPI_Wtime() - startTime;
            std::cout << "Finished ZMumpsInit at t=" << t << std::endl;
        }

        // Compute the number of nonzeros. We can worry about finding a more 
        // efficient (possibly analytical) expression later.
        int vtxChunkSize = 
            static_cast<int>(static_cast<double>(numVertices)/numProcesses);
        int myVtxOffset = vtxChunkSize*rank;
        int numLocalVertices = 
            ( rank==numProcesses-1 ? numVertices-myVtxOffset : vtxChunkSize );
        int numLocalNonzeros = 0;
        for( int vtx=myVtxOffset; vtx<myVtxOffset+numLocalVertices; ++vtx )
        {
            const int i = vtx % control.nx;
            const int j = (vtx/control.nx) % control.ny;
            const int k = vtx/(control.nx*control.ny);

            ++numLocalNonzeros; // count connection to self
            if( i != 0 ) ++numLocalNonzeros; // count connection to (i-1,j,k)
            if( j != 0 ) ++numLocalNonzeros; // count connection to (i,j-1,k)
            if( k != 0 ) ++numLocalNonzeros; // count connection to (i,j,k-1)
        }
        int numNonzeros;
        MPI_Allreduce
        ( &numLocalNonzeros, &numNonzeros, 1, MPI_INT, MPI_SUM, comm );
        if( rank == 0 )
            std::cout << "numNonzeros = " << numNonzeros << std::endl;

        for( int r=0; r<numProcesses; ++r )
        {
            if( rank == r )
            {
                std::cout << "rank " << r << ": \n" 
                          << "  vtxChunkSize=" << vtxChunkSize << "\n"
                          << "  myVtxOffset=" << myVtxOffset << "\n"
                          << "  numLocalVertices=" << numLocalVertices << "\n"
                          << "  numLocalNonzeros=" << numLocalNonzeros 
                          << std::endl;
            }
            MPI_Barrier( comm );
        }

        // Perform the analysis step
        MPI_Barrier( comm );
        if( rank == 0 )
        {
            double t = MPI_Wtime() - startTime;
            std::cout << "Starting ZMumpsAnalysis at t=" << t << std::endl;
        }
        if( rank == 0 )
        {
            // Allocate vectors for storing the row and column indices
            std::vector<int> rowIndices(numNonzeros);
            std::vector<int> colIndices(numNonzeros);

            // Fill in row and column indices each vertex at a time
            int z = 0;
            for( int vtx=0; vtx<numVertices; ++vtx )
            {
                const int i = vtx % control.nx;
                const int j = (vtx/control.nx) % control.ny;
                const int k = vtx/(control.nx*control.ny); 

                // Add the connection to ourself
                rowIndices[z] = vtx+1;
                colIndices[z] = vtx+1;
                ++z;

                // Handle the possible connection in the x direction
                if( i != 0 )
                {
                    // Add the connection to (i-1,j,k)
                    rowIndices[z] = vtx+1;
                    colIndices[z] = vtx;
                    ++z;
                }

                // Handle the possible connection in the y direction
                if( j != 0 )
                {
                    // Add the connection to (i,j-1,k)
                    rowIndices[z] = vtx+1;
                    colIndices[z] = vtx+1-control.nx;
                    ++z;
                }

                // Handle the possible connection in the z direction
                if( k != 0 )
                {
                    // Add the connection to (i,j,k-1)
                    rowIndices[z] = vtx+1;
                    colIndices[z] = vtx+1-(control.nx*control.ny);
                    ++z;
                }
            }

            // Call the analysis phase of MUMPS (use ParMetis for now)
            psp::ZMumpsHostAnalysisWithMetisOrdering
            ( handle, numVertices, numNonzeros, 
              &rowIndices[0], &colIndices[0] );
        }
        else
        {
            // Call the slave analysis routine
            psp::ZMumpsSlaveAnalysisWithMetisOrdering( handle );
        }
        MPI_Barrier( comm );
        if( rank == 0 )
        {
            double t = MPI_Wtime() - startTime;
            std::cout << "Finished ZMumpsAnalysis at t=" << t << std::endl;
        }

        // Create local portion of distributed matrix and then factor it
        int numLocalPivots;
        {
            if( rank == 0 )
            {
                double t = MPI_Wtime() - startTime;
                std::cout << "Starting matrix formation at t=" << t 
                          << std::endl;
            }

            // Give each process a chunk of the vertices
            std::vector<int> localRowIndices(numLocalNonzeros);
            std::vector<int> localColIndices(numLocalNonzeros);
            std::vector<DComplex> localA(numLocalNonzeros);

            int z = 0;
            for( int vtx=myVtxOffset; vtx<myVtxOffset+numLocalVertices; ++vtx ) 
            {
                const int i = vtx % control.nx;
                const int j = (vtx/control.nx) % control.ny;
                const int k = vtx/(control.nx*control.ny); 

                // Count the self connection
                localRowIndices[z] = vtx+1;    
                localColIndices[z] = vtx+1;
                // TODO: Fill in finite-diff approximation
                localA[z] = SampleUnitBall();
                ++z;

                if( i != 0 )
                {
                    // Count the connection to (i-1,j,k)
                    localRowIndices[z] = vtx+1;
                    localColIndices[z] = vtx;
                    // TODO: Fill in finite-diff approximation
                    localA[z] = SampleUnitBall();
                    ++z;
                }
                if( j != 0 )
                {
                    // Count the connection to (i,j-1,k)
                    localRowIndices[z] = vtx+1;
                    localColIndices[z] = vtx+1-control.nx;
                    // TODO: Fill in finite-diff approximation
                    localA[z] = SampleUnitBall();
                    ++z;
                }
                if( k != 0 )
                {
                    // Count the connection to (i,j,k-1)
                    localRowIndices[z] = vtx+1;
                    localColIndices[z] = vtx+1-control.nx*control.ny;
                    // TODO: Fill in finite-diff approximation
                    localA[z] = SampleUnitBall();
                    ++z;
                }
            }
            MPI_Barrier( comm );
            if( rank == 0 ) 
            {
                double t = MPI_Wtime() - startTime;
                std::cout << "Finished forming matrix at t=" << t << std::endl;
            }

            // Factor
            if( rank == 0 )
            {
                double t = MPI_Wtime() - startTime;
                std::cout << "Starting factorization at t=" << t << std::endl;
            }
            numLocalPivots = 
                psp::ZMumpsFactorization
                ( handle, numLocalNonzeros, &localRowIndices[0], 
                  &localColIndices[0], &localA[0] );
            MPI_Barrier( comm );
            if( rank == 0 )
            {
                double t = MPI_Wtime() - startTime;
                std::cout << "Finished factorization at t=" << t << std::endl;
            }
        }
    
        // Solve with various numbers of RHS
        for( int numRhs = 1; numRhs <= 100; numRhs *= 10 )
        {
            // Allocate space for the solution
            std::vector<DComplex> localSolutions(numRhs*numLocalPivots);
            std::vector<int> localIntegers(numLocalPivots);

            if( rank == 0 )
            {
                double t = MPI_Wtime() - startTime;
                std::cout << "Starting solve with numRhs=" << numRhs << " at t="
                          << t << std::endl;
            }
            if( rank == 0 )
            {
                // Allocate and fill the 10 RHS
                std::vector<DComplex> rhs(numRhs*numVertices);
                for( int i=0; i<numRhs*numVertices; ++i )
                {
                    // Just set each RHS to be e1
                    if( i % numVertices == 0 )
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
                ( handle, numRhs, &rhs[0], numVertices, 
                  &localSolutions[0], numLocalPivots, &localIntegers[0] );
            }
            else
            {
                psp::ZMumpsSlaveSolve
                ( handle, &localSolutions[0], numLocalPivots, 
                  &localIntegers[0] );
            }
            MPI_Barrier( comm );
            if( rank == 0 )
            {
                double t = MPI_Wtime() - startTime;
                std::cout << "Finished solve at t=" << t << std::endl;
            }
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception on rank " << rank << ": "
                  << e.what() << std::endl;
    }

    psp::ZMumpsFinalize( handle );
    MPI_Finalize();
    return 0;
}

