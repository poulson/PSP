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

using namespace psp;

void Usage()
{
    std::cout << "PanelTest <nx> <ny> <nz>\n\n"
              << "  nx: number of vertices in x direction\n"
              << "  ny: number of vertices in y direction\n"
              << "  nz: number of vertices in z direction" << std::endl;
}

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

    if( argc != 4 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }

    // Seed the random number generator
    srand( time(NULL)+rank );

    // This is just an exercise in filling the control structure
    FiniteDiffControl control;    
    control.stencil = SEVEN_POINT;
    control.nx = atoi(argv[1]);
    control.ny = atoi(argv[2]);
    control.nz = atoi(argv[3]);
    control.wx = 100.;
    control.wy = 100.;
    control.wz = 10.;
    control.omega = 0.75;
    control.C = 1.;
    control.b = 5;
    control.d = 5;
    control.frontBC = PML;
    control.rightBC = PML;
    control.backBC = PML;
    control.leftBC = PML;
    control.topBC = PML;

    mumps::Handle<DComplex> handle;
    try 
    {
        // Create a handle for an instance of MUMPS
        MPI_Barrier( comm );
        mumps::Init( handle );

        int firstLocalVertex, numLocalVertices, numVertices;
        int numLocalNonzeros, numNonzeros;
	FiniteDiffInfo
        ( MPI_COMM_WORLD, control, 
          firstLocalVertex, numLocalVertices, numVertices, 
          numLocalNonzeros, numNonzeros );
        if( rank == 0 )
        {
            std::cout << "numVertices = " << numVertices << "\n"
                      << "numNonzeros = " << numNonzeros << std::endl;
        }

        // Perform the analysis step
        MPI_Barrier( comm );
        if( rank == 0 )
        {
            double t = MPI_Wtime() - startTime;
            std::cout << "Starting MumpsAnalysis at t=" << t << std::endl;

            std::vector<int> rowIndices, colIndices;
            psp::FiniteDiffConnectivity
            ( control, numNonzeros, rowIndices, colIndices );

            // Call the analysis phase of MUMPS (use ParMetis for now)
	    mumps::HostAnalysisWithMetisOrdering
            ( handle, numVertices, numNonzeros, 
              &rowIndices[0], &colIndices[0] );
        }
        else
        {
            // Call the slave analysis routine
            mumps::SlaveAnalysisWithMetisOrdering( handle );
        }
        MPI_Barrier( comm );
        if( rank == 0 )
        {
            double t = MPI_Wtime() - startTime;
            std::cout << "Finished MumpsAnalysis at t=" << t << std::endl;
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
            for( int vtx=firstLocalVertex; 
                 vtx<firstLocalVertex+numLocalVertices; ++vtx ) 
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
                mumps::Factor
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

                mumps::HostSolve
                ( handle, numRhs, &rhs[0], numVertices, 
                  &localSolutions[0], numLocalPivots, &localIntegers[0] );
            }
            else
            {
                mumps::SlaveSolve
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

    mumps::Finalize( handle );
    MPI_Finalize();
    return 0;
}

