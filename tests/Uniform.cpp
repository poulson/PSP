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
#include "psp.hpp"
using namespace psp;

void Usage()
{
    std::cout << "Uniform <nx> <ny> <nz> <omega> <imagShift> "
                 "<numPlanesPerPanel> <fact blocksize> <solve blocksize> "
                 "<accelerate?> <SQMR?> <viz?>\n"
              << "  <nx>: Size of grid in x dimension\n"
              << "  <ny>: Size of grid in y dimension\n"
              << "  <nz>: Size of grid in z dimension\n"
              << "  <omega>: Frequency (in rad/sec) of problem\n"
              << "  <imagShift>: imaginary shift [2 pi is standard]\n"
              << "  <numPlanesPerPanel>: depth of sparse-direct solves\n"
              << "  <fact blocksize>: factorization algorithmic blocksize\n"
              << "  <solve blocksize>: solve algorithmic blocksize\n"
              << "  <accelerate?>: accelerate solves iff !=0\n"
              << "  <SQMR?>: use GMRES iff 0, otherwise SQMR\n"
              << "  <viz?>:  Visualize iff != 0\n"
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    clique::Initialize( argc, argv );
    clique::mpi::Comm comm = clique::mpi::COMM_WORLD;
    const int commSize = clique::mpi::CommSize( comm );
    const int commRank = clique::mpi::CommRank( comm );

    if( argc < 12 )
    {
        if( commRank == 0 )
            Usage();
        clique::Finalize();
        return 0;
    }
    int argNum = 1;
    const int nx = atoi( argv[argNum++] );
    const int ny = atoi( argv[argNum++] );
    const int nz = atoi( argv[argNum++] );
    const double omega = atof( argv[argNum++] );
    const double imagShift = atof( argv[argNum++] );
    const int numPlanesPerPanel = atoi( argv[argNum++] );
    const int factBlocksize = atoi( argv[argNum++] );
    const int solveBlocksize = atoi( argv[argNum++] );
    const bool accelerate = atoi( argv[argNum++] );
    const bool useSQMR = atoi( argv[argNum++] );
    const bool visualize = atoi( argv[argNum++] );

    if( commRank == 0 )
    {
        std::cout << "Running with (nx,ny,nz)=("
                  << nx << "," << ny << "," << nz << "), omega=" 
                  << omega << ", and numPlanesPerPanel=" << numPlanesPerPanel
                  << std::endl;
    }
    
    FiniteDiffControl<double> control;
    control.stencil = SEVEN_POINT;
    control.nx = nx;
    control.ny = ny;
    control.nz = nz;
    control.wx = 1;
    control.wy = 1;
    control.wz = 1;
    control.omega = omega;
    control.Cx = 1.5;
    control.Cy = 1.5;
    control.Cz = 1.5;
    control.etax = 5.0/control.nx;
    control.etay = 5.0/control.ny;
    control.etaz = 5.0/control.nz;
    control.imagShift = 2*M_PI;
    control.cutoff = 96;
    control.numPlanesPerPanel = numPlanesPerPanel;
    control.frontBC = PML;
    control.rightBC = PML;
    control.backBC = PML;
    control.leftBC = PML;
    control.topBC = DIRICHLET;

    try 
    {
        DistHelmholtz<double> helmholtz( control, comm );

        const int cubeRoot = 
            std::max(1,(int)std::floor(pow((double)commSize,0.333)));
        int px = cubeRoot;
        while( commSize % px != 0 )
            ++px;
        const int reduced = commSize / px;
        const int sqrtReduced = 
            std::max(1,(int)std::floor(sqrt((double)reduced)));
        int py = sqrtReduced;
        while( reduced % py != 0 )
            ++py;
        const int pz = reduced / py;
        if( px*py*pz != commSize )
            throw std::logic_error("Nonsensical process grid");
        else if( commRank == 0 )
            std::cout << "px=" << px << ", py=" << py << ", pz=" << pz 
                      << std::endl;

        GridData<double> slowness
        ( 1, control.nx, control.ny, control.nz, XYZ, px, py, pz, comm );
        double* localSlowness = slowness.LocalBuffer();
        const int xLocalSize = slowness.XLocalSize();
        const int yLocalSize = slowness.YLocalSize();
        const int zLocalSize = slowness.ZLocalSize();
        for( int i=0; i<xLocalSize*yLocalSize*zLocalSize; ++i )
            localSlowness[i] = 1;

        if( visualize )
        {
            if( commRank == 0 )
            {
                std::cout << "Writing slowness data...";
                std::cout.flush();
            }
            slowness.WriteVtkFiles("slowness");
            elemental::mpi::Barrier( comm );
            if( commRank == 0 )
                std::cout << "done" << std::endl;
        }

        elemental::SetBlocksize( factBlocksize );
        if( commRank == 0 )
            std::cout << "Beginning to initialize..." << std::endl;
        clique::mpi::Barrier( comm );
        const double initialStartTime = clique::mpi::Time(); 
        helmholtz.Initialize( slowness, accelerate );
        clique::mpi::Barrier( comm );
        const double initialStopTime = clique::mpi::Time();
        const double initialTime = initialStopTime - initialStartTime;
        if( commRank == 0 )
            std::cout << "Finished initialization: " << initialTime 
                      << " seconds." << std::endl;

        GridData<std::complex<double> > B
        ( 1, control.nx, control.ny, control.nz, XYZ, px, py, pz, comm );
        std::complex<double>* localB = B.LocalBuffer();
        std::memset
        ( localB, 0, 
          xLocalSize*yLocalSize*zLocalSize*sizeof(std::complex<double>) );
        const int xSource = control.nx/2;
        const int ySource = control.ny/2;
        const int zSource = control.nz/2;
        if( commRank == B.OwningProcess( xSource, ySource, zSource ) )
        {
            const int localIndex = B.LocalIndex( xSource, ySource, zSource ); 
            localB[localIndex] = 1;
        }

        if( visualize )
        {
            if( commRank == 0 )
            {
                std::cout << "Writing source data...";
                std::cout.flush();
            }
            B.WriteVtkFiles("source");
            if( commRank == 0 )
                std::cout << "done" << std::endl;
        }

        elemental::SetBlocksize( solveBlocksize );
        if( commRank == 0 )
            std::cout << "Beginning solve..." << std::endl;
        clique::mpi::Barrier( comm );
        const double solveStartTime = clique::mpi::Time();
        if( useSQMR )
            helmholtz.SolveWithSQMR( B );
        else
            helmholtz.SolveWithGMRES( B );
        clique::mpi::Barrier( comm );
        const double solveStopTime = clique::mpi::Time();
        const double solveTime = solveStopTime - solveStartTime;
        if( commRank == 0 )
            std::cout << "Finished solve: " << solveTime << " seconds." 
                      << std::endl;

        if( visualize )
        {
            if( commRank == 0 )
            {
                std::cout << "Writing solution data...";
                std::cout.flush();
            }
            B.WriteVtkFiles("solution");
            if( commRank == 0 )
                std::cout << "done" << std::endl;
        }

        helmholtz.Finalize();
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception on process " << commRank << ":\n"
                  << e.what() << std::endl;
#ifndef RELEASE
        elemental::DumpCallStack();
        clique::DumpCallStack();
#endif
    }

    clique::Finalize();
    return 0;
}
