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
#include "psp.hpp"
using namespace psp;

void Usage()
{
    std::cout << "EngquistYing-flat <N> <omega> <imagShift> <velocity> "
                 "<numPlanesPerPanel> <fact blocksize> <solve blocksize> "
                 "<accelerate?> <SQMR?> <viz?>\n"
              << "  <N>: Size of grid in each dimension\n"
              << "  <omega>: Frequency (in rad/sec) of problem\n"
              << "  <imagShift>: imaginary shift [2 pi is standard]\n"
              << "  <velocity>: Which velocity field to use, {1,2}\n"
              << "  <numPlanesPerPanel>: depth of sparse-direct solves\n"
              << "  <fact blocksize>: factorization algorithmic blocksize\n"
              << "  <solve blocksize>: solve algorithmic blocksize\n"
              << "  <accelerate?>: accelerate solves iff !=0\n"
              << "  <SQMR?>: GMRES iff 0, SQMR otherwise\n"
              << "  <full viz?>: Full visualization iff != 0\n"
              << "\n"
              << "Please see \"Sweeping preconditioner for the Helmholtz "
                 "equation: moving perfectly matched layers\" for details\n"
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    cliq::Initialize( argc, argv );
    cliq::mpi::Comm comm = cliq::mpi::COMM_WORLD;
    const int commSize = cliq::mpi::CommSize( comm );
    const int commRank = cliq::mpi::CommRank( comm );

    if( argc < 11 )
    {
        if( commRank == 0 )
            Usage();
        cliq::Finalize();
        return 0;
    }
    int argNum=1;
    const int N = atoi( argv[argNum++] );
    const double omega = atof( argv[argNum++] );
    const double imagShift = atof( argv[argNum++] );
    const int velocityModel = atoi( argv[argNum++] );
    const int numPlanesPerPanel = atoi( argv[argNum++] );
    const int factBlocksize = atoi( argv[argNum++] );
    const int solveBlocksize = atoi( argv[argNum++] );
    const bool accelerate = atoi( argv[argNum++] );
    const bool useSQMR = atoi( argv[argNum++] );
    const bool fullVisualize = atoi( argv[argNum++] );

    if( velocityModel < 1 || velocityModel > 2 )
    {
        if( commRank == 0 )
            std::cout << "Invalid velocity model choice, must be in {1,2};\n"
                      << "Please see \"Sweeping preconditioner for the "
                      << "Helmholtz equation: moving perfectly matched layers\""
                      << " for more details." << std::endl;
        cliq::Finalize();
        return 0;
    }

    if( commRank == 0 )
    {
        std::cout << "Running with N=" << N << ", omega=" << omega 
                  << ", and numPlanesPerPanel=" << numPlanesPerPanel
                  << " with velocity model " << velocityModel << std::endl;
    }
    
    FiniteDiffControl<double> control;
    control.stencil = SEVEN_POINT;
    control.nx = N;
    control.ny = N;
    control.nz = N/8;
    control.wx = 1;
    control.wy = 1;
    control.wz = 0.125;
    control.omega = omega;
    control.Cx = 1.5;
    control.Cy = 1.5;
    control.Cz = 1.5;
    control.bx = 4;
    control.by = 4;
    control.bz = 4;
    /*
    control.bx = 6;
    control.by = 6;
    control.bz = 6;
    */
    control.imagShift = imagShift;
    control.cutoff = 96;
    control.numPlanesPerPanel = numPlanesPerPanel;
    control.frontBC = PML;
    control.rightBC = PML;
    control.backBC = PML;
    control.leftBC = PML;
    control.topBC = PML;

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

        GridData<double> velocity( 1, N, N, N/8, XYZ, px, py, pz, comm );
        double* localVelocity = velocity.LocalBuffer();
        const int xLocalSize = velocity.XLocalSize();
        const int yLocalSize = velocity.YLocalSize();
        const int zLocalSize = velocity.ZLocalSize();
        const int xShift = velocity.XShift();
        const int yShift = velocity.YShift();
        const int zShift = velocity.ZShift();
        if( velocityModel == 1 )
        {
            // Converging lens
            const double center[] = { 0.5, 0.5, 0.5 };
            double arg[3];
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (N+1.0);
                arg[2] = (Z-center[2]/8)*(Z-center[2]/8);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (N+1.0);
                    arg[1] = (Y-center[1])*(Y-center[1]);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (N+1.0);
                        arg[0] = (X-center[0])*(X-center[0]);
                        
                        const int localIndex = 
                            xLocal + yLocal*xLocalSize + 
                            zLocal*xLocalSize*yLocalSize;
                        const double speed =
                            1.0 - 0.4*std::exp(-32.*(arg[0]+arg[1]+arg[2]));
                        localVelocity[localIndex] = speed / 0.8;
                    }
                }
            }
        }
        else // velocityModel == 2
        {
            // Wave guide
            const double center[] = { 0.5, 0.5 };
            double arg[2];
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (N+1.0);
                    arg[1] = (Y-center[1])*(Y-center[1]);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (N+1.0);
                        arg[0] = (X-center[0])*(X-center[0]);
                        
                        const int localIndex = 
                            xLocal + yLocal*xLocalSize + 
                            zLocal*xLocalSize*yLocalSize;
                        const double speed =
                            1.0 - 0.4*std::exp(-32.*(arg[0]+arg[1]));
                        localVelocity[localIndex] = speed / 0.8;
                    }
                }
            }
        }

        velocity.WritePlane( XY, N/16, "velocity-middleXY" );
        velocity.WritePlane( XZ, N/2,  "velocity-middleXZ" );
        velocity.WritePlane( YZ, N/2,  "velocity-middleYZ" );
        if( fullVisualize )
        {
            if( commRank == 0 )
            {
                std::cout << "Writing full velocity data...";
                std::cout.flush();
            }
            velocity.WriteVolume("velocity");
            elem::mpi::Barrier( comm );
            if( commRank == 0 )
                std::cout << "done" << std::endl;
        }

        elem::SetBlocksize( factBlocksize );
        if( commRank == 0 )
            std::cout << "Beginning to initialize..." << std::endl;
        cliq::mpi::Barrier( comm );
        const double initialStartTime = cliq::mpi::Time(); 
        helmholtz.Initialize( velocity, accelerate );
        cliq::mpi::Barrier( comm );
        const double initialStopTime = cliq::mpi::Time();
        const double initialTime = initialStopTime - initialStartTime;
        if( commRank == 0 )
            std::cout << "Finished initialization: " << initialTime 
                      << " seconds." << std::endl;

        GridData<elem::Complex<double> > 
            B( 2, N, N, N/8, XYZ, px, py, pz, comm );
        elem::Complex<double>* localB = B.LocalBuffer();
        const double dir[] = { 0., sqrt(2.)/2., sqrt(2.)/2. };
        const double center0[] = { 0.5, 0.5, 0.25/8 };
        const double center1[] = { 0.5, 0.25, 0.25/8 };
        double arg0[3];
        double arg1[3];
        const std::complex<double> imagOne( 0.0, 1.0 );
        for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
        {
            const int z = zShift + zLocal*pz;
            const double Z = z / (N+1.0);
            arg0[2] = (Z-center0[2])*(Z-center0[2]);
            arg1[2] = (Z-center1[2])*(Z-center1[2]);
            for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
            {
                const int y = yShift + yLocal*py;
                const double Y = y / (N+1.0);
                arg0[1] = (Y-center0[1])*(Y-center0[1]);
                arg1[1] = (Y-center1[1])*(Y-center1[1]);
                for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                {
                    const int x = xShift + xLocal*px;
                    const double X = x / (N+1.0);
                    arg0[0] = (X-center0[0])*(X-center0[0]);
                    arg1[0] = (X-center1[0])*(X-center1[0]);
                    
                    const int localIndex = 
                        2*(xLocal + yLocal*xLocalSize + 
                           zLocal*xLocalSize*yLocalSize);
                    // Use std::complex's for std::exp for now...
                    const std::complex<double> f0 = 
                        N*std::exp(-N*N*(arg0[0]+arg0[1]+arg0[2]));
                    const std::complex<double> f1 = 
                        N*std::exp(-2*omega*(arg1[0]+arg1[1]+arg1[2]))*
                        std::exp(omega*imagOne*(X*dir[0]+Y*dir[1]+Z*dir[2]));
                    // Conver the std::complex's to elem::Complex
                    localB[localIndex+0] = f0;
                    localB[localIndex+1] = f1;
                }
            }
        }

        B.WritePlane( XY, N/16, "source-middleXY" );
        B.WritePlane( XZ, N/2,  "source-middleXZ" );
        B.WritePlane( YZ, N/2,  "source-middleYZ" );
        if( fullVisualize )
        {
            if( commRank == 0 )
            {
                std::cout << "Writing source data...";
                std::cout.flush();
            }
            B.WriteVolume("source");
            if( commRank == 0 )
                std::cout << "done" << std::endl;
        }

        elem::SetBlocksize( solveBlocksize );
        if( commRank == 0 )
            std::cout << "Beginning solve..." << std::endl;
        cliq::mpi::Barrier( comm );
        const double solveStartTime = cliq::mpi::Time();
        if( useSQMR )
            helmholtz.SolveWithSQMR( B );
        else
            helmholtz.SolveWithGMRES( B );
        cliq::mpi::Barrier( comm );
        const double solveStopTime = cliq::mpi::Time();
        const double solveTime = solveStopTime - solveStartTime;
        if( commRank == 0 )
            std::cout << "Finished solve: " << solveTime << " seconds." 
                      << std::endl;

        B.WritePlane( XY, N/16, "solution-middleXY" );
        B.WritePlane( XZ, N/2,  "solution-middleXZ" );
        B.WritePlane( YZ, N/2,  "solution-middleYZ" );
        if( fullVisualize )
        {
            if( commRank == 0 )
            {
                std::cout << "Writing solution data...";
                std::cout.flush();
            }
            B.WriteVolume("solution");
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
        elem::DumpCallStack();
        cliq::DumpCallStack();
#endif
    }

    cliq::Finalize();
    return 0;
}
