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
    std::cout << "EngquistYing <N> <omega> <imagShift> <velocity> "
                 "<numPlanesPerPanel> <fact blocksize> <solve blocksize> "
                 "<panel scheme> <SQMR?> <viz?>\n"
              << "  <N>: Size of grid in each dimension\n"
              << "  <omega>: Frequency (in rad/sec) of problem\n"
              << "  <imagShift>: imaginary shift [2 pi is standard]\n"
              << "  <velocity>: Which velocity field to use, {1,2,3,4}\n"
              << "  <numPlanesPerPanel>: depth of sparse-direct solves\n"
              << "  <fact blocksize>: factorization algorithmic blocksize\n"
              << "  <solve blocksize>: solve algorithmic blocksize\n"
              << "  <panel scheme>: NORMAL_1D=0, FAST_2D_LDL=1\n"
              << "  <SQMR?>: GMRES iff 0, SQMR otherwise\n"
              << "  <full viz?>:  Full visualization iff != 0\n"
              << "\n"
              << "Please see \"Sweeping preconditioner for the Helmholtz "
                 "equation: moving perfectly matched layers\" for details\n"
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    psp::Initialize( argc, argv );
    psp::mpi::Comm comm = psp::mpi::COMM_WORLD;
    const int commSize = psp::mpi::CommSize( comm );
    const int commRank = psp::mpi::CommRank( comm );

    if( argc < 11 )
    {
        if( commRank == 0 )
            Usage();
        psp::Finalize();
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
    const PanelScheme panelScheme = (PanelScheme)atoi( argv[argNum++] );
    const bool useSQMR = atoi( argv[argNum++] );
    const bool fullVisualize = atoi( argv[argNum++] );

    if( velocityModel < 1 || velocityModel > 10 )
    {
        if( commRank == 0 )
            std::cout << "Invalid velocity model, must be in {1,...,10}\n"
                      << "---------------------------------------------\n"
                      << "1) Gaussian perturbation of unity\n"
                      << "2) Wave guide\n"
                      << "3) Two layers\n"
                      << "4) Cavity\n"
                      << "5) Reverse cavity\n"
                      << "6) Top half of cavity\n"
                      << "7) Bottom half of cavity\n"
                      << "8) Increasing layers\n"
                      << "9) Decreasing layers\n"
                      << "10) Sideways layers\n"
                      << std::endl;
        psp::Finalize();
        return 0;
    }

    if( commRank == 0 )
    {
        std::cout << "Running with N=" << N << ", omega=" << omega 
                  << ", and numPlanesPerPanel=" << numPlanesPerPanel
                  << " with velocity model " << velocityModel << std::endl;
    }
    
    Discretization<double> disc
    ( omega, N, N, N, 1., 1., 1., PML, PML, PML, PML, DIRICHLET );

    try 
    {
        DistHelmholtz<double> helmholtz
        ( disc, comm, imagShift, numPlanesPerPanel );

        GridData<double> velocity( 1, N, N, N, XYZ, comm );
        double* localVelocity = velocity.LocalBuffer();
        const int xLocalSize = velocity.XLocalSize();
        const int yLocalSize = velocity.YLocalSize();
        const int zLocalSize = velocity.ZLocalSize();
        const int xShift = velocity.XShift();
        const int yShift = velocity.YShift();
        const int zShift = velocity.ZShift();
        const int px = velocity.XStride();
        const int py = velocity.YStride();
        const int pz = velocity.ZStride();
        if( velocityModel == 1 )
        {
            // Converging lens
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (N+1.0);
                const double argZ = (Z-0.5)*(Z-0.5);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (N+1.0);
                    const double argY = (Y-0.5)*(Y-0.5);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (N+1.0);
                        const double argX = (X-0.5)*(X-0.5);
                        
                        const int localIndex = 
                            xLocal + yLocal*xLocalSize + 
                            zLocal*xLocalSize*yLocalSize;
                        const double speed =
                            1.0 - 0.4*std::exp(-32.*(argX+argY+argZ));
                        localVelocity[localIndex] = speed / 0.8;
                    }
                }
            }
        }
        else if( velocityModel == 2 )
        {
            // Wave guide
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (N+1.0);
                    const double argY = (Y-0.5)*(Y-0.5);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (N+1.0);
                        const double argX = (X-0.5)*(X-0.5);
                        
                        const int localIndex = 
                            xLocal + yLocal*xLocalSize + 
                            zLocal*xLocalSize*yLocalSize;
                        const double speed =
                            1.0 - 0.4*std::exp(-32.*(argX+argY));
                        localVelocity[localIndex] = speed / 0.8;
                    }
                }
            }
        }
        else if( velocityModel == 3 )
        {
            // Two layers
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (N+1.0);
                const double speed = ( Z >= 0.5 ? 8.0 : 1.0 );
                const int localOffset = zLocal*xLocalSize*yLocalSize;
                for( int i=0; i<xLocalSize*yLocalSize; ++i )
                    localVelocity[localOffset+i] = speed;
            }
        }
        else if( velocityModel == 4 )
        {
            // Cavity
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (N+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (N+1.0);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (N+1.0);
                        double speed;
                        if( X > 0.2 && X < 0.8 && 
                            Y > 0.2 && Y < 0.8 && 
                            Z > 0.2 && Z < 0.8 )
                            speed = 1;
                        else
                            speed = 8;
                        const int localIndex = 
                            xLocal + yLocal*xLocalSize + 
                            zLocal*xLocalSize*yLocalSize;
                        localVelocity[localIndex] = speed;
                    }
                }
            }
        }
        else if( velocityModel == 5 )
        {
            // Reverse-cavity
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (N+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (N+1.0);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (N+1.0);
                        double speed;
                        if( X > 0.2 && X < 0.8 && 
                            Y > 0.2 && Y < 0.8 && 
                            Z > 0.2 && Z < 0.8 )
                            speed = 8;
                        else
                            speed = 1;
                        const int localIndex = 
                            xLocal + yLocal*xLocalSize + 
                            zLocal*xLocalSize*yLocalSize;
                        localVelocity[localIndex] = speed;
                    }
                }
            }
        }
        else if( velocityModel == 6 )
        {
            // Bottom half of cavity
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (N+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (N+1.0);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (N+1.0);
                        double speed;
                        if( X > 0.2 && X < 0.8 && 
                            Y > 0.2 && Y < 0.8 && 
                            Z > 0.2 && Z < 0.8 )
                            speed = 1;
                        else if( Z > 0.5 )
                            speed = 1;
                        else
                            speed = 8;
                        const int localIndex = 
                            xLocal + yLocal*xLocalSize + 
                            zLocal*xLocalSize*yLocalSize;
                        localVelocity[localIndex] = speed;
                    }
                }
            }
        }
        else if( velocityModel == 7 )
        {
            // Top half of cavity
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (N+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (N+1.0);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (N+1.0);
                        double speed;
                        if( X > 0.2 && X < 0.8 && 
                            Y > 0.2 && Y < 0.8 && 
                            Z > 0.2 && Z < 0.8 )
                            speed = 1;
                        else if( Z < 0.5 )
                            speed = 1;
                        else
                            speed = 8;
                        const int localIndex = 
                            xLocal + yLocal*xLocalSize + 
                            zLocal*xLocalSize*yLocalSize;
                        localVelocity[localIndex] = speed;
                    }
                }
            }
        }
        else if( velocityModel == 8 )
        {
            // Increasing layers
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (N+1.0);
                double speed;
                if( Z < 0.2 )
                    speed = 1;
                else if( Z < 0.4 )
                    speed = 2;
                else if( Z < 0.6 )
                    speed = 3;
                else if( Z < 0.8 )
                    speed = 4;
                else
                    speed = 5;
                const int localOffset = zLocal*xLocalSize*yLocalSize;
                for( int i=0; i<xLocalSize*yLocalSize; ++i )
                    localVelocity[i+localOffset] = speed;
            }
        }
        else if( velocityModel == 9 )
        {
            // Decreasing layers
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (N+1.0);
                double speed;
                if( Z < 0.2 )
                    speed = 5;
                else if( Z < 0.4 )
                    speed = 4;
                else if( Z < 0.6 )
                    speed = 3;
                else if( Z < 0.8 )
                    speed = 2;
                else
                    speed = 1;
                const int localOffset = zLocal*xLocalSize*yLocalSize;
                for( int i=0; i<xLocalSize*yLocalSize; ++i )
                    localVelocity[i+localOffset] = speed;
            }
        }
        else if( velocityModel == 10 )
        {
            // Sideways layers
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*pz;
                        const double X = x / (N+1.0);
                        double speed;
                        if( X < 0.2 )
                            speed = 5;
                        else if( X < 0.4 )
                            speed = 4;
                        else if( X < 0.6 )
                            speed = 3;
                        else if( X < 0.8 )
                            speed = 2;
                        else
                            speed = 1;
                        const int localIndex = xLocal + yLocal*xLocalSize + 
                            zLocal*xLocalSize*yLocalSize;
                        localVelocity[localIndex] = speed;
                    }
                }
            }
        }

        velocity.WritePlane( XY, N/2, "velocity-middleXY" );
        velocity.WritePlane( XZ, N/2, "velocity-middleXZ" );
        velocity.WritePlane( YZ, N/2, "velocity-middleYZ" );
        if( fullVisualize )
        {
            if( commRank == 0 )
            {
                std::cout << "Writing velocity data...";
                std::cout.flush();
            }
            velocity.WriteVolume("velocity");
            mpi::Barrier( comm );
            if( commRank == 0 )
                std::cout << "done" << std::endl;
        }

        elem::SetBlocksize( factBlocksize );
        if( commRank == 0 )
            std::cout << "Beginning to initialize..." << std::endl;
        psp::mpi::Barrier( comm );
        const double initialStartTime = psp::mpi::Time(); 
        helmholtz.Initialize( velocity, panelScheme );
        psp::mpi::Barrier( comm );
        const double initialStopTime = psp::mpi::Time();
        const double initialTime = initialStopTime - initialStartTime;
        if( commRank == 0 )
            std::cout << "Finished initialization: " << initialTime 
                      << " seconds." << std::endl;

        GridData<Complex<double> > B( 2, N, N, N, XYZ, comm );
        Complex<double>* localB = B.LocalBuffer();
        const double dir[] = { 0., sqrt(2.)/2., sqrt(2.)/2. };
        const double center0[] = { 0.5, 0.5, 0.25 };
        const double center1[] = { 0.5, 0.25, 0.25 };
        double arg0[3];
        double arg1[3];
        const Complex<double> imagOne( 0.0, 1.0 );
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
                    localB[localIndex+0] =
                        N*Exp(-N*N*(arg0[0]+arg0[1]+arg0[2]));
                    localB[localIndex+1] = 
                        N*Exp(-2*omega*(arg1[0]+arg1[1]+arg1[2]))*
                        Exp(omega*imagOne*(X*dir[0]+Y*dir[1]+Z*dir[2]));
                }
            }
        }

        B.WritePlane( XY, N/2, "source-middleXY" );
        B.WritePlane( XZ, N/2, "source-middleXZ" );
        B.WritePlane( YZ, N/2, "source-middleYZ" );
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
        psp::mpi::Barrier( comm );
        const double solveStartTime = psp::mpi::Time();
        if( useSQMR )
            helmholtz.SolveWithSQMR( B );
        else
            helmholtz.SolveWithGMRES( B );
        psp::mpi::Barrier( comm );
        const double solveStopTime = psp::mpi::Time();
        const double solveTime = solveStopTime - solveStartTime;
        if( commRank == 0 )
            std::cout << "Finished solve: " << solveTime << " seconds." 
                      << std::endl;

        B.WritePlane( XY, N/2, "solution-middleXY" );
        B.WritePlane( XZ, N/2, "solution-middleXZ" );
        B.WritePlane( YZ, N/2, "solution-middleYZ" );
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
        psp::DumpCallStack();
#endif
    }

    psp::Finalize();
    return 0;
}
