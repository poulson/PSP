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
    std::cout << "UnitCube <velocity> <n> <omega> "
                 "[PML on top=true] [pmlSize=5] [sigma=1.5] [damping=7] "
                 "[# planes/panel=4] [panel scheme=1] [viz=1] "
                 "[fact blocksize=96] [solve blocksize=64]\n"
              << "\n"
              << "  velocity: which velocity field to use, {0,...,12}\n"
              << "  n: size of grid in each dimension\n"
              << "  omega: frequency (in rad/sec) of problem\n"
              << "  PML on top: PML or Dirichlet b.c. on top?\n"
              << "  PML size: number of grid points of PML\n"
              << "  sigma: maximum height of complex coordinate stretching\n"
              << "  damping: imaginary freq shift for preconditioner\n"
              << "  # planes/panel: number of planes per subdomain\n"
              << "  panel scheme: LDL_1D=0, LDL_SELINV_2D=1\n"
              << "  full viz: full volume visualization iff != 0\n"
              << "  fact blocksize: factorization algorithmic blocksize\n"
              << "  solve blocksize: solve algorithmic blocksize\n"
              << "\n"
              << "\n"
              << "velocity model:\n"
              << "---------------------------------------------\n"
              << "0) Unity\n"
              << "1) Gaussian perturbation of unity\n"
              << "2) Wave guide\n"
              << "3) Two layers\n"
              << "4) Cavity (will not converge quickly!)\n"
              << "5) Reverse cavity\n"
              << "6) Top half of cavity\n"
              << "7) Bottom half of cavity\n"
              << "8) Increasing layers\n"
              << "9) Decreasing layers\n"
              << "10) Sideways layers\n"
              << "11) Wedge\n"
              << "12) Random\n"
              << "13) Separator\n"
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    typedef Complex<double> C;

    psp::Initialize( argc, argv );
    mpi::Comm comm = mpi::COMM_WORLD;
    const int commSize = mpi::CommSize( comm );
    const int commRank = mpi::CommRank( comm );

    if( argc < 4 )
    {
        if( commRank == 0 )
            Usage();
        psp::Finalize();
        return 0;
    }
    int argNum=1;
    const int velocityModel = atoi(argv[argNum++]);
    const int n = atoi(argv[argNum++]);
    const double omega = atof(argv[argNum++]);
    const bool pmlOnTop = ( argc>argNum ? atoi(argv[argNum++]) : true );
    const int pmlSize = ( argc>argNum ? atoi(argv[argNum++]) : 5 );
    const double sigma = ( argc>argNum ? atof(argv[argNum++]) : 1.5 );
    const double damping = ( argc>argNum ? atof(argv[argNum++]) : 7. );
    const int numPlanesPerPanel = ( argc>argNum ? atoi(argv[argNum++]) : 4 );
    const PanelScheme panelScheme = 
        ( argc>argNum ? (PanelScheme)atoi(argv[argNum++]) 
                      : CLIQUE_LDL_SELINV_2D );
    const bool fullVisualize = ( argc>argNum ? atoi(argv[argNum++]) : true );
    const int factBlocksize = ( argc>argNum ? atoi( argv[argNum++] ) : 96 );
    const int solveBlocksize = ( argc>argNum ? atoi( argv[argNum++] ) : 64 );

    if( velocityModel < 0 || velocityModel > 13 )
    {
        if( commRank == 0 )
            std::cout << "Invalid velocity model, must be in {0,...,12}\n"
                      << "---------------------------------------------\n"
                      << "0) Unity\n"
                      << "1) Gaussian perturbation of unity\n"
                      << "2) Wave guide\n"
                      << "3) Two layers\n"
                      << "4) Cavity (will not converge quickly!)\n"
                      << "5) Reverse cavity\n"
                      << "6) Top half of cavity\n"
                      << "7) Bottom half of cavity\n"
                      << "8) Increasing layers\n"
                      << "9) Decreasing layers\n"
                      << "10) Sideways layers\n"
                      << "11) Wedge\n"
                      << "12) Random\n"
                      << "13) Separator\n"
                      << std::endl;
        psp::Finalize();
        return 0;
    }

    if( commRank == 0 )
    {
        std::cout << "Running with n=" << n << ", omega=" << omega 
                  << ", and numPlanesPerPanel=" << numPlanesPerPanel
                  << " with velocity model " << velocityModel << std::endl;
    }

    try 
    {
        Boundary topBC = ( pmlOnTop ? PML : DIRICHLET );
        const int pmlSize = 5;
        const double sigma = 1.5;
        Discretization<double> disc
        ( omega, n, n, n, 1., 1., 1., PML, PML, PML, PML, topBC, 
          pmlSize, sigma );

        DistHelmholtz<double> helmholtz
        ( disc, comm, damping, numPlanesPerPanel );

        DistUniformGrid<double> velocity( n, n, n, comm );
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
        switch( velocityModel )
        {
        case 0:
            if( commRank == 0 )
                std::cout << "Unit velocity model" << std::endl;
            // Unit velocity
            for( int i=0; i<xLocalSize*yLocalSize*zLocalSize; ++i )
                localVelocity[i] = 1.;
            break;
        case 1:
            if( commRank == 0 )
                std::cout << "Converging lens velocity model" << std::endl;
            // Converging lens
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
                const double argZ = (Z-0.5)*(Z-0.5);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (n+1.0);
                    const double argY = (Y-0.5)*(Y-0.5);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (n+1.0);
                        const double argX = (X-0.5)*(X-0.5);
                        
                        const int localIndex = 
                            xLocal + yLocal*xLocalSize + 
                            zLocal*xLocalSize*yLocalSize;
                        const double speed =
                            1.0 - 0.4*Exp(-32.*(argX+argY+argZ));
                        localVelocity[localIndex] = speed / 0.8;
                    }
                }
            }
            break;
        case 2:
            if( commRank == 0 )
                std::cout << "Wave guide velocity model" << std::endl;
            // Wave guide
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (n+1.0);
                    const double argY = (Y-0.5)*(Y-0.5);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (n+1.0);
                        const double argX = (X-0.5)*(X-0.5);
                        
                        const int localIndex = 
                            xLocal + yLocal*xLocalSize + 
                            zLocal*xLocalSize*yLocalSize;
                        const double speed =
                            1.0 - 0.4*Exp(-32.*(argX+argY));
                        localVelocity[localIndex] = speed / 0.8;
                    }
                }
            }
            break;
        case 3:
            if( commRank == 0 )
                std::cout << "Two layer velocity model" << std::endl;
            // Two layers
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
                const double speed = ( Z >= 0.5 ? 4.0 : 1.0 );
                const int localOffset = zLocal*xLocalSize*yLocalSize;
                for( int i=0; i<xLocalSize*yLocalSize; ++i )
                    localVelocity[localOffset+i] = speed;
            }
            break;
        case 4:
            if( commRank == 0 )
                std::cout << "Cavity velocity model (might not converge!)"
                          << std::endl;
            // Cavity
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (n+1.0);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (n+1.0);
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
            break;
        case 5:
            if( commRank == 0 )
                std::cout << "Reverse-cavity velocity model" << std::endl;
            // Reverse-cavity
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (n+1.0);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (n+1.0);
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
            break;
        case 6:
            if( commRank == 0 )
                std::cout << "Bottom half of cavity velocity model" 
                          << std::endl;
            // Bottom half of cavity
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (n+1.0);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (n+1.0);
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
            break;
        case 7:
            if( commRank == 0 )
                std::cout << "Top half of cavity velocity model" << std::endl;
            // Top half of cavity
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (n+1.0);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (n+1.0);
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
            break;
        case 8:
            if( commRank == 0 )
                std::cout << "Increasing layers velocity model" << std::endl;
            // Increasing layers
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
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
            break;
        case 9:
            if( commRank == 0 )
                std::cout << "Decreasing layers velocity model" << std::endl;
            // Decreasing layers
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
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
            break;
        case 10:
            if( commRank == 0 )
                std::cout << "Sideways layers velocity model" << std::endl;
            // Sideways layers
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (n+1.0);
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
            break;
        case 11:
            if( commRank == 0 )
                std::cout << "Wedge velocity model" << std::endl;
            // Wedge
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (n+1.0);
                    double speed;
                    if( Z <= 0.4+0.1*Y )
                        speed = 2.;
                    else if( Z <= .8-0.2*Y )
                        speed = 1.5;
                    else
                        speed = 3.;
                    const int localOffset = 
                        yLocal*xLocalSize + zLocal*xLocalSize*yLocalSize;
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                        localVelocity[localOffset+xLocal] = speed;
                }
            }
            break;
        case 12:
            if( commRank == 0 )
                std::cout << "Uniform over [1,3] velocity model" << std::endl;
            // Uniform perturbation of unity (between 1 and 3)
            for( int i=0; i<xLocalSize*yLocalSize*zLocalSize; ++i )
                localVelocity[i] = 2.+plcg::ParallelUniform<double>();
            break;
        case 13:
            if( commRank == 0 )
                std::cout << "High-velocity separator" << std::endl;
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    double speed;
                    if( y >= n/2-6 && y < n/2-4 && z <= 3*n/4 )
                        speed = 1e10;
                    else
                        speed = 1.;
                    const int localOffset = 
                        yLocal*xLocalSize + zLocal*xLocalSize*yLocalSize;
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                        localVelocity[localOffset+xLocal] = speed;
                }
            }
            break;
        default:
            throw std::runtime_error("Invalid velocity model");
        }

        velocity.WritePlane( XY, n/2, "velocity-middleXY" );
        velocity.WritePlane( XZ, n/2, "velocity-middleXZ" );
        velocity.WritePlane( YZ, n/2, "velocity-middleYZ" );
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
        mpi::Barrier( comm );
        const double initialStartTime = mpi::Time(); 
        helmholtz.Initialize( velocity, panelScheme );
        mpi::Barrier( comm );
        const double initialStopTime = mpi::Time();
        const double initialTime = initialStopTime - initialStartTime;
        if( commRank == 0 )
            std::cout << "Finished initialization: " << initialTime 
                      << " seconds." << std::endl;

        DistUniformGrid<C> B( n, n, n, comm, 4 );
        C* localB = B.LocalBuffer();
        const double center0[] = { 0.5, 0.5, 0.1 };
        const double center1[] = { 0.25, 0.25, 0.1 };
        const double center2[] = { 0.75, 0.75, 0.5 };
        const double dir[] = { 0.57735, 0.57735, -0.57735 };
        double arg0[3], arg1[3], arg2[3];
        for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
        {
            const int z = zShift + zLocal*pz;
            const double Z = z / (n+1.0);
            arg0[2] = (Z-center0[2])*(Z-center0[2]);
            arg1[2] = (Z-center1[2])*(Z-center1[2]);
            arg2[2] = (Z-center2[2])*(Z-center2[2]);
            for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
            {
                const int y = yShift + yLocal*py;
                const double Y = y / (n+1.0);
                arg0[1] = (Y-center0[1])*(Y-center0[1]);
                arg1[1] = (Y-center1[1])*(Y-center1[1]);
                arg2[1] = (Y-center2[1])*(Y-center2[1]);
                for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                {
                    const int x = xShift + xLocal*px;
                    const double X = x / (n+1.0);
                    arg0[0] = (X-center0[0])*(X-center0[0]);
                    arg1[0] = (X-center1[0])*(X-center1[0]);
                    arg2[0] = (X-center2[0])*(X-center2[0]);

                    // Point sources
                    const C f0 = n*Exp(-10*n*(arg0[0]+arg0[1]+arg0[2]));
                    const C f1 = n*Exp(-10*n*(arg1[0]+arg1[1]+arg1[2]));
                    const C f2 = n*Exp(-10*n*(arg2[0]+arg2[1]+arg2[2]));

                    // Plane wave in direction 'dir' (away from PML)
                    const C planeWave =
                        Exp(C(0,omega*(X*dir[0]+Y*dir[1]+Z*dir[2])));

                    // Gaussian beam in direction 'dir'
                    const C fBeam =
                        Exp(-4*omega*(arg2[0]+arg2[1]+arg2[2]))*planeWave;

                    const int localIndex =
                        4*(xLocal + yLocal*xLocalSize +
                           zLocal*xLocalSize*yLocalSize);

                    localB[localIndex+0] = f0;
                    localB[localIndex+1] = f0+f1+f2;
                    localB[localIndex+2] = fBeam;
                    if( x >= pmlSize && x < n-pmlSize &&
                        y >= pmlSize && y < n-pmlSize &&
                        (!pmlOnTop || z >= pmlSize) && z < n-pmlSize )
                        localB[localIndex+3] = planeWave;
                    else
                        localB[localIndex+3] = 0;
                }
            }
        }

        B.WritePlane( XY, n/2, "source-middleXY" );
        B.WritePlane( XZ, n/2, "source-middleXZ" );
        B.WritePlane( YZ, n/2, "source-middleYZ" );
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
        mpi::Barrier( comm );
        const double solveStartTime = mpi::Time();
        const int m = 20;
        const double relTol = 1e-5;
        const bool viewIterates = false;
        helmholtz.Solve( B, m, relTol, viewIterates );
        mpi::Barrier( comm );
        const double solveStopTime = mpi::Time();
        const double solveTime = solveStopTime - solveStartTime;
        if( commRank == 0 )
            std::cout << "Finished solve: " << solveTime << " seconds." 
                      << std::endl;

        B.WritePlane( XY, n/2, "solution-middleXY" );
        B.WritePlane( XZ, n/2, "solution-middleXZ" );
        B.WritePlane( YZ, n/2, "solution-middleYZ" );
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
