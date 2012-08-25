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
    std::cout << "Overthrust <nx> <ny> <nz> <omega> <px> <py> <pz> "
                 "[PML on top=true] [pmlSize=5] [sigma=1.5*20] [damping=7] "
                 "[# planes/panel=4] [panel scheme=1] [viz=1] "
                 "[fact blocksize=96] [solve blocksize=64]\n"
              << "\n"
              << "  nx,ny,nz: size of grid in each dimension\n"
              << "  omega: frequency (in rad/sec) of problem\n"
              << "  px,py,pz: 3D process grid dimensions\n"
              << "  PML on top: PML or Dirichlet b.c. on top?\n"
              << "  pmlSize: width of PML in grid points\n"
              << "  sigma: maximum height of imaginary coordinate stretching\n"
              << "  damping: imaginary freq shift for preconditioner\n"
              << "  # of planes/panel: number of planes per subdomain\n"
              << "  panel scheme: LDL_1D=0, LDL_SELINV_2D=1\n"
              << "  full viz: full volume visualization iff != 0\n" 
              << "  fact blocksize: factorization algorithmic blocksize\n"
              << "  solve blocksize: solve algorithmic blocksize\n"
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

    if( argc < 5 )
    {
        if( commRank == 0 )
            Usage();
        psp::Finalize();
        return 0;
    }
    int argNum=1;
    const int nx = atoi(argv[argNum++]);
    const int ny = atoi(argv[argNum++]);
    const int nz = atoi(argv[argNum++]);
    const double omega = atof(argv[argNum++]);
    const int px = atoi(argv[argNum++]);
    const int py = atoi(argv[argNum++]);
    const int pz = atoi(argv[argNum++]);
    const bool pmlOnTop = ( argc>argNum ? atoi(argv[argNum++]) : true );
    const int pmlSize = ( argc>argNum ? atoi(argv[argNum++]) : 5 );
    const double sigma = ( argc>argNum ? atof(argv[argNum++]) : 1.5*20 );
    const double damping = ( argc>argNum ? atof(argv[argNum++]) : 7. );
    const int numPlanesPerPanel = ( argc>argNum ? atoi(argv[argNum++]) : 4 );
    const PanelScheme panelScheme = 
        ( argc>argNum ? (PanelScheme)atoi(argv[argNum++]) 
                      : CLIQUE_LDL_SELINV_2D );
    const bool fullVisualize = ( argc>argNum ? atoi(argv[argNum++]) : true );
    const int factBlocksize = ( argc>argNum ? atoi(argv[argNum++]) : 96 );
    const int solveBlocksize = ( argc>argNum ? atoi(argv[argNum++]) : 64 );

    try 
    {
        Boundary topBC = ( pmlOnTop ? PML : DIRICHLET );
        const double wx = 20.;
        const double wy = 20.;
        const double wz = 4.65;
        Discretization<double> disc
        ( omega, nx, ny, nz, wx, wy, wz, PML, PML, PML, PML, topBC,
          pmlSize, sigma );

        DistHelmholtz<double> helmholtz
        ( disc, comm, damping, numPlanesPerPanel );

        const int nxOrig = 801;
        const int nyOrig = 801;
        const int nzOrig = 187;
        if( commRank == 0 )
            std::cout << "Loading sequential " 
                      << nxOrig << " x " << nyOrig << " x " << nzOrig 
                      << " overthrust data..." << std::endl;
        DistUniformGrid<double> velocity
        ( nxOrig, nyOrig, nzOrig, px, py, pz, comm );
        velocity.SequentialLoad("overthrust.dat");

        if( commRank == 0 )
            std::cout << "Interpolating to "
                      << nx << " x " << ny << " x " << nz << " grid..." 
                      << std::endl;
        velocity.InterpolateTo( nx, ny, nz );

        velocity.WritePlane( XY, nz/2, "velocity-middleXY" );
        velocity.WritePlane( XZ, ny/2, "velocity-middleXZ" );
        velocity.WritePlane( YZ, nx/2, "velocity-middleYZ" );
        if( fullVisualize )
        {
            if( commRank == 0 )
            {
                std::cout << "Writing full velocity data...";
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

        DistUniformGrid<C> B( nx, ny, nz, px, py, pz, comm, 4 );
        const int xShift = B.XShift();
        const int yShift = B.YShift();
        const int zShift = B.ZShift();
        const int xLocalSize = B.XLocalSize();
        const int yLocalSize = B.YLocalSize();
        const int zLocalSize = B.ZLocalSize();
        C* localB = B.LocalBuffer();
        const double center0[] = { 0.5, 0.5, 0.1 };
        const double center1[] = { 0.25, 0.25, 0.1 };
        const double center2[] = { 0.75, 0.75, 0.5 };
        const double dir[] = { 0.57735, 0.57735, -0.57735 };
        double arg0[3], arg1[3], arg2[3];
        for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
        {
            const int z = zShift + zLocal*pz;
            const double zSqueeze = disc.wz / disc.wx;
            const double Z = zSqueeze * z / (nz+1.0);
            arg0[2] = (Z-center0[2]*zSqueeze)*(Z-center0[2]*zSqueeze);
            arg1[2] = (Z-center1[2]*zSqueeze)*(Z-center1[2]*zSqueeze);
            arg2[2] = (Z-center2[2]*zSqueeze)*(Z-center2[2]*zSqueeze);
            for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
            {
                const int y = yShift + yLocal*py;
                const double Y = y / (ny+1.0);
                arg0[1] = (Y-center0[1])*(Y-center0[1]);
                arg1[1] = (Y-center1[1])*(Y-center1[1]);
                arg2[1] = (Y-center2[1])*(Y-center2[1]);
                for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                {
                    const int x = xShift + xLocal*px;
                    const double X = x / (nx+1.0);
                    arg0[0] = (X-center0[0])*(X-center0[0]);
                    arg1[0] = (X-center1[0])*(X-center1[0]);
                    arg2[0] = (X-center2[0])*(X-center2[0]);

                    // Point sources
                    const C f0 = nx*Exp(-10*nx*(arg0[0]+arg0[1]+arg0[2]));
                    const C f1 = nx*Exp(-10*nx*(arg1[0]+arg1[1]+arg1[2]));
                    const C f2 = nx*Exp(-10*nx*(arg2[0]+arg2[1]+arg2[2]));

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
                    if( x >= pmlSize && x < nx-pmlSize &&
                        y >= pmlSize && y < ny-pmlSize && 
                        (!pmlOnTop || z >= pmlSize) && z < nz-pmlSize )
                        localB[localIndex+3] = planeWave;
                    else
                        localB[localIndex+3] = 0;
                }
            }
        }

        B.WritePlane( XY, nz/2, "source-middleXY" );
        B.WritePlane( XZ, ny/2, "source-middleXZ" );
        B.WritePlane( YZ, nx/2, "source-middleYZ" );
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

        B.WritePlane( XY, nz/2, "solution-middleXY" );
        B.WritePlane( XZ, ny/2, "solution-middleXZ" );
        B.WritePlane( YZ, nx/2, "solution-middleYZ" );
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
