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
    std::cout << "Overthrust <nx> <ny> <nz> <omega> "
                 "[PML on top=false] [damping=7] [# planes/panel=4] "
                 "[panel scheme=1] [viz=0] "
                 "[fact blocksize=96] [solve blocksize=64]\n"
              << "\n"
              << "  <nx,ny,nz>: size of grid in each dimension\n"
              << "  <omega>: frequency (in rad/sec) of problem\n"
              << "  [PML on top=false]: PML or Dirichlet b.c. on top?\n"
              << "  [damping=7]: imaginary freq shift for preconditioner\n"
              << "  [# of planes/panel=4]: number of planes per subdomain\n"
              << "  [panel scheme=1]: NORMAL_1D=0, FAST_2D_LDL=1\n"
              << "  [full viz=0]: full volume visualization iff != 0\n" 
              << "  [fact blocksize=96]: factorization algorithmic blocksize\n"
              << "  [solve blocksize=64]: solve algorithmic blocksize\n"
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    psp::Initialize( argc, argv );
    psp::mpi::Comm comm = psp::mpi::COMM_WORLD;
    const int commSize = psp::mpi::CommSize( comm );
    const int commRank = psp::mpi::CommRank( comm );

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
    const bool pmlOnTop = ( argc>=6 ? atoi(argv[argNum++]) : false );
    const double damping = ( argc>=7 ? atof(argv[argNum++]) : 7. );
    const int numPlanesPerPanel = ( argc>=8 ? atoi(argv[argNum++]) : 4 );
    const PanelScheme panelScheme = 
        ( argc>=9 ? (PanelScheme)atoi(argv[argNum++]) : CLIQUE_FAST_2D_LDL );
    const bool fullVisualize = ( argc>=10 ? atoi(argv[argNum++]) : true );
    const int factBlocksize = ( argc>=11 ? atoi(argv[argNum++]) : 96 );
    const int solveBlocksize = ( argc>=12 ? atoi(argv[argNum++]) : 64 );

    try 
    {
        const double wx = 20.;
        const double wy = 20.;
        const double wz = 4.65;
        const int pmlWidth = 5;
        Boundary topBC = ( pmlOnTop ? PML : DIRICHLET );
        Discretization<double> disc
        ( omega, nx, ny, nz, wx, wy, wz, 
          PML, PML, PML, PML, topBC, pmlWidth, pmlWidth, pmlWidth );

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
        ( 1, nxOrig, nyOrig, nzOrig, XYZ, comm );
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
            psp::mpi::Barrier( comm );
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

        DistUniformGrid<Complex<double> > B( 3, nx, ny, nz, XYZ, comm );
        const int px = B.XStride();
        const int py = B.YStride();
        const int pz = B.ZStride();
        const int xShift = B.XShift();
        const int yShift = B.YShift();
        const int zShift = B.ZShift();
        const int xLocalSize = B.XLocalSize();
        const int yLocalSize = B.YLocalSize();
        const int zLocalSize = B.ZLocalSize();
        Complex<double>* localB = B.LocalBuffer();
        const double center0[] = { 0.5, 0.5, 0.1 };
        const double center1[] = { 0.25, 0.25, 0.1 };
        const double center2[] = { 0.75, 0.75, 0.1 };
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
                    
                    const int localIndex = 
                        3*(xLocal + yLocal*xLocalSize + 
                           zLocal*xLocalSize*yLocalSize);
                    const Complex<double> f0 = 
                        nx*Exp(-10*nx*(arg0[0]+arg0[1]+arg0[2]));
                    const Complex<double> f1 = 
                        nx*Exp(-10*nx*(arg1[0]+arg1[1]+arg1[2]));
                    const Complex<double> f2 = 
                        nx*Exp(-10*nx*(arg2[0]+arg2[1]+arg2[2]));
                    localB[localIndex+0] = f0;
                    localB[localIndex+1] = f1;
                    localB[localIndex+2] = f2;
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
        psp::mpi::Barrier( comm );
        const double solveStartTime = psp::mpi::Time();
        helmholtz.SolveWithGMRES( B, 20, 1e-5 );
        psp::mpi::Barrier( comm );
        const double solveStopTime = psp::mpi::Time();
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
