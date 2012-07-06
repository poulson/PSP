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
    std::cout << "Overthrust <omega> <damping> <numPlanesPerPanel> "
                 "<fact blocksize> <solve blocksize> <panelScheme> <SQMR?> "
                 "<viz?>\n"
              << "  <omega>: Frequency (in rad/sec) of problem\n"
              << "  <damping>: imaginary shift [2 pi is standard]\n"
              << "  <numPlanesPerPanel>: depth of sparse-direct solves\n"
              << "  <fact blocksize>: factorization algorithmic blocksize\n"
              << "  <solve blocksize>: solve algorithmic blocksize\n"
              << "  <panel scheme>: NORMAL_1D=0, FAST_2D_LDL=1, "
                 "COMPRESSED_2D_BLOCK_LDL=2\n"
              << "  <SQMR?>: GMRES iff 0, SQMR otherwise\n"
              << "  <full viz?>: Full visualization iff != 0\n"
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    psp::Initialize( argc, argv );
    psp::mpi::Comm comm = psp::mpi::COMM_WORLD;
    const int commSize = psp::mpi::CommSize( comm );
    const int commRank = psp::mpi::CommRank( comm );

    if( argc < 9 )
    {
        if( commRank == 0 )
            Usage();
        psp::Finalize();
        return 0;
    }
    int argNum=1;
    const double omega = atof( argv[argNum++] );
    const double damping = atof( argv[argNum++] );
    const int numPlanesPerPanel = atoi( argv[argNum++] );
    const int factBlocksize = atoi( argv[argNum++] );
    const int solveBlocksize = atoi( argv[argNum++] );
    const PanelScheme panelScheme = (PanelScheme)atoi( argv[argNum++] );
    const bool useSQMR = atoi( argv[argNum++] );
    const bool fullVisualize = atoi( argv[argNum++] );

    const int Nx = 801;
    const int Ny = 801;
    const int Nz = 187;

    if( commRank == 0 )
    {
        std::cout << "Running with omega=" << omega 
                  << ", numPlanesPerPanel=" << numPlanesPerPanel
                  << ", and damping=" << damping << std::endl;
    }
    
    Discretization<double> disc
    ( omega, Nx, Ny, Nz, 20., 20., 4.65, 
      PML, PML, PML, PML, DIRICHLET, 5, 5, 5 );

    try 
    {
        DistHelmholtz<double> helmholtz
        ( control, comm, damping, numPlanesPerPanel );

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

        GridData<double> velocity( 1, Nx, Ny, Nz, XYZ, px, py, pz, comm );
        double* localVelocity = velocity.LocalBuffer();
        const int xLocalSize = velocity.XLocalSize();
        const int yLocalSize = velocity.YLocalSize();
        const int zLocalSize = velocity.ZLocalSize();
        const int xShift = velocity.XShift();
        const int yShift = velocity.YShift();
        const int zShift = velocity.ZShift();
        std::ostringstream os;
        os << "overthrust_" << commRank << ".dat";
        std::ifstream velocityFile;
        velocityFile.open( os.str().c_str(), std::ios::in|std::ios::binary );
        velocityFile.read
        ( (char*)localVelocity, 
          xLocalSize*yLocalSize*zLocalSize*sizeof(double) );
        velocityFile.close();

        velocity.WritePlane( XY, Nz/2, "velocity-middleXY" );
        velocity.WritePlane( XZ, Ny/2, "velocity-middleXZ" );
        velocity.WritePlane( YZ, Nx/2, "velocity-middleYZ" );
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

        GridData<std::complex<double> > 
            B( 3, Nx, Ny, Nz, XYZ, px, py, pz, comm );
        std::complex<double>* localB = B.LocalBuffer();
        const double center0[] = { 0.5, 0.5, 0.1 };
        const double center1[] = { 0.25, 0.25, 0.1 };
        const double center2[] = { 0.75, 0.75, 0.1 };
        double arg0[3], arg1[3], arg2[3];
        for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
        {
            const int z = zShift + zLocal*pz;
            const double zSqueeze = control.wz / control.wx;
            const double Z = zSqueeze * z / (Nz+1.0);
            arg0[2] = (Z-center0[2]*zSqueeze)*(Z-center0[2]*zSqueeze);
            arg1[2] = (Z-center1[2]*zSqueeze)*(Z-center1[2]*zSqueeze);
            arg2[2] = (Z-center2[2]*zSqueeze)*(Z-center2[2]*zSqueeze);
            for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
            {
                const int y = yShift + yLocal*py;
                const double Y = y / (Ny+1.0);
                arg0[1] = (Y-center0[1])*(Y-center0[1]);
                arg1[1] = (Y-center1[1])*(Y-center1[1]);
                arg2[1] = (Y-center2[1])*(Y-center2[1]);
                for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                {
                    const int x = xShift + xLocal*px;
                    const double X = x / (Nx+1.0);
                    arg0[0] = (X-center0[0])*(X-center0[0]);
                    arg1[0] = (X-center1[0])*(X-center1[0]);
                    arg2[0] = (X-center2[0])*(X-center2[0]);
                    
                    const int localIndex = 
                        3*(xLocal + yLocal*xLocalSize + 
                           zLocal*xLocalSize*yLocalSize);
                    const std::complex<double> f0 = 
                        Nx*std::exp(-10*Nx*(arg0[0]+arg0[1]+arg0[2]));
                    const std::complex<double> f1 = 
                        Nx*std::exp(-10*Nx*(arg1[0]+arg1[1]+arg1[2]));
                    const std::complex<double> f2 = 
                        Nx*std::exp(-10*Nx*(arg2[0]+arg2[1]+arg2[2]));
                    localB[localIndex+0] = f0;
                    localB[localIndex+1] = f1;
                    localB[localIndex+2] = f2;
                }
            }
        }

        B.WritePlane( XY, Nz/2, "source-middleXY" );
        B.WritePlane( XZ, Ny/2, "source-middleXZ" );
        B.WritePlane( YZ, Nx/2, "source-middleYZ" );
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
            helmholtz.SolveWithGMRES( B, 20, 1e-5 );
        psp::mpi::Barrier( comm );
        const double solveStopTime = psp::mpi::Time();
        const double solveTime = solveStopTime - solveStartTime;
        if( commRank == 0 )
            std::cout << "Finished solve: " << solveTime << " seconds." 
                      << std::endl;

        B.WritePlane( XY, Nz/2, "solution-middleXY" );
        B.WritePlane( XZ, Ny/2, "solution-middleXZ" );
        B.WritePlane( YZ, Nx/2, "solution-middleYZ" );
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
