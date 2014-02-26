/*
   Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and 
   The University of Texas at Austin
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#include "psp.hpp"
using namespace psp;

int
main( int argc, char* argv[] )
{
    typedef Complex<double> C;

    psp::Initialize( argc, argv );
    mpi::Comm comm = mpi::COMM_WORLD;
    const int commRank = mpi::CommRank( comm );

    try 
    {
        const int model = Input("--model","which velocity model: 0-15",2);
        const int n = Input("--n","dimension of n x n x n grid",40);
        const double omega = Input("--omega","angular frequency",30.);
        const bool pmlOnTop = Input("--pmlOnTop","PML on top boundary?",true);
        const int pmlSize = Input("--pmlSize","number of grid points of PML",5);
        const double sigma = Input("--sigma","PML amplitude",1.5);
        const double damping = Input("--damping","damping parameter",7.);
        const int planesPerPanel = Input
            ("--planesPerPanel","number of planes to process per subdomain",4);
        const PanelScheme panelScheme = (PanelScheme) Input
            ("--panelScheme",
             "frontal scheme: 0=1D LDL, 1=1D sel. inv.",1);
        const bool fullViz = Input("--fullViz","visualize volume?",false);
        const int nbFact = Input("--nbFact","factorization blocksize",96);
        const int nbSolve = Input("--nbSolve","solve blocksize",64);
        ProcessInput();

        if( model < 0 || model > 15 )
        {
            if( commRank == 0 )
                std::cout << "Invalid velocity model, must be in {0,...,15}\n"
                          << "---------------------------------------------\n"
                          << "0) Uniform\n"
                          << "1) Converging lense\n"
                          << "2) Wave guide\n"
                          << "3) Two decreasing layers\n"
                          << "4) Two increasing layers\n"
                          << "5) Two sideways layers\n"
                          << "6) Five decreasing layers\n"
                          << "7) Five increasing layers\n"
                          << "8) Five sideways layers\n"
                          << "9) Wedge\n"
                          << "10) Random\n"
                          << "11) Separator\n"
                          << "12) Cavity (will not converge quickly!)\n"
                          << "13) Reverse cavity\n"
                          << "14) Bottom half of cavity\n"
                          << "15) Top half of cavity\n"
                          << std::endl;
            psp::Finalize();
            return 0;
        }

        if( commRank == 0 )
        {
            std::cout << "Running with n=" << n << ", omega=" << omega 
                      << ", and planesPerPanel=" << planesPerPanel
                      << " with velocity model " << model << std::endl;
        }

        Boundary topBC = ( pmlOnTop ? PML : DIRICHLET );
        Discretization<double> disc
        ( omega, n, n, n, 1., 1., 1., PML, PML, PML, PML, topBC, 
          pmlSize, sigma );

        DistHelmholtz<double> helmholtz( disc, comm, damping, planesPerPanel );

        DistUniformGrid<double> velocity( n, n, n, comm );
        double* localVelocity = velocity.Buffer();
        const int xLocalSize = velocity.XLocalSize();
        const int yLocalSize = velocity.YLocalSize();
        const int zLocalSize = velocity.ZLocalSize();
        const int xShift = velocity.XShift();
        const int yShift = velocity.YShift();
        const int zShift = velocity.ZShift();
        const int px = velocity.XStride();
        const int py = velocity.YStride();
        const int pz = velocity.ZStride();
        switch( model )
        {
        case 0:
            if( commRank == 0 )
                std::cout << "Unit" << std::endl;
            for( int i=0; i<xLocalSize*yLocalSize*zLocalSize; ++i )
                localVelocity[i] = 1.;
            break;
        case 1:
            if( commRank == 0 )
                std::cout << "Converging lens" << std::endl;
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
                std::cout << "Wave guide" << std::endl;
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
                std::cout << "Two decreasing layers" << std::endl;
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
                std::cout << "Two increasing layers" << std::endl;
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
                const double speed = ( Z < 0.5 ? 4.0 : 1.0 );
                const int localOffset = zLocal*xLocalSize*yLocalSize;
                for( int i=0; i<xLocalSize*yLocalSize; ++i )
                    localVelocity[localOffset+i] = speed;
            }
            break;
        case 5:
            if( commRank == 0 )
                std::cout << "Two sideways layers" << std::endl;
            for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
            {
                const int y = yShift + yLocal*py;
                const double Y = y / (n+1.0);
                const double speed = ( Y < 0.5 ? 4.0 : 1.0 );
                for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
                {
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int localIndex = xLocal + yLocal*xLocalSize + 
                                               zLocal*xLocalSize*yLocalSize;
                        localVelocity[localIndex] = speed;
                    }
                }
            }
            break;
        case 6:
            if( commRank == 0 )
                std::cout << "Five decreasing layers" << std::endl;
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
                double speed;
                if( Z < 0.2 )
                    speed = 1;
                else if( Z < 0.4 )
                    speed = 1.75;
                else if( Z < 0.6 )
                    speed = 2.5;
                else if( Z < 0.8 )
                    speed = 3.25;
                else
                    speed = 4;
                const int localOffset = zLocal*xLocalSize*yLocalSize;
                for( int i=0; i<xLocalSize*yLocalSize; ++i )
                    localVelocity[i+localOffset] = speed;
            }
            break;
        case 7:
            if( commRank == 0 )
                std::cout << "Five increasing layers" << std::endl;
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
                double speed;
                if( Z < 0.2 )
                    speed = 4;
                else if( Z < 0.4 )
                    speed = 3.25;
                else if( Z < 0.6 )
                    speed = 2.5;
                else if( Z < 0.8 )
                    speed = 1.75;
                else
                    speed = 1;
                const int localOffset = zLocal*xLocalSize*yLocalSize;
                for( int i=0; i<xLocalSize*yLocalSize; ++i )
                    localVelocity[i+localOffset] = speed;
            }
            break;
        case 8:
            if( commRank == 0 )
                std::cout << "Five sideways layers" << std::endl;
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
                            speed = 4;
                        else if( X < 0.4 )
                            speed = 3.25;
                        else if( X < 0.6 )
                            speed = 2.5;
                        else if( X < 0.8 )
                            speed = 1.75;
                        else
                            speed = 1;
                        const int localIndex = xLocal + yLocal*xLocalSize + 
                            zLocal*xLocalSize*yLocalSize;
                        localVelocity[localIndex] = speed;
                    }
                }
            }
            break;
        case 9:
            if( commRank == 0 )
                std::cout << "Wedge" << std::endl;
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
        case 10:
            if( commRank == 0 )
                std::cout << "Uniform random over [1,3]" << std::endl;
            for( int i=0; i<xLocalSize*yLocalSize*zLocalSize; ++i )
                localVelocity[i] = elem::SampleUniform(1.,3.);
            break;
        case 11:
            if( commRank == 0 )
                std::cout << "High-velocity separator" << std::endl;
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (n+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (n+1.0);
                    double speed;
                    if( Y >= 0.25 && Y < 0.3 && Z <= 0.75 )
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
        case 12:
            if( commRank == 0 )
                std::cout << "Cavity (might not converge!)" << std::endl;
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
        case 13:
            if( commRank == 0 )
                std::cout << "Reverse-cavity" << std::endl;
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
        case 14:
            if( commRank == 0 )
                std::cout << "Bottom half of cavity" << std::endl;
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
        case 15:
            if( commRank == 0 )
                std::cout << "Top half of cavity" << std::endl;
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
        default:
            throw std::runtime_error("Invalid velocity model");
        }

        velocity.WritePlane( XY, n/2, "velocity-middleXY" );
        velocity.WritePlane( XZ, n/2, "velocity-middleXZ" );
        velocity.WritePlane( YZ, n/2, "velocity-middleYZ" );
        if( fullViz )
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

        elem::SetBlocksize( nbFact );
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
        C* localB = B.Buffer();
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

                    localB[localIndex+0] = -f0;
                    localB[localIndex+1] = -(f0+f1+f2);
                    localB[localIndex+2] = -fBeam;
                    if( x >= pmlSize && x < n-pmlSize &&
                        y >= pmlSize && y < n-pmlSize &&
                        (!pmlOnTop || z >= pmlSize) && z < n-pmlSize )
                        localB[localIndex+3] = -planeWave;
                    else
                        localB[localIndex+3] = 0;
                }
            }
        }

        B.WritePlane( XY, n/2, "source-middleXY" );
        B.WritePlane( XZ, n/2, "source-middleXZ" );
        B.WritePlane( YZ, n/2, "source-middleYZ" );
        if( fullViz )
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

        elem::SetBlocksize( nbSolve );
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
        if( fullViz )
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
    catch( ArgException& e ) { }
    catch( std::exception& e )
    {
        std::ostringstream os;
        os << "Caught exception on process " << commRank << ":\n" << e.what()
           << std::endl;
        std::cerr << os.str();
#ifndef RELEASE
        elem::DumpCallStack();
        cliq::DumpCallStack();
        psp::DumpCallStack();
#endif
    }

    psp::Finalize();
    return 0;
}
