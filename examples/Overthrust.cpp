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
    const int commSize = mpi::CommSize( comm );

    try 
    {
        const int nx = Input("--nx","first dimension of grid",80);
        const int ny = Input("--ny","second dimension of grid",80);
        const int nz = Input("--nz","third dimension of grid",20);
        const double omega = Input("--omega","angular frequency",0.5);
        int px = Input("--px","first dimension of process grid",0);
        int py = Input("--py","second dimension of process grid",0);
        int pz = Input("--pz","third dimension of process grid",0);
        const bool pmlOnTop = Input("--pmlOnTop","PML on top boundary?",true);
        const int pmlSize = Input("--pmlSize","number of grid points of PML",4);
        const double sigma = Input("--sigma","amplitude of PML profile",1.5*20);
        const double damping = Input("--damping","damping factor",7.);
        const int m = Input("--m","GMRES restart size",20);
        const double relTol = Input("--relTol","GMRES relative tolerance",1e-5);
        const bool viewIterates = Input("--viewIts","view iterates?",false);
        const int planesPerPanel = Input
            ("--planesPerPanel","number of planes per subdomain",4);
        const PanelScheme panelScheme = (PanelScheme)Input
            ("--panelScheme",
             "frontal scheme: 0=1D LDL, 1=2D sel. inv., 2=2D block LDL",1);
        const bool fullViz = Input("--fullViz","visualize volume?",false);
        const int nbFact = Input("--nbFact","factorization blocksize",96);
        const int nbSolve = Input("--nbSolve","solve blocksize",64);
        ProcessInput();

        // Try to intelligently build a process grid in a way which matches
        // any specified dimensions
        if( px == 0 )
        {
            if( py == 0 && pz == 0 )
            {
                px = DistUniformGrid<int>::FindCubicFactor( commSize );
                py = DistUniformGrid<int>::FindQuadraticFactor( commSize/px );
                pz = commSize / (px*py);
            }
            else if( py == 0 )
            {
                px = DistUniformGrid<int>::FindQuadraticFactor( commSize/pz );
                py = commSize / (px*pz);
            }
            else if( pz == 0 )
            {
                px = DistUniformGrid<int>::FindQuadraticFactor( commSize/py );
                pz = commSize / (px*py);
            }
            else
                px = commSize / (py*pz);
        }
        else if( py == 0 )
        {
            if( pz == 0 )
            {
                py = DistUniformGrid<int>::FindQuadraticFactor( commSize/px );
                pz = commSize / (px*py);
            }
            else
                py = commSize / (px*pz);
        }
        else if( pz == 0 )
            pz = commSize / (px*py);

        Boundary topBC = ( pmlOnTop ? PML : DIRICHLET );
        const double wx = 20.;
        const double wy = 20.;
        const double wz = 4.65;
        Discretization<double> disc
        ( omega, nx, ny, nz, wx, wy, wz, PML, PML, PML, PML, topBC,
          pmlSize, sigma );

        DistHelmholtz<double> helmholtz( disc, comm, damping, planesPerPanel );

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
        if( fullViz )
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

        DistUniformGrid<C> B( nx, ny, nz, px, py, pz, comm, 4 );
        const int xShift = B.XShift();
        const int yShift = B.YShift();
        const int zShift = B.ZShift();
        const int xLocalSize = B.XLocalSize();
        const int yLocalSize = B.YLocalSize();
        const int zLocalSize = B.ZLocalSize();
        C* localB = B.Buffer();
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

                    localB[localIndex+0] = -f0;
                    localB[localIndex+1] = -(f0+f1+f2);
                    localB[localIndex+2] = -fBeam;
                    if( x >= pmlSize && x < nx-pmlSize &&
                        y >= pmlSize && y < ny-pmlSize && 
                        (!pmlOnTop || z >= pmlSize) && z < nz-pmlSize )
                        localB[localIndex+3] = -planeWave;
                    else
                        localB[localIndex+3] = 0;
                }
            }
        }

        B.WritePlane( XY, nz/2, "source-middleXY" );
        B.WritePlane( XZ, ny/2, "source-middleXZ" );
        B.WritePlane( YZ, nx/2, "source-middleYZ" );
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
