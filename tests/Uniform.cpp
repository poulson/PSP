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

int
main( int argc, char* argv[] )
{
    clique::Initialize( argc, argv );
    clique::mpi::Comm comm = clique::mpi::COMM_WORLD;
    const int commSize = clique::mpi::CommSize( comm );
    const int commRank = clique::mpi::CommRank( comm );
    
    FiniteDiffControl<double> control;
    control.stencil = SEVEN_POINT;
    control.nx = 30;
    control.ny = 30;
    control.nz = 50;
    control.wx = 6;
    control.wy = 6;
    control.wz = 10;
    control.omega = 5;
    control.Cx = 1.5*(2*M_PI);
    control.Cy = 1.5*(2*M_PI);
    control.Cz = 1.5*(2*M_PI);
    control.etax = 1.1;
    control.etay = 1.1;
    control.etaz = 1.1;
    control.imagShift = 1;
    control.cutoff = 96;
    control.numPlanesPerPanel = 5;
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

        // TODO: slowness.Visualize()???

        std::cout << "Beginning to initialize..." << std::endl;
        helmholtz.Initialize( slowness );
        std::cout << "Finished initialization." << std::endl;

        GridData<std::complex<double> > B
        ( 1, control.nx, control.ny, control.nz, XYZ, px, py, pz, comm );
        std::complex<double>* localB = B.LocalBuffer();
        for( int i=0; i<xLocalSize*yLocalSize*zLocalSize; ++i )
            localB[i] = 1;

        // TODO: B.Visualize()???

        std::cout << "Beginning solve..." << std::endl;
        helmholtz.Solve( B );
        std::cout << "Finished solve." << std::endl;

        // TODO: B.Visualize()???

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
