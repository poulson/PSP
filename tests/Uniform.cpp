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
    control.nx = 15;
    control.ny = 15;
    control.nz = 10;
    control.wx = 1;
    control.wy = 1;
    control.wz = 1;
    control.omega = 3;
    control.Cx = 1.5*(2*M_PI);
    control.Cy = 1.5*(2*M_PI);
    control.Cz = 1.5*(2*M_PI);
    control.etax = 1./6.;
    control.etay = 1./6.;
    control.etaz = 1./6.;
    control.imagShift = 1;
    control.cutoff = 16;
    control.numPlanesPerPanel = 1;
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

        if( commRank == 0 )
            std::cout << "Beginning to initialize..." << std::endl;
        helmholtz.Initialize( slowness );
        if( commRank == 0 )
            std::cout << "Finished initialization." << std::endl;

        /*
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
        */
        // TODO: REMOVE ME AFTER FINISHING DEBUGGING
        const int N = control.nx*control.ny*control.nz;
        const int localN = xLocalSize*yLocalSize*zLocalSize;
        GridData<std::complex<double> > B
        ( N, control.nx, control.ny, control.nz, XYZ, px, py, pz, comm );
        std::complex<double>* localB = B.LocalBuffer();
        std::memset( localB, 0, N*localN*sizeof(std::complex<double>) );
        const int xShift = B.XShift();
        const int yShift = B.YShift();
        const int zShift = B.ZShift();
        for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
        {
            for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
            {
                for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                {
                    const int x = xShift + xLocal*px;
                    const int y = yShift + yLocal*py;
                    const int z = zShift + zLocal*pz;
                    const int natural = x + y*control.nx + 
                                        z*control.nx*control.ny;
                    const int localIndex = B.LocalIndex( x, y, z );
                    localB[localIndex+natural] = 1;
                }
            }
        }
        double localNorm = elemental::blas::Nrm2( N*localN, localB, 1 );
        double localNormSquared = localNorm*localNorm;
        double normSquared;
        elemental::mpi::AllReduce
        ( &localNormSquared, &normSquared, 1, elemental::mpi::SUM, comm );
        if( commRank == 0 )
            std::cout << "Frobenius norm: " << sqrt(normSquared) << std::endl;

        // TODO: B.Visualize()???

        if( commRank == 0 )
            std::cout << "Beginning solve..." << std::endl;
        const int maxIterations = 500;
        helmholtz.Solve( B, QMR, maxIterations );
        if( commRank == 0 )
            std::cout << "Finished solve." << std::endl;

        // TODO: B.Visualize()???

        // TODO: REMOVE ME AFTER FINISHING DEBUGGING
        localNorm = elemental::blas::Nrm2( N*localN, localB, 1 );
        localNormSquared = localNorm*localNorm;
        elemental::mpi::AllReduce
        ( &localNormSquared, &normSquared, 1, elemental::mpi::SUM, comm );
        if( commRank == 0 )
            std::cout << "Frobenius norm: " << sqrt(normSquared) << std::endl;

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
