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
    std::cout << "Interpolate <velocity> <m1> <m2> <m3> <n1> <n2> <n3>"
              << "  <velocity>: Which velocity field to use, {0,...,12}\n"
              << "  <m1,m2,m3>: dimensions of original velocity field\n"
              << "  <n1,n2,n3>: dimensions of interpolated velocity field\n"
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
              << std::endl;
}

int
main( int argc, char* argv[] )
{
    psp::Initialize( argc, argv );
    mpi::Comm comm = mpi::COMM_WORLD;
    const int commSize = mpi::CommSize( comm );
    const int commRank = mpi::CommRank( comm );

    if( argc < 8 )
    {
        if( commRank == 0 )
            Usage();
        psp::Finalize();
        return 0;
    }
    int argNum=1;
    const int velocityModel = atoi(argv[argNum++]);
    const int m1 = atoi(argv[argNum++]);
    const int m2 = atoi(argv[argNum++]);
    const int m3 = atoi(argv[argNum++]);
    const int n1 = atoi(argv[argNum++]);
    const int n2 = atoi(argv[argNum++]);
    const int n3 = atoi(argv[argNum++]);

    if( velocityModel < 0 || velocityModel > 12 )
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
                      << std::endl;
        psp::Finalize();
        return 0;
    }

    try 
    {
        DistUniformGrid<double> velocity( 1, m1, m2, m3, XYZ, comm );
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
                const double Z = z / (m3+1.0);
                const double argZ = (Z-0.5)*(Z-0.5);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (m2+1.0);
                    const double argY = (Y-0.5)*(Y-0.5);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (m1+1.0);
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
                    const double Y = y / (m2+1.0);
                    const double argY = (Y-0.5)*(Y-0.5);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (m1+1.0);
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
            break;
        case 3:
            if( commRank == 0 )
                std::cout << "Two layer velocity model" << std::endl;
            // Two layers
            for( int zLocal=0; zLocal<zLocalSize; ++zLocal )
            {
                const int z = zShift + zLocal*pz;
                const double Z = z / (m3+1.0);
                const double speed = ( Z >= 0.5 ? 8.0 : 1.0 );
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
                const double Z = z / (m3+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (m2+1.0);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (m1+1.0);
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
                const double Z = z / (m3+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (m2+1.0);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (m1+1.0);
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
                const double Z = z / (m3+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (m2+1.0);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (m1+1.0);
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
                const double Z = z / (m3+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (m2+1.0);
                    for( int xLocal=0; xLocal<xLocalSize; ++xLocal )
                    {
                        const int x = xShift + xLocal*px;
                        const double X = x / (m1+1.0);
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
                const double Z = z / (m3+1.0);
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
                const double Z = z / (m3+1.0);
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
                        const double X = x / (m1+1.0);
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
                const double Z = z / (m3+1.0);
                for( int yLocal=0; yLocal<yLocalSize; ++yLocal )
                {
                    const int y = yShift + yLocal*py;
                    const double Y = y / (m2+1.0);
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
        default:
            throw std::runtime_error("Invalid velocity model");
        }

        if( commRank == 0 )
        {
            std::cout << "Writing original velocity data...";
            std::cout.flush();
        }
        velocity.WritePlane( XY, m3/2, "original-middleXY" );
        velocity.WritePlane( XZ, m2/2, "original-middleXZ" );
        velocity.WritePlane( YZ, m1/2, "original-middleYZ" );
        velocity.WriteVolume("original");
        mpi::Barrier( comm );
        if( commRank == 0 )
            std::cout << "done" << std::endl;

        velocity.InterpolateTo( n1, n2, n3 );

        if( commRank == 0 )
        {
            std::cout << "Writing interpolated velocity data...";
            std::cout.flush();
        }
        velocity.WritePlane( XY, n3/2, "interpolated-middleXY" );
        velocity.WritePlane( XZ, n2/2, "interpolated-middleXZ" );
        velocity.WritePlane( YZ, n1/2, "interpolated-middleYZ" );
        velocity.WriteVolume("interpolated");
        mpi::Barrier( comm );
        if( commRank == 0 )
            std::cout << "done" << std::endl;
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
