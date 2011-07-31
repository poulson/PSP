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

void Usage()
{
    std::cout << "MultiplyHMat <xSize> <ySize> <zSize> <numLevels> "
                 "<strongly admissible?> <maxRank> "
                 "<print?> <print structure?> <multiply identity?>" 
              << std::endl;
}

template<typename Real>
void
FormRow
( int x, int y, int z, int xSize, int ySize, int zSize, 
  std::vector< std::complex<Real> >& row, std::vector<int>& colIndices )
{
    typedef std::complex<Real> Scalar;
    const int rowIdx = x + xSize*y + xSize*ySize*z;

    row.resize( 0 );
    colIndices.resize( 0 );

    // Set up the diagonal entry
    colIndices.push_back( rowIdx );
    row.push_back( (Scalar)8 );

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        colIndices.push_back( (x-1) + xSize*y + xSize*ySize*z );
        row.push_back( (Scalar)-1 );
    }

    // Back connection to (x+1,y,z)
    if( x != xSize-1 )
    {
        colIndices.push_back( (x+1) + xSize*y + xSize*ySize*z );
        row.push_back( (Scalar)-1 );
    }

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        colIndices.push_back( x + xSize*(y-1) + xSize*ySize*z );
        row.push_back( (Scalar)-1 );
    }

    // Right connection to (x,y+1,z)
    if( y != ySize-1 )
    {
        colIndices.push_back( x + xSize*(y+1) + xSize*ySize*z );
        row.push_back( (Scalar)-1 );
    }

    // Top connection to (x,y,z-1)
    if( z != 0 )
    {
        colIndices.push_back( x + xSize*y + xSize*ySize*(z-1) );
        row.push_back( (Scalar)-1 );
    }

    // Bottom connection to (x,y,z+1)
    if( z != zSize-1 )
    {
        colIndices.push_back( x + xSize*y + xSize*ySize*(z+1) );
        row.push_back( (Scalar)-1 );
    }
}

int
main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    psp::UInt64 seed;
    seed.d[0] = 17U;
    seed.d[1] = 21U;
    psp::SeedSerialLcg( seed );

    const int commSize = psp::mpi::CommSize( MPI_COMM_WORLD );
    const int commRank = psp::mpi::CommRank( MPI_COMM_WORLD );
    if( commSize != 1 )
    {
        if( commRank == 0 )
            std::cerr << "This test must be run with a single MPI process" 
                      << std::endl;
        MPI_Finalize();
        return 0;
    }

    if( argc < 10 )
    {
        Usage();
        MPI_Finalize();
        return 0;
    }
    const int xSize = atoi( argv[1] );
    const int ySize = atoi( argv[2] );
    const int zSize = atoi( argv[3] );
    const int numLevels = atoi( argv[4] );
    const bool stronglyAdmissible = atoi( argv[5] );
    const int maxRank = atoi( argv[6] );
    const bool print = atoi( argv[7] );
    const bool printStructure = atoi( argv[8] );
    const bool multiplyIdentity = atoi( argv[9] );

    const int m = xSize*ySize*zSize;
    const int n = xSize*ySize*zSize;

    std::cout << "----------------------------------------------------\n"
              << "Testing H-matrix mult using generated matrices      \n"
              << "----------------------------------------------------" 
              << std::endl;
    try
    {
        typedef std::complex<double> Scalar;
        typedef psp::Quasi2dHMat<Scalar,false> Quasi2d;

        psp::Sparse<Scalar> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;

        std::vector<int> map;
        Quasi2d::BuildNaturalToHierarchicalMap
        ( map, xSize, ySize, zSize, numLevels );

        std::vector<int> inverseMap( m );
        for( int i=0; i<m; ++i )
            inverseMap[map[i]] = i;

        std::cout << "Filling sparse matrices...";
        std::cout.flush();
        double fillStartTime = psp::mpi::WallTime();
        std::vector<Scalar> row;
        std::vector<int> colIndices;
        for( int i=0; i<m; ++i )
        {
            S.rowOffsets.push_back( S.nonzeros.size() );
            const int iNatural = inverseMap[i];
            const int x = iNatural % xSize;
            const int y = (iNatural/xSize) % ySize;
            const int z = iNatural/(xSize*ySize);

            FormRow
            ( x, y, z, xSize, ySize, zSize, row, colIndices );

            for( unsigned j=0; j<row.size(); ++j )
            {
                S.nonzeros.push_back( row[j] );
                S.columnIndices.push_back( map[colIndices[j]] );
            }
        }
        S.rowOffsets.push_back( S.nonzeros.size() );
        double fillStopTime = psp::mpi::WallTime();
        std::cout << "done: " << fillStopTime-fillStartTime << " seconds." 
                  << std::endl;

        // Convert to H-matrix form
        std::cout << "Constructing H-matrix...";
        std::cout.flush();
        double constructStartTime = psp::mpi::WallTime();
        Quasi2d A
        ( S, numLevels, maxRank, stronglyAdmissible, xSize, ySize, zSize );
        double constructStopTime = psp::mpi::WallTime();
        std::cout << "done: " << constructStopTime-constructStartTime 
                  << " seconds." << std::endl;

        // Invert H-matrix and make a copy
        std::cout << "Inverting H-matrix and making copy...";
        std::cout.flush();
        double invertStartTime = psp::mpi::WallTime();
        A.DirectInvert();
        Quasi2d B;
        B.CopyFrom( A );
        double invertStopTime = psp::mpi::WallTime();
        std::cout << "done: " << invertStopTime-invertStartTime 
                  << " seconds." << std::endl;
        if( print )
            A.Print("A");
        if( printStructure )
        {
            A.LatexWriteStructure("A_structure");
            A.MScriptWriteStructure("A_structure");
        }

        // Attempt to multiply the two matrices
        std::cout << "Multiplying H-matrices...";
        std::cout.flush();
        double multStartTime = psp::mpi::WallTime();
        Quasi2d C;
        A.Multiply( (Scalar)1, B, C );
        double multStopTime = psp::mpi::WallTime();
        std::cout << "done: " << multStopTime-multStartTime
                  << " seconds." << std::endl;
        if( printStructure )
        {
            C.LatexWriteStructure("C_structure");
            C.MScriptWriteStructure("C_structure");
        }

        // Check that CX = ABX for an arbitrary X
        std::cout << "Checking consistency: " << std::endl;
        psp::Dense<Scalar> X;
        if( multiplyIdentity )
        {
            X.Resize( m, n );
            psp::hmat_tools::Scale( (Scalar)0, X );
            for( int j=0; j<n; ++j )
                X.Set( j, j, (Scalar)1 );
        }
        else
        {
            const int numRhs = 30;
            X.Resize( m, numRhs );
            psp::SerialGaussianRandomVectors( X );
        }
        
        psp::Dense<Scalar> Y, Z;
        // Y := AZ := ABX
        B.Multiply( (Scalar)1, X, Z );
        A.Multiply( (Scalar)1, Z, Y );
        // Z := CX
        C.Multiply( (Scalar)1, X, Z );

        if( print )
        {
            std::ofstream YFile( "Y.m" );
            std::ofstream ZFile( "Z.m" );

            YFile << "Y=[\n";
            Y.Print( YFile, "" );
            YFile << "];\n";

            ZFile << "Z=[\n";
            Z.Print( ZFile, "" );
            ZFile << "];\n";
        }

        // Compute the error norms and put Z := Y - Z
        double infTruth=0, infError=0, 
               L1Truth=0, L1Error=0, 
               L2SquaredTruth=0, L2SquaredError=0;
        for( int j=0; j<X.Width(); ++j )
        {
            for( int i=0; i<m; ++i )
            {
                const std::complex<double> truth = Y.Get(i,j);
                const std::complex<double> error = truth - Z.Get(i,j);
                const double truthMag = psp::Abs( truth );
                const double errorMag = psp::Abs( error );
                Z.Set( i, j, error );

                // RHS norms
                infTruth = std::max(infTruth,truthMag);
                L1Truth += truthMag;
                L2SquaredTruth += truthMag*truthMag;
                // Error norms
                infError = std::max(infError,errorMag);
                L1Error += errorMag;
                L2SquaredError += errorMag*errorMag;
            }
        }
        std::cout << "||ABX||_oo    = " << infTruth << "\n"
                  << "||ABX||_1     = " << L1Truth << "\n"
                  << "||ABX||_2     = " << sqrt(L2SquaredTruth) << "\n"
                  << "||CX-ABX||_oo = " << infError << "\n"
                  << "||CX-ABX||_1  = " << L1Error << "\n"
                  << "||CX-ABX||_2  = " << sqrt(L2SquaredError) 
                  << std::endl;

        if( print )
        {
            std::ofstream EFile( "E.m" );
            EFile << "E=[\n"; 
            Z.Print( EFile, "" );
            EFile << "];\n";
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        psp::DumpCallStack();
#endif
    }
    
    MPI_Finalize();
    return 0;
}

