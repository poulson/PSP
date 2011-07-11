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
    std::cout << "Ghost <xSize> <ySize> <zSize> <numLevels> "
                 "<strongly admissible?> <r> <print?> <print structure?> "
                 "<multiply identity?>" << std::endl;
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
    const int commRank = psp::mpi::CommRank( MPI_COMM_WORLD );
    const int commSize = psp::mpi::CommSize( MPI_COMM_WORLD );

    psp::UInt64 seed;
    seed.d[0] = 17U;
    seed.d[1] = 21U;
    psp::SeedParallelLcg( commRank, commSize, seed );

    if( argc < 10 )
    {
        if( commRank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const int xSize = atoi( argv[1] );
    const int ySize = atoi( argv[2] );
    const int zSize = atoi( argv[3] );
    const int numLevels = atoi( argv[4] );
    const bool stronglyAdmissible = atoi( argv[5] );
    const int r = atoi( argv[6] );
    const bool print = atoi( argv[7] );
    const bool printStructure = atoi( argv[8] );
    const bool multiplyIdentity = atoi( argv[9] );

    const int m = xSize*ySize*zSize;
    const int n = xSize*ySize*zSize;

    if( commRank == 0 )
    {
        std::cout << "----------------------------------------------------\n"
                  << "Testing complex double Quasi2dHMat packing/unpacking\n"
                  << "into DistQuasi2dHMat                                \n"
                  << "----------------------------------------------------" 
                  << std::endl;
    }
    try
    {
        typedef std::complex<double> Scalar;
        typedef psp::Quasi2dHMat<Scalar,false> Quasi2d;
        typedef psp::DistQuasi2dHMat<Scalar,false> DistQuasi2d;

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

        if( commRank == 0 )
        {
            std::cout << "Filling sparse matrices...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
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
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double fillStopTime = psp::mpi::WallTime();
        if( commRank == 0 )
        {
            std::cout << "done: " << fillStopTime-fillStartTime << " seconds." 
                      << std::endl;
        }

        // Convert to H-matrix form
        if( commRank == 0 )
        {
            std::cout << "Constructing H-matrices in serial...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double constructStartTime = psp::mpi::WallTime();
        Quasi2d ASerial
        ( S, numLevels, r, stronglyAdmissible, xSize, ySize, zSize );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double constructStopTime = psp::mpi::WallTime();
        if( commRank == 0 )
            std::cout << "done: " << constructStopTime-constructStartTime 
                      << " seconds." << std::endl;

        // Invert H-matrix
        if( commRank == 0 )
        {
            std::cout << "Inverting H-matrices in serial...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double invertStartTime = psp::mpi::WallTime();
        ASerial.DirectInvert();
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double invertStopTime = psp::mpi::WallTime();
        if( commRank == 0 )
        {
            std::cout << "done: " << invertStopTime-invertStartTime 
                      << " seconds." << std::endl;
            if( print )
                ASerial.Print("ASerial");
            if( printStructure )
            {
                ASerial.LatexWriteStructure("ASerial_structure");
                ASerial.MScriptWriteStructure("ASerial_structure");
            }
        }

        // Set up our subcommunicators and compute the packed sizes
        DistQuasi2d::Teams teams( MPI_COMM_WORLD );
        std::vector<std::size_t> packedSizes;
        DistQuasi2d::PackedSizes( packedSizes, ASerial, teams ); 
        const std::size_t myMaxSize = 
            *(std::max_element( packedSizes.begin(), packedSizes.end() ));

        // Pack for a DistQuasi2dHMat
        if( commRank == 0 )
        {
            std::cout << "Packing H-matrix for distribution...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double packStartTime = psp::mpi::WallTime();
        std::vector<psp::byte> sendBuffer( commSize*myMaxSize );
        std::vector<psp::byte*> packedPieces( commSize );
        for( int i=0; i<commSize; ++i )
            packedPieces[i] = &sendBuffer[i*myMaxSize];
        DistQuasi2d::Pack( packedPieces, ASerial, teams );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double packStopTime = psp::mpi::WallTime();
        if( commRank == 0 )
            std::cout << "done: " << packStopTime-packStartTime << " seconds."
                      << std::endl;

        // Compute the maximum package size
        int myIntMaxSize, intMaxSize;
        {
            myIntMaxSize = myMaxSize;
            psp::mpi::AllReduce
            ( &myIntMaxSize, &intMaxSize, 1, MPI_MAX, MPI_COMM_WORLD );
        }
        if( commRank == 0 )
        {
            std::cout << "Maximum per-process message size: " 
                      << ((double)intMaxSize)/(1024.*1024.) << " MB." 
                      << std::endl;
        }
 
        // AllToAll
        if( commRank == 0 )
        {
            std::cout << "AllToAll redistribution...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double allToAllStartTime = psp::mpi::WallTime();
        std::vector<psp::byte> recvBuffer( commSize*intMaxSize );
        psp::mpi::AllToAll
        ( &sendBuffer[0], myIntMaxSize, &recvBuffer[0], intMaxSize,
          MPI_COMM_WORLD );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double allToAllStopTime = psp::mpi::WallTime();
        if( commRank == 0 )
        {
            std::cout << "done: " << allToAllStopTime-allToAllStartTime
                      << " seconds." << std::endl;
        }

        // Unpack our part of the matrix defined by process 0 twice
        if( commRank == 0 )
        {
            std::cout << "Unpacking...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double unpackStartTime = psp::mpi::WallTime();
        DistQuasi2d A( &recvBuffer[0], teams );
        DistQuasi2d B( &recvBuffer[0], teams );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double unpackStopTime = psp::mpi::WallTime();
        if( commRank == 0 )
        {
            std::cout << "done: " << unpackStopTime-unpackStartTime
                      << " seconds." << std::endl;
        }
        if( printStructure )
        {
            A.LatexWriteLocalStructure("A_structure");
            A.MScriptWriteLocalStructure("A_structure");
        }

        // Attempt to multiply the two matrices
        if( commRank == 0 )
        {
            std::cout << "Multiplying distributed H-matrices...";
            std::cout.flush();
        }
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double multStartTime = psp::mpi::WallTime();
        DistQuasi2d C( teams );
        A.Multiply( (Scalar)1, B, C );
        psp::mpi::Barrier( MPI_COMM_WORLD );
        double multStopTime = psp::mpi::WallTime();
        if( commRank == 0 )
        {
            std::cout << "done: " << multStopTime-multStartTime
                      << " seconds." << std::endl;
        }
        if( printStructure )
        {
            C.LatexWriteLocalStructure("C_ghosted_structure");
            C.MScriptWriteLocalStructure("C_ghosted_structure");
        }

        // Check that CX = ABX for an arbitrary X
        if( commRank == 0 )
            std::cout << "Checking consistency: " << std::endl;
        psp::mpi::Barrier( MPI_COMM_WORLD );
        const int localHeight = A.LocalHeight();
        const int localWidth = A.LocalWidth();
        if( localHeight != localWidth )
            throw std::logic_error("A was not locally square");

        psp::Dense<Scalar> XLocal;
        if( multiplyIdentity )
        {
            const int firstLocalRow = A.FirstLocalRow();
            XLocal.Resize( localHeight, n );
            psp::hmat_tools::Scale( (Scalar)0, XLocal );
            for( int j=firstLocalRow; j<firstLocalRow+localHeight; ++j )
                XLocal.Set( j-firstLocalRow, j, (Scalar)1 );
        }
        else
        {
            const int numRhs = 30;
            XLocal.Resize( localHeight, numRhs );
            psp::ParallelGaussianRandomVectors( XLocal );
        }
        
        psp::Dense<Scalar> YLocal, ZLocal;
        // Y := AZ := ABX
        B.Multiply( (Scalar)1, XLocal, ZLocal );
        A.Multiply( (Scalar)1, ZLocal, YLocal );
        // Z := CX
        C.Multiply( (Scalar)1, XLocal, ZLocal );

        if( print )
        {
            std::ostringstream sY, sZ;
            sY << "YLocal_" << commRank << ".m";
            sZ << "ZLocal_" << commRank << ".m";
            std::ofstream YFile( sY.str().c_str() );
            std::ofstream ZFile( sZ.str().c_str() );

            YFile << "YLocal" << commRank << "=[\n";
            YLocal.Print( YFile, "" );
            YFile << "];\n";

            ZFile << "ZLocal" << commRank << "=[\n";
            ZLocal.Print( ZFile, "" );
            ZFile << "];\n";
        }

        // Compute the error norms and put ZLocal = YLocal-ZLocal
        double myNormVars[6] = { 0., 0., 0., 0., 0., 0. };
        for( int j=0; j<XLocal.Width(); ++j )
        {
            for( int i=0; i<localHeight; ++i )
            {
                const std::complex<double> truth = YLocal.Get(i,j);
                const std::complex<double> error = truth - ZLocal.Get(i,j);
                const double truthMag = psp::Abs( truth );
                const double errorMag = psp::Abs( error );
                ZLocal.Set( i, j, error );

                // RHS norms
                myNormVars[0] = std::max(myNormVars[0],truthMag);
                myNormVars[1] += truthMag;
                myNormVars[2] += truthMag*truthMag;
                // Error norms
                myNormVars[3] = std::max(myNormVars[3],errorMag);
                myNormVars[4] += errorMag;
                myNormVars[5] += errorMag*errorMag;
            }
        }
        double normVars[6];
        psp::mpi::Reduce( myNormVars, normVars, 6, 0, MPI_SUM, MPI_COMM_WORLD );
        if( commRank == 0 )
        {
            const double infTruth = normVars[0];
            const double L1Truth = normVars[1];
            const double L2Truth = sqrt(normVars[2]);
            const double infError = normVars[3];
            const double L1Error = normVars[4];
            const double L2Error = sqrt(normVars[5]);
            std::cout << "||ABX||_oo    = " << infTruth << "\n"
                      << "||ABX||_1     = " << L1Truth << "\n"
                      << "||ABX||_2     = " << L2Truth << "\n"
                      << "||CX-ABX||_oo = " << infError << "\n"
                      << "||CX-ABX||_1  = " << L1Error << "\n"
                      << "||CX-ABX||_2  = " << L2Error << std::endl;
        }

        if( print )
        {
            std::ostringstream sE;
            sE << "ELocal_" << commRank << ".m";
            std::ofstream EFile( sE.str().c_str() );

            EFile << "ELocal" << commRank << "=[\n";
            ZLocal.Print( EFile, "" );
            EFile << "];\n";
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Process " << commRank << " caught message: " << e.what() 
                  << std::endl;
#ifndef RELEASE
        psp::DumpCallStack();
#endif
    }
    
    MPI_Finalize();
    return 0;
}

