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
    std::cout << "Invert <xSize> <ySize> <zSize> <numLevels> <r> <print?>" 
              << std::endl;
}

const double omega = 10.0;
const double C = 1.5*(2*M_PI);

template<typename Real>
std::complex<Real>
sInv( int k, int b, int size )
{
    if( (k+1)<b )
    {
        const Real delta = b-(k+1);
        const Real h = 1. / (size+1);
        const Real realPart = 1;
        const Real imagPart = ::C*delta*delta/(b*b*b*h*::omega);
        return std::complex<Real>(realPart,imagPart);
    }
    else if( k>(size-b) )
    {
        const Real delta = k-(size-b);
        const Real h = 1. / (size+1);
        const Real realPart = 1;
        const Real imagPart = ::C*delta*delta/(b*b*b*h*::omega);
        return std::complex<Real>(realPart,imagPart);
    }
    else
        return 1;
}

template<typename Real>
void
FormRow
( Real imagShift, 
  int x, int y, int z, int xSize, int ySize, int zSize, int pmlSize,
  std::vector< std::complex<Real> >& row, std::vector<int>& colIndices )
{
    typedef std::complex<Real> Scalar;
    const int rowIdx = x + xSize*y + xSize*ySize*z;
    const Real hx = 1.0 / (xSize+1);
    const Real hy = 1.0 / (ySize+1);
    const Real hz = 1.0 / (zSize+1);

    const Scalar s1InvL = sInv<Real>( x-1, pmlSize, xSize );
    const Scalar s1InvM = sInv<Real>( x,   pmlSize, xSize );
    const Scalar s1InvR = sInv<Real>( x+1, pmlSize, xSize );
    const Scalar s2InvL = sInv<Real>( y-1, pmlSize, ySize );
    const Scalar s2InvM = sInv<Real>( y,   pmlSize, ySize );
    const Scalar s2InvR = sInv<Real>( y+1, pmlSize, ySize );
    const Scalar s3InvL = sInv<Real>( z-1, pmlSize, zSize );
    const Scalar s3InvM = sInv<Real>( z,   pmlSize, zSize );
    const Scalar s3InvR = sInv<Real>( z+1, pmlSize, zSize );

    // Compute all of the x-shifted terms
    const Scalar xTempL = s2InvM*s3InvM/s1InvL;
    const Scalar xTempM = s2InvM*s3InvM/s1InvM;
    const Scalar xTempR = s2InvM*s3InvM/s1InvR;
    const Scalar xTermL = (xTempL+xTempM)/(2*hx*hx);
    const Scalar xTermR = (xTempR+xTempM)/(2*hx*hx);

    // Compute all of the y-shifted terms
    const Scalar yTempL = s1InvM*s3InvM/s2InvL;
    const Scalar yTempM = s1InvM*s3InvM/s2InvM;
    const Scalar yTempR = s1InvM*s3InvM/s2InvR;
    const Scalar yTermL = (yTempL+yTempM)/(2*hy*hy);
    const Scalar yTermR = (yTempR+yTempM)/(2*hy*hy);

    // Compute all of the z-shifted terms
    const Scalar zTempL = s1InvM*s2InvM/s3InvL;
    const Scalar zTempM = s1InvM*s2InvM/s3InvM;
    const Scalar zTempR = s1InvM*s2InvM/s3InvR;
    const Scalar zTermL = (zTempL+zTempM)/(2*hz*hz);
    const Scalar zTermR = (zTempR+zTempM)/(2*hz*hz);

    // Compute the center term
    const Scalar alpha = 1;
    const Scalar centerTerm = -(xTermL+xTermR+yTermL+yTermR+zTermL+zTermR) +
        (::omega*alpha)*(::omega*alpha)*s1InvM*s2InvM*s3InvM + 
        std::complex<Real>(0,imagShift);

    row.resize( 0 );
    colIndices.resize( 0 );

    // Set up the diagonal entry
    colIndices.push_back( rowIdx );
    row.push_back( centerTerm );

    // Front connection to (x-1,y,z)
    if( x != 0 )
    {
        colIndices.push_back( (x-1) + xSize*y + xSize*ySize*z );
        row.push_back( xTermL );
    }

    // Back connection to (x+1,y,z)
    if( x != xSize-1 )
    {
        colIndices.push_back( (x+1) + xSize*y + xSize*ySize*z );
        row.push_back( xTermR );
    }

    // Left connection to (x,y-1,z)
    if( y != 0 )
    {
        colIndices.push_back( x + xSize*(y-1) + xSize*ySize*z );
        row.push_back( yTermL );
    }

    // Right connection to (x,y+1,z)
    if( y != ySize-1 )
    {
        colIndices.push_back( x + xSize*(y+1) + xSize*ySize*z );
        row.push_back( yTermR );
    }

    // Top connection to (x,y,z-1)
    if( z != 0 )
    {
        colIndices.push_back( x + xSize*y + xSize*ySize*(z-1) );
        row.push_back( zTermL );
    }

    // Bottom connection to (x,y,z+1)
    if( z != zSize-1 )
    {
        colIndices.push_back( x + xSize*y + xSize*ySize*(z+1) );
        row.push_back( zTermR );
    }
}

int
main( int argc, char* argv[] )
{
    if( argc < 7 )
    {
        Usage();
        return 0;
    }
    const int xSize = atoi( argv[1] );
    const int ySize = atoi( argv[2] );
    const int zSize = atoi( argv[3] );
    const int numLevels = atoi( argv[4] );
    const int r = atoi( argv[5] );
    const bool print = atoi( argv[6] );

    const int m = xSize*ySize*zSize;
    const int n = xSize*ySize*zSize;

    std::cout << "--------------------------------------------------\n"
              << "Testing complex double Quasi2dHMatrix inversion   \n"
              << "--------------------------------------------------" 
              << std::endl;
    try
    {
        typedef std::complex<double> Scalar;
        typedef psp::Quasi2dHMatrix<Scalar,false> Quasi2d;

        psp::SparseMatrix<Scalar> S;
        S.height = m;
        S.width = n;
        S.symmetric = false;
        const int pmlSize = 5;
        const double imagShift = 1.0;

        std::vector<int> map;

        Quasi2d::BuildNaturalToHierarchicalMap
        ( map, xSize, ySize, zSize, numLevels );

        std::vector<int> inverseMap( m );
        for( int i=0; i<m; ++i )
            inverseMap[map[i]] = i;

        std::cout << "Filling sparse matrix...";
        std::cout.flush();
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
            ( imagShift, 
              x, y, z, xSize, ySize, zSize, pmlSize, row, colIndices );

            for( unsigned j=0; j<row.size(); ++j )
            {
                S.nonzeros.push_back( row[j] );
                S.columnIndices.push_back( map[colIndices[j]] );
            }
        }
        S.rowOffsets.push_back( S.nonzeros.size() );
        std::cout << "done." << std::endl;
        if( print )
            S.Print( "S" );

        // Convert to H-matrix form
        std::cout << "Constructing H-matrix...";
        std::cout.flush();
        Quasi2d H( S, numLevels, r, true, xSize, ySize, zSize );
        //Quasi2d H( S, numLevels, r, false, xSize, ySize, zSize );
        std::cout << "done" << std::endl;
        if( print )
        {
            H.Print( "H" );
            H.PrintStructure( "H Structure" );
        }

        // Test against a vector of all 1's
        psp::Vector<Scalar> x;
        x.Resize( m );
        Scalar* xBuffer = x.Buffer();
        for( int i=0; i<m; ++i )
            xBuffer[i] = 1.0;
        if( print )
            x.Print( "x" );
        std::cout << "Multiplying H-matrix by a vector...";
        std::cout.flush();
        psp::Vector<Scalar> y;
        H.MapVector( 1.0, x, y );
        std::cout << "done" << std::endl;
        if( print )
            y.Print( "y := H x ~= S x" );

        // Make a copy for inversion
        std::cout << "Making a copy of the H-matrix for inversion...";
        std::cout.flush();
        Quasi2d invH;
        invH.CopyFrom( H );
        std::cout << "done" << std::endl;

        // Perform a direct inversion
        std::cout << "Directly inverting the H-matrix...";
        std::cout.flush();
        invH.DirectInvert();
        std::cout << "done" << std::endl;
        if( print )
            invH.Print( "inv(H)" );

        // Test for discrepancies in x and inv(H) H x
        std::cout << "Multiplying the inverse by a vector...";
        std::cout.flush();
        psp::Vector<Scalar> z;
        invH.MapVector( 1.0, y, z );
        std::cout << "done" << std::endl;
        if( print )
        {
            y.Print( "y := H x ~= S x" );
            z.Print( "z := inv(H) H x ~= x" );
        }
        const double xNormL1 = m;
        const double xNormL2 = sqrt( m );
        const double xNormLInf = 1.0;
        double errorNormL1, errorNormL2, errorNormLInf;
        {
            errorNormL1 = 0;
            errorNormLInf = 0;

            Scalar* zBuffer = z.Buffer();
            double errorNormL2Squared = 0;
            for( int i=0; i<m; ++i )
            {
                const double deviation = std::abs( zBuffer[i] - 1.0 );
                errorNormL1 += deviation;
                errorNormL2Squared += deviation*deviation;
                errorNormLInf = std::max( errorNormLInf, deviation );
            }
            errorNormL2 = std::sqrt( errorNormL2Squared );
        }
        std::cout << "||e||_1 =  " << errorNormL1 << "\n"
                  << "||e||_2 =  " << errorNormL2 << "\n"
                  << "||e||_oo = " << errorNormLInf << "\n"
                  << "\n"
                  << "||e||_1  / ||x||_1  = " << errorNormL1/xNormL1 << "\n"
                  << "||e||_2  / ||x||_2  = " << errorNormL2/xNormL2 << "\n"
                  << "||e||_oo / ||x||_oo = " << errorNormLInf/xNormLInf << "\n"
                  << std::endl;
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught message: " << e.what() << std::endl;
#ifndef RELEASE
        psp::DumpCallStack();
#endif
    }
    
    return 0;
}
