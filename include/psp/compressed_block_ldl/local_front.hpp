/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_LOCAL_FRONT_COMPRESSION_HPP
#define PSP_LOCAL_FRONT_COMPRESSION_HPP

namespace psp {

template<typename Real>
void CompressFront
( CompressedFront<Complex<Real> >& front, 
  int depth, bool isLeaf, bool useQR, Real tolA, Real tolB );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

namespace internal {

template<typename Real>
void CompressSVD
( Matrix<Complex<Real>>& U, Matrix<Real>& s, Matrix<Complex<Real>>& V, 
  Real tolerance )
{
    typedef Complex<Real> C;

    // Compress
    const Real twoNorm = elem::MaxNorm( s );
    const Real cutoff = twoNorm*tolerance;
    const int k = s.Height();
    int numKeptModes = k;
    for( int i=1; i<k; ++i )
    {
        if( s.Get(i,0) <= cutoff )
        {
            numKeptModes = i;
            break;
        }
    }
    U.Resize( U.Height(), numKeptModes );
    s.Resize( numKeptModes, 1 );
    V.Resize( V.Height(), numKeptModes );

    const int worldRank = mpi::CommRank( mpi::COMM_WORLD );
    if( worldRank == 0 && numKeptModes > 0 )
    {
        std::ostringstream msg;
        msg << "kept " << numKeptModes << "/" << k << " modes, "
            << (Real(100)*numKeptModes)/Real(k) << "%" << std::endl;
        std::cout << msg.str();
    }
}

template<typename Real> 
inline void CompressBlock
( Matrix<Complex<Real>>& A, 
  std::vector<Matrix<Complex<Real>>>& greens, 
  std::vector<Matrix<Complex<Real>>>& coefficients, 
  int depth, bool useQR, Real tolerance )
{
    DEBUG_ONLY(CallStackEntry cse("internal::CompressBlock"))
    typedef Complex<Real> C;

    // Shuffle A into tall-skinny form
    const int s1 = A.Height() / depth;
    const int s2 = A.Width() / depth;
    Matrix<Complex<Real>> Z( s1*s2, depth*depth );
    for( int j2=0; j2<depth; ++j2 )
    {
        for( int j1=0; j1<depth; ++j1 )
        {
            for( int i2=0; i2<s2; ++i2 )
            {
                C* ZCol = Z.Buffer(i2*s1,j1+j2*depth);
                const C* ACol = A.LockedBuffer(j1*s1,i2+j2*s2);
                elem::MemCopy( ZCol, ACol, s1 );
            }
        }
    }

    // Turn compute a low-rank factorization, Z := U V^H, using an SVD of Z
    // (or an SVD of its 'R' from a QR factorization)
    Matrix<C> U, V;
    const int m = Z.Height();
    const int n = Z.Width();
    if( m > 1.5*n )
    {
        // QR
        Matrix<C> t;
        elem::QR( Z, t );
        
        // Compress
        Matrix<Real> s;
        Matrix<C> ZT, W;
        View( ZT, Z, 0, 0, n, n );
        W = ZT;
        elem::MakeTriangular( UPPER, W );
        elem::SVD( W, s, V, useQR );
        internal::CompressSVD( W, s, V, tolerance );
        elem::DiagonalScale( RIGHT, NORMAL, s, V );

        // Reexpand (TODO: Think about explicitly expanded reflectors)
        const int numKeptModes = s.Height();
        U.Resize( m, numKeptModes );
        Matrix<C> UT, UB;
        elem::PartitionDown
        ( U, UT,
             UB, n ); 
        UT = W;
        MakeZeros( UB );
        elem::ApplyPackedReflectors
        ( LEFT, LOWER, VERTICAL, BACKWARD, UNCONJUGATED, 0, Z, t, U );
    }
    else
    {
        Matrix<Real> s;
        U = Z;
        elem::SVD( U, s, V, useQR );
        internal::CompressSVD( U, s, V, tolerance );
        elem::DiagonalScale( RIGHT, NORMAL, s, V );
    }

    // Unshuffle each column of U into an s1 x s2 matrix in 'greens'
    // Unshuffle each column of V into a depth x depth matrix in 'coefficients'
    const int numKeptModes = U.Width();
    greens.resize( numKeptModes );
    coefficients.resize( numKeptModes );
    for( int t=0; t<numKeptModes; ++t )    
    {
        Matrix<C>& G = greens[t];
        Matrix<C>& D = coefficients[t];

        G.Resize( s1, s2 );
        D.Resize( depth, depth );

        // Unshuffle U 
        for( int i2=0; i2<s2; ++i2 )
        {
            C* GCol = G.Buffer(0,i2);
            const C* UCol = U.LockedBuffer(i2*s1,t);
            elem::MemCopy( GCol, UCol, s1 );
        }

        // Unshuffle V
        for( int j2=0; j2<depth; ++j2 )
            for( int j1=0; j1<depth; ++j1 )
                D.Set(j1,j2,Conj(V.Get(j1+j2*depth,t)));
    }
}

template<typename Real>
void SparsifyBlock
( Matrix<Complex<Real>>& B, 
  std::vector<int>& rows, 
  std::vector<int>& cols, 
  std::vector<Complex<Real>>& values ) 
{
    DEBUG_ONLY(CallStackEntry entry("internal::SparsifyBlock"))
    typedef Complex<Real> C;

    // Count the total number of nonzeros
    int numNonzeros=0;
    const Complex<Real> zero( 0, 0 );
    const int width = B.Width();
    const int height = B.Height();
    for( int j=0; j<width; ++j )
        for( int i=0; i<height; ++i )
            if( B.Get(i,j) != zero )
                ++numNonzeros;

    const int worldRank = mpi::CommRank( mpi::COMM_WORLD );
    if( worldRank == 0 && height != 0 && width != 0 )
    {
        std::ostringstream msg;
        msg << "kept " << numNonzeros << "/" << height*width << " entries, "
            << (Real(100)*numNonzeros)/Real(height*width) << "%" << std::endl;
        std::cout << msg.str();
    }

    // Allocate the space
    rows.resize( numNonzeros );
    cols.resize( numNonzeros );
    values.resize( numNonzeros );

    // Fill the data
    numNonzeros=0;
    for( int j=0; j<width; ++j )
    {
        for( int i=0; i<height; ++i )
        {
            if( B.Get(i,j) != zero )
            {
                rows[numNonzeros] = i;
                cols[numNonzeros] = j;
                values[numNonzeros] = B.Get(i,j);
                ++numNonzeros;
            }
        }
    }
}

} // namespace internal

template<typename Real>
void CompressFront
( CompressedFront<Complex<Real>>& front, 
  int depth, bool isLeaf, bool useQR, Real tolA, Real tolB )
{
    DEBUG_ONLY(CallStackEntry entry("CompressFront"))
    const int snSize = front.frontL.Width();
    Matrix<Complex<Real>> A, B;
    elem::PartitionDown
    ( front.frontL, A,
                    B, snSize );
    front.sT = A.Height() / depth;
    front.sB = B.Height() / depth;
    front.depth = depth;
    front.isLeaf = isLeaf;
    internal::CompressBlock
    ( A, front.AGreens, front.ACoefficients, depth, useQR, tolA );
    if( isLeaf )
        internal::SparsifyBlock
        ( B, front.BRows, front.BCols, front.BValues );
    else
        internal::CompressBlock
        ( B, front.BGreens, front.BCoefficients, depth, useQR, tolB );
    front.frontL.Empty();
}

} // namespace psp

#endif // PSP_LOCAL_FRONT_COMPRESSION_HPP
