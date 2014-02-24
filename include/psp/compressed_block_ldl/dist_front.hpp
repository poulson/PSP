/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_DIST_FRONT_COMPRESSION_HPP
#define PSP_DIST_FRONT_COMPRESSION_HPP

namespace psp {

template<typename Real>
void CompressFront
( DistCompressedFront<Complex<Real>>& front, int depth, bool useQR,
  Real tolA, Real tolB );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

namespace internal {

template<typename Real>
void CompressSVD
( DistMatrix<Complex<Real>,STAR,STAR>& U, 
  DistMatrix<Real,STAR,STAR>& s, 
  DistMatrix<Complex<Real>,STAR,STAR>& V,
  Real tolerance )
{
    typedef Complex<Real> C;

    // Compress
    const Real twoNorm = elem::MaxNorm( s.Matrix() );
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
( DistMatrix<Complex<Real>>& A, 
  std::vector<DistMatrix<Complex<Real>>>& greens, 
  std::vector<DistMatrix<Complex<Real>,STAR,STAR>>& coefficients,
  int depth, bool useQR, Real tolerance )
{
    DEBUG_ONLY(CallStackEntry entry("internal::CompressBlock"))
    typedef Complex<Real> C;
    const Grid& g = A.Grid();
    const int gridRank = g.Rank();
    const int gridSize = g.Size();
    const int gridHeight = g.Height();
    const int gridWidth = g.Width();

    // Shuffle A into tall-skinny form
    const int s1 = A.Height() / depth;
    const int s2 = A.Width() / depth;
    DistMatrix<C> Z( s1*s2, depth*depth, g );
    DistMatrix<C,VC,STAR> Z_VC_STAR( g );
    {
        // Count the total amount of send data from A
        std::vector<int> sendCounts( gridSize, 0 );
        if( gridHeight >= s1 && gridWidth >= s2 )
        {
            const int ALocalHeight = A.LocalHeight();
            const int ALocalWidth = A.LocalWidth();
            const int AColShift = A.ColShift();
            const int ARowShift = A.RowShift();
            const int AColStride = A.ColStride();
            const int ARowStride = A.RowStride();

            // We should end up packing and unpacking in the same order by 
            // default since each process will only own one entry per s1 x s2
            // block of A
            for( int jLocal=0; jLocal<ALocalWidth; ++jLocal )
            {
                const int j = ARowShift + jLocal*ARowStride;
                const int i2 = j % s2;

                for( int iLocal=0; iLocal<ALocalHeight; ++iLocal )
                {
                    const int i = AColShift + iLocal*AColStride;
                    const int i1 = i % s1;

                    const int newProc = (i1+i2*s1) % gridSize;
                    ++sendCounts[newProc];
                }
            }
        }
        else
        {
            DistMatrix<C> D( g );
            for( int j2=0; j2<depth; ++j2 )
            {
                for( int j1=0; j1<depth; ++j1 )
                {
                    LockedView( D, A, j1*s1, j2*s2, s1, s2 );
                    const int DLocalHeight = D.LocalHeight();
                    const int DLocalWidth = D.LocalWidth();
                    const int DColShift = D.ColShift();
                    const int DRowShift = D.RowShift();
                    const int DColStride = D.ColStride();
                    const int DRowStride = D.RowStride();
                    for( int jLocal=0; jLocal<DLocalWidth; ++jLocal )
                    {
                        const int i2 = DRowShift + jLocal*DRowStride;
                        for( int iLocal=0; iLocal<DLocalHeight; ++iLocal )
                        {
                            const int i1 = DColShift + iLocal*DColStride;
                            const int newProc = (i1+i2*s1) % gridSize;
                            ++sendCounts[newProc];
                        }
                    }
                }
            }
        }

        // Set up the send displacements and count the total amount
        int totalSendCount=0;
        std::vector<int> sendDispls( gridSize );
        for( int proc=0; proc<gridSize; ++proc )
        {
            sendDispls[proc] = totalSendCount;
            totalSendCount += sendCounts[proc];
        }

        // Allocate the send buffer and then pack it
        std::vector<C> sendBuffer( totalSendCount );
        std::vector<int> offsets = sendDispls;
        if( gridHeight >= s1 && gridWidth >= s2 )
        {
            const int ALocalHeight = A.LocalHeight();
            const int ALocalWidth = A.LocalWidth();
            const int AColShift = A.ColShift();
            const int ARowShift = A.RowShift();
            const int AColStride = A.ColStride();
            const int ARowStride = A.RowStride();

            // We should end up packing and unpacking in the same order by 
            // default since each process will only own one entry per s1 x s2
            // block of A
            for( int jLocal=0; jLocal<ALocalWidth; ++jLocal )
            {
                const int j = ARowShift + jLocal*ARowStride;
                const int i2 = j % s2;

                for( int iLocal=0; iLocal<ALocalHeight; ++iLocal )
                {
                    const int i = AColShift + iLocal*AColStride;
                    const int i1 = i % s1;

                    const int newProc = (i1+i2*s1) % gridSize;
                    const C value = A.GetLocal(iLocal,jLocal);
                    sendBuffer[offsets[newProc]] = value;
                    ++offsets[newProc];
                }
            }
        }
        else
        {
            DistMatrix<C> D( g );
            for( int j2=0; j2<depth; ++j2 )
            {
                for( int j1=0; j1<depth; ++j1 )
                {
                    LockedView( D, A, j1*s1, j2*s2, s1, s2 );
                    const int DLocalHeight = D.LocalHeight();
                    const int DLocalWidth = D.LocalWidth();
                    const int DColShift = D.ColShift();
                    const int DRowShift = D.RowShift();
                    const int DColStride = D.ColStride();
                    const int DRowStride = D.RowStride();
                    for( int jLocal=0; jLocal<DLocalWidth; ++jLocal )
                    {
                        const int i2 = DRowShift + jLocal*DRowStride;
                        for( int iLocal=0; iLocal<DLocalHeight; ++iLocal )
                        {
                            const int i1 = DColShift + iLocal*DColStride;
                            const int newProc = (i1+i2*s1) % gridSize;
                            const C value = D.GetLocal(iLocal,jLocal);
                            sendBuffer[offsets[newProc]] = value;
                            ++offsets[newProc];
                        }
                    }
                }
            }
        }

        // Count the recv data
        std::vector<int> recvCounts( gridSize, 0 );
        const int AColAlign = A.ColAlign();
        const int ARowAlign = A.RowAlign();
        const int ZLocalHeight = Length( s1*s2, gridRank, gridSize );
        for( int j2=0; j2<depth; ++j2 )
        {
            for( int j1=0; j1<depth; ++j1 )
            {
                for( int iLocal=0; iLocal<ZLocalHeight; ++iLocal )
                {
                    const int i = gridRank + iLocal*gridSize;
                    const int i1 = i % s1;
                    const int i2 = i / s1;

                    const int origRow = (i1+j1*s1+AColAlign) % gridHeight;
                    const int origCol = (i2+j2*s2+ARowAlign) % gridWidth;
                    const int origProc = origRow + origCol*gridHeight;
                    ++recvCounts[origProc];
                }
            }
        }

        // Set up the recv displacements and count the total amount
        int totalRecvCount=0;
        std::vector<int> recvDispls( gridSize );
        for( int proc=0; proc<gridSize; ++proc )
        {
            recvDispls[proc] = totalRecvCount;
            totalRecvCount += recvCounts[proc];
        }

        // Communicate (and free the send data)
        std::vector<C> recvBuffer( totalRecvCount );
        mpi::AllToAll
        ( &sendBuffer[0], &sendCounts[0], &sendDispls[0],
          &recvBuffer[0], &recvCounts[0], &recvDispls[0], g.Comm() );
        sendBuffer.clear();
        sendCounts.clear();
        sendDispls.clear();

        // Unpack the recv data into Z[VC,* ]
        Z_VC_STAR.Resize( s1*s2, depth*depth );
        offsets = recvDispls;

        for( int j2=0; j2<depth; ++j2 )
        {
            for( int j1=0; j1<depth; ++j1 )
            {
                for( int iLocal=0; iLocal<ZLocalHeight; ++iLocal )
                {
                    const int i = gridRank + iLocal*gridSize;
                    const int i1 = i % s1;
                    const int i2 = i / s1;

                    const int origCol = (i2+j2*s2+ARowAlign) % gridWidth;
                    const int origRow = (i1+j1*s1+AColAlign) % gridHeight;
                    const int origProc = origRow + origCol*gridHeight;

                    const C value = recvBuffer[offsets[origProc]];
                    Z_VC_STAR.SetLocal( iLocal, j1+j2*depth, value );
                    ++offsets[origProc];
                }
            }
        }
    }
    // TODO: TSQR here on Z[VC,* ]?
    Z = Z_VC_STAR;
    Z_VC_STAR.Empty();

    DistMatrix<C> U( g );
    DistMatrix<C,STAR,STAR> V_STAR_STAR( g );
    const int m = Z.Height();
    const int n = Z.Width();
    if( m > 1.5*n )
    {
        // QR
        DistMatrix<C,MD,STAR> t( g );
        elem::QR( Z, t );

        // Compress
        DistMatrix<C> ZT( g );
        View( ZT, Z, 0, 0, n, n );

        DistMatrix<Real,STAR,STAR> s_STAR_STAR( n, 1, g );
        DistMatrix<C,STAR,STAR> W_STAR_STAR( ZT );
        elem::MakeTriangular( UPPER, W_STAR_STAR );
        V_STAR_STAR.Resize( n, n );
        elem::SVD
        ( W_STAR_STAR.Matrix(), s_STAR_STAR.Matrix(), V_STAR_STAR.Matrix(), 
          useQR );
        internal::CompressSVD
        ( W_STAR_STAR, s_STAR_STAR, V_STAR_STAR, tolerance );
        DiagonalScale( RIGHT, NORMAL, s_STAR_STAR, V_STAR_STAR );

        // Reexpand (TODO: Think about explicitly expanded reflectors)
        const int numKeptModes = s_STAR_STAR.Height();
        U.Resize( m, numKeptModes );
        DistMatrix<C> UT( g ), UB( g );
        elem::PartitionDown
        ( U, UT,
             UB, n );
        UT = W_STAR_STAR;
        MakeZeros( UB );
        elem::ApplyPackedReflectors
        ( LEFT, LOWER, VERTICAL, BACKWARD, UNCONJUGATED, 0, Z, t, U );
    }
    else
    {
        const int k = std::min( m, n );
        DistMatrix<Real,STAR,STAR> s_STAR_STAR( k, 1, g);
        DistMatrix<C,STAR,STAR> U_STAR_STAR( Z );
        V_STAR_STAR.Resize( n, k );
        elem::SVD
        ( U_STAR_STAR.Matrix(), s_STAR_STAR.Matrix(), V_STAR_STAR.Matrix(), 
          useQR );
        internal::CompressSVD
        ( U_STAR_STAR, s_STAR_STAR, V_STAR_STAR, tolerance );
        U = U_STAR_STAR;
        DiagonalScale( RIGHT, NORMAL, s_STAR_STAR, V_STAR_STAR );
    }

    // Unpack the columns of U and V into 'greens' and 'coefficients'
    const int numKeptModes = U.Width();
    greens.resize( numKeptModes ); 
    coefficients.resize( numKeptModes );
    for( int t=0; t<numKeptModes; ++t )
    {
        DistMatrix<C>& G = greens[t];
        DistMatrix<C,STAR,STAR>& D = coefficients[t];

        G.SetGrid( g );
        D.SetGrid( g );
        G.Resize( s1, s2 );
        D.Resize( depth, depth );

        // Unshuffle U 
        // (if the number of processes evenly divided s1, this would just be
        //  a scatter within process rows)
        // 
        // TODO: Optimize this...it will have an extremely high latency cost.
        //
        DistMatrix<C> UStrip( g ), GStrip( g );
        for( int i2=0; i2<s2; ++i2 )
        {
            LockedView( UStrip, U, i2*s1, t, s1, 1 );
            View( GStrip, G, 0, i2, s1, 1 );
            GStrip = UStrip; 
        }

        // Unshuffle V
        for( int j2=0; j2<depth; ++j2 )
            for( int j1=0; j1<depth; ++j1 )
                D.Set( j1, j2, Conj(V_STAR_STAR.Get(j1+j2*depth,t)) );
    }
}

} // namespace internal

template<typename Real>
void CompressFront
( DistCompressedFront<Complex<Real>>& front, int depth, bool useQR, 
  Real tolA, Real tolB )
{
    DEBUG_ONLY(CallStackEntry entry("CompressFront"))
    const Grid& grid = front.frontL.Grid();
    const int snSize = front.frontL.Width();
    DistMatrix<Complex<Real>> A(grid), B(grid);
    elem::PartitionDown
    ( front.frontL, A,
                    B, snSize );
    front.sT = A.Height() / depth;
    front.sB = B.Height() / depth;
    front.depth = depth;
    front.grid = &grid;
    internal::CompressBlock
    ( A, front.AGreens, front.ACoefficients, depth, useQR, tolA );
    internal::CompressBlock
    ( B, front.BGreens, front.BCoefficients, depth, useQR, tolB );
    front.frontL.Empty();
}

} // namespace psp

#endif // PSP_DIST_FRONT_COMPRESSION_HPP
