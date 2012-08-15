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
#ifndef PSP_DIST_FRONT_COMPRESSION_HPP
#define PSP_DIST_FRONT_COMPRESSION_HPP 1

namespace psp {

template<typename R>
void DistFrontCompression
( DistCompressedFront<Complex<R> >& front, int depth, bool useQR=false );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

namespace internal {

template<typename R>
void CompressSVD
( DistMatrix<Complex<R>,STAR,STAR>& U, 
  DistMatrix<R,STAR,STAR>& s, 
  DistMatrix<Complex<R>,STAR,STAR>& V,
  R tolerance )
{
    typedef Complex<R> C;

    // Compress
    const R twoNorm = elem::Norm( s.LocalMatrix(), elem::MAX_NORM );
    const R cutoff = twoNorm*tolerance;
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
    U.ResizeTo( U.Height(), numKeptModes );
    s.ResizeTo( numKeptModes, 1 );
    V.ResizeTo( V.Height(), numKeptModes );

    const int worldRank = mpi::CommRank( mpi::COMM_WORLD );
    if( worldRank == 0 && numKeptModes > 0 )
    {
        std::ostringstream msg;
        msg << "kept " << numKeptModes << "/" << k << " modes, "
            << ((R)100*numKeptModes)/((R)k) << "%" << std::endl;
        std::cout << msg.str();
    }
}

template<typename R> 
inline void DistBlockCompression
( DistMatrix<Complex<R> >& A, 
  std::vector<DistMatrix<Complex<R> > >& greens, 
  std::vector<DistMatrix<Complex<R>,STAR,STAR> >& coefficients,
  int depth, R tolerance, bool useQR=false )
{
#ifndef RELEASE
    PushCallStack("DistBlockCompression");
#endif
    typedef Complex<R> C;
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
                    D.LockedView( A, j1*s1, j2*s2, s1, s2 );
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
                    D.LockedView( A, j1*s1, j2*s2, s1, s2 );
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
        const int AColAlignment = A.ColAlignment();
        const int ARowAlignment = A.RowAlignment();
        const int ZLocalHeight = LocalLength( s1*s2, gridRank, gridSize );
        for( int j2=0; j2<depth; ++j2 )
        {
            for( int j1=0; j1<depth; ++j1 )
            {
                for( int iLocal=0; iLocal<ZLocalHeight; ++iLocal )
                {
                    const int i = gridRank + iLocal*gridSize;
                    const int i1 = i % s1;
                    const int i2 = i / s1;

                    const int origRow = (i1+j1*s1+AColAlignment) % gridHeight;
                    const int origCol = (i2+j2*s2+ARowAlignment) % gridWidth;
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
        Z_VC_STAR.ResizeTo( s1*s2, depth*depth );
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

                    const int origCol = (i2+j2*s2+ARowAlignment) % gridWidth;
                    const int origRow = (i1+j1*s1+AColAlignment) % gridHeight;
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
        ZT.View( Z, 0, 0, n, n );

        DistMatrix<R,STAR,STAR> s_STAR_STAR( n, 1, g );
        DistMatrix<C,STAR,STAR> W_STAR_STAR( ZT );
        elem::MakeTrapezoidal( LEFT, UPPER, 0, W_STAR_STAR );
        V_STAR_STAR.ResizeTo( n, n );
        elem::SVD
        ( W_STAR_STAR.LocalMatrix(),
          s_STAR_STAR.LocalMatrix(),
          V_STAR_STAR.LocalMatrix(), useQR );
        internal::CompressSVD
        ( W_STAR_STAR, s_STAR_STAR, V_STAR_STAR, tolerance );
        DiagonalScale( RIGHT, NORMAL, s_STAR_STAR, V_STAR_STAR );

        // Reexpand (TODO: Think about explicitly expanded reflectors)
        const int numKeptModes = s_STAR_STAR.Height();
        U.ResizeTo( m, numKeptModes );
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
        DistMatrix<R,STAR,STAR> s_STAR_STAR( k, 1, g);
        DistMatrix<C,STAR,STAR> U_STAR_STAR( Z );
        V_STAR_STAR.ResizeTo( n, k );
        elem::SVD
        ( U_STAR_STAR.LocalMatrix(), 
          s_STAR_STAR.LocalMatrix(), 
          V_STAR_STAR.LocalMatrix(), useQR );
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
        G.ResizeTo( s1, s2 );
        D.ResizeTo( depth, depth );

        // Unshuffle U 
        // (if the number of processes evenly divided s1, this would just be
        //  a scatter within process rows)
        // 
        // TODO: Optimize this...it will have an extremely high latency cost.
        //
        DistMatrix<C> UStrip( g ), GStrip( g );
        for( int i2=0; i2<s2; ++i2 )
        {
            UStrip.LockedView( U, i2*s1, t, s1, 1 );
            GStrip.View( G, 0, i2, s1, 1 );
            GStrip = UStrip; 
        }

        // Unshuffle V
        for( int j2=0; j2<depth; ++j2 )
            for( int j1=0; j1<depth; ++j1 )
                D.Set( j1, j2, Conj(V_STAR_STAR.Get(j1+j2*depth,t)) );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace internal

template<typename R>
void DistFrontCompression
( DistCompressedFront<Complex<R> >& front, int depth, bool useQR )
{
#ifndef RELEASE
    PushCallStack("DistFrontCompression");
#endif
    const R tolA = 0.02;
    const R tolB = 0.1;

    const Grid& grid = front.frontL.Grid();
    const int snSize = front.frontL.Width();
    DistMatrix<Complex<R> > A(grid), B(grid);
    elem::PartitionDown
    ( front.frontL, A,
                    B, snSize );
    front.sT = A.Height() / depth;
    front.sB = B.Height() / depth;
    front.depth = depth;
    front.grid = &grid;
    internal::DistBlockCompression
    ( A, front.AGreens, front.ACoefficients, depth, tolA, useQR );
    internal::DistBlockCompression
    ( B, front.BGreens, front.BCoefficients, depth, tolB, useQR );
    front.frontL.Empty();
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_DIST_FRONT_COMPRESSION_HPP
