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
#ifndef PSP_LOCAL_FRONT_COMPRESSION_HPP
#define PSP_LOCAL_FRONT_COMPRESSION_HPP 1

namespace psp {

template<typename R>
void LocalFrontCompression
( LocalCompressedFront<Complex<R> >& front, 
  int depth, bool isLeaf, bool useQR=false );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

namespace internal {

template<typename R>
void CompressSVD
( Matrix<Complex<R> >& U, Matrix<R>& s, Matrix<Complex<R> >& V, 
  R tolerance )
{
    typedef Complex<R> C;

    // Compress
    const R twoNorm = elem::Norm( s, elem::MAX_NORM );
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
inline void LocalBlockCompression
( Matrix<Complex<R> >& A, 
  std::vector<Matrix<Complex<R> > >& greens, 
  std::vector<Matrix<Complex<R> > >& coefficients, 
  int depth, R tolerance, bool useQR )
{
#ifndef RELEASE
    PushCallStack("internal::LocalBlockCompression");
#endif
    typedef Complex<R> C;

    // Shuffle A into tall-skinny form
    const int s1 = A.Height() / depth;
    const int s2 = A.Width() / depth;
    Matrix<Complex<R> > Z( s1*s2, depth*depth );
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
        Matrix<R> s;
        Matrix<C> ZT, W;
        ZT.View( Z, 0, 0, n, n );
        W = ZT;
        elem::MakeTrapezoidal( LEFT, UPPER, 0, W );
        elem::SVD( W, s, V, useQR );
        internal::CompressSVD( W, s, V, tolerance );
        elem::DiagonalScale( RIGHT, NORMAL, s, V );

        // Reexpand (TODO: Think about explicitly expanded reflectors)
        const int numKeptModes = s.Height();
        U.ResizeTo( m, numKeptModes );
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
        Matrix<R> s;
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

        G.ResizeTo( s1, s2 );
        D.ResizeTo( depth, depth );

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
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
void LocalBlockSparsify
( Matrix<Complex<R> >& B, 
  std::vector<int>& rows, 
  std::vector<int>& cols, 
  std::vector<Complex<R> >& values ) 
{
#ifndef RELEASE
    PushCallStack("internal::LocalBlockSparsify");
#endif
    typedef Complex<R> C;

    // Count the total number of nonzeros
    int numNonzeros=0;
    const Complex<R> zero( 0, 0 );
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
            << ((R)100*numNonzeros)/((R)height*width) << "%" << std::endl;
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
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace internal

template<typename R>
void LocalFrontCompression
( LocalCompressedFront<Complex<R> >& front, 
  int depth, bool isLeaf, bool useQR )
{
#ifndef RELEASE
    PushCallStack("LocalFrontCompression");
#endif
    const R tolA = 0.02;
    const R tolB = 0.1;

    const int snSize = front.frontL.Width();
    Matrix<Complex<R> > A, B;
    elem::PartitionDown
    ( front.frontL, A,
                    B, snSize );
    front.sT = A.Height() / depth;
    front.sB = B.Height() / depth;
    front.depth = depth;
    front.isLeaf = isLeaf;
    internal::LocalBlockCompression
    ( A, front.AGreens, front.ACoefficients, depth, tolA, useQR );
    if( isLeaf )
        internal::LocalBlockSparsify
        ( B, front.BRows, front.BCols, front.BValues );
    else
        internal::LocalBlockCompression
        ( B, front.BGreens, front.BCoefficients, depth, tolB, useQR );
    front.frontL.Empty();
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_LOCAL_FRONT_COMPRESSION_HPP
