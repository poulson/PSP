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
( Matrix<Complex<R> >& A, 
  std::vector<Matrix<Complex<R> > >& greens, 
  std::vector<Matrix<Complex<R> > >& coefficients,
  int depth, bool useQR=false );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

namespace internal {

template<typename R>
void CompressSVD
( Matrix<Complex<R> >& U, Matrix<R>& s, Matrix<Complex<R> >& V )
{
    typedef Complex<R> C;
    const R tolerance = 0.02;

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

} // namespace internal

template<typename R> 
inline void LocalFrontCompression
( Matrix<Complex<R> >& A, 
  std::vector<Matrix<Complex<R> > >& greens, 
  std::vector<Matrix<Complex<R> > >& coefficients, 
  int depth, bool useQR )
{
#ifndef RELEASE
    PushCallStack("LocalFrontCompression");
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
    if( Z.Height() > 1.5*Z.Width() )
    {
        // QR
        Matrix<C> t;
        elem::QR( Z, t );
        
        // Compress
        Matrix<R> s;
        Matrix<C> ZT, W;
        ZT.View( Z, 0, 0, Z.Width(), Z.Width() );
        W = ZT;
        elem::MakeTrapezoidal( LEFT, UPPER, 0, W );
        internal::CompressSVD( W, s, V );
        elem::DiagonalScale( RIGHT, NORMAL, s, V );

        // Reexpand (TODO: Think about explicitly expanded reflectors)
        const int numKeptModes = s.Height();
        U.ResizeTo( Z.Height(), numKeptModes );
        Matrix<C> UT, UB;
        elem::PartitionDown
        ( U, UT,
             UB, Z.Width() ); 
        UT = W;
        MakeZeros( UB );
        elem::ApplyPackedReflectors
        ( LEFT, LOWER, VERTICAL, BACKWARD, UNCONJUGATED, 0, Z, t, U );
    }
    else if( Z.Width() > 1.5*Z.Height() )
    {
        // LQ
        Matrix<C> t;
        elem::LQ( Z, t );
        
        // Compress
        Matrix<R> s;
        Matrix<C> ZL, W;
        ZL.View( Z, 0, 0, Z.Height(), Z.Height() );
        U = ZL;
        elem::MakeTrapezoidal( LEFT, LOWER, 0, U );
        internal::CompressSVD( U, s, W );
        elem::DiagonalScale( RIGHT, NORMAL, s, U );

        // Reexpand (TODO: Think about explicitly expanded reflectors)
        const int numKeptModes = s.Height();
        V.ResizeTo( Z.Width(), numKeptModes );
        Matrix<C> VT, VB;
        elem::PartitionDown 
        ( V, VT,
             VB, Z.Height() );
        VT = W;
        MakeZeros( VB );
        // TODO: Think about whether or not this should be conjugated
        elem::ApplyPackedReflectors
        ( LEFT, UPPER, HORIZONTAL, BACKWARD, CONJUGATED, 0, Z, t, V );
    }
    else
    {
        Matrix<R> s;
        U = Z;
        elem::SVD( U, s, V, useQR );
        internal::CompressSVD( U, s, V );
        elem::DiagonalScale( RIGHT, NORMAL, s, V );
    }

    // Unshuffle each column of U into an s1 x s2 matrix in 'greens'
    // Unshuffle each column of V into a depth x depth matrix in 'coefficients'
    const int numKept = U.Width();
    greens.resize( numKept );
    coefficients.resize( numKept );
    for( int t=0; t<numKept; ++t )    
    {
        greens[t].ResizeTo( s1, s2 );
        coefficients[t].ResizeTo( depth, depth );

        for( int i2=0; i2<s2; ++i2 )
        {
            C* greenCol = greens[t].Buffer(0,i2);
            const C* UCol = U.LockedBuffer(i2*s1,t);
            elem::MemCopy( greenCol, UCol, s1 );
        }
        for( int j2=0; j2<depth; ++j2 )
        {
            C* coefficientCol = coefficients[t].Buffer(0,j2);
            const C* VCol = V.LockedBuffer(j2*depth,t);
            elem::MemCopy( coefficientCol, VCol, depth );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// TODO: Sparse leaf-level B compression

} // namespace psp

#endif // PSP_LOCAL_FRONT_COMPRESSION_HPP
