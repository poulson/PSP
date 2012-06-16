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
#ifndef PSP_LOCAL_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
#define PSP_LOCAL_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP 1

namespace psp {

template<typename F>
inline void
LocalFrontCompressedBlockLowerForwardSolve
( const LocalCompressedFront<F>& front, Matrix<F>& X )
{
#ifndef RELEASE
    PushCallStack("LocalFrontCompressedBlockLowerForwardSolve");
#endif
    const int numKeptModesA = front.AGreens.size();
    const int numKeptModesB = front.BGreens.size();

    const int sT = front.sT;     
    const int depth = front.depth;
    const int snSize = sT*depth;
    const int numRhs = X.Width();

    Matrix<F> XT,
              XB;
    elem::PartitionDown
    ( X, XT,
         XB, snSize );

    // XT := inv(ATL) XT
    //     = \sum_t (CA_t o GA_t) XT
    Matrix<F> YT, ZT;
    YT = XT;
    MakeZeros( XT );
    Zeros( snSize, numRhs, ZT );
    Matrix<F> XTBlock, YTBlock, ZTBlock;
    for( int t=0; t<numKeptModesA; ++t )
    {
        const Matrix<F>& GA = front.AGreens[t];
        const Matrix<F>& CA = front.ACoefficients[t];

        for( int j=0; j<depth; ++j )
        {
            ZTBlock.View( ZT, j*sT, 0, sT, numRhs );
            YTBlock.LockedView( YT, j*sT, 0, sT, numRhs );
            elem::Gemm( NORMAL, NORMAL, (F)1, GA, YTBlock, (F)0, ZTBlock );
        }

        for( int i=0; i<depth; ++i )
        {
            XTBlock.View( XT, i*sT, 0, sT, numRhs );
            for( int j=0; j<depth; ++j )
            {
                ZTBlock.LockedView( ZT, j*sT, 0, sT, numRhs );
                elem::Axpy( CA.Get(i,j), ZTBlock, XTBlock );
            }
        }
    }
    YT.Empty();
    ZT.Empty();

    // XB := XB - LB XT
    //     = XB - \sum_t (CB_t o GB_t) XT
    if( numKeptModesB != 0 )
    {
        const int sB = front.sB;

        Matrix<F> ZB;
        Zeros( XB.Height(), numRhs, ZB );
        Matrix<F> XBBlock, ZBBlock;
        for( int t=0; t<numKeptModesB; ++t )
        {
            const Matrix<F>& GB = front.BGreens[t];
            const Matrix<F>& CB = front.BCoefficients[t];

            for( int j=0; j<depth; ++j )
            {
                ZBBlock.View( ZB, j*sB, 0, sB, numRhs );
                XTBlock.LockedView( XT, j*sT, 0, sT, numRhs );
                elem::Gemm( NORMAL, NORMAL, (F)1, GB, XTBlock, (F)0, ZBBlock );
            }

            for( int i=0; i<depth; ++i )
            {
                XBBlock.View( XB, i*sB, 0, sB, numRhs );
                for( int j=0; j<depth; ++j )
                {
                    ZBBlock.LockedView( ZB, j*sB, 0, sB, numRhs );
                    elem::Axpy( -CB.Get(i,j), ZBBlock, XBBlock );
                }
            }
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void
LocalFrontCompressedBlockLowerBackwardSolve
( Orientation orientation, const LocalCompressedFront<F>& front, Matrix<F>& X )
{
#ifndef RELEASE
    PushCallStack("LocalFrontCompressedBlockLowerBackwardSolve");
#endif
    const int numKeptModesA = front.AGreens.size();
    const int numKeptModesB = front.BGreens.size();

    if( numKeptModesB == 0 )
    {
#ifndef RELEASE
        PopCallStack();
#endif
        return;
    }

    const int sT = front.sT;
    const int sB = front.sB;
    const int depth = front.depth;
    const int snSize = sT*depth;
    const int numRhs = X.Width();
    const bool conjugate = ( orientation==ADJOINT ? true : false );

    Matrix<F> XT,
              XB;
    elem::PartitionDown
    ( X, XT,
         XB, snSize );

    // YT := LB^[T/H] XB,
    //     = \sum_t (CB_t o GB_t)^[T/H] XB
    //     = \sum_t (CB_t^[T/H] o GB_t^[T/H]) XB
    Matrix<F> YT, ZT;
    Zeros( snSize, numRhs, YT );
    Zeros( snSize, numRhs, ZT );
    Matrix<F> XBBlock, YTBlock, ZTBlock;
    for( int t=0; t<numKeptModesB; ++t )
    {
        const Matrix<F>& GB = front.BGreens[t];
        const Matrix<F>& CB = front.BCoefficients[t];

        for( int j=0; j<depth; ++j )    
        {
            ZTBlock.View( ZT, j*sT, 0, sT, numRhs );
            XBBlock.LockedView( XB, j*sB, 0, sB, numRhs );
            elem::Gemm( orientation, NORMAL, (F)1, GB, XBBlock, (F)0, ZTBlock );
        }

        for( int i=0; i<depth; ++i )
        {
            YTBlock.View( YT, i*sT, 0, sT, numRhs );
            for( int j=0; j<depth; ++j )
            {
                ZTBlock.LockedView( ZT, j*sT, 0, sT, numRhs );
                const F entry = CB.Get(j,i);
                const F scalar = ( conjugate ? Conj(entry) : entry );
                elem::Axpy( scalar, ZTBlock, YTBlock );
            }
        }
    }

    // XT := XT - inv(ATL) YT
    //     = XT - \sum_t (CA_t o GA_t) YT
    Matrix<F> XTBlock;
    for( int t=0; t<numKeptModesA; ++t )
    {
        const Matrix<F>& GA = front.AGreens[t];
        const Matrix<F>& CA = front.ACoefficients[t];

        for( int j=0; j<depth; ++j )
        {
            ZTBlock.View( ZT, j*sT, 0, sT, numRhs );
            YTBlock.LockedView( YT, j*sT, 0, sT, numRhs );
            elem::Gemm( NORMAL, NORMAL, (F)1, GA, YTBlock, (F)0, ZTBlock );
        }

        for( int i=0; i<depth; ++i )
        {
            XTBlock.View( XT, i*sT, 0, sT, numRhs );
            for( int j=0; j<depth; ++j )
            {
                ZTBlock.LockedView( ZT, j*sT, 0, sT, numRhs );
                elem::Axpy( -CA.Get(i,j), ZTBlock, XTBlock );
            }
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_LOCAL_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
