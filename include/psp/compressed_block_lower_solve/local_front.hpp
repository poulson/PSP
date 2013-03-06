/*
   Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and 
   The University of Texas at Austin
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#ifndef PSP_LOCAL_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
#define PSP_LOCAL_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP 1

namespace psp {

template<typename F>
inline void
FrontCompressedBlockLowerForwardSolve
( const CompressedFront<F>& front, Matrix<F>& X )
{
#ifndef RELEASE
    PushCallStack("FrontCompressedBlockLowerForwardSolve");
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
            View( ZTBlock, ZT, j*sT, 0, sT, numRhs );
            LockedView( YTBlock, YT, j*sT, 0, sT, numRhs );
            elem::Gemm( NORMAL, NORMAL, F(1), GA, YTBlock, F(0), ZTBlock );
        }

        for( int i=0; i<depth; ++i )
        {
            View( XTBlock, XT, i*sT, 0, sT, numRhs );
            for( int j=0; j<depth; ++j )
            {
                LockedView( ZTBlock, ZT, j*sT, 0, sT, numRhs );
                elem::Axpy( CA.Get(i,j), ZTBlock, XTBlock );
            }
        }
    }
    YT.Empty();
    ZT.Empty();

    // XB := XB - LB XT
    //     = XB - \sum_t (CB_t o GB_t) XT
    if( front.isLeaf )
    {
        const int numNonzeros = front.BValues.size();
        for( int k=0; k<numNonzeros; ++k )
        {
            const int i = front.BRows[k];
            const int j = front.BCols[k];
            const F value = front.BValues[k];
            for( int l=0; l<numRhs; ++l )
                XB.Update( i, l, -value*XT.Get(j,l) );
        }
    }
    else if( numKeptModesB != 0 )
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
                View( ZBBlock, ZB, j*sB, 0, sB, numRhs );
                LockedView( XTBlock, XT, j*sT, 0, sT, numRhs );
                elem::Gemm( NORMAL, NORMAL, F(1), GB, XTBlock, F(0), ZBBlock );
            }

            for( int i=0; i<depth; ++i )
            {
                View( XBBlock, XB, i*sB, 0, sB, numRhs );
                for( int j=0; j<depth; ++j )
                {
                    LockedView( ZBBlock, ZB, j*sB, 0, sB, numRhs );
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
FrontCompressedBlockLowerBackwardSolve
( Orientation orientation, const CompressedFront<F>& front, Matrix<F>& X )
{
#ifndef RELEASE
    PushCallStack("FrontCompressedBlockLowerBackwardSolve");
#endif
    const int numKeptModesA = front.AGreens.size();

    const bool isLeaf = front.isLeaf;
    const int numKeptModesB = ( isLeaf ? 0 : front.BGreens.size() );
    const int numNonzeros = ( isLeaf ? front.BRows.size() : 0 );

    if( (isLeaf && numNonzeros==0) || (!isLeaf && numKeptModesB==0) )
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

    // YT := LB^[T/H] XB
    Matrix<F> YT, ZT, YTBlock, ZTBlock;
    Zeros( snSize, numRhs, YT );
    Zeros( snSize, numRhs, ZT );
    if( isLeaf )
    {
        for( int k=0; k<numNonzeros; ++k )
        {
            const int i = front.BRows[k];
            const int j = front.BCols[k];
            const F value = front.BValues[k];
            const F scalar = ( conjugate ? Conj(value) : value );
            for( int l=0; l<numRhs; ++l )
                YT.Update( j, l, scalar*XB.Get(i,l) );
        }
    }
    else
    {
        // YT := LB^[T/H] XB
        //     = \sum_t (CB_t o GB_t)^[T/H] XB
        //     = \sum_t (CB_t^[T/H] o GB_t^[T/H]) XB
        Matrix<F> XBBlock;
        for( int t=0; t<numKeptModesB; ++t )
        {
            const Matrix<F>& GB = front.BGreens[t];
            const Matrix<F>& CB = front.BCoefficients[t];

            for( int j=0; j<depth; ++j )    
            {
                View( ZTBlock, ZT, j*sT, 0, sT, numRhs );
                LockedView( XBBlock, XB, j*sB, 0, sB, numRhs );
                elem::Gemm
                ( orientation, NORMAL, F(1), GB, XBBlock, F(0), ZTBlock );
            }

            for( int i=0; i<depth; ++i )
            {
                View( YTBlock, YT, i*sT, 0, sT, numRhs );
                for( int j=0; j<depth; ++j )
                {
                    LockedView( ZTBlock, ZT, j*sT, 0, sT, numRhs );
                    const F entry = CB.Get(j,i);
                    const F scalar = ( conjugate ? Conj(entry) : entry );
                    elem::Axpy( scalar, ZTBlock, YTBlock );
                }
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
            View( ZTBlock, ZT, j*sT, 0, sT, numRhs );
            LockedView( YTBlock, YT, j*sT, 0, sT, numRhs );
            elem::Gemm( NORMAL, NORMAL, F(1), GA, YTBlock, F(0), ZTBlock );
        }

        for( int i=0; i<depth; ++i )
        {
            View( XTBlock, XT, i*sT, 0, sT, numRhs );
            for( int j=0; j<depth; ++j )
            {
                LockedView( ZTBlock, ZT, j*sT, 0, sT, numRhs );
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
