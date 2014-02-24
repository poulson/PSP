/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_DIST_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
#define PSP_DIST_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP

namespace psp {

// NOTE: This parallelization is known to be grossly suboptimal and will be
//       replaced over the next few months.

template<typename F>
inline void
FrontCompressedBlockLowerForwardSolve
( const DistCompressedFront<F>& front, DistMatrix<F,VC,STAR>& X )
{
    DEBUG_ONLY(CallStackEntry entry("FrontCompressedBlockLowerForwardSolve"))
    const Grid& g = *front.grid;
    const int numKeptModesA = front.AGreens.size();
    const int numKeptModesB = front.BGreens.size();
    
    const int sT = front.sT;
    const int depth = front.depth;
    const int snSize = sT*depth;
    const int numRhs = X.Width();

    DistMatrix<F,VC,STAR> XT(g), XB(g);
    elem::PartitionDown( X, XT, XB, snSize );

    // XT := inv(ATL) XT
    //     = \sum_t (CA_t o GA_t) XT
    std::vector<DistMatrix<F,VC,STAR>> XTBlocks, ZTBlocks;
    XTBlocks.resize( depth );
    ZTBlocks.resize( depth );
    for( int i=0; i<depth; ++i )
    {
        XTBlocks[i].SetGrid( g );
        ZTBlocks[i].SetGrid( g );
        Zeros( XTBlocks[i], sT, numRhs );
        Zeros( ZTBlocks[i], sT, numRhs );
    }
    DistMatrix<F,VC,STAR> XTBlock(g);
    DistMatrix<F,MR,STAR> XTBlock_MR_STAR(g);
    DistMatrix<F,MC,STAR> ZTBlock_MC_STAR( g );
    Zeros( ZTBlock_MC_STAR, sT, numRhs );
    for( int t=0; t<numKeptModesA; ++t )
    {
        const DistMatrix<F>& GA = front.AGreens[t];
        const DistMatrix<F,STAR,STAR>& CA = front.ACoefficients[t];

        XTBlock_MR_STAR.AlignWith( GA );
        for( int j=0; j<depth; ++j )
        {
            LockedView( XTBlock, XT, j*sT, 0, sT, numRhs );
            XTBlock_MR_STAR = XTBlock;
            elem::LocalGemm
            ( NORMAL, NORMAL, 
              F(1), GA, XTBlock_MR_STAR, F(0), ZTBlock_MC_STAR );
            ZTBlocks[j].SumScatterFrom( ZTBlock_MC_STAR );
        }
        XTBlock_MR_STAR.FreeAlignments();

        for( int i=0; i<depth; ++i )
            for( int j=0; j<depth; ++j )
                elem::Axpy( CA.Get(i,j), ZTBlocks[j], XTBlocks[i] );
    }
    // Now that each block of XT has been formed, perform the necessary 
    // redistributions to fix alignments
    for( int i=0; i<depth; ++i )
    {
        View( XTBlock, XT, i*sT, 0, sT, numRhs );
        XTBlock = XTBlocks[i];
    }
    XTBlocks.clear();
    ZTBlocks.clear();
    XTBlock_MR_STAR.Empty();
    ZTBlock_MC_STAR.Empty();

    // XB := XB - LB XT
    //     = XB - \sum_t (CB_t o GB_t) XT
    if( numKeptModesB != 0 )
    {
        const int sB = front.sB;

        std::vector<DistMatrix<F,VC,STAR>> XBUpdates, ZBBlocks;
        XBUpdates.resize( depth );
        ZBBlocks.resize( depth );
        for( int i=0; i<depth; ++i )
        {
            XBUpdates[i].SetGrid( g );
            ZBBlocks[i].SetGrid( g );
            Zeros( XBUpdates[i], sB, numRhs );
            Zeros( ZBBlocks[i], sB, numRhs );
        }
        DistMatrix<F,VC,STAR> XTBlock(g);
        DistMatrix<F,MR,STAR> XTBlock_MR_STAR(g);
        DistMatrix<F,MC,STAR> ZBBlock_MC_STAR( g );
        Zeros( ZBBlock_MC_STAR, sB, numRhs );
        for( int t=0; t<numKeptModesB; ++t )
        {
            const DistMatrix<F>& GB = front.BGreens[t];
            const DistMatrix<F,STAR,STAR>& CB = front.BCoefficients[t];

            XTBlock_MR_STAR.AlignWith( GB );
            for( int j=0; j<depth; ++j )
            {
                LockedView( XTBlock, XT, j*sT, 0, sT, numRhs );
                XTBlock_MR_STAR = XTBlock;
                elem::LocalGemm
                ( NORMAL, NORMAL, 
                  F(1), GB, XTBlock_MR_STAR, F(0), ZBBlock_MC_STAR );
                ZBBlocks[j].SumScatterFrom( ZBBlock_MC_STAR );
            }
            XTBlock_MR_STAR.FreeAlignments();
            
            for( int i=0; i<depth; ++i )
                for( int j=0; j<depth; ++j )
                    elem::Axpy( CB.Get(i,j), ZBBlocks[j], XBUpdates[i] );
        }
        // Now that the updates have been formed, perform the redistributions
        // necessary to subtract them from XB
        DistMatrix<F,VC,STAR> XBBlock(g);
        for( int i=0; i<depth; ++i )
        {
            View( XBBlock, XB, i*sB, 0, sB, numRhs );
            elem::Axpy( F(-1), XBUpdates[i], XBBlock );
        }
    }
}

template<typename F>
inline void
FrontCompressedBlockLowerBackwardSolve
( const DistCompressedFront<F>& front, DistMatrix<F,VC,STAR>& X,
  bool conjugate=false )
{
    DEBUG_ONLY(CallStackEntry entry("FrontCompressedBlockLowerBackwardSolve"))
    const Grid& g = *front.grid;
    const int numKeptModesA = front.AGreens.size();
    const int numKeptModesB = front.BGreens.size();

    if( numKeptModesB == 0 )
        return;

    const int sT = front.sT;
    const int sB = front.sB;
    const int depth = front.depth;
    const int snSize = sT*depth;
    const int numRhs = X.Width();
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );

    DistMatrix<F,VC,STAR> XT(g), XB(g);
    elem::PartitionDown( X, XT, XB, snSize );

    // YT := LB^[T/H] XB
    //     = \sum_t (CB_t o GB_t)^[T/H] XB
    //     = \sum_t (CB_t^[T/H] o GB_t^[T/H]) XB
    DistMatrix<F,VC,STAR> YT( snSize, numRhs, g );
    {
        std::vector<DistMatrix<F,VR,STAR> > YTBlocks, ZTBlocks;
        YTBlocks.resize( depth );
        ZTBlocks.resize( depth );
        for( int i=0; i<depth; ++i )
        {
            YTBlocks[i].SetGrid( g );
            ZTBlocks[i].SetGrid( g );
            Zeros( YTBlocks[i], sT, numRhs );
            Zeros( ZTBlocks[i], sT, numRhs );
        }
        DistMatrix<F,VC,STAR> XBBlock(g);
        DistMatrix<F,MC,STAR> XBBlock_MC_STAR(g);
        DistMatrix<F,MR,STAR> ZTBlock_MR_STAR(g);
        Zeros( ZTBlock_MR_STAR, sT, numRhs );
        for( int t=0; t<numKeptModesB; ++t )
        {
            const DistMatrix<F>& GB = front.BGreens[t];
            const DistMatrix<F,STAR,STAR>& CB = front.BCoefficients[t];

            XBBlock_MC_STAR.AlignWith( GB );
            for( int j=0; j<depth; ++j )
            {
                LockedView( XBBlock, XB, j*sB, 0, sB, numRhs );
                XBBlock_MC_STAR = XBBlock;
                elem::LocalGemm
                ( orientation, NORMAL,
                  F(1), GB, XBBlock_MC_STAR, F(0), ZTBlock_MR_STAR );
                ZTBlocks[j].SumScatterFrom( ZTBlock_MR_STAR );
            }
            XBBlock_MC_STAR.FreeAlignments();

            for( int i=0; i<depth; ++i )
            {
                for( int j=0; j<depth; ++j )
                {
                    const F entry = CB.Get(j,i);
                    const F scalar = ( conjugate ? Conj(entry) : entry );
                    elem::Axpy( scalar, ZTBlocks[j], YTBlocks[i] );
                }
            }
        }

        // Each YT block was accumulated in a [VR,* ] distribution for 
        // efficiency reasons, but we must now redistribute them into [VC,* ]
        // distributions and put them in YT
        DistMatrix<F,VC,STAR> YTBlock(g);
        for( int i=0; i<depth; ++i )
        {
            View( YTBlock, YT, i*sT, 0, sT, numRhs );
            YTBlock = YTBlocks[i];
        }
    }

    // XT := XT - inv(ATL) YT
    //     = XT - \sum_t (CA_t o GA_t) YT
    std::vector<DistMatrix<F,VC,STAR> > XTUpdates, ZTBlocks;
    XTUpdates.resize( depth );
    ZTBlocks.resize( depth );
    for( int i=0; i<depth; ++i )
    {
        XTUpdates[i].SetGrid( g );
        ZTBlocks[i].SetGrid( g );
        Zeros( XTUpdates[i], sT, numRhs );
        Zeros( ZTBlocks[i], sT, numRhs );
    }
    DistMatrix<F,VC,STAR> YTBlock(g);
    DistMatrix<F,MR,STAR> YTBlock_MR_STAR(g);
    DistMatrix<F,MC,STAR> ZTBlock_MC_STAR(g);
    Zeros( ZTBlock_MC_STAR, sT, numRhs );
    for( int t=0; t<numKeptModesA; ++t )
    {
        const DistMatrix<F>& GA = front.AGreens[t];
        const DistMatrix<F,STAR,STAR>& CA = front.ACoefficients[t];

        YTBlock_MR_STAR.AlignWith( GA );
        for( int j=0; j<depth; ++j )
        {
            LockedView( YTBlock, YT, j*sT, 0, sT, numRhs );
            YTBlock_MR_STAR = YTBlock;
            elem::LocalGemm
            ( NORMAL, NORMAL,
              F(1), GA, YTBlock_MR_STAR, F(0), ZTBlock_MC_STAR );
            ZTBlocks[j].SumScatterFrom( ZTBlock_MC_STAR );
        }
        YTBlock_MR_STAR.FreeAlignments();

        for( int i=0; i<depth; ++i )
            for( int j=0; j<depth; ++j )
                elem::Axpy( CA.Get(i,j), ZTBlocks[j], XTUpdates[i] );
    }
    // Now that the updates have been accumulated in XTUpdates, subtract them
    // from XT
    DistMatrix<F,VC,STAR> XTBlock(g);
    for( int i=0; i<depth; ++i )
    {
        View( XTBlock, XT, i*sT, 0, sT, numRhs );
        elem::Axpy( F(-1), XTUpdates[i], XTBlock );
    }
}

} // namespace psp

#endif // ifndef PSP_DIST_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
