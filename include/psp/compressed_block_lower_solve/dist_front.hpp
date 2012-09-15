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
#ifndef PSP_DIST_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
#define PSP_DIST_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP 1

namespace psp {

template<typename F>
inline void
DistFrontCompressedBlockLowerForwardSolve
( const DistCompressedFront<F>& front, DistMatrix<F,VC,STAR>& X )
{
#ifndef RELEASE
    PushCallStack("DistFrontCompressedBlockLowerForwardSolve");
#endif
    const Grid& g = *front.grid;
    const int numKeptModesA = front.AGreens.size();
    const int numKeptModesB = front.BGreens.size();
    
    const int sT = front.sT;
    const int depth = front.depth;
    const int snSize = sT*depth;
    const int numRhs = X.Width();

    DistMatrix<F,VC,STAR> XT(g),
                          XB(g);
    elem::PartitionDown
    ( X, XT,
         XB, snSize );

    // XT := inv(ATL) XT
    //     = \sum_t (CA_t o GA_t) XT
    std::vector<DistMatrix<F,VC,STAR> > XTBlocks, ZTBlocks;
    XTBlocks.resize( depth );
    ZTBlocks.resize( depth );
    for( int i=0; i<depth; ++i )
    {
        XTBlocks[i].SetGrid( g );
        ZTBlocks[i].SetGrid( g );
        Zeros( sT, numRhs, XTBlocks[i] );
        Zeros( sT, numRhs, ZTBlocks[i] );
    }
    DistMatrix<F,VC,STAR> XTBlock(g);
    DistMatrix<F,MR,STAR> XTBlock_MR_STAR(g);
    DistMatrix<F,MC,STAR> ZTBlock_MC_STAR( g );
    Zeros( sT, numRhs, ZTBlock_MC_STAR );
    for( int t=0; t<numKeptModesA; ++t )
    {
        const DistMatrix<F>& GA = front.AGreens[t];
        const DistMatrix<F,STAR,STAR>& CA = front.ACoefficients[t];

        XTBlock_MR_STAR.AlignWith( GA );
        for( int j=0; j<depth; ++j )
        {
            XTBlock.LockedView( XT, j*sT, 0, sT, numRhs );
            XTBlock_MR_STAR = XTBlock;
            elem::internal::LocalGemm
            ( NORMAL, NORMAL, 
              (F)1, GA, XTBlock_MR_STAR, (F)0, ZTBlock_MC_STAR );
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
        XTBlock.View( XT, i*sT, 0, sT, numRhs );
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

        std::vector<DistMatrix<F,VC,STAR> > XBUpdates, ZBBlocks;
        XBUpdates.resize( depth );
        ZBBlocks.resize( depth );
        for( int i=0; i<depth; ++i )
        {
            XBUpdates[i].SetGrid( g );
            ZBBlocks[i].SetGrid( g );
            Zeros( sB, numRhs, XBUpdates[i] );
            Zeros( sB, numRhs, ZBBlocks[i] );
        }
        DistMatrix<F,VC,STAR> XTBlock(g);
        DistMatrix<F,MR,STAR> XTBlock_MR_STAR(g);
        DistMatrix<F,MC,STAR> ZBBlock_MC_STAR( g );
        Zeros( sB, numRhs, ZBBlock_MC_STAR );
        for( int t=0; t<numKeptModesB; ++t )
        {
            const DistMatrix<F>& GB = front.BGreens[t];
            const DistMatrix<F,STAR,STAR>& CB = front.BCoefficients[t];

            XTBlock_MR_STAR.AlignWith( GB );
            for( int j=0; j<depth; ++j )
            {
                XTBlock.LockedView( XT, j*sT, 0, sT, numRhs );
                XTBlock_MR_STAR = XTBlock;
                elem::internal::LocalGemm
                ( NORMAL, NORMAL, 
                  (F)1, GB, XTBlock_MR_STAR, (F)0, ZBBlock_MC_STAR );
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
            XBBlock.View( XB, i*sB, 0, sB, numRhs );
            elem::Axpy( (F)-1, XBUpdates[i], XBBlock );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void
DistFrontCompressedBlockLowerBackwardSolve
( Orientation orientation, 
  const DistCompressedFront<F>& front, DistMatrix<F,VC,STAR>& X )
{
#ifndef RELEASE
    PushCallStack("DistFrontCompressedBlockLowerBackwardSolve");
#endif
    const Grid& g = *front.grid;
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
    const bool conjugated = ( orientation==ADJOINT ? true : false );

    DistMatrix<F,VC,STAR> XT(g),
                          XB(g);
    elem::PartitionDown
    ( X, XT,
         XB, snSize );

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
            Zeros( sT, numRhs, YTBlocks[i] );
            Zeros( sT, numRhs, ZTBlocks[i] );
        }
        DistMatrix<F,VC,STAR> XBBlock(g);
        DistMatrix<F,MC,STAR> XBBlock_MC_STAR(g);
        DistMatrix<F,MR,STAR> ZTBlock_MR_STAR(g);
        Zeros( sT, numRhs, ZTBlock_MR_STAR );
        for( int t=0; t<numKeptModesB; ++t )
        {
            const DistMatrix<F>& GB = front.BGreens[t];
            const DistMatrix<F,STAR,STAR>& CB = front.BCoefficients[t];

            XBBlock_MC_STAR.AlignWith( GB );
            for( int j=0; j<depth; ++j )
            {
                XBBlock.LockedView( XB, j*sB, 0, sB, numRhs );
                XBBlock_MC_STAR = XBBlock;
                elem::internal::LocalGemm
                ( orientation, NORMAL,
                  (F)1, GB, XBBlock_MC_STAR, (F)0, ZTBlock_MR_STAR );
                ZTBlocks[j].SumScatterFrom( ZTBlock_MR_STAR );
            }
            XBBlock_MC_STAR.FreeAlignments();

            for( int i=0; i<depth; ++i )
            {
                for( int j=0; j<depth; ++j )
                {
                    const F entry = CB.Get(j,i);
                    const F scalar = ( conjugated ? Conj(entry) : entry );
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
            YTBlock.View( YT, i*sT, 0, sT, numRhs );
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
        Zeros( sT, numRhs, XTUpdates[i] );
        Zeros( sT, numRhs, ZTBlocks[i] );
    }
    DistMatrix<F,VC,STAR> YTBlock(g);
    DistMatrix<F,MR,STAR> YTBlock_MR_STAR(g);
    DistMatrix<F,MC,STAR> ZTBlock_MC_STAR(g);
    Zeros( sT, numRhs, ZTBlock_MC_STAR );
    for( int t=0; t<numKeptModesA; ++t )
    {
        const DistMatrix<F>& GA = front.AGreens[t];
        const DistMatrix<F,STAR,STAR>& CA = front.ACoefficients[t];

        YTBlock_MR_STAR.AlignWith( GA );
        for( int j=0; j<depth; ++j )
        {
            YTBlock.LockedView( YT, j*sT, 0, sT, numRhs );
            YTBlock_MR_STAR = YTBlock;
            elem::internal::LocalGemm
            ( NORMAL, NORMAL,
              (F)1, GA, YTBlock_MR_STAR, (F)0, ZTBlock_MC_STAR );
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
        XTBlock.View( XT, i*sT, 0, sT, numRhs );
        elem::Axpy( (F)-1, XTUpdates[i], XTBlock );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_DIST_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP