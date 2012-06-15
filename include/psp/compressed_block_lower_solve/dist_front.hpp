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
    //     = (C_A o G_A) XT
    DistMatrix<F,VC,STAR> YT( XT );
    MakeZeros( XT );
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
    DistMatrix<F,VC,STAR> XTBlock(g), YTBlock(g);
    DistMatrix<F,MR,STAR> YTBlock_MR_STAR(g);
    DistMatrix<F,MC,STAR> ZTBlock_MC_STAR( g );
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
        {
            for( int j=0; j<depth; ++j )
            {
                const F scalar = CA.Get(i,j);
                elem::Axpy( scalar, ZTBlocks[j], XTBlocks[i] );
            }
            XTBlock.View( XT, i*sT, 0, sT, numRhs );
            XTBlock = XTBlocks[i];
        }
    }
    XTBlocks.clear();
    ZTBlocks.clear();
    YTBlock_MR_STAR.Empty();
    ZTBlock_MC_STAR.Empty();

    // XB := XB - LB XT
    //     = XB - (C_B o G_B) XT
    if( numKeptModesB != 0 )
    {
        const int sB = front.sB;

        DistMatrix<F,VC,STAR> YB(g), ZB(g);
        std::vector<DistMatrix<F,VC,STAR> > XBBlocks, ZBBlocks;
        XBBlocks.resize( depth );
        ZBBlocks.resize( depth );
        for( int i=0; i<depth; ++i )
        {
            XBBlocks[i].SetGrid( g );
            ZBBlocks[i].SetGrid( g );
            Zeros( sB, numRhs, XBBlocks[i] );
            Zeros( sB, numRhs, ZBBlocks[i] );
        }
        DistMatrix<F,VC,STAR> XTBlock(g), XBBlock(g);
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
            {
                for( int j=0; j<depth; ++j )
                {
                    const F scalar = CB.Get(i,j);
                    elem::Axpy( scalar, ZBBlocks[j], XBBlocks[i] );
                }
                XBBlock.View( XB, i*sB, 0, sB, numRhs );
                elem::Axpy( (F)-1, XBBlocks[i], XBBlock );
            }
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
    PushCallStack();
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

    const Grid& g = *front.grid;

    // YT := LB^[T/H] XB
    //     = (C_B o G_B)^[T/H] XB
    //     = (C_B^[T/H] o G_B^[T/H]) XB
    // TODO

    // XT := XT - inv(ATL) YT
    //     = XT - (C_A o G_A) YT
    // TODO
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_DIST_FRONT_COMPRESSED_BLOCK_LOWER_SOLVE_HPP
