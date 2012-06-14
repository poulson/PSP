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
#ifndef PSP_LOCAL_COMPRESSED_BLOCK_LDL_HPP
#define PSP_LOCAL_COMPRESSED_BLOCK_LDL_HPP 1

namespace psp {

template<typename F> 
void LocalCompressedBlockLDL
( Orientation orientation, 
  cliq::symbolic::SymmFact& S, CompressedFrontTree<F>& L, int depth );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename F> 
inline void LocalCompressedBlockLDL
( Orientation orientation, 
  cliq::symbolic::SymmFact& S, CompressedFrontTree<F>& L, int depth )
{
    using namespace cliq::symbolic;
#ifndef RELEASE
    PushCallStack("LocalCompressedBlockLDL");
    if( orientation == NORMAL )
        throw std::logic_error("LDL must be (conjugate-)transposed");
#endif
    const int numLocalSupernodes = S.local.supernodes.size();
    for( int s=0; s<numLocalSupernodes; ++s )
    {
        LocalSymmFactSupernode& sn = S.local.supernodes[s];
        const int updateSize = sn.lowerStruct.size();
        LocalCompressedFront<F>& front = L.local.fronts[s];
        Matrix<F>& frontL = front.frontL;
        Matrix<F>& frontBR = front.work;
        frontBR.Empty();
#ifndef RELEASE
        if( frontL.Height() != sn.size+updateSize ||
            frontL.Width() != sn.size )
            throw std::logic_error("Front was not the proper size");
#endif

        // Add updates from children (if they exist)
        elem::Zeros( updateSize, updateSize, frontBR );
        const int numChildren = sn.children.size();
        if( numChildren == 2 )
        {
            const int leftIndex = sn.children[0];
            const int rightIndex = sn.children[1];
            Matrix<F>& leftUpdate = L.local.fronts[leftIndex].work;
            Matrix<F>& rightUpdate = L.local.fronts[rightIndex].work;

            // Add the left child's update matrix
            const int leftUpdateSize = leftUpdate.Height();
            for( int jChild=0; jChild<leftUpdateSize; ++jChild )
            {
                const int jFront = sn.leftChildRelIndices[jChild];
                for( int iChild=0; iChild<leftUpdateSize; ++iChild )
                {
                    const int iFront = sn.leftChildRelIndices[iChild];
                    const F value = leftUpdate.Get(iChild,jChild);
                    if( jFront < sn.size )
                        frontL.Update( iFront, jFront, value );
                    else if( iFront >= sn.size )
                        frontBR.Update( iFront-sn.size, jFront-sn.size, value );
                }
            }
            leftUpdate.Empty();

            // Add the right child's update matrix
            const int rightUpdateSize = rightUpdate.Height();
            for( int jChild=0; jChild<rightUpdateSize; ++jChild )
            {
                const int jFront = sn.rightChildRelIndices[jChild];
                for( int iChild=0; iChild<rightUpdateSize; ++iChild )
                {
                    const int iFront = sn.rightChildRelIndices[iChild];
                    const F value = rightUpdate.Get(iChild,jChild);
                    if( jFront < sn.size )
                        frontL.Update( iFront, jFront, value );
                    else if( iFront >= sn.size )
                        frontBR.Update( iFront-sn.size, jFront-sn.size, value );
                }
            }
            rightUpdate.Empty();
        }

        // Call the custom partial block LDL
        cliq::numeric::LocalFrontBlockLDL( orientation, frontL, frontBR );

        // Compress the A and B fronts
        Matrix<F> A, B;
        elem::PartitionDown
        ( frontL, A,
                  B, sn.size );
        LocalFrontCompression( A, front.AGreens, depth );
        if( numChildren == 0 )
        {
            // TODO: Use sparse compression at the leaf-level
            LocalFrontCompression( B, front.BGreens, depth );
        }
        else
            LocalFrontCompression( B, front.BGreens, depth );
        // TODO: Uncomment after compressed solve works
        //frontL.Empty();
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_LOCAL_COMPRESSED_BLOCK_LDL_HPP
