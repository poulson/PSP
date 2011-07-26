/*
   Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
   of a sweeping preconditioner for 3d Helmholtz equations.

   Copyright (C) 2011 Jack Poulson, Lexing Ying, and
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

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::SetToRandom()
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::SetToRandom");
#endif
    switch( _block.type )
    {
    case NODE:
    {
        Node& node = *_block.data.N;
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                node.Child(t,s).SetToRandom();
        break;
    }
    case NODE_SYMMETRIC:
    {
#ifndef RELEASE
        throw std::logic_error("Symmetry not yet supported.");
#endif
        break;
    }
    case DENSE:
        ParallelGaussianRandomVectors( *_block.data.D );
        break;
    case LOW_RANK:
    {
        LowRank<Scalar,Conjugated>& F = *_block.data.F;
        const int maxRank = MaxRank();
        const int height = F.U.Height();
        const int width = F.V.Height();

        F.U.Resize( height, maxRank );
        F.V.Resize( width,  maxRank );
        ParallelGaussianRandomVectors( F.U );
        ParallelGaussianRandomVectors( F.V );
        break;
    }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

