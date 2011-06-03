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
#include "psp.hpp"

namespace { 

psp::ExpandedUInt64 lcgValue;

// Manually import Knuth's multiplication constant, 6364136223846793005,
// and his additive constant, 1442695040888963407
const psp::ExpandedUInt64 lcgMultValue = { 32557U, 19605U, 62509U, 22609U }; 
const psp::ExpandedUInt64 lcgAddValue = { 33103U, 63335U, 31614U, 5125U };

psp::ExpandedUInt64 teamMultValue, teamAddValue, myLcgValue;

} // anonymous namespace

// Return floor( log2(x) )
psp::UInt32 psp::Log2( UInt32 x )
{
    static const psp::UInt32 MultiplyDeBruijnBitPosition[32] = 
    {
      0U, 9U, 1U, 10U, 13U, 21U, 2U, 29U, 11U, 14U, 16U, 18U, 22U, 25U, 3U, 30U,
      8U, 12U, 20U, 28U, 15U, 17U, 24U, 7U, 19U, 27U, 23U, 6U, 26U, 5U, 4U, 31U
    };

    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return MultiplyDeBruijnBitPosition[(UInt32)(x*0x07C4ACDDU)>>27];
}

// Return log2(x), where x is a power of 2
psp::UInt32 psp::Log2PowerOfTwo( UInt32 x )
{
    static const int MultiplyDeBruijnBitPosition2[32] = 
    {
      0U, 1U, 28U, 2U, 29U, 14U, 24U, 3U, 30U, 22U, 20U, 15U, 25U, 17U, 4U, 8U, 
      31U, 27U, 13U, 23U, 21U, 19U, 16U, 7U, 26U, 12U, 18U, 6U, 11U, 5U, 10U, 
      9U
    };

    return MultiplyDeBruijnBitPosition[(UInt32)(x*0x077CB531U)>>27];
}

// x := x/2
void psp::Halve( ExpandedUInt64& x )
{
    x[0] >>= 1;
    x[0] |= ((x[1]&1) << 16);
    x[1] >>= 1;
    x[1] |= ((x[2]&1) << 16);
    x[2] >>= 1;
    x[2] |= ((x[3]&1) << 16);
    x[3] >>= 1;
    // The 16th bit of x[3] must be zero since we require the 17th and above
    // to have been zero upon entry.
}

// Return x^n with O(log2(n)) work. This is the "Right-to-left binary method for
// exponentiation" from Knuth's 'Seminumerical Algorithms' volume of TAOCP.
psp::ExpandedUInt64 psp::IntegerPowerWith64BitMod
( ExpandedUInt64 x, ExpandedUInt64 n )
{
    ExpandedUInt64 N=n, Z=x, Y={1U,0U,0U,0U};
    while( 1 )
    {
        Halve( N );
        Y = MultiplyWith64BitMod( Z, Y );
        if( N[0]==0 && N[1]==0 && N[2]==0 && N[3]==0 )
            break;
        Z = MultiplyWith64BitMod( Z, Z );
    }
    return Y;
}

void psp::SeedParallelLcg( UInt32 rank, UInt32 commSize, UInt64 globalSeed )
{
    // Compute a^rank and a^commSize in O(log2(commSize)) work.
    const ExpandedUInt64 myMultValue = 
        IntegerPowerWith64BitMod( ::lcgMultValue, Expand(rank) );
    ::teamMultValue = 
        IntegerPowerWith64BitMod( ::lcgMultValue, Expand(commSize) );

    // Compute (a^rank-1)/(a-1) and (a^commSize-1)/(a-1) in O(commSize) work.
    // This could almost certainly be optimized, but its execution time is 
    // probably ignorable.
    ExpandedUInt64 Y, one={1U,0U,0U,0U};
    Y = one;
    for( int j=0; j<rank; ++j )
    {
        Y = MultWith64BitMod( Y, ::lcgMultValue );
        Y = AddWith64BitMod( Y, one );
    }
    const ExpandedUInt64 myAddValue = Y;
    for( int j=rank; j<commSize; ++j )
    {
        Y = MultWith64BitMod( Y, ::lcgMultValue );
        Y = AddWith64BitMod( Y, one );
    }
    ::teamAddValue = Y;

    // Set our local value equal to 
    //     X_rank := a^rank X_0 + (a^rank-1)/(a-1) c mod 2^64
    // where X_0 is 'globalSeed'.
    ::myLcgValue = Expand( globalSeed );
    Lcg( myMultValue, myAddValue, ::myLcgValue );
}

// Return a uniform sample from [0,2^64)
psp::UInt64 psp::ParallelLcg()
{
    Lcg( ::teamMultValue, ::teamAddValue, ::myLcgValue ); 
    return Expand( ::myLcgValue );
}

// TODO: Figure out how to convert UInt64 into sample form [0,1].
/*
template<>
float psp::ParallelUniform<float>()
{

}

template<>
double psp::ParallelUniform<double>()
{

}
*/

