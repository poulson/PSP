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

// Manually import Knuth's multiplication constant, 6364136223846793005
const psp::ExpandedUInt64 lcgMultValue = { 32557U, 19605U, 62509U, 22609U }; 

// Manually import Knuth's additive constant, 1442695040888963407
const psp::ExpandedUInt64 lcgAddValue = { 33103U, 63335U, 31614U, 5125U };

} // anonymous namespace

psp::UInt32 psp::Log2( UInt32 a )
{
    static const psp::UInt32 MultiplyDeBruijnBitPosition[32] = 
    {
      0U, 9U, 1U, 10U, 13U, 21U, 2U, 29U, 11U, 14U, 16U, 18U, 22U, 25U, 3U, 30U,
      8U, 12U, 20U, 28U, 15U, 17U, 24U, 7U, 19U, 27U, 23U, 6U, 26U, 5U, 4U, 31U
    };

    a |= a >> 1;
    a |= a >> 2;
    a |= a >> 4;
    a |= a >> 8;
    a |= a >> 16;
    return MultiplyDeBruijnBitPosition[(UInt32)(a*0x07C4ACDDU)>>27];
}

psp::UInt32 psp::Log2PowerOfTwo( UInt32 a )
{
    static const int MultiplyDeBruijnBitPosition2[32] = 
    {
      0U, 1U, 28U, 2U, 29U, 14U, 24U, 3U, 30U, 22U, 20U, 15U, 25U, 17U, 4U, 8U, 
      31U, 27U, 13U, 23U, 21U, 19U, 16U, 7U, 26U, 12U, 18U, 6U, 11U, 5U, 10U, 
      9U
    };

    return MultiplyDeBruijnBitPosition[(UInt32)(a*0x077CB531U)>>27];
}

void psp::SeedParallelLcg( UInt32 rank, UInt32 commSize, UInt64 globalSeed )
{
    const UInt32 P = psp::NextPowerOfTwo( commSize );
    const UInt32 log2P = Log2PowerOfTwo( P );
    const UInt32 log2Period = 64U - log2P;

    ExpandedUInt32 period;
    if( log2Period < 48U ) // 32 <= log2Period < 48
    {
        period[0] = 0U;    
        period[1] = 0U;
        period[2] = 1U << (log2Period-32U);
        period[3] = 0U;
    }
    else // 48 <= log2Period <= 64
    {
        period[0] = 0U;
        period[1] = 0U;
        period[2] = 0U;
        period[3] = 1U << (log2Period-48U);
    }
    ExpandedUInt32 myOffset = MultiplyWith64BitMod( period, Expand(rank) );
    ExpandedUInt32 myStart = AddWith64BitMod( myOffset, Expand(globalSeed) );
    
    // Compute the first term in our recurrence, using the formula
    //
    // U_n = a^n U_0 + ((a^n - 1)/(a - 1)) c mod m,
    //
    // where we force U_0 = 0. Our main computation is then to find 
    // a^n mod m using recursive exponential doubling mod m.
    throw std::logic_error("Routine not finished");
    // HERE
}

