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
#ifndef PSP_RANDOM_HPP
#define PSP_RANDOM_HPP 1

#include "psp/building_blocks/environment.hpp"

namespace psp {

// C++98 does not have 64-bit integer support. We instead store each 16-bits
// as a 32-bit unsigned integer in order to allow for multiplication and 
// addition without overflow.
typedef unsigned UInt32;
typedef UInt32 UInt64[2];
typedef UInt32 ExpandedUInt64[4];

UInt32 NextPowerOfTwo( UInt32 a );
UInt32 Log2( UInt32 a );
UInt32 Log2PowerOfTwo( UInt32 a );
UInt32 Lower16Bits( UInt32 a );
UInt32 Upper16Bits( UInt32 a );

ExpandedUInt64 Expand( UInt32 a );
ExpandedUInt64 Expand( UInt64 a );
UInt64 Deflate( ExpandedUInt64 a );

// Carry the upper 16 bits in each of the four 32-bit UInt32's used in the
// expanded representation of a UInt64. The topmost 16 bits are simply zeroed.
void CarryUpper16Bits( ExpandedUInt64& a );

// Add two expanded UInt64's mod 2^64
ExpandedUInt64 AddWith64BitMod( ExpandedUInt64 a, ExpandedUInt64 b );

// Multiply two expanded UInt64's mod 2^64
ExpandedUInt64 MultiplyWith64BitMod( ExpandedUInt64 a, ExpandedUInt64 b );

void SeedParallelLcg( UInt32 rank, UInt32 commSize );
ExpandedUInt64 CurrentLcgValue();
ExpandedUInt64 NextLcgValue();

} // namespace psp

//----------------------------------------------------------------------------//
// Header implementations                                                     //
//----------------------------------------------------------------------------//

inline UInt32
psp::NextPowerOfTwo( UInt32 a )
{
    --a;
    a |= a >> 1;
    a |= a >> 2;
    a |= a >> 4;
    a |= a >> 8;
    a |= a >> 16;
    ++a;
    a += ( a == 0 );

    return a;
}

inline UInt32
psp::Lower16Bits( UInt32 a )
{
    return a & 0xFFFF;
}

inline UInt32
psp::Upper16Bits( UInt32 a )
{
    return (a >> 16) & 0xFFFF;
}

inline ExpandedUInt64
psp::Expand( UInt32 a )
{
    ExpandedUInt64 b;
    b[0] = LowerBits( a );
    b[1] = UpperBits( a );
    b[2] = 0U;
    b[3] = 0U;
}

inline ExpandedUInt64
psp::Expand( UInt64 a )
{
    ExpandedUInt64 b;
    b[0] = Lower16Bits( a[0] );
    b[1] = Upper16Bits( a[0] );
    b[2] = Lower16Bits( a[1] );
    b[3] = Upper16Bits( a[1] );

    return b;
}

inline UInt64
psp::Deflate( ExpandedUInt64 a )
{
    UInt64 b;
    b[0] = a[0] + ( a[1] << 16 );
    b[1] = a[2] + ( a[3] << 16 );

    return b;
}

inline void
psp::CarryUpper16Bits( ExpandedUInt64& c )
{
    c[1] += Upper16Bits(c[0]);
    c[0] = Lower16Bits(c[0]);
    c[2] += Upper16Bits(c[1]);
    c[1] = Lower16Bits(c[1]);
    c[3] += Upper16Bits(c[2]);
    c[2] = Lower16Bits(c[2]);
    c[3] = Lower16Bits(c[3]);
}

// Multiply two expanded UInt64's (mod 2^64)
//
// We do so by breaking the 64-bit integers into 16-bit pieces so that
// products can be safely be computed with 32-bit unsigned integers.
//
// a = 2^48 a3 + 2^32 a2 + 2^16 a1 + 2^0 a0,
// b = 2^48 b3 + 2^32 b2 + 2^16 b1 + 2^0 b0,
// where a_j, b_j < 2^16.
//
// Then,
// a b = 
//   2^96 ( a3 b3 ) + 
//   2^80 ( a3 b2 + b3 a2 ) + 
//   2^64 ( a3 b1 + a2 b2 + a1 b3 ) +
//   2^48 ( a3 b0 + a2 b1 + a1 b2 + a0 b3 ) +
//   2^32 ( a2 b0 + a1 b1 + a0 b2 ) + 
//   2^16 ( a1 b0 + a0 b1 ) +
//   2^0  ( a0 b0 )
//
// Since c := a b (mod 2^64), only the last four terms must be computed.
//
// NOTE: k (mod 2^n) may be quickly computed using k & (2^n - 1),
//       and 2^16-1 = 65535 = 0xFFFF.
inline ExpandedUInt64 
psp::MultiplyWith64BitMod( ExpandedUInt64 a, ExpandedUInt64 b )
{
    UInt32 temp;
    ExpandedUInt64 c;

    // c = 2^0 a0 b0
    temp = a[0]*b[0];
    c[0] = Lower16Bits( temp );
    c[1] = Upper16Bits( temp );

    // c += 2^16 ( a1 b0 + a0 b1 )
    temp = a[1]*b[0];
    c[1] += Lower16Bits( temp );
    c[2] =  Upper16Bits( temp );
    temp = a[0]*b[1];
    c[1] += Lower16Bits( temp );
    c[2] += Upper16Bits( temp );

    // c += 2^32 ( a2 b0 + a1 b1 + a0 b2 )
    temp = a[2]*b[0];
    c[2] += Lower16Bits( temp );
    c[3] =  Upper16Bits( temp );
    temp = a[1]*b[1];
    c[2] += Lower16Bits( temp );
    c[3] += Upper16Bits( temp );
    temp = a[0]*b[2];
    c[2] += Lower16Bits( temp );
    c[3] += Upper16Bits( temp );

    // c += 2^48 ( a3 b0 + a2 b1 + a1 b2 + a0 b3 )
    temp = a[3]*b[0];
    c[3] += Lower16Bits( temp );
    temp = a[2]*b[1];
    c[3] += Lower16Bits( temp );
    temp = a[1]*b[2];
    c[3] += Lower16Bits( temp );
    temp = a[0]*b[3];
    c[3] += Lower16Bits( temp );

    CarryUpper16Bits( c );
    return c;
}

inline ExpandedUInt64 
psp::AddWith64BitMod( ExpandedUInt64 a, ExpandedUInt64 b )
{
    ExpandedUInt64 c;
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
    c[2] = a[2] + b[2];
    c[3] = a[3] + b[3];

    CarryUpper16Bits( c );
    return c;
}

#endif // PSP_RANDOM_HPP
