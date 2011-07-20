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

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::Adjoint()
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::Adjoint");
#endif
    // This requires communication and is not yet written
    throw std::logic_error("DistQuasi2dHMat::Adjoint is not yet written");
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
void
psp::DistQuasi2dHMat<Scalar,Conjugated>::AdjointFrom
( const DistQuasi2dHMat<Scalar,Conjugated>& B )
{
#ifndef RELEASE
    PushCallStack("DistQuasi2dHMat::AdjointFrom");
#endif
    DistQuasi2dHMat<Scalar,Conjugated>& A = *this;

    A._numLevels = B._numLevels;
    A._maxRank = B._maxRank;
    A._targetOffset = B._targetOffset;
    A._sourceOffset = B._sourceOffset;
    A._stronglyAdmissible = B._stronglyAdmissible;

    A._xSizeTarget = B._xSizeTarget;
    A._ySizeTarget = B._ySizeTarget;
    A._xSizeSource = B._xSizeSource;
    A._ySizeSource = B._ySizeSource;
    A._zSize = B._zSize;

    A._xTarget = B._xTarget;
    A._yTarget = B._yTarget;
    A._xSource = B._xSource;
    A._ySource = B._ySource;

    A._teams = B._teams;
    A._level = B._level;
    A._inTargetTeam = B._inTargetTeam;
    A._inSourceTeam = B._inSourceTeam;
    A._targetRoot = B._targetRoot;
    A._sourceRoot = B._sourceRoot;

    A._block.Clear();
    A._block.type = B._block.type;

    // This requires communication and is not yet written
    throw std::logic_error("DistQuasi2dHMat::AdjointFrom is not yet written");

#ifndef RELEASE
    PopCallStack();
#endif
}

