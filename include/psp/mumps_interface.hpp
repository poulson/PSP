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

   Additional permissions under GNU GPL version 3 section 7

   If you modify this Program, or any covered work, by linking or combining it
   with MUMPS and/or ParMetis (or modified versions of those libraries),
   containing parts covered by the terms of the respective licenses of MUMPS
   and ParMetis, the licensors of this Program grant you additional permission
   to convey the resulting work. {Corresponding Source for a non-source form of
   such a combination shall include the source code for the parts of MUMPS and
   ParMetis used as well as that of the covered work.}
*/
#ifndef PSP_MUMPS_INTERFACE_H
#define PSP_MUMPS_INTERFACE_H 1

#include "zmumps_c.h"

#include <vector>

extern "C" {

typedef mumps_complex SComplex;    
typedef mumps_double_complex DComplex;

} // extern "C"

namespace psp {
namespace mumps {

template<typename F> struct Handle;
template<> struct Handle<DComplex> { ZMUMPS_STRUC_C _internal; };

void 
Init
( MPI_Comm comm, Handle<DComplex>& handle );

void 
SlaveAnalysisWithManualOrdering
( Handle<DComplex>& handle );

void 
HostAnalysisWithManualOrdering
( Handle<DComplex>& handle,
  int numVertices, int numNonzeros, 
  int* rowIndices, int* colIndices, int* ordering );

void 
SlaveAnalysisWithMetisOrdering
( Handle<DComplex>& handle );

void 
HostAnalysisWithMetisOrdering
( Handle<DComplex>& handle, 
  int numVertices, int numNonzeros,
  int* rowIndices, int* colIndices );

int 
Factor
( Handle<DComplex>& handle, 
  int numLocalNonzeros, int* localRowIndices, int* localColIndices,
  DComplex* localABuffer );

void
SlaveSolve
( Handle<DComplex>& handle, 
  DComplex* localSolutionBuffer, int localSolutionLDim, 
  int* localIntegerBuffer );

void
HostSolve
( Handle<DComplex>& handle, 
  int numRhs, DComplex* rhsBuffer, int rhsLDim, 
              DComplex* localSolutionBuffer, int localSolutionLDim,
  int* localIntegerBuffer );

void
Finalize
( Handle<DComplex>& handle );

} // namespace mumps
} // namespace psp

#endif // PSP_MUMPS_INTERFACE_H
