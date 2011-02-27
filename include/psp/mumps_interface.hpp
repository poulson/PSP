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

#include "smumps_c.h"
#include "dmumps_c.h"
#include "cmumps_c.h"
#include "zmumps_c.h"

#include <vector>

extern "C" {

typedef mumps_complex SComplex;    
typedef mumps_double_complex DComplex;

} // extern "C"

namespace psp {
namespace mumps {

template<typename F> struct Handle;
template<> struct Handle<float>    { SMUMPS_STRUC_C _internal; };
template<> struct Handle<double>   { DMUMPS_STRUC_C _internal; };
template<> struct Handle<SComplex> { CMUMPS_STRUC_C _internal; };
template<> struct Handle<DComplex> { ZMUMPS_STRUC_C _internal; };

template<typename F>
void 
Init
( MPI_Comm comm, Handle<F>& h );

template<typename F>
void 
SlaveAnalysisWithManualOrdering
( Handle<F>& h );

template<typename F>
void 
HostAnalysisWithManualOrdering
( Handle<F>& h,
  int numVertices, int numNonzeros, 
  int* rowIndices, int* colIndices, int* ordering );

template<typename F>
void 
SlaveAnalysisWithMetisOrdering
( Handle<F>& h );

template<typename F>
void 
HostAnalysisWithMetisOrdering
( Handle<F>& h, 
  int numVertices, int numNonzeros,
  int* rowIndices, int* colIndices );

template<typename F>
int 
Factor
( Handle<F>& h, 
  int numLocalNonzeros, int* localRowIndices, int* localColIndices,
  F* localABuffer );

template<typename F>
void
SlaveSolve
( Handle<F>& h, 
  F* localSolutionBuffer, int localSolutionLDim, 
  int* localIntegerBuffer );

template<typename F>
void
HostSolve
( Handle<F>& h, 
  int numRhs, F* rhsBuffer, int rhsLDim, 
              F* localSolutionBuffer, int localSolutionLDim,
  int* localIntegerBuffer );

template<typename F>
void
Finalize
( Handle<F>& h );

// You should not directly use these routines. They are for internal usage.
void Call( Handle<float>& h );
void Call( Handle<double>& h );
void Call( Handle<SComplex>& h );
void Call( Handle<DComplex>& h );

} // namespace mumps
} // namespace psp

#endif // PSP_MUMPS_INTERFACE_H
