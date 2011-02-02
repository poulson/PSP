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

#include <vector>

extern "C" {
struct SComplex { float r; float i; };    
struct DComplex { double r; double i; };    
}

// We mirror MUMPS's C interface definition here so that we can avoid having to 
// use their headers when using psp
#define MUMPS_VERSION "4.9.2"
#define MUMPS_VERSION_MAX_LEN 14
extern "C"
struct ZMUMPS_STRUC_C {
    int sym;
    int par;
    int job;
    int comm_fortran;
    int icntl[40];
    double cntl[15]; 
    int n;

    int nz_alloc;

    // Assembled entry
    int nz;
    int* irn;
    int* jcn;
    DComplex* a;

    // Distributed entry
    int nz_loc;
    int* irn_loc;
    int* jcn_loc;
    DComplex* a_loc;

    // Element entry
    int nelt;
    int* eltptr;
    int* eltvar;
    DComplex* a_elt;

    // Ordering, if given by user
    int* perm_in;

    // Orderings returned to user
    int* sym_perm; // symmetric permutation
    int* uns_perm; // column permutation

    // Scaling (input only in this version)
    double* colsca;
    double* rowsca;

    // RHS, solution, output data and statistics
    DComplex* rhs;
    DComplex* redrhs;
    DComplex* rhs_sparse;
    DComplex* sol_loc;
    int* irhs_sparse;
    int* irhs_ptr;
    int* isol_loc;
    int nrhs;
    int lrhs;
    int lredrhs;
    int nz_rhs;
    int lsol_loc;
    int schur_mloc;
    int schur_nloc;
    int schur_lld;
    int mblock;
    int nblock;
    int nprow;
    int npcol;
    int info[40];
    int infog[40];
    int rinfo[20];
    int rinfog[20];

    // Null space
    int deficiency; // misspelled?
    int* pivnul_list;
    int* mapping;

    // Schur
    int size_schur;
    int* listvar_shchur;
    DComplex* schur;

    // Internal parameters
    int instance_number;
    DComplex* wk_user;

    // Version number: length=14 in FORTRAN + 1 for final \0 and + 1 for 
    // alignment
    char version_number[MUMPS_VERSION_MAX_LEN +2];

    // For out-of-core
    char ooc_tmpdir[256];
    char ooc_prefix[64];

    // To save the matrix in matrix market format
    char write_problem[256];
    int lwk_user;
};

// This is the prototype of the double-complex MUMPS routine, but it should 
// not be called directly from within psp
extern "C" void
zmumps_c( ZMUMPS_STRUC_C* handle );

namespace psp {

template<typename F> struct MumpsHandle;
template<> struct MumpsHandle<DComplex> { ZMUMPS_STRUC_C _internal; };

void 
MumpsInit
( MumpsHandle<DComplex>& handle );

void 
MumpsSlaveAnalysisWithManualOrdering
( MumpsHandle<DComplex>& handle );

void 
MumpsHostAnalysisWithManualOrdering
( MumpsHandle<DComplex>& handle,
  int numVertices, int numNonzeros, 
  int* rowIndices, int* colIndices, int* ordering );

void 
MumpsSlaveAnalysisWithMetisOrdering
( MumpsHandle<DComplex>& handle );

void 
MumpsHostAnalysisWithMetisOrdering
( MumpsHandle<DComplex>& handle, 
  int numVertices, int numNonzeros,
  int* rowIndices, int* colIndices );

int 
MumpsFactorization
( MumpsHandle<DComplex>& handle, 
  int numLocalNonzeros, int* localRowIndices, int* localColIndices,
  DComplex* localABuffer );

void
MumpsSlaveSolve
( MumpsHandle<DComplex>& handle, 
  DComplex* localSolutionBuffer, int localSolutionLDim, 
  int* localIntegerBuffer );

void
MumpsHostSolve
( MumpsHandle<DComplex>& handle, 
  int numRhs, DComplex* rhsBuffer, int rhsLDim, 
              DComplex* localSolutionBuffer, int localSolutionLDim,
  int* localIntegerBuffer );

void
MumpsFinalize
( MumpsHandle<DComplex>& handle );

} // namespace psp

#endif // PSP_MUMPS_INTERFACE_H
