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
#include "psp.hpp"
#include <sstream>
#include <stdexcept>

namespace {
int 
TranslateCommFromCToF( MPI_Comm comm )
{
    int fortranComm;
#if defined(HAVE_MPI_COMM_C2F)
    fortranComm = MPI_Comm_c2f( comm );
#elif defined(HAVE_MPIR_FROM_POINTER)
    fortranComm = MPIR_FromPointer( comm );
#else
    fortranComm = (int)comm;
#endif
    return fortranComm;
}
} // anonymous namespace

#define USE_COMM_WORLD -987654
void
psp::mumps::Init
( MPI_Comm comm, psp::mumps::Handle<DComplex>& handle )
{
    ZMUMPS_STRUC_C& h = handle._internal;

    // Signal that we would like to initialize an instance of MUMPS
    h.job = -1; 

    // Signal that our matrix is general symmetric (not Hermitian)
    h.sym = 2;

    // Have the host participate in the factorization/solve as well
    h.par = 1; 

    // Convert the specified communicator so that it can be used by Fortran
    h.comm_fortran = TranslateCommFromCToF( comm );

    // Define the input/output settings
    h.icntl[0] = 6; // default error output stream (- to suppress)
    h.icntl[1] = 0; // default diagnostic output stream (- to suppress)
    h.icntl[2] = 6; // default global output stream (- to suppress)
    h.icntl[3] = 2; // default information level (- to suppress)

    // Initialize the instance
    zmumps_c( &h );

    // Summary of returned info:
    //
    // h.info[0]: 0 if call was successful
    // h.info[1]: holds additional info if error or warning
    // h.infog[0]: 0 if successful, negative if error, positive if warning
    // h.infog[1]: additional info about error or warning
    if( h.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::MumpsInit returned with info[0]=" << h.info[0];
        throw std::runtime_error( msg.str() );
    }
}

void
psp::mumps::SlaveAnalysisWithManualOrdering
( psp::mumps::Handle<DComplex>& handle )
{
    ZMUMPS_STRUC_C& h = handle._internal;

    // Signal that we would like to analyze a matrix
    h.job = 1;

    // Fill the control variables
    h.icntl[4] = 0; // default host anlalysis method
    h.icntl[5] = 7; // default permutation/scaling option
    h.icntl[6] = 7; // manual ordering is supplied
    h.icntl[7] = 77; // let MUMPS determine the scaling strategy
    h.icntl[11] = 0; // default general symmetric ordering strategy
    h.icntl[12] = 0; // default ScaLAPACK usage on root (important param)
    h.icntl[13] = 20; // percent increase in estimated workspace
    h.icntl[17] = 2; // host analyzes, but any distribution is allowed
    h.icntl[18] = 0; // do not return the Schur complement
    h.icntl[27] = 1; // our manual analysis is considered sequential
    h.icntl[28] = 0; // parallel analysis parameter (meaningless for us)
    h.cntl[3] = -1.0; // threshold for static pivoting (?)

    // Analyze the matrix
    zmumps_c( &h );

    // Summary of returned info:
    //
    // h.rinfo[0]: estimated flops on this process for elimination
    // h.info[0]: 0 if call was successful
    // h.info[1]: holds additional info if error or warning
    // h.info[2]: estimated local entries needed for factorization; 
    //            if negative, it is the number of million entries
    // h.info[3]: estimated local integers needed for factorization
    // h.info[4]: estimated max local front size
    // h.info[5]: number of nodes in the complete tree
    // h.info[6]: min estimated size of integer array IS for factorization
    // h.info[7]: min estimated size of array S for fact.; if neg, millions
    // h.info[14]: estimated MB of in-core work space for fact./solve
    // h.info[16]: estimated MB of OOC work space for fact./solve
    // h.info[18]: estimated size of IS to run OOC factorization
    // h.info[19]: estimated size of S to run OOC fact; if neg, in millions
    // h.info[23]: estimated entries in factors on this process;
    //             if negative, is is the number of million entries
    // h.rinfog[0]: estimated total flops for elimination process
    // h.infog[0]: 0 if successful, negative if error, positive if warning
    // h.infog[1]: additional info about error or warning
    // h.infog[2]: total estimated workspace for factors; if negative, ...
    // h.infog[3]: total estimated integer workspace for factors
    // h.infog[4]: estimated max front size in the complete tree
    // h.infog[5]: number of nodes in complete tree
    // h.infog[6]: ordering method used
    // h.infog[7]: structural symmetry in percent
    // h.infog[15]: estimated max MB of MUMPS data for in-core fact.
    // h.infog[16]: estimated total MB of MUMPS data for in-core fact.
    // h.infog[19]: estimated number of entries in factors; if neg, ...
    // h.infog[22]: value of h.icntl[5] effectively used
    // h.infog[23]: value of h.icntl[11] effectively used
    // h.infog[25]: estimated max MB of MUMPS data for OOC factorization
    // h.infog[26]: estimated total MB of MUMPS data for OOC factorization
    // h.infog[32]: effective value for h.icntl[7]    
    if( h.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::MumpsSlaveAnalysisWithManualOrdering returned with "
               "info[0]=" << h.info[0];
        throw std::runtime_error( msg.str() );
    }
}

// For the chosen analysis method, MUMPS requires the row/column indices and 
// the specified ordering to reside entirely on the host...
void
psp::mumps::HostAnalysisWithManualOrdering
( psp::mumps::Handle<DComplex>& handle,
  int numVertices, int numNonzeros,
  int* rowIndices, int* colIndices, int* ordering )
{
    ZMUMPS_STRUC_C& h = handle._internal;

    // Signal that we would like to analyze a matrix
    h.job = 1; 

    // Fill in the size of the system
    h.n = numVertices;
    h.nz = numNonzeros;

    // Set the row/column index buffers
    h.irn = rowIndices;
    h.jcn = colIndices;

    // Set the manual ordering buffer
    h.perm_in = ordering;

    // Fill the control variables
    h.icntl[4] = 0; // default host anlalysis method
    h.icntl[5] = 7; // default permutation/scaling option
    h.icntl[6] = 7; // manual ordering is supplied
    h.icntl[7] = 77; // let MUMPS determine the scaling strategy
    h.icntl[11] = 0; // default general symmetric ordering strategy
    h.icntl[12] = 0; // default ScaLAPACK usage on root (important param)
    h.icntl[13] = 20; // percent increase in estimated workspace
    h.icntl[17] = 2; // host analyzes, but any distribution is allowed
    h.icntl[18] = 0; // do not return the Schur complement
    h.icntl[27] = 1; // our manual analysis is considered sequential
    h.icntl[28] = 0; // parallel analysis parameter (meaningless for us)
    h.cntl[3] = -1.0; // threshold for static pivoting (?)

    // Analyze the matrix
    zmumps_c( &h );

    // Summary of returned info:
    //
    // h.rinfo[0]: estimated flops on this process for elimination
    // h.info[0]: 0 if call was successful
    // h.info[1]: holds additional info if error or warning
    // h.info[2]: estimated local entries needed for factorization; 
    //            if negative, it is the number of million entries
    // h.info[3]: estimated local integers needed for factorization
    // h.info[4]: estimated max local front size
    // h.info[5]: number of nodes in the complete tree
    // h.info[6]: min estimated size of integer array IS for factorization
    // h.info[7]: min estimated size of array S for fact.; if neg, millions
    // h.info[14]: estimated MB of in-core work space for fact./solve
    // h.info[16]: estimated MB of OOC work space for fact./solve
    // h.info[18]: estimated size of IS to run OOC factorization
    // h.info[19]: estimated size of S to run OOC fact; if neg, in millions
    // h.info[23]: estimated entries in factors on this process;
    //             if negative, is is the number of million entries
    // h.rinfog[0]: estimated total flops for elimination process
    // h.infog[0]: 0 if successful, negative if error, positive if warning
    // h.infog[1]: additional info about error or warning
    // h.infog[2]: total estimated workspace for factors; if negative, ...
    // h.infog[3]: total estimated integer workspace for factors
    // h.infog[4]: estimated max front size in the complete tree
    // h.infog[5]: number of nodes in complete tree
    // h.infog[6]: ordering method used
    // h.infog[7]: structural symmetry in percent
    // h.infog[15]: estimated max MB of MUMPS data for in-core fact.
    // h.infog[16]: estimated total MB of MUMPS data for in-core fact.
    // h.infog[19]: estimated number of entries in factors; if neg, ...
    // h.infog[22]: value of h.icntl[5] effectively used
    // h.infog[23]: value of h.icntl[11] effectively used
    // h.infog[25]: estimated max MB of MUMPS data for OOC factorization
    // h.infog[26]: estimated total MB of MUMPS data for OOC factorization
    // h.infog[32]: effective value for h.icntl[7]
    if( h.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::MumpsHostAnalysisWithManualOrdering returned with "
               "info[0]=" << h.info[0];
        throw std::runtime_error( msg.str() );
    }
}

void
psp::mumps::SlaveAnalysisWithMetisOrdering
( psp::mumps::Handle<DComplex>& handle )
{
    ZMUMPS_STRUC_C& h = handle._internal;

    // Signal that we would like to analyze a matrix
    h.job = 1; 

    // Fill the control variables
    h.icntl[4] = 0; // default host analysis method
    h.icntl[5] = 7; // default permutation/scaling option
    h.icntl[6] = 5; // use Metis for the ordering
    h.icntl[7] = 77; // let MUMPS determine the scaling strategy
    h.icntl[11] = 0; // default general symmetric ordering strategy
    h.icntl[12] = 0; // default ScaLAPACK usage on root (important param)
    h.icntl[13] = 20; // percent increase in estimated workspace
    h.icntl[17] = 2; // host analyzes, but any distribution is allowed
    h.icntl[18] = 0; // do not return the Schur complement
    h.icntl[27] = 2; // perform parallel analysis
    h.icntl[28] = 2; // use ParMetis for the parallel analysis
    h.cntl[3] = -1.0; // threshold for static pivoting (?)

    // Analyze the matrix
    zmumps_c( &h );

    // Summary of the returned info:
    //
    // h.rinfo[0]: estimated flops on this process for elimination
    // h.info[0]: 0 if call was successful
    // h.info[1]: holds additional info if error or warning
    // h.info[2]: estimated local entries needed for factorization; 
    //            if negative, it is the number of million entries
    // h.info[3]: estimated local integers needed for factorization
    // h.info[4]: estimated max local front size
    // h.info[5]: number of nodes in the complete tree
    // h.info[6]: min estimated size of integer array IS for factorization
    // h.info[7]: min estimated size of array S for fact.; if neg, millions
    // h.info[14]: estimated MB of in-core work space for fact./solve
    // h.info[16]: estimated MB of OOC work space for fact./solve
    // h.info[18]: estimated size of IS to run OOC factorization
    // h.info[19]: estimated size of S to run OOC fact; if neg, in millions
    // h.info[23]: estimated entries in factors on this process;
    //             if negative, is is the number of million entries
    // h.rinfog[0]: estimated total flops for elimination process
    // h.infog[0]: 0 if successful, negative if error, positive if warning
    // h.infog[1]: additional info about error or warning
    // h.infog[2]: total estimated workspace for factors; if negative, ...
    // h.infog[3]: total estimated integer workspace for factors
    // h.infog[4]: estimated max front size in the complete tree
    // h.infog[5]: number of nodes in complete tree
    // h.infog[6]: ordering method used
    // h.infog[7]: structural symmetry in percent
    // h.infog[15]: estimated max MB of MUMPS data for in-core fact.
    // h.infog[16]: estimated total MB of MUMPS data for in-core fact.
    // h.infog[19]: estimated number of entries in factors; if neg, ...
    // h.infog[22]: value of h.icntl[5] effectively used
    // h.infog[23]: value of h.icntl[11] effectively used
    // h.infog[25]: estimated max MB of MUMPS data for OOC factorization
    // h.infog[26]: estimated total MB of MUMPS data for OOC factorization
    // h.infog[32]: effective value for h.icntl[7]
    if( h.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::MumpsSlaveAnalysisWithMetisOrdering returned with "
               "info[0]=" << h.info[0];
        throw std::runtime_error( msg.str() );
    }
}

// For the chosen analysis method, MUMPS requires the row/column indices to 
// reside entirely on the host...
void
psp::mumps::HostAnalysisWithMetisOrdering
( psp::mumps::Handle<DComplex>& handle, 
  int numVertices, int numNonzeros, 
  int* rowIndices, int* colIndices )
{
    ZMUMPS_STRUC_C& h = handle._internal;

    // Signal that we would like to analyze a matrix
    h.job = 1; 

    // Fill in the size of the system
    h.n = numVertices;
    h.nz = numNonzeros;

    // Set the row/column index buffers
    h.irn = rowIndices;
    h.jcn = colIndices;

    // Fill the control variables
    h.icntl[4] = 0; // default host analysis method
    h.icntl[5] = 7; // default permutation/scaling option
    h.icntl[6] = 5; // use Metis for the ordering
    h.icntl[7] = 77; // let MUMPS determine the scaling strategy
    h.icntl[11] = 0; // default general symmetric ordering strategy
    h.icntl[12] = 0; // default ScaLAPACK usage on root (important param)
    h.icntl[13] = 20; // percent increase in estimated workspace
    h.icntl[17] = 2; // host analyzes, but any distribution is allowed
    h.icntl[18] = 0; // do not return the Schur complement
    h.icntl[27] = 2; // perform parallel analysis
    h.icntl[28] = 2; // use ParMetis for the parallel analysis
    h.cntl[3] = -1.0; // threshold for static pivoting (?)

    // Analyze the matrix
    zmumps_c( &h );

    // Summary of returned info:
    //
    // h.rinfo[0]: estimated flops on this process for elimination
    // h.info[0]: 0 if call was successful
    // h.info[1]: holds additional info if error or warning
    // h.info[2]: estimated local entries needed for factorization; 
    //            if negative, it is the number of million entries
    // h.info[3]: estimated local integers needed for factorization
    // h.info[4]: estimated max local front size
    // h.info[5]: number of nodes in the complete tree
    // h.info[6]: min estimated size of integer array IS for factorization
    // h.info[7]: min estimated size of array S for fact.; if neg, millions
    // h.info[14]: estimated MB of in-core work space for fact./solve
    // h.info[16]: estimated MB of OOC work space for fact./solve
    // h.info[18]: estimated size of IS to run OOC factorization
    // h.info[19]: estimated size of S to run OOC fact; if neg, in millions
    // h.info[23]: estimated entries in factors on this process;
    //             if negative, is is the number of million entries
    // h.rinfog[0]: estimated total flops for elimination process
    // h.infog[0]: 0 if successful, negative if error, positive if warning
    // h.infog[1]: additional info about error or warning
    // h.infog[2]: total estimated workspace for factors; if negative, ...
    // h.infog[3]: total estimated integer workspace for factors
    // h.infog[4]: estimated max front size in the complete tree
    // h.infog[5]: number of nodes in complete tree
    // h.infog[6]: ordering method used
    // h.infog[7]: structural symmetry in percent
    // h.infog[15]: estimated max MB of MUMPS data for in-core fact.
    // h.infog[16]: estimated total MB of MUMPS data for in-core fact.
    // h.infog[19]: estimated number of entries in factors; if neg, ...
    // h.infog[22]: value of h.icntl[5] effectively used
    // h.infog[23]: value of h.icntl[11] effectively used
    // h.infog[25]: estimated max MB of MUMPS data for OOC factorization
    // h.infog[26]: estimated total MB of MUMPS data for OOC factorization
    // h.infog[32]: effective value for h.icntl[7]
    if( h.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::MumpsHostAnalysisWithMetisOrdering returned with "
               "info[0]=" << h.info[0];
        throw std::runtime_error( msg.str() );
    }
}

int 
psp::mumps::Factor
( psp::mumps::Handle<DComplex>& handle, 
  int numLocalNonzeros, int* localRowIndices, int* localColIndices,
  DComplex* localABuffer )
{
    ZMUMPS_STRUC_C& h = handle._internal;

    // Signal that we would like to factor a matrix
    h.job = 2;

    // Describe the distribution
    h.nz_loc = numLocalNonzeros;
    h.irn_loc = localRowIndices;
    h.jcn_loc = localColIndices;
    h.a_loc = localABuffer;

    // Fill the control variables
    h.icntl[21] = 0; // in-core factorization
    h.icntl[22] = 5 * h.infog[25]; // workspace size
    h.icntl[23] = 0; // null-pivots cause the factorization to fail.
                     // We might want to change this because our 
                     // factorization is only being used as a 
                     // preconditioner.
    h.cntl[0] = 0.01; // default relative threshold for pivoting
                      // higher values lead to more fill-in
    h.cntl[2] = 0.0; // null-pivot flag (irrelevant since we ignore them)
    h.cntl[4] = 0.0; // fixation for null-pivots (?)

    // Factor the matrix
    zmumps_c( &h );

    // Summary of returned info:
    //
    // h.rinfo[1]: flops on this process for assembly
    // h.rinfo[2]: flops on this process for elimination
    // h.info[0]: 0 if call was successful
    // h.info[1]: holds additional info if error or warning
    // h.info[8]: number of local entries for storing factorization;
    //            if negative, number of entries in millions
    // h.info[9]: size of integer space for storing factorization
    // h.info[10]: order of largest frontal matrix on this process
    // h.info[11]: number of negative pivots on this process
    // h.info[12]: number of postponed eliminations because of numerics
    // h.info[13]: number of memory compresses
    // h.info[15]: size in MB of MUMPS-allocated data during in-core fact.
    // h.info[17]: size in MB of MUMPS-allocated data during OOC fact.
    // h.info[20]: effective space for S; if negative, in millions
    // h.info[21]: MB of memory used for factorization
    // h.info[22]: number of pivots eliminated on this process.
    //             set ISOL_LOC to it and SOL_LOC = LSOL_LOC x NRHS,
    //             where LSOL_LOC >= h.info[22]
    // h.info[24]: number of tiny pivots on this process
    // h.info[26]: number of entries in factors on this process;
    //             if negative, then number in millions
    // h.rinfog[1]: total flops for assembly
    // h.rinfog[2]: total flops for the elimination process
    // h.infog[0]: 0 if successful, negative if error, positive if warning
    // h.infog[1]: additional info about error or warning
    // h.infog[8]: total workspace for factors; if negative, in millions
    // h.infog[9]: total integer workspace for factors
    // h.infog[10]: order of largest frontal matrix
    // h.infog[11]: total number of negative pivots
    // h.infog[12]: total number of delayed pivots (if > 10%, num. problem)
    // h.infog[13]: total number of memory compresses
    // h.infog[17]: max MB of MUMPS data needed for factorization
    // h.infog[18]: total MB of MUMPS data needed for factorization
    // h.infog[20]: max MB of memory used during factorization
    // h.infog[21]: total MB of memory used during factorization
    // h.infog[24]: number of tiny pivots
    // h.infog[27]: number of null pivots encountered
    // h.infog[28]: effective number of entries in factors; if neg, ...
    if( h.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::MumpsFactorization returned with info[0]="
            << h.info[0];
        throw std::runtime_error( msg.str() );
    }

    return h.info[22];
}

void
psp::mumps::SlaveSolve
( psp::mumps::Handle<DComplex>& handle, 
  DComplex* localSolutionBuffer, int localSolutionLDim, 
  int* localIntegerBuffer ) 
{
    ZMUMPS_STRUC_C& h = handle._internal;

    // Signal that we would like to solve
    h.job = 3;    

    // Fill the solution buffers
    h.sol_loc = localSolutionBuffer;
    h.lsol_loc = localSolutionLDim;
    h.isol_loc = localIntegerBuffer;

    // Fill the control variables
    h.icntl[8] = 1; // solve Ax=b, not A^Tx=b
    h.icntl[9] = 0; // no iterative refinement for distributed solutions
    h.icntl[10] = 0; // do not return statistics b/c they are expensive
    h.icntl[13] = 20; // percent increase in estimated workspace
    h.icntl[19] = 0; // we will input our RHS's as dense vectors
    h.icntl[20] = 1; // the RHS's will be kept distributed
    h.icntl[24] = 0; // we do not need to handle null-pivots yet
    h.icntl[25] = 0; // do not worry about the interface Schur complement
    h.icntl[26] = 128; // IMPORTANT: handles the number of RHS's that can
                       // be simultaneously solved
    h.cntl[1] = 0; // iterative refinement is irrelavant with multiple RHS

    // Solve the system
    zmumps_c( &h );

    // Summary of the returned information:
    //
    // h.info[0]: 0 if call was successful
    // h.info[1]: holds additional info if error or warning
    // h.info[25]: MB of work space used for solution on this process
    //             (maximum and sum returned by infog[29] and infog[30])
    // h.rinfog[3-10]: only set if error analysis is on
    // h.infog[0]: 0 if successful, negative if error, positive if warning
    // h.infog[1]: additional info about error or warning
    // h.infog[14]: number of steps of iterative refinement (zero for us)
    // h.infog[29]: max MB of effective memory for solution
    // h.infog[30]: total MB of effective memory for solution
   
    if( h.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::MumpsSlaveSolve returned with info[0]=" << h.info[0];
        throw std::runtime_error( msg.str() );
    }
}

// MUMPS requires the RHS to reside entirely on the host...
void
psp::mumps::HostSolve
( psp::mumps::Handle<DComplex>& handle,
  int numRhs, DComplex* rhsBuffer, int rhsLDim, 
              DComplex* localSolutionBuffer, int localSolutionLDim,
  int* localIntegerBuffer ) 
{
    ZMUMPS_STRUC_C& h = handle._internal;

    // Signal that we would like to solve
    h.job = 3;    

    // Set up the RHS
    h.nrhs = numRhs;
    h.rhs = rhsBuffer;
    h.lrhs = rhsLDim;

    // Set up the solution buffers
    h.sol_loc = localSolutionBuffer;
    h.lsol_loc = localSolutionLDim;
    h.isol_loc = localIntegerBuffer;

    // Fill the control variables
    h.icntl[8] = 1; // solve Ax=b, not A^Tx=b
    h.icntl[9] = 0; // no iterative refinement for distributed solutions
    h.icntl[10] = 0; // do not return statistics b/c they are expensive
    h.icntl[13] = 20; // percent increase in estimated workspace
    h.icntl[19] = 0; // we will input our RHS's as dense vectors
    h.icntl[20] = 1; // the RHS's will be kept distributed
    h.icntl[24] = 0; // we do not need to handle null-pivots yet
    h.icntl[25] = 0; // do not worry about the interface Schur complement
    h.icntl[26] = 128; // IMPORTANT: handles the number of RHS's that can
                       // be simultaneously solved
    h.cntl[1] = 0; // iterative refinement is irrelavant with multiple RHS

    // Solve the system
    zmumps_c( &h );

    // Summary of returned info:
    //
    // h.info[0]: 0 if call was successful
    // h.info[1]: holds additional info if error or warning
    // h.info[25]: MB of work space used for solution on this process
    //             (maximum and sum returned by infog[29] and infog[30])
    // h.rinfog[3-10]: only set if error analysis is on
    // h.infog[0]: 0 if successful, negative if error, positive if warning
    // h.infog[1]: additional info about error or warning
    // h.infog[14]: number of steps of iterative refinement (zero for us)
    // h.infog[29]: max MB of effective memory for solution
    // h.infog[30]: total MB of effective memory for solution
    
    if( h.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::MumpsHostSolve returned with info[0]=" << h.info[0];
        throw std::runtime_error( msg.str() );
    }
}

void
psp::mumps::Finalize( psp::mumps::Handle<DComplex>& handle )
{
    ZMUMPS_STRUC_C& h = handle._internal;

    // Signal that we are destroying a MUMPS instance
    h.job = -2;

    // Destroy the instance
    zmumps_c( &h );

    // Summary of returned info:
    //
    // h.info[0]: 0 if call was successful
    // h.info[1]: holds additional info if error or warning
    // h.infog[0]: 0 if successful, negative if error, positive if warning
    // h.infog[1]: additional info about error or warning
    
    if( h.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::MumpsFinalize returned with info[0]=" << h.info[0];
        throw std::runtime_error( msg.str() );
    }
}

