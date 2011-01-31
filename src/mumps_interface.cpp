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

#define USE_COMM_WORLD -987654
void
psp::ZMumpsInit
( psp::ZMumpsHandle& handle )
{
    // Signal that we would like to initialize an instance of MUMPS
    handle.job = -1; 

    // Signal that our matrix is general symmetric (not Hermitian)
    handle.sym = 2;

    // Have the host participate in the factorization/solve as well
    handle.par = 1; 

    // Use the entire MPI_COMM_WORLD communicator
    handle.comm_fortran = USE_COMM_WORLD;

    // Define the input/output settings
    handle.icntl[0] = 6; // default error output stream (- to suppress)
    handle.icntl[1] = 0; // default diagnostic output stream (- to suppress)
    handle.icntl[2] = 6; // default global output stream (- to suppress)
    //handle.icntl[3] = 2; // default information level (- to suppress)
    handle.icntl[3] = 4; // TODO: Switch back to '2' after debugging

    // Initialize the instance
    zmumps_c( &handle );

    // Summary of returned info:
    //
    // handle.info[0]: 0 if call was successful
    // handle.info[1]: holds additional info if error or warning
    // handle.info[0]: 0 if successful, negative if error, positive if warning
    // handle.infog[1]: additional info about error or warning
    if( handle.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::ZMumpsInit returned with info[0]=" << handle.info[0];
        throw std::runtime_error( msg.str() );
    }
}

void
psp::ZMumpsSlaveAnalysisWithManualOrdering
( psp::ZMumpsHandle& handle )
{
    // Signal that we would like to analyze a matrix
    handle.job = 1;

    // Fill the control variables
    handle.icntl[4] = 0; // default host anlalysis method
    handle.icntl[5] = 7; // default permutation/scaling option
    handle.icntl[6] = 7; // manual ordering is supplied
    handle.icntl[7] = 77; // let MUMPS determine the scaling strategy
    handle.icntl[11] = 0; // default general symmetric ordering strategy
    handle.icntl[12] = 0; // default ScaLAPACK usage on root (important param)
    handle.icntl[13] = 20; // percent increase in estimated workspace
    handle.icntl[17] = 2; // host analyzes, but any distribution is allowed
    handle.icntl[18] = 0; // do not return the Schur complement
    handle.icntl[27] = 1; // our manual analysis is considered sequential
    handle.icntl[28] = 0; // parallel analysis parameter (meaningless for us)
    handle.cntl[3] = -1.0; // threshold for static pivoting (?)

    // Analyze the matrix
    zmumps_c( &handle );

    // Summary of returned info:
    //
    // handle.rinfo[0]: estimated flops on this process for elimination
    // handle.info[0]: 0 if call was successful
    // handle.info[1]: holds additional info if error or warning
    // handle.info[2]: estimated local entries needed for factorization; 
    //                 if negative, it is the number of million entries
    // handle.info[3]: estimated local integers needed for factorization
    // handle.info[4]: estimated max local front size
    // handle.info[5]: number of nodes in the complete tree
    // handle.info[6]: min estimated size of integer array IS for factorization
    // handle.info[7]: min estimated size of array S for fact.; if neg, millions
    // handle.info[14]: estimated MB of in-core work space for fact./solve
    // handle.info[16]: estimated MB of OOC work space for fact./solve
    // handle.info[18]: estimated size of IS to run OOC factorization
    // handle.info[19]: estimated size of S to run OOC fact; if neg, in millions
    // handle.info[23]: estimated entries in factors on this process;
    //                  if negative, is is the number of million entries
    // handle.rinfog[0]: estimated total flops for elimination process
    // handle.infog[0]: 0 if successful, negative if error, positive if warning
    // handle.infog[1]: additional info about error or warning
    // handle.infog[2]: total estimated workspace for factors; if negative, ...
    // handle.infog[3]: total estimated integer workspace for factors
    // handle.infog[4]: estimated max front size in the complete tree
    // handle.infog[5]: number of nodes in complete tree
    // handle.infog[6]: ordering method used
    // handle.infog[7]: structural symmetry in percent
    // handle.infog[15]: estimated max MB of MUMPS data for in-core fact.
    // handle.infog[16]: estimated total MB of MUMPS data for in-core fact.
    // handle.infog[19]: estimated number of entries in factors; if neg, ...
    // handle.infog[22]: value of handle.icntl[5] effectively used
    // handle.infog[23]: value of handle.icntl[11] effectively used
    // handle.infog[25]: estimated max MB of MUMPS data for OOC factorization
    // handle.infog[26]: estimated total MB of MUMPS data for OOC factorization
    // handle.infog[32]: effective value for handle.icntl[7]    
    if( handle.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::ZMumpsSlaveAnalysisWithManualOrdering returned with "
               "info[0]=" << handle.info[0];
        throw std::runtime_error( msg.str() );
    }
}

// For the chosen analysis method, MUMPS requires the row/column indices and 
// the specified ordering to reside entirely on the host...
void
psp::ZMumpsHostAnalysisWithManualOrdering
( psp::ZMumpsHandle& handle,
  int numVertices, int numNonzeros,
  int* rowIndices, int* colIndices, int* ordering )
{
    // Signal that we would like to analyze a matrix
    handle.job = 1; 

    // Fill in the size of the system
    handle.n = numVertices;
    handle.nz = numNonzeros;

    // Set the row/column index buffers
    handle.irn = rowIndices;
    handle.jcn = colIndices;

    // Set the manual ordering buffer
    handle.perm_in = ordering;

    // Fill the control variables
    handle.icntl[4] = 0; // default host anlalysis method
    handle.icntl[5] = 7; // default permutation/scaling option
    handle.icntl[6] = 7; // manual ordering is supplied
    handle.icntl[7] = 77; // let MUMPS determine the scaling strategy
    handle.icntl[11] = 0; // default general symmetric ordering strategy
    handle.icntl[12] = 0; // default ScaLAPACK usage on root (important param)
    handle.icntl[13] = 20; // percent increase in estimated workspace
    handle.icntl[17] = 2; // host analyzes, but any distribution is allowed
    handle.icntl[18] = 0; // do not return the Schur complement
    handle.icntl[27] = 1; // our manual analysis is considered sequential
    handle.icntl[28] = 0; // parallel analysis parameter (meaningless for us)
    handle.cntl[3] = -1.0; // threshold for static pivoting (?)

    // Analyze the matrix
    zmumps_c( &handle );

    // Summary of returned info:
    //
    // handle.rinfo[0]: estimated flops on this process for elimination
    // handle.info[0]: 0 if call was successful
    // handle.info[1]: holds additional info if error or warning
    // handle.info[2]: estimated local entries needed for factorization; 
    //                 if negative, it is the number of million entries
    // handle.info[3]: estimated local integers needed for factorization
    // handle.info[4]: estimated max local front size
    // handle.info[5]: number of nodes in the complete tree
    // handle.info[6]: min estimated size of integer array IS for factorization
    // handle.info[7]: min estimated size of array S for fact.; if neg, millions
    // handle.info[14]: estimated MB of in-core work space for fact./solve
    // handle.info[16]: estimated MB of OOC work space for fact./solve
    // handle.info[18]: estimated size of IS to run OOC factorization
    // handle.info[19]: estimated size of S to run OOC fact; if neg, in millions
    // handle.info[23]: estimated entries in factors on this process;
    //                  if negative, is is the number of million entries
    // handle.rinfog[0]: estimated total flops for elimination process
    // handle.infog[0]: 0 if successful, negative if error, positive if warning
    // handle.infog[1]: additional info about error or warning
    // handle.infog[2]: total estimated workspace for factors; if negative, ...
    // handle.infog[3]: total estimated integer workspace for factors
    // handle.infog[4]: estimated max front size in the complete tree
    // handle.infog[5]: number of nodes in complete tree
    // handle.infog[6]: ordering method used
    // handle.infog[7]: structural symmetry in percent
    // handle.infog[15]: estimated max MB of MUMPS data for in-core fact.
    // handle.infog[16]: estimated total MB of MUMPS data for in-core fact.
    // handle.infog[19]: estimated number of entries in factors; if neg, ...
    // handle.infog[22]: value of handle.icntl[5] effectively used
    // handle.infog[23]: value of handle.icntl[11] effectively used
    // handle.infog[25]: estimated max MB of MUMPS data for OOC factorization
    // handle.infog[26]: estimated total MB of MUMPS data for OOC factorization
    // handle.infog[32]: effective value for handle.icntl[7]
    if( handle.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::ZMumpsHostAnalysisWithManualOrdering returned with "
               "info[0]=" << handle.info[0];
        throw std::runtime_error( msg.str() );
    }
}

void
psp::ZMumpsSlaveAnalysisWithMetisOrdering
( psp::ZMumpsHandle& handle )
{
    // Signal that we would like to analyze a matrix
    handle.job = 1; 

    // Fill the control variables
    handle.icntl[4] = 0; // default host analysis method
    handle.icntl[5] = 7; // default permutation/scaling option
    handle.icntl[6] = 5; // use Metis for the ordering
    handle.icntl[7] = 77; // let MUMPS determine the scaling strategy
    handle.icntl[11] = 0; // default general symmetric ordering strategy
    handle.icntl[12] = 0; // default ScaLAPACK usage on root (important param)
    handle.icntl[13] = 20; // percent increase in estimated workspace
    handle.icntl[17] = 2; // host analyzes, but any distribution is allowed
    handle.icntl[18] = 0; // do not return the Schur complement
    handle.icntl[27] = 2; // perform parallel analysis
    handle.icntl[28] = 2; // use ParMetis for the parallel analysis
    handle.cntl[3] = -1.0; // threshold for static pivoting (?)

    // Analyze the matrix
    zmumps_c( &handle );

    // Summary of the returned info:
    //
    // handle.rinfo[0]: estimated flops on this process for elimination
    // handle.info[0]: 0 if call was successful
    // handle.info[1]: holds additional info if error or warning
    // handle.info[2]: estimated local entries needed for factorization; 
    //                 if negative, it is the number of million entries
    // handle.info[3]: estimated local integers needed for factorization
    // handle.info[4]: estimated max local front size
    // handle.info[5]: number of nodes in the complete tree
    // handle.info[6]: min estimated size of integer array IS for factorization
    // handle.info[7]: min estimated size of array S for fact.; if neg, millions
    // handle.info[14]: estimated MB of in-core work space for fact./solve
    // handle.info[16]: estimated MB of OOC work space for fact./solve
    // handle.info[18]: estimated size of IS to run OOC factorization
    // handle.info[19]: estimated size of S to run OOC fact; if neg, in millions
    // handle.info[23]: estimated entries in factors on this process;
    //                  if negative, is is the number of million entries
    // handle.rinfog[0]: estimated total flops for elimination process
    // handle.infog[0]: 0 if successful, negative if error, positive if warning
    // handle.infog[1]: additional info about error or warning
    // handle.infog[2]: total estimated workspace for factors; if negative, ...
    // handle.infog[3]: total estimated integer workspace for factors
    // handle.infog[4]: estimated max front size in the complete tree
    // handle.infog[5]: number of nodes in complete tree
    // handle.infog[6]: ordering method used
    // handle.infog[7]: structural symmetry in percent
    // handle.infog[15]: estimated max MB of MUMPS data for in-core fact.
    // handle.infog[16]: estimated total MB of MUMPS data for in-core fact.
    // handle.infog[19]: estimated number of entries in factors; if neg, ...
    // handle.infog[22]: value of handle.icntl[5] effectively used
    // handle.infog[23]: value of handle.icntl[11] effectively used
    // handle.infog[25]: estimated max MB of MUMPS data for OOC factorization
    // handle.infog[26]: estimated total MB of MUMPS data for OOC factorization
    // handle.infog[32]: effective value for handle.icntl[7]
    if( handle.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::ZMumpsSlaveAnalysisWithMetisOrdering returned with "
               "info[0]=" << handle.info[0];
        throw std::runtime_error( msg.str() );
    }
}

// For the chosen analysis method, MUMPS requires the row/column indices to 
// reside entirely on the host...
void
psp::ZMumpsHostAnalysisWithMetisOrdering
( psp::ZMumpsHandle& handle, 
  int numVertices, int numNonzeros, 
  int* rowIndices, int* colIndices )
{
    // Signal that we would like to analyze a matrix
    handle.job = 1; 

    // Fill in the size of the system
    handle.n = numVertices;
    handle.nz = numNonzeros;

    // Set the row/column index buffers
    handle.irn = rowIndices;
    handle.jcn = colIndices;

    // Fill the control variables
    handle.icntl[4] = 0; // default host analysis method
    handle.icntl[5] = 7; // default permutation/scaling option
    handle.icntl[6] = 5; // use Metis for the ordering
    handle.icntl[7] = 77; // let MUMPS determine the scaling strategy
    handle.icntl[11] = 0; // default general symmetric ordering strategy
    handle.icntl[12] = 0; // default ScaLAPACK usage on root (important param)
    handle.icntl[13] = 20; // percent increase in estimated workspace
    handle.icntl[17] = 2; // host analyzes, but any distribution is allowed
    handle.icntl[18] = 0; // do not return the Schur complement
    handle.icntl[27] = 2; // perform parallel analysis
    handle.icntl[28] = 2; // use ParMetis for the parallel analysis
    handle.cntl[3] = -1.0; // threshold for static pivoting (?)

    // Analyze the matrix
    zmumps_c( &handle );

    // Summary of returned info:
    //
    // handle.rinfo[0]: estimated flops on this process for elimination
    // handle.info[0]: 0 if call was successful
    // handle.info[1]: holds additional info if error or warning
    // handle.info[2]: estimated local entries needed for factorization; 
    //                 if negative, it is the number of million entries
    // handle.info[3]: estimated local integers needed for factorization
    // handle.info[4]: estimated max local front size
    // handle.info[5]: number of nodes in the complete tree
    // handle.info[6]: min estimated size of integer array IS for factorization
    // handle.info[7]: min estimated size of array S for fact.; if neg, millions
    // handle.info[14]: estimated MB of in-core work space for fact./solve
    // handle.info[16]: estimated MB of OOC work space for fact./solve
    // handle.info[18]: estimated size of IS to run OOC factorization
    // handle.info[19]: estimated size of S to run OOC fact; if neg, in millions
    // handle.info[23]: estimated entries in factors on this process;
    //                  if negative, is is the number of million entries
    // handle.rinfog[0]: estimated total flops for elimination process
    // handle.infog[0]: 0 if successful, negative if error, positive if warning
    // handle.infog[1]: additional info about error or warning
    // handle.infog[2]: total estimated workspace for factors; if negative, ...
    // handle.infog[3]: total estimated integer workspace for factors
    // handle.infog[4]: estimated max front size in the complete tree
    // handle.infog[5]: number of nodes in complete tree
    // handle.infog[6]: ordering method used
    // handle.infog[7]: structural symmetry in percent
    // handle.infog[15]: estimated max MB of MUMPS data for in-core fact.
    // handle.infog[16]: estimated total MB of MUMPS data for in-core fact.
    // handle.infog[19]: estimated number of entries in factors; if neg, ...
    // handle.infog[22]: value of handle.icntl[5] effectively used
    // handle.infog[23]: value of handle.icntl[11] effectively used
    // handle.infog[25]: estimated max MB of MUMPS data for OOC factorization
    // handle.infog[26]: estimated total MB of MUMPS data for OOC factorization
    // handle.infog[32]: effective value for handle.icntl[7]
    if( handle.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::ZMumpsHostAnalysisWithMetisOrdering returned with "
               "info[0]=" << handle.info[0];
        throw std::runtime_error( msg.str() );
    }
}

int 
psp::ZMumpsFactorization
( psp::ZMumpsHandle& handle, 
  int numLocalNonzeros, int* localRowIndices, int* localColIndices,
  DComplex* localABuffer )
{
    // Signal that we would like to factor a matrix
    handle.job = 2;

    // Describe the distribution
    handle.nz_loc = numLocalNonzeros;
    handle.irn_loc = localRowIndices;
    handle.jcn_loc = localColIndices;
    handle.a_loc = localABuffer;

    // Fill the control variables
    handle.icntl[21] = 0; // in-core factorization
    handle.icntl[22] = 5 * handle.infog[25]; // workspace size
    handle.icntl[23] = 0; // null-pivots cause the factorization to fail.
                          // We might want to change this because our 
                          // factorization is only being used as a 
                          // preconditioner.
    handle.cntl[0] = 0.01; // default relative threshold for pivoting
                           // higher values lead to more fill-in
    handle.cntl[2] = 0.0; // null-pivot flag (irrelevant since we ignore them)
    handle.cntl[4] = 0.0; // fixation for null-pivots (?)

    // Factor the matrix
    zmumps_c( &handle );

    // Summary of returned info:
    //
    // handle.rinfo[1]: flops on this process for assembly
    // handle.rinfo[2]: flops on this process for elimination
    // handle.info[0]: 0 if call was successful
    // handle.info[1]: holds additional info if error or warning
    // handle.info[8]: number of local entries for storing factorization;
    //                 if negative, number of entries in millions
    // handle.info[9]: size of integer space for storing factorization
    // handle.info[10]: order of largest frontal matrix on this process
    // handle.info[11]: number of negative pivots on this process
    // handle.info[12]: number of postponed eliminations because of numerics
    // handle.info[13]: number of memory compresses
    // handle.info[15]: size in MB of MUMPS-allocated data during in-core fact.
    // handle.info[17]: size in MB of MUMPS-allocated data during OOC fact.
    // handle.info[20]: effective space for S; if negative, in millions
    // handle.info[21]: MB of memory used for factorization
    // handle.info[22]: number of pivots eliminated on this process.
    //                  set ISOL_LOC to it and SOL_LOC = LSOL_LOC x NRHS,
    //                  where LSOL_LOC >= handle.info[22]
    // handle.info[24]: number of tiny pivots on this process
    // handle.info[26]: number of entries in factors on this process;
    //                  if negative, then number in millions
    // handle.rinfog[1]: total flops for assembly
    // handle.rinfog[2]: total flops for the elimination process
    // handle.infog[0]: 0 if successful, negative if error, positive if warning
    // handle.infog[1]: additional info about error or warning
    // handle.infog[8]: total workspace for factors; if negative, in millions
    // handle.infog[9]: total integer workspace for factors
    // handle.infog[10]: order of largest frontal matrix
    // handle.infog[11]: total number of negative pivots
    // handle.infog[12]: total number of delayed pivots (if > 10%, num. problem)
    // handle.infog[13]: total number of memory compresses
    // handle.infog[17]: max MB of MUMPS data needed for factorization
    // handle.infog[18]: total MB of MUMPS data needed for factorization
    // handle.infog[20]: max MB of memory used during factorization
    // handle.infog[21]: total MB of memory used during factorization
    // handle.infog[24]: number of tiny pivots
    // handle.infog[27]: number of null pivots encountered
    // handle.infog[28]: effective number of entries in factors; if neg, ...
    if( handle.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::ZMumpsFactorization returned with info[0]="
            << handle.info[0];
        throw std::runtime_error( msg.str() );
    }

    return handle.info[22];
}

void
psp::ZMumpsSlaveSolve
( psp::ZMumpsHandle& handle, 
  DComplex* localSolutionBuffer, int localSolutionLDim, 
  int* localIntegerBuffer ) 
{
    // Signal that we would like to solve
    handle.job = 3;    

    // Fill the solution buffers
    handle.sol_loc = localSolutionBuffer;
    handle.lsol_loc = localSolutionLDim;
    handle.isol_loc = localIntegerBuffer;

    // Fill the control variables
    handle.icntl[8] = 1; // solve Ax=b, not A^Tx=b
    handle.icntl[9] = 0; // no iterative refinement for distributed solutions
    handle.icntl[10] = 0; // do not return statistics b/c they are expensive
    handle.icntl[13] = 20; // percent increase in estimated workspace
    handle.icntl[19] = 0; // we will input our RHS's as dense vectors
    handle.icntl[20] = 1; // the RHS's will be kept distributed
    handle.icntl[24] = 0; // we do not need to handle null-pivots yet
    handle.icntl[25] = 0; // do not worry about the interface Schur complement
    handle.icntl[26] = 128; // IMPORTANT: handles the number of RHS's that can
                            // be simultaneously solved
    handle.cntl[1] = 0; // iterative refinement is irrelavant with multiple RHS

    // Solve the system
    zmumps_c( &handle );

    // Summary of the returned information:
    //
    // handle.info[0]: 0 if call was successful
    // handle.info[1]: holds additional info if error or warning
    // handle.info[25]: MB of work space used for solution on this process
    //                  (maximum and sum returned by infog[29] and infog[30])
    // handle.rinfog[3-10]: only set if error analysis is on
    // handle.infog[0]: 0 if successful, negative if error, positive if warning
    // handle.infog[1]: additional info about error or warning
    // handle.infog[14]: number of steps of iterative refinement (zero for us)
    // handle.infog[29]: max MB of effective memory for solution
    // handle.infog[30]: total MB of effective memory for solution
    
    if( handle.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::ZMumpsSlaveSolve returned with info[0]=" << handle.info[0];
        throw std::runtime_error( msg.str() );
    }
}

// MUMPS requires the RHS to reside entirely on the host...
void
psp::ZMumpsHostSolve
( psp::ZMumpsHandle& handle,
  int numRhs, DComplex* rhsBuffer, int rhsLDim, 
              DComplex* localSolutionBuffer, int localSolutionLDim,
  int* localIntegerBuffer ) 
{
    // Signal that we would like to solve
    handle.job = 3;    

    // Set up the RHS
    handle.nrhs = numRhs;
    handle.rhs = rhsBuffer;
    handle.lrhs = rhsLDim;

    // Set up the solution buffers
    handle.sol_loc = localSolutionBuffer;
    handle.lsol_loc = localSolutionLDim;
    handle.isol_loc = localIntegerBuffer;

    // Fill the control variables
    handle.icntl[8] = 1; // solve Ax=b, not A^Tx=b
    handle.icntl[9] = 0; // no iterative refinement for distributed solutions
    handle.icntl[10] = 0; // do not return statistics b/c they are expensive
    handle.icntl[13] = 20; // percent increase in estimated workspace
    handle.icntl[19] = 0; // we will input our RHS's as dense vectors
    handle.icntl[20] = 1; // the RHS's will be kept distributed
    handle.icntl[24] = 0; // we do not need to handle null-pivots yet
    handle.icntl[25] = 0; // do not worry about the interface Schur complement
    handle.icntl[26] = 128; // IMPORTANT: handles the number of RHS's that can
                            // be simultaneously solved
    handle.cntl[1] = 0; // iterative refinement is irrelavant with multiple RHS

    // Solve the system
    zmumps_c( &handle );

    // Summary of returned info:
    //
    // handle.info[0]: 0 if call was successful
    // handle.info[1]: holds additional info if error or warning
    // handle.info[25]: MB of work space used for solution on this process
    //                  (maximum and sum returned by infog[29] and infog[30])
    // handle.rinfog[3-10]: only set if error analysis is on
    // handle.infog[0]: 0 if successful, negative if error, positive if warning
    // handle.infog[1]: additional info about error or warning
    // handle.infog[14]: number of steps of iterative refinement (zero for us)
    // handle.infog[29]: max MB of effective memory for solution
    // handle.infog[30]: total MB of effective memory for solution
    
    if( handle.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::ZMumpsHostSolve returned with info[0]=" << handle.info[0];
        throw std::runtime_error( msg.str() );
    }
}

void
psp::ZMumpsFinalize( psp::ZMumpsHandle& handle )
{
    // Signal that we are destroying a MUMPS instance
    handle.job = -2;

    // Destroy the instance
    zmumps_c( &handle );

    // Summary of returned info:
    //
    // handle.info[0]: 0 if call was successful
    // handle.info[1]: holds additional info if error or warning
    // handle.infog[0]: 0 if successful, negative if error, positive if warning
    // handle.infog[1]: additional info about error or warning
    
    if( handle.info[0] != 0 )
    {
        std::ostringstream msg;
        msg << "psp::ZMumpsFinalize returned with info[0]=" << handle.info[0];
        throw std::runtime_error( msg.str() );
    }
}

