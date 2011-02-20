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
#include <iostream>

// Include the PETSc Krylov SubsPace (KSP) solvers
#include "petscksp.h"

void Usage()
{
    std::cout << "PetscTest <nx> <ny> <nz>\n\n"
              << "  nx: number of vertices in x direction\n"
              << "  ny: number of vertices in y direction\n"
              << "  nz: number of vertices in z direction" << std::endl;
}

static char help[] = "A simple PETSc test.\n\n";

extern "C" {
struct FiniteDiffSweepingPC
{
    psp::FiniteDiffControl control;
    // TODO: parallel info? MUMPS handle?
};
}

PetscErrorCode
FiniteDiffSweepingPCSetUp(PC pc)
{
    PetscErrorCode ierr;

    // TODO: Fill in MUMPS factorization steps

    return ierr;
}

PetscErrorCode 
FiniteDiffSweepingPCApply(PC pc,Vec x,Vec y)
{
    PetscErrorCode ierr;

    FiniteDiffSweepingPC *context;
    ierr = PCShellGetContext(pc,(void**)&context); CHKERRQ(ierr);

    // TODO: Apply sweeping preconditioner. This will require a modified
    //       version of the Vec scatter to zero routine.

    return ierr;
}

PetscErrorCode
FiniteDiffSweepingPCDestroy(PC pc)
{
    PetscErrorCode ierr;

    FiniteDiffSweepingPC* context;
    ierr = PCShellGetContext(pc,(void**)&context);

    // TODO: Call MUMPS destruction routines

    return ierr;
}

int
main( int argc, char* argv[] )
{
    PetscInitialize( &argc, &argv, (char*)0, help );

    PetscErrorCode ierr;
    PetscMPIInt size, rank;
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);

    if( argc != 4 )
    {
        if( rank == 0 )    
	    Usage();
	PetscFinalize();
	return 0;
    }

    PetscInt argNum = 0;
    PetscInt nx = atoi(argv[++argNum]);
    PetscInt ny = atoi(argv[++argNum]);
    PetscInt nz = atoi(argv[++argNum]);
    PetscInt n = nx*ny*nz;

    // With 7 point stencil and symmetry, we only have 4 nonzeros per row. We 
    // can assume ~2 will be in the diagonal block and ~2 will be outside it.
    // Pad each by 1 to avoid underallocation. PETSc requires us to use a 
    // blocked symmetric matrix, so we set the blocksize to 1.
    Mat A;
    ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
    ierr = MatSetType(A,MATMPISBAIJ); CHKERRQ(ierr);
    ierr = MatMPISBAIJSetPreallocation(A,1,3,PETSC_NULL,3,PETSC_NULL); 
    CHKERRQ(ierr);
    PetscInt iStart, iEnd;
    ierr = MatGetOwnershipRange(A,&iStart,&iEnd); CHKERRQ(ierr);

    // TODO: Fill 7-point stencil for our index range here

    // Create the approx. solution (x), exact solution (u), and RHS (b) 
    // vectors
    Vec x, u, b;
    ierr = VecCreate(PETSC_COMM_WORLD,&b); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)b,"RHS"); CHKERRQ(ierr);
    ierr = VecSetSizes(b,PETSC_DECIDE,n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(b); CHKERRQ(ierr);
    ierr = VecDuplicate(b,&x); CHKERRQ(ierr);
    ierr = VecDuplicate(b,&u); CHKERRQ(ierr);
    VecSet(u,1.0);
    MatMult(A,u,b);

    // Set up the Krylov solver
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);
    PC pc;
    KSPGetPC(ksp,&pc);
    KSPSetTolerances(ksp,1.e-6,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
    FiniteDiffSweepingPC context;
    // TODO: Fill and set context here
    PCSetType(pc,PCSHELL);
    PCShellSetSetUp(pc,FiniteDiffSweepingPCSetUp);
    PCShellSetApply(pc,FiniteDiffSweepingPCApply);
    PCShellSetDestroy(pc,FiniteDiffSweepingPCDestroy);

    // Optionally override our KSP options from the commandline. We should
    // specify GMRES vs. TFQMR from the commandline rather than hardcoding it.
    KSPSetFromOptions(ksp);

    // Solve the system
    KSPSolve(ksp,b,x);

    // Check the solution
    VecAXPY(x,-1.0,u);
    PetscReal norm;
    VecNorm(x,NORM_2,&norm);
    PetscInt its;
    KSPGetIterationNumber(ksp,&its);
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A iterations %D\n",norm,its);

    // Free work space
    KSPDestroy(ksp);
    VecDestroy(u); VecDestroy(x);
    VecDestroy(b); MatDestroy(A);

    PetscFinalize();
    return 0;
}
