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

PetscErrorCode
CustomPCSetUp( PC pc )
{
    PetscErrorCode ierr;

    psp::FiniteDiffSweepingPC* context;
    ierr = PCShellGetContext( pc, (void**)&context ); CHKERRQ(ierr);

    context->Init();

    return ierr;
}

PetscErrorCode 
CustomPCApply(PC pc,Vec x,Vec y)
{
    PetscErrorCode ierr;

    psp::FiniteDiffSweepingPC* context;
    ierr = PCShellGetContext( pc, (void**)&context ); CHKERRQ(ierr);

    context->Apply( x, y );

    return ierr;
}

PetscErrorCode
CustomPCDestroy(PC pc)
{
    PetscErrorCode ierr;

    psp::FiniteDiffSweepingPC* context;
    ierr = PCShellGetContext( pc, (void**)&context ); CHKERRQ(ierr);

    context->Destroy();

    return ierr;
}

int
main( int argc, char* argv[] )
{
    PetscInitialize( &argc, &argv, PETSC_NULL, PETSC_NULL );

    PetscErrorCode ierr;
    PetscMPIInt size, rank;
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);

    if( argc < 4 )
    {
        if( rank == 0 )    
	    Usage();
	PetscFinalize();
	return 0;
    }

    PetscInt argNum = 0;
    const PetscInt nx = atoi(argv[++argNum]);
    const PetscInt ny = atoi(argv[++argNum]);
    const PetscInt nz = atoi(argv[++argNum]);
    const PetscInt n = nx*ny*nz;

    psp::FiniteDiffControl control;
    control.stencil = psp::SEVEN_POINT;
    control.nx = nx;
    control.ny = ny;
    control.nz = nz;
    control.wx = 15;
    control.wy = 15;
    control.wz = 15;
    control.omega = 10;
    control.Cx = 1;
    control.Cy = 1;
    control.Cz = 1;
    control.etax = 1;
    control.etay = 1;
    control.etaz = 1;
    control.imagShift = 1;
    control.planesPerPanel = 5;
    control.frontBC = psp::PML;
    control.rightBC = psp::PML;
    control.backBC = psp::PML;
    control.leftBC = psp::PML;
    control.bottomBC = psp::PML;

    psp::FiniteDiffSweepingPC context
    ( PETSC_COMM_WORLD, numProcessRows, numProcessCols, control, solver );

    Vec slowness;
    const PetscInt localSize = context.GetLocalSize();
    ierr = VecCreate(PETSC_COMM_WORLD,&slowness); CHKERRW(ierr);
    ierr = PetscObjectSetName((PetscObject)slowness,"slowness"); CHKERRQ(ierr);
    ierr = VecSetSizes(slowness,localSize,n); CHKERRQ(ierr);
    // TODO: Fill slowness vector here. We should probably start with all 1's.
    VecSet(slowness,1.0);

    // Set up the approximate inverse and the original matrix
    Mat A;
    context.Init( slowness, A );

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
    PCSetType(pc,PCSHELL);
    PCSetContext(pc,&context);
    PCShellSetSetUp(pc,CustomPCSetUp);
    PCShellSetApply(pc,CustomPCApply);
    PCShellSetDestroy(pc,CustomPCDestroy);

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

