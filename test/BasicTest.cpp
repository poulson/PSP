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
    std::cout << "PetscTest <nx> <ny> <nz> <wx> <wy> <wz> <etax> <etay> <etaz> "
              << "\n          <planesPerPanel> <numProcRows> <numProcCols>\n"
              << "  nx: number of vertices in x direction\n"
              << "  ny: number of vertices in y direction\n"
              << "  nz: number of vertices in z direction\n" 
              << "  wx: width of box in x direction\n"
              << "  wy: width of box in y direction\n"
              << "  wz: width of box in z direction\n"
              << "  etax: width of PML in x direction\n"
              << "  etay: width of PML in y direction\n"
              << "  etaz: width of PML in z direction\n"
              << "  planesPerPanel: number of xy planes to process per panel\n"
              << "  numProcessRows\n"
              << "  numProcessCols\n" << std::endl;
}

PetscErrorCode 
CustomPCApply( PC pc, Vec x, Vec y )
{
    PetscErrorCode ierr;

    psp::FiniteDiffSweepingPC* context;
    ierr = PCShellGetContext( pc, (void**)&context ); CHKERRQ(ierr);

    context->Apply( x, y );

    return ierr;
}

int
main( int argc, char* argv[] )
{
    PetscInitialize( &argc, &argv, PETSC_NULL, PETSC_NULL );

    PetscMPIInt size, rank;
    MPI_Comm_size( PETSC_COMM_WORLD, &size );
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank ); 

    if( argc < 13 )
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
    const PetscReal wx = atof(argv[++argNum]);
    const PetscReal wy = atof(argv[++argNum]);
    const PetscReal wz = atof(argv[++argNum]);
    const PetscReal etax = atof(argv[++argNum]);
    const PetscReal etay = atof(argv[++argNum]);
    const PetscReal etaz = atof(argv[++argNum]);
    const PetscInt planesPerPanel = atoi(argv[++argNum]);
    const PetscInt numProcessRows = atoi(argv[++argNum]);
    const PetscInt numProcessCols = atoi(argv[++argNum]);

    psp::FiniteDiffControl control;
    control.stencil = psp::SEVEN_POINT;
    control.nx = nx;
    control.ny = ny;
    control.nz = nz;
    control.wx = wx;
    control.wy = wy;
    control.wz = wz;
    control.omega = 4*M_PI;
    control.Cx = 1.5*(2*M_PI);
    control.Cy = 1.5*(2*M_PI);
    control.Cz = 1.5*(2*M_PI);
    control.etax = etax;
    control.etay = etay;
    control.etaz = etaz;
    control.imagShift = 1;
    control.planesPerPanel = planesPerPanel;
    control.frontBC = psp::PML;
    control.rightBC = psp::PML;
    control.backBC = psp::PML;
    control.leftBC = psp::PML;
    control.bottomBC = psp::PML;

    //psp::SparseDirectSolver solver = psp::MUMPS_SYMMETRIC;
    psp::SparseDirectSolver solver = psp::MUMPS;

    psp::FiniteDiffSweepingPC context
    ( PETSC_COMM_WORLD, numProcessRows, numProcessCols, control, solver );

    const PetscInt localSize = context.GetLocalSize();
    Vec slowness;
    VecCreate( PETSC_COMM_WORLD, &slowness );
    VecSetSizes( slowness, localSize, n ); 
    VecSetType( slowness, VECMPI ); 
    PetscObjectSetName( (PetscObject)slowness, "slowness" );

    // TODO: Fill slowness vector here. We should probably start with all 1's.
    VecSet(slowness,1.0);

    // Set up the approximate inverse and the original matrix
    Mat A;
    if( rank == 0 )
    {
        std::cout << "Initializing preconditioner...";
        std::cout.flush();
    }
    context.Init( slowness, A );
    if( rank == 0 )
        std::cout << "done." << std::endl;
    const PetscReal oneNorm = MatNorm( A, NORM_1 );
    const PetscReal infNorm = MatNorm( A, NORM_INFINITY );
    const PetscReal frobNorm = MatNorm( A, NORM_FROBENIUS );
    if( rank == 0 )
    {
        std::cout << "||A||_1  = " << oneNorm << "\n"
                  << "||A||_oo = " << infNorm << "\n"
                  << "||A||_F  = " << frobNorm << std::endl;
    }

    // Create the approx. solution (x), exact solution (u), and RHS (b) 
    // vectors
    Vec x, u, b;
    MatGetVecs( A, &x, PETSC_NULL );
    VecDuplicate( x, &u );
    VecDuplicate( x, &b );
    PetscObjectSetName( (PetscObject)x, "approx solution" );
    PetscObjectSetName( (PetscObject)u, "solution" );
    PetscObjectSetName( (PetscObject)b, "RHS" );
    VecSet( u, 1.0 );
    MatMult( A, u, b );

    // Set up the Krylov solver with a preconditioner
    KSP ksp;
    KSPCreate( PETSC_COMM_WORLD, &ksp );
    KSPSetOperators( ksp, A, A, DIFFERENT_NONZERO_PATTERN );
    KSPSetType( ksp, KSPGMRES );
    PC pc;
    KSPGetPC( ksp, &pc );
    KSPSetTolerances( ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT );
    PCSetType( pc, PCSHELL );
    PCShellSetApply( pc, CustomPCApply );
    PCShellSetContext( pc, &context );

    // Solve the system with a preconditioner
    if( rank == 0 )
    {
        std::cout << "Solving with a preconditioner...";
        std::cout.flush();
    }
    KSPSolve( ksp, b, x );
    if( rank == 0 )
        std::cout << "done." << std::endl;

    // Check the solution
    VecAXPY( x, -1.0, u );
    PetscReal norm;
    VecNorm( x, NORM_2, &norm );
    PetscInt its;
    KSPGetIterationNumber( ksp, &its );
    if( rank == 0 )
    {
        std::cout << "Using sweeping preconditioner:\n"
                  << "Norm of error = " << norm 
                  << ", # of iterations = " << its << "\n" << std::endl;
    }

    // Reset u and b
    VecSet( u, 1.0 );
    MatMult( A, u, b );

    // Set up the Krylov solver WITHOUT a preconditioner
    KSP kspWithout;
    KSPCreate( PETSC_COMM_WORLD, &kspWithout );
    KSPSetOperators( kspWithout, A, A, DIFFERENT_NONZERO_PATTERN );
    KSPSetType( kspWithout, KSPGMRES );
    KSPSetTolerances
    ( kspWithout, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT );
    PC pcWithout;
    KSPGetPC( kspWithout, &pcWithout );
    PCSetType( pcWithout, PCNONE );

    // Solve the system WITHOUT a preconditioner
    if( rank == 0 )
    {
        std::cout << "Solving WITHOUT a preconditioner...";
        std::cout.flush();
    }
    KSPSolve( kspWithout, b, x );
    if( rank == 0 )
        std::cout << "done." << std::endl;

    // Check the solution
    VecAXPY( x, -1.0, u );
    VecNorm( x, NORM_2, &norm );
    KSPGetIterationNumber( kspWithout, &its );
    if( rank == 0 )
    {
        std::cout << "With no preconditioner:\n"
                  << "Norm of error = " << norm 
                  << ", # of iterations = " << its << "\n" << std::endl;
    }

    // Free work space
    context.Destroy();
    KSPDestroy( ksp );
    KSPDestroy( kspWithout );
    VecDestroy( u ); 
    VecDestroy( x );
    VecDestroy( b ); 
    MatDestroy( A );

    PetscFinalize();
    return 0;
}

