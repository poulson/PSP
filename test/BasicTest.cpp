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
    std::cout << "BasicTest <nx> <ny> <nz> <wx> <wy> <wz> <omega> "
              << "\n          <etax> <etay> <etaz> <planesPerPanel>"
              << "\n          <numProcRows> <numProcCols> "
              << "\n          <storeSlowness?> <storeRhs?> <storeSolution?>\n"
              << "  nx: number of vertices in x direction\n"
              << "  ny: number of vertices in y direction\n"
              << "  nz: number of vertices in z direction\n" 
              << "  wx: width of box in x direction\n"
              << "  wy: width of box in y direction\n"
              << "  wz: width of box in z direction\n"
              << "  omega: frequency of the problem\n"
              << "  etax: width of PML in x direction\n"
              << "  etay: width of PML in y direction\n"
              << "  etaz: width of PML in z direction\n"
              << "  planesPerPanel: number of xy planes to process per panel\n"
              << "  numProcessRows\n"
              << "  numProcessCols\n" 
              << "  storeSlowness?: store slowness vec in parallel VTK file?\n"
              << "  storeRhs?: store RHS vec in parallel VTK file?\n"
              << "  storeSolution?: store solution vec in parallel VTK file?\n"
              << std::endl;
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

    if( argc < 17 )
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
    const PetscReal omega = atof(argv[++argNum]);
    const PetscReal etax = atof(argv[++argNum]);
    const PetscReal etay = atof(argv[++argNum]);
    const PetscReal etaz = atof(argv[++argNum]);
    const PetscInt planesPerPanel = atoi(argv[++argNum]);
    const PetscInt numProcessRows = atoi(argv[++argNum]);
    const PetscInt numProcessCols = atoi(argv[++argNum]);
    const bool storeSlowness = atoi(argv[++argNum]);
    const bool storeRhs = atoi(argv[++argNum]);
    const bool storeSolution = atoi(argv[++argNum]);

    psp::FiniteDiffControl control;
    control.stencil = psp::SEVEN_POINT;
    control.nx = nx;
    control.ny = ny;
    control.nz = nz;
    control.wx = wx;
    control.wy = wy;
    control.wz = wz;
    control.omega = omega;
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
    
    // Fill the slowness with a Gaussian
    {
        PetscScalar* slownessLocalData;
        VecGetArray( slowness, &slownessLocalData );

        const PetscInt myXOffset  = context.GetMyXOffset();
        const PetscInt myYOffset  = context.GetMyYOffset();
        const PetscInt myXPortion = context.GetMyXPortion();
        const PetscInt myYPortion = context.GetMyYPortion();
        const PetscReal hx = context.GetXSpacing();
        const PetscReal hy = context.GetYSpacing();
        const PetscReal hz = context.GetZSpacing();
        for( PetscInt z=0; z<control.nz; ++z )
        {
            const PetscReal Z = z*hz - 0.5;
            for( PetscInt yLocal=0; yLocal<myYPortion; ++yLocal )
            {
                const PetscInt y = myYOffset + yLocal;
                const PetscReal Y = y*hy - 0.5;
                for( PetscInt xLocal=0; xLocal<myXPortion; ++xLocal )
                {
                    const PetscInt x = myXOffset + xLocal;
                    const PetscReal X = x*hx - 0.5;
                    PetscScalar gamma = 1 - 0.5*std::exp(-32*(X*X+Y*Y+Z*Z));
                    slownessLocalData[xLocal+myXPortion*yLocal+
                                      myXPortion*myYPortion*z] = gamma;
                }
            }
        }

        VecRestoreArray( slowness, &slownessLocalData );
    }
    if( storeSlowness )
    {
        if( rank == 0 )
        {
            std::cout << "Writing slowness vector to file...";
            std::cout.flush();
        }
        context.WriteParallelVtkFile( slowness, "slowness" );
        if( rank == 0 )
            std::cout << "done." << std::endl;
    }

    // Set up the approximate inverse and the original matrix
    Mat A;
    MPI_Barrier( PETSC_COMM_WORLD );
    const PetscReal initStartTime = MPI_Wtime();
    if( rank == 0 )
    {
        std::cout << "Initializing preconditioner...";
        std::cout.flush();
    }
    context.Init( slowness, A );
    MPI_Barrier( PETSC_COMM_WORLD );
    if( rank == 0 )
    {
        const PetscReal initStopTime = MPI_Wtime();
        std::cout << "done, time=" << initStopTime-initStartTime << std::endl;
    }
    const PetscReal oneNorm = MatNorm( A, NORM_1 );
    const PetscReal infNorm = MatNorm( A, NORM_INFINITY );
    const PetscReal frobNorm = MatNorm( A, NORM_FROBENIUS );
    if( rank == 0 )
    {
        std::cout << "||A||_1  = " << oneNorm << "\n"
                  << "||A||_oo = " << infNorm << "\n"
                  << "||A||_F  = " << frobNorm << std::endl;
    }

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

    // Build an RHS and solve
    Vec x, b;
    MatGetVecs( A, &x, PETSC_NULL );
    VecDuplicate( x, &b );
    PetscObjectSetName( (PetscObject)x, "approx solution" );
    PetscObjectSetName( (PetscObject)b, "RHS" );
    {
        PetscScalar* bLocalData;
        VecGetArray( b, &bLocalData );

        const PetscInt myXOffset  = context.GetMyXOffset();
        const PetscInt myYOffset  = context.GetMyYOffset();
        const PetscInt myXPortion = context.GetMyXPortion();
        const PetscInt myYPortion = context.GetMyYPortion();
        const PetscReal hx = context.GetXSpacing();
        const PetscReal hy = context.GetYSpacing();
        const PetscReal hz = context.GetZSpacing();
        for( PetscInt z=0; z<control.nz; ++z )
        {
            const PetscReal Z = z*hz - 0.5;
            for( PetscInt yLocal=0; yLocal<myYPortion; ++yLocal )
            {
                const PetscInt y = myYOffset + yLocal;
                const PetscReal Y = y*hy - 0.5;
                for( PetscInt xLocal=0; xLocal<myXPortion; ++xLocal )
                {
                    const PetscInt x = myXOffset + xLocal;
                    const PetscReal X = x*hx - 0.5;
                    PetscScalar gamma = 
                      std::exp(-control.nx*control.nx*(X*X+Y*Y+Z*Z));
                    bLocalData[xLocal+myXPortion*yLocal+
                               myXPortion*myYPortion*z] = gamma;
                }
            }
        }

        VecRestoreArray( b, &bLocalData );
    }
    if( storeRhs )
    {
        if( rank == 0 )
        {
            std::cout << "Writing RHS vector to file...";
            std::cout.flush();
        }
        context.WriteParallelVtkFile( b, "rhs" );
        if( rank == 0 )
            std::cout << "done." << std::endl;
    }

    // Solve the system with a preconditioner
    MPI_Barrier( PETSC_COMM_WORLD );
    const PetscReal solveStartTime = MPI_Wtime();
    if( rank == 0 )
    {
        std::cout << "Solving with a preconditioner...";
        std::cout.flush();
    }
    KSPSolve( ksp, b, x );
    PetscInt its;
    KSPGetIterationNumber( ksp, &its );
    MPI_Barrier( PETSC_COMM_WORLD );
    if( rank == 0 )
    {
        const PetscReal solveStopTime = MPI_Wtime();
        std::cout << "needed " << its << " its and " 
                  << solveStopTime-solveStartTime << " secs." << std::endl;
    }
    if( storeSolution )
    {
        if( rank == 0 )
        {
            std::cout << "Writing solution vector to file...";
            std::cout.flush();
        }
        context.WriteParallelVtkFile( x, "solution" );
        if( rank == 0 )
            std::cout << "done." << std::endl;
    }

    // Manufacture an RHS from a chosen solution
    Vec u;
    VecDuplicate( x, &u );
    PetscObjectSetName( (PetscObject)u, "solution" );
    VecSet( u, 1.0 );
    MatMult( A, u, b );

    // Solve the system with a preconditioner
    if( rank == 0 )
    {
        std::cout << "Solving using manufactured RHS with a preconditioner...";
        std::cout.flush();
    }
    KSPSolve( ksp, b, x );
    if( rank == 0 )
        std::cout << "done." << std::endl;

    // Check the solution
    VecAXPY( x, -1.0, u );
    PetscReal norm;
    VecNorm( x, NORM_2, &norm );
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
    MPI_Barrier( PETSC_COMM_WORLD );
    const PetscReal noPrecSolveStartTime = MPI_Wtime();
    if( rank == 0 )
    {
        std::cout << "Solving WITHOUT a preconditioner...";
        std::cout.flush();
    }
    KSPSolve( kspWithout, b, x );
    MPI_Barrier( PETSC_COMM_WORLD );
    if( rank == 0 )
    {
        const PetscReal noPrecSolveStopTime = MPI_Wtime();
        std::cout << "done, time=" << noPrecSolveStopTime-noPrecSolveStartTime 
                  << std::endl;
    }

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

