/* Example program using the C interface to the 
 * double complex arithmetic version of MUMPS, zmumps_c.
 * We solve the system A x = RHS with
 *   A = diag(1 2) and RHS = [1 4]^T
 * Solution is [1 2]^T */
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include "mpi.h"
#include "zmumps_c.h"
#define JOB_INIT -1
#define JOB_END -2
#define USE_COMM_WORLD -987654

int main(int argc, char ** argv)
{
  ZMUMPS_STRUC_C id;
  int n = 2;
  int nz = 2;
  int irn[] = {1,2};
  int jcn[] = {1,2};
  mumps_double_complex a[2]; 
  mumps_double_complex rhs[2];
  /*
  double a[4];
  double rhs[4];
  */

  int myid, ierr;
  ierr = MPI_Init(&argc, &argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  /* Define A and rhs */
  rhs[0].r = 1.0; rhs[0].i = 0.0;
  rhs[1].r = 4.0; rhs[1].i = 0.0;
  /*
  rhs[0] = 1.0; rhs[1] = 0.0;
  rhs[2] = 4.0; rhs[3] = 0.0;
  */

  a[0].r = 1.0; a[0].i = 0.0;
  a[1].r = 2.0; a[1].i = 0.0;
  /*
  a[0] = 1.0; a[1] = 0.0;
  a[2] = 2.0; a[3] = 0.0;
  */

  /* Initialize a MUMPS instance. Use MPI_COMM_WORLD */
  id.job=JOB_INIT; id.par=1; id.sym=0;id.comm_fortran=USE_COMM_WORLD;
  zmumps_c(&id);
  /* Define the problem on the host */
  if (myid == 0) {
    id.n = n; id.nz =nz; id.irn=irn; id.jcn=jcn;
    id.a = a; id.rhs = rhs;
  }
#define ICNTL(I) icntl[(I)-1] /* macro s.t. indices match documentation */
/* No outputs */
  id.ICNTL(1)=-1; id.ICNTL(2)=-1; id.ICNTL(3)=-1; id.ICNTL(4)=0;
/* Call the MUMPS package. */
  id.job=6;
  zmumps_c(&id);
  id.job=JOB_END; zmumps_c(&id); /* Terminate instance */
  if (myid == 0) {
    printf("Solution is : (%8.2f  %8.2f)\n", rhs[0].r,rhs[1].r);
    /*printf("Solution is: (%8.2f %8.2f)\n", rhs[0], rhs[2]);*/
  }
  ierr = MPI_Finalize();
  return 0;
}
