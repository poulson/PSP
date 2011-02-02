#include <cstdlib>
#include <iostream>
#include "mpi.h"
using namespace std;

extern "C"
{
    void
    Cblacs_pinfo( int* rank, int* p );

    void
    Cblacs_get( int context, int request, int* value );

    void
    Cblacs_gridinit( int* context, char* order, int r, int c );

    void
    Cblacs_gridinfo( int context, int* r, int* c, int* myRow, int* myCol );

    void
    Cblacs_gridexit( int context );

    void
    descinit_
    ( int* desc, int* m, int* n, int* mb, int* nb,
      int* colAlign, int* rowAlign, int* context, int* localLDim, int* err );

    void
    pdpotrf_
    ( char* uplo, int* m, double* A, int* iA, int* jA, int* ADesc, int* info );
}

void Usage()
{
    cout << "ScaLAPACK Cholesky Factorization\n\n"
         << "  ScaLAPACK-Cholesky <r> <c> <shape> <m> <nb>\n\n"
         << "  <r>:      number of process rows\n"
         << "  <c>:      number of process cols\n"
         << "  <shape>:  {L,U}\n"
         << "  <m>: problem size to test\n"
         << "  <nb>:     blocksize" << endl;
}

int
main( int argc, char* argv[] )
{
    int rankMPI;
    int one = 1; 
    int zero = 0;

    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rankMPI );

    if( argc < 6 )
    {
        if( rankMPI == 0 )
	    Usage();
        MPI_Finalize();      
        return 0;
    }

    int r = atoi(argv[1]);
    int c = atoi(argv[2]);
    char shape = *argv[3];
    int m = atoi(argv[4]);
    int nb = atoi(argv[5]);

    if( rankMPI == 0 )
    {
        cout << "r = " << r << endl;
        cout << "c = " << c << endl;
        cout << "shape = " << shape << endl;
        cout << "m = " << m << endl;
        cout << "nb = " << nb << endl;
    }

    int p, rank, myRow, myCol;
    int context;
    char order = 'C';

    Cblacs_pinfo( &rank, &p );
    Cblacs_get( 0, 0, &context );
    Cblacs_gridinit( &context, &order, r, c );
    Cblacs_gridinfo( context, &r, &c, &myRow, &myCol );

    int ALocLDim = m/r+nb;
    int err;
    int ADesc[9];
    descinit_
    ( ADesc, &m, &m, &nb, &nb, &zero, &zero, &context, &ALocLDim, &err );

    double* ALoc = new double[ALocLDim*(m/c+nb)];

    // Make A diagonally dominant
    for( int jLoc=0; jLoc<m/c+nb; ++jLoc )
    {
        const int jBlock = jLoc/nb;
        const int jStart = (myCol-1+jBlock*c)*nb;
        const int jOffset = jLoc%nb;

        for( int iLoc=0; iLoc<ALocLDim; ++iLoc )
        {
            const int iBlock = iLoc/nb;
            const int iStart = (myRow-1+iBlock*r)*nb;
            const int iOffset = iLoc%nb;

            if( iStart+iOffset == jStart+jOffset )
                ALoc[iLoc+jLoc*ALocLDim] = m+drand48();
            else
                ALoc[iLoc+jLoc*ALocLDim] = (2*drand48())-1.;
        }
    }

    if( rank == 0 )
    {
        cout << "Calling pdpotrf...";
        cout.flush();
    }
    MPI_Barrier( MPI_COMM_WORLD );

    int info;
    double startTime = MPI_Wtime();
    pdpotrf_
    ( &shape, &m, ALoc, &one, &one, ADesc, &info );
    MPI_Barrier( MPI_COMM_WORLD );
    double stopTime = MPI_Wtime();
    double runTime = stopTime-startTime;
    double gFlops = (1./3.*m*m*m)/(1.e9*runTime);
    if( rank == 0 )
        cout << "DONE. GFlops = " << gFlops << endl;

    delete[] ALoc;

    Cblacs_gridexit( context );
    MPI_Finalize();

    return 0;
}

