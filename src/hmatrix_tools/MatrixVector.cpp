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

// Dense y := alpha A x + beta y
template<typename Scalar>
void psp::hmatrix_tools::MatrixVector
( Scalar alpha, const DenseMatrix<Scalar>& A,
                const std::vector<Scalar>& x,
  Scalar beta,        std::vector<Scalar>& y )
{
    if( A.symmetric )
    {
        blas::Symv
        ( 'L', A.m, alpha, &A.buffer[0], A.m, &x[0], 1, beta, &y[0], 1 );
    }
    else
    {
        blas::Gemv
        ( 'N', A.m, A.n, 1, &A.buffer[0], A.m, &x[0], 1, beta, &y[0], 1 );
    }
}

// Dense y := alpha A x
template<typename Scalar>
void psp::hmatrix_tools::MatrixVector
( Scalar alpha, const DenseMatrix<Scalar>& A, 
                const std::vector<Scalar>& x,
                      std::vector<Scalar>& y )
{
    y.resize( x.size() );
    if( A.symmetric )
    {
        blas::Symv
        ( 'L', A.m, alpha, &A.buffer[0], A.m, &x[0], 1, 0, &y[0], 1 );
    }
    else
    {
        blas::Gemv
        ( 'N', A.m, A.n, alpha, &A.buffer[0], A.m, &x[0], 1, 0, &y[0], 1 );
    }
}

// Low-rank y := alpha A x + beta y
template<typename Scalar>
void psp::hmatrix_tools::MatrixVector
( Scalar alpha, const FactorMatrix<Scalar>& A, 
                const std::vector<Scalar>& x,
  Scalar beta,        std::vector<Scalar>& y )
{
    const int r = A.r;
    std::vector<Scalar> t(r);

    // Form t := alpha (A.V)^H x
    blas::Gemv( 'C', A.n, A.r, alpha, &A.V[0], A.n, &x[0], 1, 0, &t[0], 1 );

    // Form y := (A.U) t + beta y
    blas::Gemv( 'N', A.m, A.r, 1, &A.U[0], A.m, &t[0], 1, beta, &y[0], 1 );
}

// Low-rank y := alpha A x
template<typename Scalar>
void psp::hmatrix_tools::MatrixVector
( Scalar alpha, const FactorMatrix<Scalar>& A, 
                const std::vector<Scalar>& x,
                      std::vector<Scalar>& y )
{
    const int r = A.r;
    std::vector<Scalar> t(r);

    // Form t := alpha (A.V)^H x
    blas::Gemv( 'C', A.n, A.r, alpha, &A.V[0], A.n, &x[0], 1, 0, &t[0], 1 );

    // Form y := (A.U) t
    y.resize( x.size() );
    blas::Gemv( 'N', A.m, A.r, 1, &A.U[0], A.m, &t[0], 1, 0, &y[0], 1 );
}

template void psp::hmatrix_tools::MatrixVector
( float alpha, const DenseMatrix<float>& A,
               const std::vector<float>& x,
  float beta,        std::vector<float>& y );
template void psp::hmatrix_tools::MatrixVector
( double alpha, const DenseMatrix<double>& A,
                const std::vector<double>& x,
  double beta,        std::vector<double>& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const std::vector< std::complex<float> >& x,
  std::complex<float> beta,        std::vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const std::vector< std::complex<double> >& x,
  std::complex<double> beta,        std::vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixVector
( float alpha, const DenseMatrix<float>& A,
               const std::vector<float>& x,
                     std::vector<float>& y );
template void psp::hmatrix_tools::MatrixVector
( double alpha, const DenseMatrix<double>& A,
                const std::vector<double>& x,
                      std::vector<double>& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<float> alpha, const DenseMatrix< std::complex<float> >& A,
                             const std::vector< std::complex<float> >& x,
                                   std::vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<double> alpha, const DenseMatrix< std::complex<double> >& A,
                              const std::vector< std::complex<double> >& x,
                                    std::vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixVector
( float alpha, const FactorMatrix<float>& A,
               const std::vector<float>& x,
  float beta,        std::vector<float>& y );
template void psp::hmatrix_tools::MatrixVector
( double alpha, const FactorMatrix<double>& A,
                const std::vector<double>& x,
  double beta,        std::vector<double>& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<float> alpha, const FactorMatrix< std::complex<float> >& A,
                             const std::vector< std::complex<float> >& x,
  std::complex<float> beta,        std::vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<double> alpha, const FactorMatrix< std::complex<double> >& A,
                              const std::vector< std::complex<double> >& x,
  std::complex<double> beta,        std::vector< std::complex<double> >& y );

template void psp::hmatrix_tools::MatrixVector
( float alpha, const FactorMatrix<float>& A,
               const std::vector<float>& x,
                     std::vector<float>& y );
template void psp::hmatrix_tools::MatrixVector
( double alpha, const FactorMatrix<double>& A,
                const std::vector<double>& x,
                      std::vector<double>& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<float> alpha, const FactorMatrix< std::complex<float> >& A,
                             const std::vector< std::complex<float> >& x,
                                   std::vector< std::complex<float> >& y );
template void psp::hmatrix_tools::MatrixVector
( std::complex<double> alpha, const FactorMatrix< std::complex<double> >& A,
                              const std::vector< std::complex<double> >& x,
                                    std::vector< std::complex<double> >& y );

