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

template<typename Scalar>
void psp::hmatrix_tools::Scale
( Scalar alpha, Vector<Scalar>& x )
{
    blas::Scal( x.Size(), alpha, x.Buffer(), 1 );
}

template<typename Scalar>
void psp::hmatrix_tools::Scale
( Scalar alpha, DenseMatrix<Scalar>& D )
{
    const int m = D.Height();
    const int n = D.Width();
    if( D.Symmetric() )
    {
        for( int j=0; j<n; ++j )
            blas::Scal( m-j, alpha, D.Buffer(j,j), 1 );
    }
    else
    {
        for( int j=0; j<n; ++j ) 
            blas::Scal( m, alpha, D.Buffer(0,j), 1 );
    }
}

template<typename Scalar,bool Conjugate>
void psp::hmatrix_tools::Scale
( Scalar alpha, FactorMatrix<Scalar,Conjugate>& F )
{
    blas::Scal( F.m*F.r, alpha, &F.U[0], 1 );
}

template void psp::hmatrix_tools::Scale
( float alpha, Vector<float>& x );
template void psp::hmatrix_tools::Scale
( double alpha, Vector<double>& x );
template void psp::hmatrix_tools::Scale
( std::complex<float> alpha, Vector< std::complex<float> >& x );
template void psp::hmatrix_tools::Scale
( std::complex<double> alpha, Vector< std::complex<double> >& x );

template void psp::hmatrix_tools::Scale
( float alpha, DenseMatrix<float>& D );
template void psp::hmatrix_tools::Scale
( double alpha, DenseMatrix<double>& D );
template void psp::hmatrix_tools::Scale
( std::complex<float> alpha, DenseMatrix< std::complex<float> >& D );
template void psp::hmatrix_tools::Scale
( std::complex<double> alpha, DenseMatrix< std::complex<double> >& D );

template void psp::hmatrix_tools::Scale
( float alpha, FactorMatrix<float,false>& F );
template void psp::hmatrix_tools::Scale
( float alpha, FactorMatrix<float,true>& F );
template void psp::hmatrix_tools::Scale
( double alpha, FactorMatrix<double,false>& F );
template void psp::hmatrix_tools::Scale
( double alpha, FactorMatrix<double,true>& F );
template void psp::hmatrix_tools::Scale
( std::complex<float> alpha, FactorMatrix<std::complex<float>,false>& F );
template void psp::hmatrix_tools::Scale
( std::complex<float> alpha, FactorMatrix<std::complex<float>,true>& F );
template void psp::hmatrix_tools::Scale
( std::complex<double> alpha, FactorMatrix<std::complex<double>,false>& F );
template void psp::hmatrix_tools::Scale
( std::complex<double> alpha, FactorMatrix<std::complex<double>,true>& F );
