/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef LAPACK_POLAR_HPP
#define LAPACK_POLAR_HPP

#include "elemental/lapack-like/HermitianFunction.hpp"
#include "elemental/lapack-like/SVD.hpp"

namespace elem {

//
// Compute the polar decomposition of A, A = Q P, where Q is unitary and P is 
// Hermitian positive semi-definite. On exit, A is overwritten with Q.
//

template<typename F>
inline void
Polar( Matrix<F>& A, Matrix<F>& P )
{
#ifndef RELEASE
    PushCallStack("Polar");
#endif
    typedef typename Base<F>::type R;
    const int n = A.Width();

    // Get the SVD of A
    Matrix<R> s;
    Matrix<F> U, V;
    U = A;
    SVD( U, s, V );

    // Form Q := U V^H in A
    MakeZeros( A );
    Gemm( NORMAL, ADJOINT, F(1), U, V, F(0), A );

    // Form P := V Sigma V^H in P
    Zeros( n, n, P );
    hermitian_function::ReformHermitianMatrix( LOWER, P, s, V );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void
Polar( DistMatrix<F>& A, DistMatrix<F>& P )
{
#ifndef RELEASE
    PushCallStack("Polar");
#endif
    typedef typename Base<F>::type R;
    const Grid& g = A.Grid();
    const int n = A.Width();

    // Get the SVD of A
    DistMatrix<R,VR,STAR> s(g);
    DistMatrix<F> U(g), V(g);
    U = A;
    SVD( U, s, V );

    // Form Q := U V^H in A
    MakeZeros( A );
    Gemm( NORMAL, ADJOINT, F(1), U, V, F(0), A );

    // Form P := V Sigma V^H in P
    Zeros( n, n, P );
    hermitian_function::ReformHermitianMatrix( LOWER, P, s, V );
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace elem

#endif // ifndef LAPACK_POLAR_HPP
