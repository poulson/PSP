/*
   Copyright (c) 2009-2012, Jack Poulson
   All rights reserved.

   This file is part of Elemental.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    - Neither the name of the owner nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

namespace elem {

namespace internal {

template<typename F>
inline SafeProduct<F> 
SafeDeterminantWithOverwrite( Matrix<F>& A )
{
#ifndef RELEASE
    PushCallStack("internal::SafeDeterminantWithOverwrite");
#endif
    if( A.Height() != A.Width() )
        throw std::logic_error
        ("Cannot compute determinant of nonsquare matrix");

    typedef typename Base<F>::type R;
    const int n = A.Height();
    const R scale(n);
    SafeProduct<F> det( n );

    try
    {
        Matrix<int> p;
        LU( A, p ); 
        const bool isOdd = PivotParity( p );
        
        Matrix<F> d;
        A.GetDiagonal( d );
        for( int i=0; i<n; ++i )
        {
            const F delta = d.Get(i,0);
            R alpha = Abs(delta);
            det.rho *= delta/alpha;
            det.kappa += Log(alpha)/scale;
        }
        if( isOdd )
            det.rho = -det.rho;
    }
    catch( SingularMatrixException& e )
    {
        det.rho = 0;
        det.kappa = 0;
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return det;
}

template<typename F> 
inline SafeProduct<F> 
SafeDeterminantWithOverwrite( DistMatrix<F>& A )
{
#ifndef RELEASE
    PushCallStack("internal::SafeDeterminantWithOverwrite");
#endif
    if( A.Height() != A.Width() )
        throw std::logic_error
        ("Cannot compute determinant of nonsquare matrix");

    typedef typename Base<F>::type R;
    const int n = A.Height();
    const R scale(n);
    SafeProduct<F> det( n );
    const Grid& g = A.Grid();

    try
    {
        DistMatrix<int,VC,STAR> p(g);
        LU( A, p );
        const bool isOdd = PivotParity( p );

        DistMatrix<F,MD,STAR> d(g);
        A.GetDiagonal( d );
        F localRho = 1;
        R localKappa = 0; 
        if( d.InDiagonal() )
        {
            const int nLocalDiag = d.LocalHeight();
            for( int iLocal=0; iLocal<nLocalDiag; ++iLocal )
            {
                const F delta = d.GetLocal(iLocal,0);
                R alpha = Abs(delta);
                localRho *= delta/alpha;
                localKappa += Log(alpha)/scale;
            }
        }
        mpi::AllReduce( &localRho, &det.rho, 1, mpi::PROD, g.VCComm() );
        mpi::AllReduce( &localKappa, &det.kappa, 1, mpi::SUM, g.VCComm() );
        if( isOdd )
            det.rho = -det.rho;
    }
    catch( SingularMatrixException& e )
    {
        det.rho = 0;
        det.kappa = 0;
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return det;
}

} // namespace internal

template<typename F>
inline SafeProduct<F> 
SafeDeterminant( const Matrix<F>& A )
{
#ifndef RELEASE
    PushCallStack("SafeDeterminant");
#endif
    Matrix<F> B( A );
    SafeProduct<F> det = internal::SafeDeterminantWithOverwrite( B ); 
#ifndef RELEASE
    PopCallStack();
#endif
    return det;
}

template<typename F>
inline SafeProduct<F> 
SafeDeterminant( Matrix<F>& A, bool canOverwrite )
{
#ifndef RELEASE
    PushCallStack("SafeDeterminant");
#endif
    Matrix<F> B;
    if( canOverwrite )
        B.View( A );
    else
        B = A;
    SafeProduct<F> det = internal::SafeDeterminantWithOverwrite( B ); 
#ifndef RELEASE
    PopCallStack();
#endif
    return det;
}

template<typename F>
inline F Determinant( const Matrix<F>& A )
{
#ifndef RELEASE
    PushCallStack("Determinant");
#endif
    SafeProduct<F> safeDet = SafeDeterminant( A );
    F det = safeDet.rho * Exp(safeDet.kappa*safeDet.n);
#ifndef RELEASE
    PopCallStack();
#endif
    return det;
}

template<typename F>
inline F Determinant( Matrix<F>& A, bool canOverwrite )
{
#ifndef RELEASE
    PushCallStack("Determinant");
#endif
    SafeProduct<F> safeDet = SafeDeterminant( A, canOverwrite );
    F det = safeDet.rho * Exp(safeDet.kappa*safeDet.n);
#ifndef RELEASE
    PopCallStack();
#endif
    return det;
}

template<typename F> 
inline SafeProduct<F> 
SafeDeterminant( const DistMatrix<F>& A )
{
#ifndef RELEASE
    PushCallStack("SafeDeterminant");
#endif
    DistMatrix<F> B( A );
    SafeProduct<F> det = internal::SafeDeterminantWithOverwrite( B );
#ifndef RELEASE
    PopCallStack();
#endif
    return det;
}

template<typename F> 
inline SafeProduct<F> 
SafeDeterminant( DistMatrix<F>& A, bool canOverwrite )
{
#ifndef RELEASE
    PushCallStack("SafeDeterminant");
#endif
    DistMatrix<F> B( A.Grid() );
    if( canOverwrite )
        B.View( A );
    else
        B = A;
    SafeProduct<F> det = internal::SafeDeterminantWithOverwrite( B );
#ifndef RELEASE
    PopCallStack();
#endif
    return det;
}

template<typename F> 
inline F Determinant( const DistMatrix<F>& A )
{
#ifndef RELEASE
    PushCallStack("Determinant");
#endif
    SafeProduct<F> safeDet = SafeDeterminant( A );
    F det = safeDet.rho * Exp(safeDet.kappa*safeDet.n);
#ifndef RELEASE
    PopCallStack();
#endif
    return det;
}

template<typename F> 
inline F Determinant( DistMatrix<F>& A, bool canOverwrite )
{
#ifndef RELEASE
    PushCallStack("Determinant");
#endif
    SafeProduct<F> safeDet = SafeDeterminant( A, canOverwrite );
    F det = safeDet.rho * Exp(safeDet.kappa*safeDet.n);
#ifndef RELEASE
    PopCallStack();
#endif
    return det;
}

} // namespace elem
