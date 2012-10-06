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

template<typename R>
inline void
HouseholderSolve
( Orientation orientation, 
  Matrix<R>& A, const Matrix<R>& B,
                      Matrix<R>& X )
{
#ifndef RELEASE
    PushCallStack("HouseholderSolve");
#endif
    // TODO: Add scaling
    const int m = A.Height();
    const int n = A.Width();
    if( orientation == NORMAL )
    {
        if( m != B.Height() )
            throw std::logic_error("A and B do not conform");

        if( m >= n )
        {
            // Overwrite A with its packed QR factorization
            QR( A );

            // Copy B into X
            X = B;

            // Apply Q' to X
            ApplyPackedReflectors( LEFT, LOWER, VERTICAL, FORWARD, 0, A, X );

            // Shrink X to its new height
            X.ResizeTo( n, X.Width() );

            // Solve against R (checking for singularities)
            Matrix<R> AT;
            AT.LockedView( A, 0, 0, n, n );
            Trsm( LEFT, UPPER, NORMAL, NON_UNIT, R(1), AT, X, true );
        }
        else
        {
            // Overwrite A with its packed LQ factorization
            LQ( A );

            // Copy B into X
            X.ResizeTo( n, B.Width() );
            Matrix<R> XT,
                      XB;
            PartitionDown( X, XT,
                              XB, m );
            XT = B;
            Zero( XB );

            // Solve against L (checking for singularities)
            Matrix<R> AL;
            AL.LockedView( A, 0, 0, m, m );
            Trsm( LEFT, LOWER, NORMAL, NON_UNIT, R(1), AL, XT, true );

            // Apply Q' to X 
            ApplyPackedReflectors( LEFT, UPPER, HORIZONTAL, BACKWARD, 0, A, X );
        }
    }
    else // orientation == ADJOINT
    {
        if( n != B.Height() )
            throw std::logic_error("A and B do not conform");

        if( m >= n )
        {
            // Overwrite A with its packed QR factorization
            QR( A );

            // Copy B into X
            X.ResizeTo( m, B.Width() );
            Matrix<R> XT,
                      XB;
            PartitionDown( X, XT,
                              XB, n );
            XT = B; 
            Zero( XB );

            // Solve against R' (checking for singularities)
            Matrix<R> AT;
            AT.LockedView( A, 0, 0, n, n );
            Trsm( LEFT, UPPER, ADJOINT, NON_UNIT, R(1), AT, XT, true );

            // Apply Q to X
            ApplyPackedReflectors( LEFT, LOWER, VERTICAL, BACKWARD, 0, A, X );
        }
        else
        {
            // Overwrite A with its packed LQ factorization
            LQ( A );

            // Copy B into X
            X = B;

            // Apply Q to X
            ApplyPackedReflectors( LEFT, UPPER, HORIZONTAL, FORWARD, 0, A, X );

            // Shrink X to its new size
            X.ResizeTo( m, X.Width() );

            // Solve against L' (check for singularities)
            Matrix<R> AL;
            AL.LockedView( A, 0, 0, m, m );
            Trsm( LEFT, LOWER, ADJOINT, NON_UNIT, R(1), AL, X, true );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
inline void
HouseholderSolve
( Orientation orientation, 
  DistMatrix<R>& A, const DistMatrix<R>& B,
                          DistMatrix<R>& X )
{
#ifndef RELEASE
    PushCallStack("HouseholderSolve");
    if( A.Grid() != B.Grid() || A.Grid() != X.Grid() )
        throw std::logic_error("Grids do not match");
#endif
    const Grid& g = A.Grid();

    // TODO: Add scaling
    const int m = A.Height();
    const int n = A.Width();
    if( orientation == NORMAL )
    {
        if( m != B.Height() )
            throw std::logic_error("A and B do not conform");

        if( m >= n )
        {
            // Overwrite A with its packed QR factorization
            QR( A );

            // Copy B into X
            X = B;

            // Apply Q' to X
            ApplyPackedReflectors
            ( LEFT, LOWER, VERTICAL, FORWARD, 0, A, X );

            // Shrink X to its new height
            X.ResizeTo( n, X.Width() );

            // Solve against R (checking for singularities)
            DistMatrix<R> AT( g );
            AT.LockedView( A, 0, 0, n, n );
            Trsm( LEFT, UPPER, NORMAL, NON_UNIT, R(1), AT, X, true );
        }
        else
        {
            // Overwrite A with its packed LQ factorization
            LQ( A );

            // Copy B into X
            X.ResizeTo( n, B.Width() );
            DistMatrix<R> XT( g ),
                          XB( g );
            PartitionDown( X, XT,
                              XB, m );
            XT = B;
            Zero( XB );

            // Solve against L (checking for singularities)
            DistMatrix<R> AL( g );
            AL.LockedView( A, 0, 0, m, m );
            Trsm( LEFT, LOWER, NORMAL, NON_UNIT, R(1), AL, XT, true );

            // Apply Q' to X 
            ApplyPackedReflectors( LEFT, UPPER, HORIZONTAL, BACKWARD, 0, A, X );
        }
    }
    else // orientation == ADJOINT
    {
        if( n != B.Height() )
            throw std::logic_error("A and B do not conform");

        if( m >= n )
        {
            // Overwrite A with its packed QR factorization
            QR( A );

            // Copy B into X
            X.ResizeTo( m, B.Width() );
            DistMatrix<R> XT( g ),
                          XB( g );
            PartitionDown( X, XT,
                              XB, n );
            XT = B; 
            Zero( XB );

            // Solve against R' (checking for singularities)
            DistMatrix<R> AT( g );
            AT.LockedView( A, 0, 0, n, n );
            Trsm( LEFT, UPPER, ADJOINT, NON_UNIT, R(1), AT, XT, true );

            // Apply Q to X
            ApplyPackedReflectors( LEFT, LOWER, VERTICAL, BACKWARD, 0, A, X );
        }
        else
        {
            // Overwrite A with its packed LQ factorization
            LQ( A );

            // Copy B into X
            X = B;

            // Apply Q to X
            ApplyPackedReflectors( LEFT, UPPER, HORIZONTAL, FORWARD, 0, A, X );

            // Shrink X to its new size
            X.ResizeTo( m, X.Width() );

            // Solve against L' (check for singularities)
            DistMatrix<R> AL( g );
            AL.LockedView( A, 0, 0, m, m );
            Trsm( LEFT, LOWER, ADJOINT, NON_UNIT, R(1), AL, X, true );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R> 
inline void
HouseholderSolve
( Orientation orientation, 
  Matrix<Complex<R> >& A, 
  const Matrix<Complex<R> >& B,
        Matrix<Complex<R> >& X )
{
#ifndef RELEASE
    PushCallStack("HouseholderSolve");
    if( orientation == TRANSPOSE )
        throw std::logic_error("Invalid orientation");
#endif
    typedef Complex<R> C;

    // TODO: Add scaling
    const int m = A.Height();
    const int n = A.Width();
    Matrix<C> t;
    if( orientation == NORMAL )
    {
        if( m != B.Height() )
            throw std::logic_error("A and B do not conform");

        if( m >= n )
        {
            // Overwrite A with its packed QR factorization (and store the 
            // corresponding Householder scalars in t)
            QR( A, t );

            // Copy B into X
            X = B;

            // Apply Q' to X
            ApplyPackedReflectors
            ( LEFT, LOWER, VERTICAL, FORWARD, CONJUGATED, 0, A, t, X );

            // Shrink X to its new height
            X.ResizeTo( n, X.Width() );

            // Solve against R (checking for singularities)
            Matrix<C> AT;
            AT.LockedView( A, 0, 0, n, n );
            Trsm( LEFT, UPPER, NORMAL, NON_UNIT, C(1), AT, X, true );
        }
        else
        {
            // Overwrite A with its packed LQ factorization (and store the
            // corresponding Householder scalars in it)
            LQ( A, t );

            // Copy B into X
            X.ResizeTo( n, B.Width() );
            Matrix<C> XT,
                      XB;
            PartitionDown( X, XT,
                              XB, m );
            XT = B;
            Zero( XB );

            // Solve against L (checking for singularities)
            Matrix<C> AL;
            AL.LockedView( A, 0, 0, m, m );
            Trsm( LEFT, LOWER, NORMAL, NON_UNIT, C(1), AL, XT, true );

            // Apply Q' to X 
            ApplyPackedReflectors
            ( LEFT, UPPER, HORIZONTAL, BACKWARD, CONJUGATED, 0, A, t, X );
        }
    }
    else // orientation == ADJOINT
    {
        if( n != B.Height() )
            throw std::logic_error("A and B do not conform");

        if( m >= n )
        {
            // Overwrite A with its packed QR factorization (and store the 
            // corresponding Householder scalars in t)
            QR( A, t );

            // Copy B into X
            X.ResizeTo( m, B.Width() );
            Matrix<C> XT,
                      XB;
            PartitionDown( X, XT,
                              XB, n );
            XT = B;
            Zero( XB );

            // Solve against R' (checking for singularities)
            Matrix<C> AT;
            AT.LockedView( A, 0, 0, n, n );
            Trsm( LEFT, UPPER, ADJOINT, NON_UNIT, C(1), AT, XT, true );

            // Apply Q to X
            ApplyPackedReflectors
            ( LEFT, LOWER, VERTICAL, BACKWARD, UNCONJUGATED, 0, A, t, X );
        }
        else
        {
            // Overwrite A with its packed LQ factorization (and store the
            // corresponding Householder scalars in t)
            LQ( A, t );

            // Copy B into X
            X = B;

            // Apply Q to X
            ApplyPackedReflectors
            ( LEFT, UPPER, HORIZONTAL, FORWARD, UNCONJUGATED, 0, A, t, X );

            // Shrink X to its new height
            X.ResizeTo( m, X.Width() );

            // Solve against L' (check for singularities)
            Matrix<C> AL;
            AL.LockedView( A, 0, 0, m, m );
            Trsm( LEFT, LOWER, ADJOINT, NON_UNIT, C(1), AL, X, true );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R> 
inline void
HouseholderSolve
( Orientation orientation, 
  DistMatrix<Complex<R> >& A, 
  const DistMatrix<Complex<R> >& B,
        DistMatrix<Complex<R> >& X )
{
#ifndef RELEASE
    PushCallStack("HouseholderSolve");
    if( A.Grid() != B.Grid() || A.Grid() != X.Grid() )
        throw std::logic_error("Grids do not match");
    if( orientation == TRANSPOSE )
        throw std::logic_error("Invalid orientation");
#endif
    typedef Complex<R> C;
    const Grid& g = A.Grid();

    // TODO: Add scaling
    const int m = A.Height();
    const int n = A.Width();
    DistMatrix<C,MD,STAR> t( g );
    if( orientation == NORMAL )
    {
        if( m != B.Height() )
            throw std::logic_error("A and B do not conform");

        if( m >= n )
        {
            // Overwrite A with its packed QR factorization (and store the 
            // corresponding Householder scalars in t)
            QR( A, t );

            // Copy B into X
            X = B;

            // Apply Q' to X
            ApplyPackedReflectors
            ( LEFT, LOWER, VERTICAL, FORWARD, CONJUGATED, 0, A, t, X );

            // Shrink X to its new height
            X.ResizeTo( n, X.Width() );

            // Solve against R (checking for singularities)
            DistMatrix<C> AT( g );
            AT.LockedView( A, 0, 0, n, n );
            Trsm( LEFT, UPPER, NORMAL, NON_UNIT, C(1), AT, X, true );
        }
        else
        {
            // Overwrite A with its packed LQ factorization (and store the
            // corresponding Householder scalars in it)
            LQ( A, t );

            // Copy B into X
            X.ResizeTo( n, B.Width() );
            DistMatrix<C> XT( g ),
                          XB( g );
            PartitionDown( X, XT,
                              XB, m );
            XT = B;
            Zero( XB );

            // Solve against L (checking for singularities)
            DistMatrix<C> AL( g );
            AL.LockedView( A, 0, 0, m, m );
            Trsm( LEFT, LOWER, NORMAL, NON_UNIT, C(1), AL, XT, true );

            // Apply Q' to X 
            ApplyPackedReflectors
            ( LEFT, UPPER, HORIZONTAL, BACKWARD, CONJUGATED, 0, A, t, X );
        }
    }
    else // orientation == ADJOINT
    {
        if( n != B.Height() )
            throw std::logic_error("A and B do not conform");

        if( m >= n )
        {
            // Overwrite A with its packed QR factorization (and store the 
            // corresponding Householder scalars in t)
            QR( A, t );

            // Copy B into X
            X.ResizeTo( m, B.Width() );
            DistMatrix<C> XT( g ),
                          XB( g );
            PartitionDown( X, XT,
                              XB, n );
            XT = B;
            Zero( XB );

            // Solve against R' (checking for singularities)
            DistMatrix<C> AT( g );
            AT.LockedView( A, 0, 0, n, n );
            Trsm( LEFT, UPPER, ADJOINT, NON_UNIT, C(1), AT, XT, true );

            // Apply Q to X
            ApplyPackedReflectors
            ( LEFT, LOWER, VERTICAL, BACKWARD, UNCONJUGATED, 0, A, t, X );
        }
        else
        {
            // Overwrite A with its packed LQ factorization (and store the
            // corresponding Householder scalars in t)
            LQ( A, t );

            // Copy B into X
            X = B;

            // Apply Q to X
            ApplyPackedReflectors
            ( LEFT, UPPER, HORIZONTAL, FORWARD, UNCONJUGATED, 0, A, t, X );

            // Shrink X to its new height
            X.ResizeTo( m, X.Width() );

            // Solve against L' (check for singularities)
            DistMatrix<C> AL( g );
            AL.LockedView( A, 0, 0, m, m );
            Trsm( LEFT, LOWER, ADJOINT, NON_UNIT, C(1), AL, X, true );
        }
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace elem
