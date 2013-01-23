/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef LAPACK_LDL_HPP
#define LAPACK_LDL_HPP

#include "elemental/lapack-like/internal.hpp"

#include "./LDL/Var3.hpp"

namespace elem {

namespace internal {

template<typename F>
inline void
LocalLDL
( Orientation orientation,
  DistMatrix<F,STAR,STAR>& A, DistMatrix<F,STAR,STAR>& d )
{
#ifndef RELEASE
    PushCallStack("internal::LocalLDL");
    if( d.Viewing() && (d.Height() != A.Height() || d.Width() != 1) )
        throw std::logic_error
        ("d must be a column vector of the same height as A");
#endif
    if( !d.Viewing() )
        d.ResizeTo( A.Height(), 1 );
    LDLVar3( orientation, A.LocalMatrix(), d.LocalMatrix() );
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace internal

template<typename F>
inline void
LDLH( Matrix<F>& A )
{
#ifndef RELEASE
    PushCallStack("LDLH");
#endif
    Matrix<F> d;
    LDLH( A, d );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void 
LDLH( DistMatrix<F>& A )
{
#ifndef RELEASE
    PushCallStack("LDLH");
#endif
    DistMatrix<F,MC,STAR> d( A.Grid() );
    LDLH( A, d );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void
LDLH( Matrix<F>& A, Matrix<F>& d )
{
#ifndef RELEASE
    PushCallStack("LDLH");
#endif
    internal::LDLVar3( ADJOINT, A, d );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void 
LDLH( DistMatrix<F>& A, DistMatrix<F,MC,STAR>& d )
{
#ifndef RELEASE
    PushCallStack("LDLH");
#endif
    internal::LDLVar3( ADJOINT, A, d );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void
LDLT( Matrix<F>& A )
{
#ifndef RELEASE
    PushCallStack("LDLT");
#endif
    Matrix<F> d;
    LDLT( A, d );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void 
LDLT( DistMatrix<F>& A )
{
#ifndef RELEASE
    PushCallStack("LDLT");
#endif
    DistMatrix<F,MC,STAR> d( A.Grid() );
    LDLT( A, d );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void
LDLT( Matrix<F>& A, Matrix<F>& d )
{
#ifndef RELEASE
    PushCallStack("LDLT");
#endif
    internal::LDLVar3( TRANSPOSE, A, d );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename F>
inline void 
LDLT( DistMatrix<F>& A, DistMatrix<F,MC,STAR>& d )
{
#ifndef RELEASE
    PushCallStack("LDLT");
#endif
    internal::LDLVar3( TRANSPOSE, A, d );
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace elem

#endif // ifndef LAPACK_LDL_HPP
