/*
   Copyright (c) 2009-2011, Jack Poulson
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

#include "./Syr2k/LocalTriangularRank2K.hpp"
#include "./Syr2k/Syr2kLN.hpp"
#include "./Syr2k/Syr2kLT.hpp"
#include "./Syr2k/Syr2kUN.hpp"
#include "./Syr2k/Syr2kUT.hpp"

template<typename T>
inline void
elemental::basic::Syr2k
( Shape shape, 
  Orientation orientation,
  T alpha, const DistMatrix<T,MC,MR>& A,
           const DistMatrix<T,MC,MR>& B,
  T beta,        DistMatrix<T,MC,MR>& C )
{
#ifndef RELEASE
    PushCallStack("basic::Syr2k");
    if( orientation == ADJOINT )
        throw std::logic_error("Syr2k accepts Normal and Transpose options");
#endif
    if( shape == LOWER && orientation == NORMAL )
        basic::internal::Syr2kLN( alpha, A, B, beta, C );
    else if( shape == LOWER )
        basic::internal::Syr2kLT( alpha, A, B, beta, C );
    else if( orientation == NORMAL )
        basic::internal::Syr2kUN( alpha, A, B, beta, C );
    else
        basic::internal::Syr2kUT( alpha, A, B, beta, C );
#ifndef RELEASE
    PopCallStack();
#endif
}

