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

namespace elemental {

// Distributed E := alpha (A^{T/H} B + C^{T/H} D) + beta E
template<typename T>
inline void
internal::Trr2kTNTN
( UpperOrLower uplo,
  Orientation orientationOfA, Orientation orientationOfC,
  T alpha, const DistMatrix<T,MC,MR>& A, const DistMatrix<T,MC,MR>& B,
           const DistMatrix<T,MC,MR>& C, const DistMatrix<T,MC,MR>& D,
  T beta,        DistMatrix<T,MC,MR>& E )
{
#ifndef RELEASE
    PushCallStack("internal::Trr2kTNTN");
    if( E.Height() != E.Width()  || A.Height() != C.Height() ||
        A.Width()  != E.Height() || C.Width()  != E.Height() ||
        B.Width()  != E.Width()  || D.Width()  != E.Width()  ||
        A.Height() != B.Height() || C.Height() != D.Height() )
        throw std::logic_error("Nonconformal Trr2kTNTN");
#endif
    const Grid& g = E.Grid();

    DistMatrix<T,MC,MR> AT(g),  A0(g),
                        AB(g),  A1(g),
                                A2(g);
    DistMatrix<T,MC,MR> BT(g),  B0(g),
                        BB(g),  B1(g),
                                B2(g);

    DistMatrix<T,MC,MR> CT(g),  C0(g),
                        CB(g),  C1(g),
                                C2(g);
    DistMatrix<T,MC,MR> DT(g),  D0(g),
                        DB(g),  D1(g),
                                D2(g);

    DistMatrix<T,STAR,MC  > A1_STAR_MC(g);
    DistMatrix<T,MR,  STAR> B1Trans_MR_STAR(g);
    DistMatrix<T,STAR,MC  > C1_STAR_MC(g);
    DistMatrix<T,MR,  STAR> D1Trans_MR_STAR(g);

    LockedPartitionDown
    ( A, AT,
         AB, 0 );
    LockedPartitionDown
    ( B, BT,
         BB, 0 );
    LockedPartitionDown
    ( C, CT,
         CB, 0 );
    LockedPartitionDown
    ( D, DT,
         DB, 0 );
    while( AT.Height() < A.Height() )
    {
        LockedRepartitionDown
        ( AT,  A0,
         /**/ /**/
               A1,
          AB,  A2 );
        LockedRepartitionDown
        ( BT,  B0,
         /**/ /**/
               B1,
          BB,  B2 );
        LockedRepartitionDown
        ( CT,  C0,
         /**/ /**/
               C1,
          CB,  C2 );
        LockedRepartitionDown
        ( DT,  D0,
         /**/ /**/
               D1,
          DB,  D2 );

        A1_STAR_MC.AlignWith( E );
        B1Trans_MR_STAR.AlignWith( E );
        C1_STAR_MC.AlignWith( E );
        D1Trans_MR_STAR.AlignWith( E );
        //--------------------------------------------------------------------//
        A1_STAR_MC = A1;
        C1_STAR_MC = C1;
        B1Trans_MR_STAR.TransposeFrom( B1 );
        D1Trans_MR_STAR.TransposeFrom( D1 );
        internal::LocalTrr2k
        ( uplo, orientationOfA, TRANSPOSE, orientationOfC, TRANSPOSE,
          alpha, A1_STAR_MC, B1Trans_MR_STAR, 
                 C1_STAR_MC, D1Trans_MR_STAR,
          beta,  E );
        //--------------------------------------------------------------------//
        D1Trans_MR_STAR.FreeAlignments();
        C1_STAR_MC.FreeAlignments();
        B1Trans_MR_STAR.FreeAlignments();
        A1_STAR_MC.FreeAlignments();

        SlideLockedPartitionDown
        ( DT,  D0,
               D1,
         /**/ /**/
          DB,  D2 );
        SlideLockedPartitionDown
        ( CT,  C0,
               C1,
         /**/ /**/
          CB,  C2 );
        SlideLockedPartitionDown
        ( BT,  B0,
               B1,
         /**/ /**/
          BB,  B2 );
        SlideLockedPartitionDown
        ( AT,  A0,
               A1,
         /**/ /**/
          AB,  A2 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace elemental
