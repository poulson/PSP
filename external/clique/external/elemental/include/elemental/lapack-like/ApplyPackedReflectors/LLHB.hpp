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

//
// Since applying Householder transforms from vectors stored bottom-to-top
// implies that we will be forming a generalization of 
//
//   (I - tau_0 v_0^H v_0) (I - tau_1 v_1^H v_1) = 
//   I - tau_0 v_0^H v_0 - tau_1 v_1^H v_1 + (tau_0 tau_1 v_0 v_1^H) v_0^H v_1 =
//   I - [ v_0^H, v_1^H ] [  tau_0, -tau_0 tau_1 v_0 v_1^H ] [ v_0 ]
//                        [  0,      tau_1                 ] [ v_1 ],
//
// which has a upper-triangular center matrix, say S, we will form S as 
// the inverse of a matrix T, which can easily be formed as
// 
//   triu(T) = triu( V V^H ),  diag(T) = 1/t or 1/conj(t),
//
// where V is the matrix of Householder vectors and t is the vector of scalars.
// V is stored row-wise in the matrix.
//

template<typename R>
inline void
ApplyPackedReflectorsLLHB
( int offset, const Matrix<R>& H, Matrix<R>& A )
{
#ifndef RELEASE
    PushCallStack("internal::ApplyPackedReflectorsLLHB");
    if( offset > 0 || offset < -H.Width() )
        throw std::logic_error("Transforms out of bounds");
    if( H.Width() != A.Height() )
        throw std::logic_error
        ("Width of transforms must equal height of target matrix");
#endif
    Matrix<R>
        HTL, HTR,  H00, H01, H02,  HPan, HPanCopy,
        HBL, HBR,  H10, H11, H12,
                   H20, H21, H22;
    Matrix<R> ATop;

    Matrix<R> SInv, Z;

    LockedPartitionUpDiagonal
    ( H, HTL, HTR,
         HBL, HBR, 0 );
    while( HBR.Height() < H.Height() && HBR.Width() < H.Width() )
    {
        LockedRepartitionUpDiagonal
        ( HTL, /**/ HTR,  H00, H01, /**/ H02,
               /**/       H10, H11, /**/ H12,
         /*************/ /******************/
          HBL, /**/ HBR,  H20, H21, /**/ H22 );

        const int HPanWidth = H10.Width() + H11.Width();
        const int HPanOffset = 
            std::min( H11.Height(), std::max(-offset-H00.Height(),0) );
        const int HPanHeight = H11.Height()-HPanOffset;
        HPan.LockedView( H, H00.Height()+HPanOffset, 0, HPanHeight, HPanWidth );

        ATop.View( A, 0, 0, HPanWidth, A.Width() );

        Zeros( HPanHeight, ATop.Width(), Z );
        Zeros( HPanHeight, HPanHeight, SInv );
        //--------------------------------------------------------------------//
        HPanCopy = HPan;
        MakeTrapezoidal( RIGHT, LOWER, offset, HPanCopy );
        SetDiagonalToOne( RIGHT, offset, HPanCopy );

        Syrk( UPPER, NORMAL, (R)1, HPanCopy, (R)0, SInv );
        HalveMainDiagonal( SInv );

        Gemm( NORMAL, NORMAL, (R)1, HPanCopy, ATop, (R)0, Z );
        Trsm( LEFT, UPPER, NORMAL, NON_UNIT, (R)1, SInv, Z );
        Gemm( TRANSPOSE, NORMAL, (R)-1, HPanCopy, Z, (R)1, ATop );
        //--------------------------------------------------------------------//

        SlideLockedPartitionUpDiagonal
        ( HTL, /**/ HTR,  H00, /**/ H01, H02,
         /*************/ /******************/
               /**/       H10, /**/ H11, H12,
          HBL, /**/ HBR,  H20, /**/ H21, H22 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
inline void
ApplyPackedReflectorsLLHB
( int offset, 
  const DistMatrix<R,MC,MR>& H,
        DistMatrix<R,MC,MR>& A )
{
#ifndef RELEASE
    PushCallStack("internal::ApplyPackedReflectorsLLHB");
    if( H.Grid() != A.Grid() )
        throw std::logic_error("{H,A} must be distributed over the same grid");
    if( offset > 0 || offset < -H.Width() )
        throw std::logic_error("Transforms out of bounds");
    if( H.Width() != A.Height() )
        throw std::logic_error
        ("Width of transforms must equal height of target matrix");
#endif
    const Grid& g = H.Grid();

    DistMatrix<R,MC,MR>
        HTL(g), HTR(g),  H00(g), H01(g), H02(g),  HPan(g), HPanCopy(g),
        HBL(g), HBR(g),  H10(g), H11(g), H12(g),
                         H20(g), H21(g), H22(g);
    DistMatrix<R,MC,MR> ATop(g);

    DistMatrix<R,STAR,VR  > HPan_STAR_VR(g);
    DistMatrix<R,STAR,MC  > HPan_STAR_MC(g); 
    DistMatrix<R,STAR,STAR> SInv_STAR_STAR(g);
    DistMatrix<R,STAR,MR  > Z_STAR_MR(g);
    DistMatrix<R,STAR,VR  > Z_STAR_VR(g);

    LockedPartitionUpDiagonal
    ( H, HTL, HTR,
         HBL, HBR, 0 );
    while( HBR.Height() < H.Height() && HBR.Width() < H.Width() )
    {
        LockedRepartitionUpDiagonal
        ( HTL, /**/ HTR,  H00, H01, /**/ H02,
               /**/       H10, H11, /**/ H12,
         /*************/ /******************/
          HBL, /**/ HBR,  H20, H21, /**/ H22 );

        const int HPanWidth = H10.Width() + H11.Width();
        const int HPanOffset = 
            std::min( H11.Height(), std::max(-offset-H00.Height(),0) );
        const int HPanHeight = H11.Height()-HPanOffset;
        HPan.LockedView( H, H00.Height()+HPanOffset, 0, HPanHeight, HPanWidth );

        ATop.View( A, 0, 0, HPanWidth, A.Width() );

        HPan_STAR_MC.AlignWith( ATop );
        Z_STAR_MR.AlignWith( ATop );
        Z_STAR_VR.AlignWith( ATop );
        Zeros( HPanHeight, ATop.Width(), Z_STAR_MR );
        Zeros( HPanHeight, HPanHeight, SInv_STAR_STAR );
        //--------------------------------------------------------------------//
        HPanCopy = HPan;
        MakeTrapezoidal( RIGHT, LOWER, offset, HPanCopy );
        SetDiagonalToOne( RIGHT, offset, HPanCopy );

        HPan_STAR_VR = HPanCopy;
        Syrk
        ( UPPER, NORMAL,
          (R)1, HPan_STAR_VR.LockedLocalMatrix(),
          (R)0, SInv_STAR_STAR.LocalMatrix() );
        SInv_STAR_STAR.SumOverGrid();
        HalveMainDiagonal( SInv_STAR_STAR );

        HPan_STAR_MC = HPan_STAR_VR;
        LocalGemm
        ( NORMAL, NORMAL, 
          (R)1, HPan_STAR_MC, ATop, (R)0, Z_STAR_MR );
        Z_STAR_VR.SumScatterFrom( Z_STAR_MR );

        LocalTrsm
        ( LEFT, UPPER, NORMAL, NON_UNIT,
          (R)1, SInv_STAR_STAR, Z_STAR_VR );

        Z_STAR_MR = Z_STAR_VR;
        LocalGemm
        ( TRANSPOSE, NORMAL,
          (R)-1, HPan_STAR_MC, Z_STAR_MR, (R)1, ATop );
        //--------------------------------------------------------------------//
        HPan_STAR_MC.FreeAlignments();
        Z_STAR_MR.FreeAlignments();
        Z_STAR_VR.FreeAlignments();

        SlideLockedPartitionUpDiagonal
        ( HTL, /**/ HTR,  H00, /**/ H01, H02,
         /*************/ /******************/
               /**/       H10, /**/ H11, H12,
          HBL, /**/ HBR,  H20, /**/ H21, H22 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R> 
inline void
ApplyPackedReflectorsLLHB
( Conjugation conjugation, int offset, 
  const Matrix<Complex<R> >& H,
  const Matrix<Complex<R> >& t,
        Matrix<Complex<R> >& A )
{
#ifndef RELEASE
    PushCallStack("internal::ApplyPackedReflectorsLLHB");
    if( offset > 0 || offset < -H.Width() )
        throw std::logic_error("Transforms out of bounds");
    if( H.Width() != A.Height() )
        throw std::logic_error
        ("Width of transforms must equal height of target matrix");
    if( t.Height() != H.DiagonalLength( offset ) )
        throw std::logic_error("t must be the same length as H's offset diag");
#endif
    typedef Complex<R> C;

    Matrix<C>
        HTL, HTR,  H00, H01, H02,  HPan, HPanCopy,
        HBL, HBR,  H10, H11, H12,
                   H20, H21, H22;
    Matrix<C> ATop;
    Matrix<C>
        tT,  t0,
        tB,  t1,
             t2;

    Matrix<C> SInv, Z;

    LockedPartitionUpDiagonal
    ( H, HTL, HTR,
         HBL, HBR, 0 );
    LockedPartitionUp
    ( t, tT,
         tB, 0 );
    while( HBR.Height() < H.Height() && HBR.Width() < H.Width() )
    {
        LockedRepartitionUpDiagonal
        ( HTL, /**/ HTR,  H00, H01, /**/ H02,
               /**/       H10, H11, /**/ H12,
         /*************/ /******************/
          HBL, /**/ HBR,  H20, H21, /**/ H22 );

        const int HPanWidth = H10.Width() + H11.Width();
        const int HPanOffset = 
            std::min( H11.Height(), std::max(-offset-H00.Height(),0) );
        const int HPanHeight = H11.Height()-HPanOffset;
        HPan.LockedView( H, H00.Height()+HPanOffset, 0, HPanHeight, HPanWidth );

        LockedRepartitionUp
        ( tT,  t0,
               t1,
         /**/ /**/ 
          tB,  t2, HPanHeight );

        ATop.View( A, 0, 0, HPanWidth, A.Width() );

        Zeros( HPanHeight, ATop.Width(), Z );
        Zeros( HPanHeight, HPanHeight, SInv );
        //--------------------------------------------------------------------//
        HPanCopy = HPan;
        MakeTrapezoidal( RIGHT, LOWER, offset, HPanCopy );
        SetDiagonalToOne( RIGHT, offset, HPanCopy );

        Herk( UPPER, NORMAL, (C)1, HPanCopy, (C)0, SInv );
        FixDiagonal( conjugation, t1, SInv );

        Gemm( NORMAL, NORMAL, (C)1, HPanCopy, ATop, (C)0, Z );
        Trsm( LEFT, UPPER, NORMAL, NON_UNIT, (C)1, SInv, Z );
        Gemm( ADJOINT, NORMAL, (C)-1, HPanCopy, Z, (C)1, ATop );
        //--------------------------------------------------------------------//

        SlideLockedPartitionUpDiagonal
        ( HTL, /**/ HTR,  H00, /**/ H01, H02,
         /*************/ /******************/
               /**/       H10, /**/ H11, H12,
          HBL, /**/ HBR,  H20, /**/ H21, H22 );
    
        SlideLockedPartitionUp
        ( tT,  t0,
         /**/ /**/
               t1,
          tB,  t2 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R> 
inline void
ApplyPackedReflectorsLLHB
( Conjugation conjugation, int offset, 
  const DistMatrix<Complex<R>,MC,MR  >& H,
  const DistMatrix<Complex<R>,MD,STAR>& t,
        DistMatrix<Complex<R>,MC,MR  >& A )
{
#ifndef RELEASE
    PushCallStack("internal::ApplyPackedReflectorsLLHB");
    if( H.Grid() != t.Grid() || t.Grid() != A.Grid() )
        throw std::logic_error
        ("H, t, and A must be distributed over the same grid");
    if( offset > 0 || offset < -H.Width() )
        throw std::logic_error("Transforms out of bounds");
    if( H.Width() != A.Height() )
        throw std::logic_error
        ("Width of transforms must equal height of target matrix");
    if( t.Height() != H.DiagonalLength( offset ) )
        throw std::logic_error("t must be the same length as H's offset diag");
    if( !t.AlignedWithDiagonal( H, offset ) )
        throw std::logic_error("t must be aligned with H's offset diagonal");
#endif
    typedef Complex<R> C;
    const Grid& g = H.Grid();

    DistMatrix<C,MC,MR>
        HTL(g), HTR(g),  H00(g), H01(g), H02(g),  HPan(g), HPanCopy(g),
        HBL(g), HBR(g),  H10(g), H11(g), H12(g),
                         H20(g), H21(g), H22(g);
    DistMatrix<C,MC,MR> ATop(g);
    DistMatrix<C,MD,STAR>
        tT(g),  t0(g),
        tB(g),  t1(g),
                t2(g);

    DistMatrix<C,STAR,VR  > HPan_STAR_VR(g);
    DistMatrix<C,STAR,MC  > HPan_STAR_MC(g); 
    DistMatrix<C,STAR,STAR> t1_STAR_STAR(g);
    DistMatrix<C,STAR,STAR> SInv_STAR_STAR(g);
    DistMatrix<C,STAR,MR  > Z_STAR_MR(g);
    DistMatrix<C,STAR,VR  > Z_STAR_VR(g);

    LockedPartitionUpDiagonal
    ( H, HTL, HTR,
         HBL, HBR, 0 );
    LockedPartitionUp
    ( t, tT,
         tB, 0 );
    while( HBR.Height() < H.Height() && HBR.Width() < H.Width() )
    {
        LockedRepartitionUpDiagonal
        ( HTL, /**/ HTR,  H00, H01, /**/ H02,
               /**/       H10, H11, /**/ H12,
         /*************/ /******************/
          HBL, /**/ HBR,  H20, H21, /**/ H22 );

        const int HPanWidth = H10.Width() + H11.Width();
        const int HPanOffset = 
            std::min( H11.Height(), std::max(-offset-H00.Height(),0) );
        const int HPanHeight = H11.Height()-HPanOffset;
        HPan.LockedView( H, H00.Height()+HPanOffset, 0, HPanHeight, HPanWidth );

        LockedRepartitionUp
        ( tT,  t0,
               t1,
         /**/ /**/ 
          tB,  t2, HPanHeight );

        ATop.View( A, 0, 0, HPanWidth, A.Width() );

        HPan_STAR_MC.AlignWith( ATop );
        Z_STAR_MR.AlignWith( ATop );
        Z_STAR_VR.AlignWith( ATop );
        Zeros( HPanHeight, ATop.Width(), Z_STAR_MR );
        Zeros( HPanHeight, HPanHeight, SInv_STAR_STAR );
        //--------------------------------------------------------------------//
        HPanCopy = HPan;
        MakeTrapezoidal( RIGHT, LOWER, offset, HPanCopy );
        SetDiagonalToOne( RIGHT, offset, HPanCopy );

        HPan_STAR_VR = HPanCopy;
        Herk
        ( UPPER, NORMAL,
          (C)1, HPan_STAR_VR.LockedLocalMatrix(),
          (C)0, SInv_STAR_STAR.LocalMatrix() );
        SInv_STAR_STAR.SumOverGrid();
        t1_STAR_STAR = t1;
        FixDiagonal( conjugation, t1_STAR_STAR, SInv_STAR_STAR );

        HPan_STAR_MC = HPan_STAR_VR;
        LocalGemm
        ( NORMAL, NORMAL, 
          (C)1, HPan_STAR_MC, ATop, (C)0, Z_STAR_MR );
        Z_STAR_VR.SumScatterFrom( Z_STAR_MR );

        LocalTrsm
        ( LEFT, UPPER, NORMAL, NON_UNIT,
          (C)1, SInv_STAR_STAR, Z_STAR_VR );

        Z_STAR_MR = Z_STAR_VR;
        LocalGemm
        ( ADJOINT, NORMAL,
          (C)-1, HPan_STAR_MC, Z_STAR_MR, (C)1, ATop );
        //--------------------------------------------------------------------//
        HPan_STAR_MC.FreeAlignments();
        Z_STAR_MR.FreeAlignments();
        Z_STAR_VR.FreeAlignments();

        SlideLockedPartitionUpDiagonal
        ( HTL, /**/ HTR,  H00, /**/ H01, H02,
         /*************/ /******************/
               /**/       H10, /**/ H11, H12,
          HBL, /**/ HBR,  H20, /**/ H21, H22 );
    
        SlideLockedPartitionUp
        ( tT,  t0,
         /**/ /**/
               t1,
          tB,  t2 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace internal
} // namespace elem
