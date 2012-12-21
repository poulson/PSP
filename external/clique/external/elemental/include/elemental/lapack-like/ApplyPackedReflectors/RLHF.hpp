/*
   Copyright (c) 2009-2012, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/

namespace elem {
namespace internal {

//
// Since applying Householder transforms from vectors stored top-to-bottom
// implies that we will be forming a generalization of
//
//   (I - tau_0 v_0^H v_0) (I - tau_1 v_1^H v_1) = 
//   I - tau_0 v_0^H v_0 - tau_1 v_1^H v_1 + (tau_0 tau_1 v_0 v_1^H) v_0^H v_1 =
//   I - [ v_0^H, v_1^H ] [ tau_0, -tau_0 tau_1 v_0 v_1^H ] [ v_0 ]
//                        [ 0,      tau_1                 ] [ v_1 ],
//
// which has an upper-triangular center matrix, say S, we will form S as 
// the inverse of a matrix T, which can easily be formed as
// 
//   triu(T) = triu( V V^H ),  diag(T) = 1/t or 1/conj(t),
//
// where V is the matrix of Householder vectors and t is the vector of scalars.
//

template<typename R>
inline void
ApplyPackedReflectorsRLHF
( int offset, const Matrix<R>& H, Matrix<R>& A )
{
#ifndef RELEASE
    PushCallStack("internal::ApplyPackedReflectorsRLHF");
    if( offset > 0 || offset < -H.Width() )
        throw std::logic_error("Transforms out of bounds");
    if( H.Width() != A.Width() )
        throw std::logic_error
        ("Width of transforms must equal width of target matrix");
#endif
    Matrix<R>
        HTL, HTR,  H00, H01, H02,  HPan, HPanCopy,
        HBL, HBR,  H10, H11, H12,
                   H20, H21, H22;
    Matrix<R> ALeft;

    Matrix<R> SInv, Z;

    LockedPartitionDownDiagonal
    ( H, HTL, HTR,
         HBL, HBR, 0 );
    while( HTL.Height() < H.Height() && HTL.Width() < H.Width() )
    {
        LockedRepartitionDownDiagonal
        ( HTL, /**/ HTR,  H00, /**/ H01, H02,
         /*************/ /******************/
               /**/       H10, /**/ H11, H12,
          HBL, /**/ HBR,  H20, /**/ H21, H22 );

        const int HPanWidth = H10.Width() + H11.Width();
        const int HPanOffset = 
            std::min( H11.Height(), std::max(-offset-H00.Height(),0) );
        const int HPanHeight = H11.Height()-HPanOffset;
        HPan.LockedView( H, H00.Height()+HPanOffset, 0, HPanHeight, HPanWidth );

        ALeft.View( A, 0, 0, A.Height(), HPanWidth );

        Zeros( ALeft.Height(), HPan.Height(), Z );
        Zeros( HPan.Height(), HPan.Height(), SInv );
        //--------------------------------------------------------------------//
        HPanCopy = HPan;
        MakeTrapezoidal( RIGHT, LOWER, offset, HPanCopy );
        SetDiagonalToOne( RIGHT, offset, HPanCopy );

        Syrk( UPPER, NORMAL, R(1), HPanCopy, R(0), SInv );
        HalveMainDiagonal( SInv );

        Gemm( NORMAL, TRANSPOSE, R(1), ALeft, HPanCopy, R(0), Z );
        Trsm( RIGHT, UPPER, NORMAL, NON_UNIT, R(1), SInv, Z );
        Gemm( NORMAL, NORMAL, R(-1), Z, HPanCopy, R(1), ALeft );
        //--------------------------------------------------------------------//

        SlideLockedPartitionDownDiagonal
        ( HTL, /**/ HTR,  H00, H01, /**/ H02,
               /**/       H10, H11, /**/ H12,
         /*************/ /******************/
          HBL, /**/ HBR,  H20, H21, /**/ H22 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R>
inline void
ApplyPackedReflectorsRLHF
( int offset, 
  const DistMatrix<R>& H,
        DistMatrix<R>& A )
{
#ifndef RELEASE
    PushCallStack("internal::ApplyPackedReflectorsRLHF");
    if( H.Grid() != A.Grid() )
        throw std::logic_error("{H,A} must be distributed over the same grid");
    if( offset > 0 || offset < -H.Width() )
        throw std::logic_error("Transforms out of bounds");
    if( H.Width() != A.Width() )
        throw std::logic_error
        ("Width of transforms must equal width of target matrix");
#endif
    const Grid& g = H.Grid();

    DistMatrix<R>
        HTL(g), HTR(g),  H00(g), H01(g), H02(g),  HPan(g), HPanCopy(g),
        HBL(g), HBR(g),  H10(g), H11(g), H12(g),
                         H20(g), H21(g), H22(g);
    DistMatrix<R> ALeft(g);

    DistMatrix<R,STAR,VR  > HPan_STAR_VR(g);
    DistMatrix<R,STAR,MR  > HPan_STAR_MR(g);
    DistMatrix<R,STAR,STAR> SInv_STAR_STAR(g);
    DistMatrix<R,STAR,MC  > ZTrans_STAR_MC(g);
    DistMatrix<R,STAR,VC  > ZTrans_STAR_VC(g);

    LockedPartitionDownDiagonal
    ( H, HTL, HTR,
         HBL, HBR, 0 );
    while( HTL.Height() < H.Height() && HTL.Width() < H.Width() )
    {
        LockedRepartitionDownDiagonal
        ( HTL, /**/ HTR,  H00, /**/ H01, H02,
         /*************/ /******************/
               /**/       H10, /**/ H11, H12,
          HBL, /**/ HBR,  H20, /**/ H21, H22 );

        const int HPanWidth = H10.Width() + H11.Width();
        const int HPanOffset = 
            std::min( H11.Height(), std::max(-offset-H00.Height(),0) );
        const int HPanHeight = H11.Height()-HPanOffset;
        HPan.LockedView( H, H00.Height()+HPanOffset, 0, HPanHeight, HPanWidth );

        ALeft.View( A, 0, 0, A.Height(), HPanWidth );

        HPan_STAR_MR.AlignWith( ALeft );
        ZTrans_STAR_MC.AlignWith( ALeft );
        ZTrans_STAR_VC.AlignWith( ALeft );
        Zeros( HPan.Height(), ALeft.Height(), ZTrans_STAR_MC );
        Zeros( HPan.Height(), HPan.Height(), SInv_STAR_STAR );
        //--------------------------------------------------------------------//
        HPanCopy = HPan;
        MakeTrapezoidal( RIGHT, LOWER, offset, HPanCopy );
        SetDiagonalToOne( RIGHT, offset, HPanCopy );

        HPan_STAR_VR = HPanCopy;
        Syrk
        ( UPPER, NORMAL,
          R(1), HPan_STAR_VR.LockedLocalMatrix(),
          R(0), SInv_STAR_STAR.LocalMatrix() );
        SInv_STAR_STAR.SumOverGrid();
        HalveMainDiagonal( SInv_STAR_STAR );

        HPan_STAR_MR = HPan_STAR_VR;
        LocalGemm
        ( NORMAL, TRANSPOSE,
          R(1), HPan_STAR_MR, ALeft, R(0), ZTrans_STAR_MC );
        ZTrans_STAR_VC.SumScatterFrom( ZTrans_STAR_MC );

        LocalTrsm
        ( LEFT, UPPER, TRANSPOSE, NON_UNIT,
          R(1), SInv_STAR_STAR, ZTrans_STAR_VC );

        ZTrans_STAR_MC = ZTrans_STAR_VC;
        LocalGemm
        ( TRANSPOSE, NORMAL,
          R(-1), ZTrans_STAR_MC, HPan_STAR_MR, R(1), ALeft );
        //--------------------------------------------------------------------//
        HPan_STAR_MR.FreeAlignments();
        ZTrans_STAR_MC.FreeAlignments();
        ZTrans_STAR_VC.FreeAlignments();

        SlideLockedPartitionDownDiagonal
        ( HTL, /**/ HTR,  H00, H01, /**/ H02,
               /**/       H10, H11, /**/ H12,
         /*************/ /******************/
          HBL, /**/ HBR,  H20, H21, /**/ H22 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R> 
inline void
ApplyPackedReflectorsRLHF
( Conjugation conjugation, int offset, 
  const Matrix<Complex<R> >& H,
  const Matrix<Complex<R> >& t,
        Matrix<Complex<R> >& A )
{
#ifndef RELEASE
    PushCallStack("internal::ApplyPackedReflectorsRLHF");
    if( offset > 0 || offset < -H.Width() )
        throw std::logic_error("Transforms out of bounds");
    if( H.Width() != A.Width() )
        throw std::logic_error
        ("Width of transforms must equal width of target matrix");
    if( t.Height() != H.DiagonalLength( offset ) )
        throw std::logic_error("t must be the same length as H's offset diag");
#endif
    typedef Complex<R> C;

    Matrix<C>
        HTL, HTR,  H00, H01, H02,  HPan, HPanCopy,
        HBL, HBR,  H10, H11, H12,
                   H20, H21, H22;
    Matrix<C> ALeft;
    Matrix<C>
        tT,  t0,
        tB,  t1,
             t2;

    Matrix<C> SInv, Z;

    LockedPartitionDownDiagonal
    ( H, HTL, HTR,
         HBL, HBR, 0 );
    LockedPartitionDown
    ( t, tT,
         tB, 0 );
    while( HTL.Height() < H.Height() && HTL.Width() < H.Width() )
    {
        LockedRepartitionDownDiagonal
        ( HTL, /**/ HTR,  H00, /**/ H01, H02,
         /*************/ /******************/
               /**/       H10, /**/ H11, H12,
          HBL, /**/ HBR,  H20, /**/ H21, H22 );

        const int HPanWidth = H10.Width() + H11.Width();
        const int HPanOffset = 
            std::min( H11.Height(), std::max(-offset-H00.Height(),0) );
        const int HPanHeight = H11.Height()-HPanOffset;
        HPan.LockedView( H, H00.Height()+HPanOffset, 0, HPanHeight, HPanWidth );

        LockedRepartitionDown
        ( tT,  t0,
         /**/ /**/
               t1,
          tB,  t2, HPanHeight );

        ALeft.View( A, 0, 0, A.Height(), HPanWidth );

        Zeros( ALeft.Height(), HPan.Height(), Z );
        Zeros( HPan.Height(), HPan.Height(), SInv );
        //--------------------------------------------------------------------//
        HPanCopy = HPan;
        MakeTrapezoidal( RIGHT, LOWER, offset, HPanCopy );
        SetDiagonalToOne( RIGHT, offset, HPanCopy );

        Herk( UPPER, NORMAL, C(1), HPanCopy, C(0), SInv );
        FixDiagonal( conjugation, t1, SInv );

        Gemm( NORMAL, ADJOINT, C(1), ALeft, HPanCopy, C(0), Z );
        Trsm( RIGHT, UPPER, NORMAL, NON_UNIT, C(1), SInv, Z );
        Gemm( NORMAL, NORMAL, C(-1), Z, HPanCopy, C(1), ALeft );
        //--------------------------------------------------------------------//

        SlideLockedPartitionDownDiagonal
        ( HTL, /**/ HTR,  H00, H01, /**/ H02,
               /**/       H10, H11, /**/ H12,
         /*************/ /******************/
          HBL, /**/ HBR,  H20, H21, /**/ H22 );

        SlideLockedPartitionDown
        ( tT,  t0,
               t1,
         /**/ /**/
          tB,  t2 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename R> 
inline void
ApplyPackedReflectorsRLHF
( Conjugation conjugation, int offset, 
  const DistMatrix<Complex<R> >& H,
  const DistMatrix<Complex<R>,MD,STAR>& t,
        DistMatrix<Complex<R> >& A )
{
#ifndef RELEASE
    PushCallStack("internal::ApplyPackedReflectorsRLHF");
    if( H.Grid() != t.Grid() || t.Grid() != A.Grid() )
        throw std::logic_error
        ("{H,t,A} must be distributed over the same grid");
    if( offset > 0 || offset < -H.Width() )
        throw std::logic_error("Transforms out of bounds");
    if( H.Width() != A.Width() )
        throw std::logic_error
        ("Width of transforms must equal width of target matrix");
    if( t.Height() != H.DiagonalLength( offset ) )
        throw std::logic_error("t must be the same length as H's offset diag");
    if( !t.AlignedWithDiagonal( H, offset ) )
        throw std::logic_error("t must be aligned with H's 'offset' diagonal");
#endif
    typedef Complex<R> C;
    const Grid& g = H.Grid();

    DistMatrix<C>
        HTL(g), HTR(g),  H00(g), H01(g), H02(g),  HPan(g), HPanCopy(g),
        HBL(g), HBR(g),  H10(g), H11(g), H12(g),
                         H20(g), H21(g), H22(g);
    DistMatrix<C> ALeft(g);
    DistMatrix<C,MD,STAR>
        tT(g),  t0(g),
        tB(g),  t1(g),
                t2(g);

    DistMatrix<C,STAR,VR  > HPan_STAR_VR(g);
    DistMatrix<C,STAR,MR  > HPan_STAR_MR(g);
    DistMatrix<C,STAR,STAR> t1_STAR_STAR(g);
    DistMatrix<C,STAR,STAR> SInv_STAR_STAR(g);
    DistMatrix<C,STAR,MC  > ZAdj_STAR_MC(g);
    DistMatrix<C,STAR,VC  > ZAdj_STAR_VC(g);

    LockedPartitionDownDiagonal
    ( H, HTL, HTR,
         HBL, HBR, 0 );
    LockedPartitionDown
    ( t, tT,
         tB, 0 );
    while( HTL.Height() < H.Height() && HTL.Width() < H.Width() )
    {
        LockedRepartitionDownDiagonal
        ( HTL, /**/ HTR,  H00, /**/ H01, H02,
         /*************/ /******************/
               /**/       H10, /**/ H11, H12,
          HBL, /**/ HBR,  H20, /**/ H21, H22 );

        const int HPanWidth = H10.Width() + H11.Width();
        const int HPanOffset = 
            std::min( H11.Height(), std::max(-offset-H00.Height(),0) );
        const int HPanHeight = H11.Height()-HPanOffset;
        HPan.LockedView( H, H00.Height()+HPanOffset, 0, HPanHeight, HPanWidth );

        LockedRepartitionDown
        ( tT,  t0,
         /**/ /**/
               t1,
          tB,  t2, HPanHeight );

        ALeft.View( A, 0, 0, A.Height(), HPanWidth );

        HPan_STAR_MR.AlignWith( ALeft );
        ZAdj_STAR_MC.AlignWith( ALeft );
        ZAdj_STAR_VC.AlignWith( ALeft );
        Zeros( HPan.Height(), ALeft.Height(), ZAdj_STAR_MC );
        Zeros( HPan.Height(), HPan.Height(), SInv_STAR_STAR );
        //--------------------------------------------------------------------//
        HPanCopy = HPan;
        MakeTrapezoidal( RIGHT, LOWER, offset, HPanCopy );
        SetDiagonalToOne( RIGHT, offset, HPanCopy );

        HPan_STAR_VR = HPanCopy;
        Herk
        ( UPPER, NORMAL,
          C(1), HPan_STAR_VR.LockedLocalMatrix(),
          C(0), SInv_STAR_STAR.LocalMatrix() );
        SInv_STAR_STAR.SumOverGrid();
        t1_STAR_STAR = t1;
        FixDiagonal( conjugation, t1_STAR_STAR, SInv_STAR_STAR );

        HPan_STAR_MR = HPan_STAR_VR;
        LocalGemm
        ( NORMAL, ADJOINT,
          C(1), HPan_STAR_MR, ALeft, C(0), ZAdj_STAR_MC );
        ZAdj_STAR_VC.SumScatterFrom( ZAdj_STAR_MC );

        LocalTrsm
        ( LEFT, UPPER, ADJOINT, NON_UNIT,
          C(1), SInv_STAR_STAR, ZAdj_STAR_VC );

        ZAdj_STAR_MC = ZAdj_STAR_VC;
        LocalGemm
        ( ADJOINT, NORMAL,
          C(-1), ZAdj_STAR_MC, HPan_STAR_MR, C(1), ALeft );
        //--------------------------------------------------------------------//
        HPan_STAR_MR.FreeAlignments();
        ZAdj_STAR_MC.FreeAlignments();
        ZAdj_STAR_VC.FreeAlignments();

        SlideLockedPartitionDownDiagonal
        ( HTL, /**/ HTR,  H00, H01, /**/ H02,
               /**/       H10, H11, /**/ H12,
         /*************/ /******************/
          HBL, /**/ HBR,  H20, H21, /**/ H22 );

        SlideLockedPartitionDown
        ( tT,  t0,
               t1,
         /**/ /**/
          tB,  t2 );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace internal
} // namespace elem
