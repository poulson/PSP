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

namespace elemental {
namespace basic {
namespace trrk_util {

#ifndef RELEASE
// Local C := alpha A B + beta C
template<typename T>
inline void 
CheckInputNN
( const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C )
{
    if( A.Height() != C.Height() || B.Width()  != C.Width() ||
        A.Width()  != B.Height() || A.Height() != B.Width() )
    {
        std::ostringstream msg;
        msg << "Nonconformal LocalTrrk: \n"
            << "  A ~ " << A.Height() << " x "
                        << A.Width()  << "\n"
            << "  B ~ " << B.Height() << " x "
                        << B.Width()  << "\n"
            << "  C ~ " << C.Height() << " x " << C.Width() << "\n";
        throw std::logic_error( msg.str().c_str() );
    }
}

// Distributed C := alpha A B + beta C
template<typename T>
inline void 
CheckInput
( const DistMatrix<T,MC,  STAR>& A, 
  const DistMatrix<T,STAR,MR  >& B,
  const DistMatrix<T,MC,  MR  >& C )
{
    if( A.Grid() != B.Grid() || B.Grid() != C.Grid() )
        throw std::logic_error
        ("A, B, and C must be distributed over the same grid");
    if( A.Height() != C.Height() || B.Width()  != C.Width() ||
        A.Width()  != B.Height() || A.Height() != B.Width() )
    {
        std::ostringstream msg;
        msg << "Nonconformal LocalTrrk: \n"
            << "  A[MC,* ] ~ " << A.Height() << " x "
                               << A.Width()  << "\n"
            << "  B[* ,MR] ~ " << B.Height() << " x "
                               << B.Width()  << "\n"
            << "  C[MC,MR] ~ " << C.Height() << " x " << C.Width() << "\n";
        throw std::logic_error( msg.str().c_str() );
    }
    if( A.ColAlignment() != C.ColAlignment() ||
        B.RowAlignment() != C.RowAlignment() )
    {
        std::ostringstream msg;
        msg << "Misaligned LocalTrrk: \n"
            << "  A[MC,* ] ~ " << A.ColAlignment() << "\n"
            << "  B[* ,MR] ~ " << B.RowAlignment() << "\n"
            << "  C[MC,MR] ~ " << C.ColAlignment() << " , " <<
                                  C.RowAlignment() << "\n";
        throw std::logic_error( msg.str().c_str() );
    }
}

// Local C := alpha A B^{T/H} + beta C
template<typename T>
inline void
CheckInputNT
( Orientation orientationOfB,
  const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C )
{
    if( orientationOfB == NORMAL )
        throw std::logic_error("B must be (Conjugate)Transpose'd");
    if( A.Height() != C.Height() || B.Height() != C.Width() ||
        A.Width()  != B.Width()  || A.Height() != B.Height() )
    {
        std::ostringstream msg;
        msg << "Nonconformal LocalTrrk: \n"
            << "  A ~ " << A.Height() << " x "
                        << A.Width()  << "\n"
            << "  B ~ " << B.Height() << " x "
                        << B.Width()  << "\n"
            << "  C ~ " << C.Height() << " x " << C.Width() << "\n";
        throw std::logic_error( msg.str().c_str() );
    }
}

// Distributed C := alpha A B^{T/H} + beta C
template<typename T>
inline void
CheckInput
( Orientation orientationOfB,
  const DistMatrix<T,MC,STAR>& A,
  const DistMatrix<T,MR,STAR>& B,
  const DistMatrix<T,MC,MR  >& C )
{
    if( orientationOfB == NORMAL )
        throw std::logic_error("B[MR,* ] must be (Conjugate)Transpose'd");
    if( A.Grid() != B.Grid() || B.Grid() != C.Grid() )
        throw std::logic_error
        ("A, B, and C must be distributed over the same grid");
    if( A.Height() != C.Height() || B.Height() != C.Width() ||
        A.Width()  != B.Width()  || A.Height() != B.Height() )
    {
        std::ostringstream msg;
        msg << "Nonconformal LocalTrrk: \n"
            << "  A[MC,* ] ~ " << A.Height() << " x "
                               << A.Width()  << "\n"
            << "  B[MR,* ] ~ " << B.Height() << " x "
                               << B.Width()  << "\n"
            << "  C[MC,MR] ~ " << C.Height() << " x " << C.Width() << "\n";
        throw std::logic_error( msg.str().c_str() );
    }
    if( A.ColAlignment() != C.ColAlignment() ||
        B.ColAlignment() != C.RowAlignment() )
    {
        std::ostringstream msg;
        msg << "Misaligned LocalTrrk: \n"
            << "  A[MC,* ] ~ " << A.ColAlignment() << "\n"
            << "  B[MR,* ] ~ " << B.ColAlignment() << "\n"
            << "  C[MC,MR] ~ " << C.ColAlignment() << " , " <<
                                  C.RowAlignment() << "\n";
        throw std::logic_error( msg.str().c_str() );
    }
}

// Local C := alpha A^{T/H} B + beta C
template<typename T>
inline void
CheckInputTN
( Orientation orientationOfA,
  const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C )
{
    if( orientationOfA == NORMAL )
        throw std::logic_error("A must be (Conjugate)Transpose'd");
    if( A.Width() != C.Height() || B.Width() != C.Width() ||
        A.Height() != B.Height() || A.Width() != B.Width() )
    {
        std::ostringstream msg;
        msg << "Nonconformal LocalTrrk: \n"
            << "  A ~ " << A.Height() << " x "
                        << A.Width()  << "\n"
            << "  B ~ " << B.Height() << " x "
                        << B.Width()  << "\n"
            << "  C ~ " << C.Height() << " x " << C.Width() << "\n";
        throw std::logic_error( msg.str().c_str() );
    }
}

// Distributed C := alpha A^{T/H} B + beta C
template<typename T>
inline void
CheckInput
( Orientation orientationOfA,
  const DistMatrix<T,STAR,MC>& A,
  const DistMatrix<T,STAR,MR>& B,
  const DistMatrix<T,MC,  MR>& C )
{
    if( orientationOfA == NORMAL )
        throw std::logic_error("A[* ,MC] must be (Conjugate)Transpose'd");
    if( A.Grid() != B.Grid() || B.Grid() != C.Grid() )
        throw std::logic_error
        ("A, B, and C must be distributed over the same grid");
    if( A.Width() != C.Height() || B.Width() != C.Width() ||
        A.Height() != B.Height() || A.Width() != B.Width() )
    {
        std::ostringstream msg;
        msg << "Nonconformal LocalTrrk: \n"
            << "  A[* ,MC] ~ " << A.Height() << " x "
                               << A.Width()  << "\n"
            << "  B[* ,MR] ~ " << B.Height() << " x "
                               << B.Width()  << "\n"
            << "  C[MC,MR] ~ " << C.Height() << " x " << C.Width() << "\n";
        throw std::logic_error( msg.str().c_str() );
    }
    if( A.RowAlignment() != C.ColAlignment() ||
        B.RowAlignment() != C.RowAlignment() )
    {
        std::ostringstream msg;
        msg << "Misaligned LocalTrrk: \n"
            << "  A[* ,MC] ~ " << A.RowAlignment() << "\n"
            << "  B[* ,MR] ~ " << B.RowAlignment() << "\n"
            << "  C[MC,MR] ~ " << C.ColAlignment() << " , " <<
                                  C.RowAlignment() << "\n";
        throw std::logic_error( msg.str().c_str() );
    }
}

// Local C := alpha A^{T/H} B^{T/H} + beta C
template<typename T>
inline void
CheckInputTT
( Orientation orientationOfA,
  Orientation orientationOfB,
  const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C )
{
    if( orientationOfA == NORMAL )
        throw std::logic_error("A must be (Conjugate)Transpose'd");
    if( orientationOfB == NORMAL )
        throw std::logic_error("B must be (Conjugate)Transpose'd");
    if( A.Width() != C.Height() || B.Height() != C.Width() ||
        A.Height() != B.Width() || A.Width() != B.Height() )
    {
        std::ostringstream msg;
        msg << "Nonconformal LocalTrrk: \n"
            << "  A ~ " << A.Height() << " x "
                        << A.Width()  << "\n"
            << "  B ~ " << B.Height() << " x "
                        << B.Width()  << "\n"
            << "  C ~ " << C.Height() << " x " << C.Width() << "\n";
        throw std::logic_error( msg.str().c_str() );
    }
}

// Distributed C := alpha A^{T/H} B^{T/H} + beta C
template<typename T>
inline void
CheckInput
( Orientation orientationOfA,
  Orientation orientationOfB,
  const DistMatrix<T,STAR,MC  >& A,
  const DistMatrix<T,MR,  STAR>& B,
  const DistMatrix<T,MC,  MR  >& C )
{
    if( orientationOfA == NORMAL )
        throw std::logic_error("A[* ,MC] must be (Conjugate)Transpose'd");
    if( orientationOfB == NORMAL )
        throw std::logic_error("B[MR,* ] must be (Conjugate)Transpose'd");
    if( A.Grid() != B.Grid() || B.Grid() != C.Grid() )
        throw std::logic_error
        ("A, B, and C must be distributed over the same grid");
    if( A.Width() != C.Height() || B.Height() != C.Width() ||
        A.Height() != B.Width() || A.Width() != B.Height() )
    {
        std::ostringstream msg;
        msg << "Nonconformal LocalTrrk: \n"
            << "  A[* ,MC] ~ " << A.Height() << " x "
                               << A.Width()  << "\n"
            << "  B[MR,* ] ~ " << B.Height() << " x "
                               << B.Width()  << "\n"
            << "  C[MC,MR] ~ " << C.Height() << " x " << C.Width() << "\n";
        throw std::logic_error( msg.str().c_str() );
    }
    if( A.RowAlignment() != C.ColAlignment() ||
        B.ColAlignment() != C.RowAlignment() )
    {
        std::ostringstream msg;
        msg << "Misaligned LocalTrrk: \n"
            << "  A[* ,MC] ~ " << A.RowAlignment() << "\n"
            << "  B[MR,* ] ~ " << B.ColAlignment() << "\n"
            << "  C[MC,MR] ~ " << C.ColAlignment() << " , " <<
                                  C.RowAlignment() << "\n";
        throw std::logic_error( msg.str().c_str() );
    }
}
#endif // WITHOUT_COMPLEX

// Local C := alpha A B + beta C
template<typename T>
inline void
TrrkNNKernel
( UpperOrLower uplo, 
  T alpha, const Matrix<T>& A, const Matrix<T>& B,
  T beta,        Matrix<T>& C )
{
#ifndef RELEASE
    PushCallStack("TrrkNNKernel");
    CheckInputNN( A, B, C );
#endif
    Matrix<T> AT,
              AB;
    Matrix<T> BL, BR;
    Matrix<T> CTL, CTR,
              CBL, CBR;
    Matrix<T> DTL, DBR;

    const unsigned half = C.Height()/2;
    basic::Scal( beta, C );
    LockedPartitionDown
    ( A, AT,
         AB, half );
    LockedPartitionRight( B, BL, BR, half );
    PartitionDownDiagonal
    ( C, CTL, CTR,
         CBL, CBR, half );

    DTL.ResizeTo( CTL.Height(), CTL.Width() );
    DBR.ResizeTo( CBR.Height(), CBR.Width() );
    //------------------------------------------------------------------------//
    if( uplo == LOWER )
        basic::Gemm( NORMAL, NORMAL, alpha, AB, BL, (T)1, CBL );
    else
        basic::Gemm( NORMAL, NORMAL, alpha, AT, BR, (T)1, CTR );

    basic::Gemm( NORMAL, NORMAL, alpha, AT, BL, (T)0, DTL );
    DTL.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DTL, CTL );

    basic::Gemm( NORMAL, NORMAL, alpha, AB, BR, (T)0, DBR );
    DBR.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DBR, CBR );
    //------------------------------------------------------------------------//
#ifndef RELEASE
    PopCallStack();
#endif
}

// Distributed C := alpha A B + beta C
template<typename T>
inline void
LocalTrrkKernel
( UpperOrLower uplo, 
  T alpha, const DistMatrix<T,MC,  STAR>& A,
           const DistMatrix<T,STAR,MR  >& B,
  T beta,        DistMatrix<T,MC,  MR  >& C )
{
#ifndef RELEASE
    PushCallStack("LocalTrrkKernel");
    CheckInput( A, B, C );
#endif
    const Grid& g = C.Grid();

    DistMatrix<T,MC,STAR> AT(g), 
                          AB(g);
    DistMatrix<T,STAR,MR> BL(g), BR(g);
    DistMatrix<T,MC,MR> CTL(g), CTR(g),
                        CBL(g), CBR(g);
    DistMatrix<T,MC,MR> DTL(g), DBR(g);

    const unsigned half = C.Height()/2;
    C.ScaleTrapezoid( beta, LEFT, uplo );
    LockedPartitionDown
    ( A, AT,
         AB, half );
    LockedPartitionRight( B, BL, BR, half );
    PartitionDownDiagonal
    ( C, CTL, CTR,
         CBL, CBR, half );

    DTL.AlignWith( CTL );
    DBR.AlignWith( CBR );
    DTL.ResizeTo( CTL.Height(), CTL.Width() );
    DBR.ResizeTo( CBR.Height(), CBR.Width() );
    //------------------------------------------------------------------------//
    if( uplo == LOWER )
    {
        basic::internal::LocalGemm
        ( NORMAL, NORMAL, alpha, AB, BL, (T)1, CBL );
    }
    else
    {
        basic::internal::LocalGemm
        ( NORMAL, NORMAL, alpha, AT, BR, (T)1, CTR );
    }

    basic::internal::LocalGemm
    ( NORMAL, NORMAL, alpha, AT, BL, (T)0, DTL );

    DTL.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DTL, CTL );

    basic::internal::LocalGemm
    ( NORMAL, NORMAL, alpha, AB, BR, (T)0, DBR );

    DBR.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DBR, CBR );
    //------------------------------------------------------------------------//
#ifndef RELEASE
    PopCallStack();
#endif
}

// Local C := alpha A B^{T/H} + beta C
template<typename T>
inline void
TrrkNTKernel
( UpperOrLower uplo,
  Orientation orientationOfB,
  T alpha, const Matrix<T>& A, const Matrix<T>& B,
  T beta,        Matrix<T>& C )
{
#ifndef RELEASE
    PushCallStack("TrrkNTKernel");
    CheckInputNT( orientationOfB, A, B, C );
#endif
    Matrix<T> AT,
              AB;
    Matrix<T> BT,
              BB;
    Matrix<T> CTL, CTR,
              CBL, CBR;
    Matrix<T> DTL, DBR;

    const unsigned half = C.Height()/2;
    C.ScaleTrapezoid( beta, LEFT, uplo );
    LockedPartitionDown
    ( A, AT,
         AB, half );
    LockedPartitionDown
    ( B, BT, 
         BB, half );
    PartitionDownDiagonal
    ( C, CTL, CTR,
         CBL, CBR, half );

    DTL.ResizeTo( CTL.Height(), CTL.Width() );
    DBR.ResizeTo( CBR.Height(), CBR.Width() );
    //------------------------------------------------------------------------//
    if( uplo == LOWER )
        basic::Gemm( NORMAL, orientationOfB, alpha, AB, BT, (T)1, CBL );
    else
        basic::Gemm( NORMAL, orientationOfB, alpha, AT, BB, (T)1, CTR );

    basic::Gemm( NORMAL, orientationOfB, alpha, AT, BT, (T)0, DTL );
    // TODO: AxpyTrapezoidal?
    DTL.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DTL, CTL );

    basic::Gemm( NORMAL, orientationOfB, alpha, AB, BB, (T)0, DBR );
    // TODO: AxpyTrapezoidal?
    DBR.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DBR, CBR );
    //------------------------------------------------------------------------//
#ifndef RELEASE
    PopCallStack();
#endif
}

// Distributed C := alpha A B^{T/H} + beta C
template<typename T>
inline void
LocalTrrkKernel
( UpperOrLower uplo,
  Orientation orientationOfB,
  T alpha, const DistMatrix<T,MC,STAR>& A,
           const DistMatrix<T,MR,STAR>& B,
  T beta,        DistMatrix<T,MC,MR  >& C )
{
#ifndef RELEASE
    PushCallStack("LocalTrrkKernel");
    CheckInput( orientationOfB, A, B, C );
#endif
    const Grid& g = C.Grid();

    DistMatrix<T,MC,STAR> AT(g),
                          AB(g);
    DistMatrix<T,MR,STAR> BT(g), 
                          BB(g);
    DistMatrix<T,MC,MR> CTL(g), CTR(g),
                        CBL(g), CBR(g);
    DistMatrix<T,MC,MR> DTL(g), DBR(g);

    const unsigned half = C.Height()/2;
    C.ScaleTrapezoid( beta, LEFT, uplo );
    LockedPartitionDown
    ( A, AT,
         AB, half );
    LockedPartitionDown
    ( B, BT, 
         BB, half );
    PartitionDownDiagonal
    ( C, CTL, CTR,
         CBL, CBR, half );

    DTL.AlignWith( CTL );
    DBR.AlignWith( CBR );
    DTL.ResizeTo( CTL.Height(), CTL.Width() );
    DBR.ResizeTo( CBR.Height(), CBR.Width() );
    //------------------------------------------------------------------------//
    if( uplo == LOWER )
    {
        basic::internal::LocalGemm
        ( NORMAL, orientationOfB, alpha, AB, BT, (T)1, CBL );
    }
    else
    {
        basic::internal::LocalGemm
        ( NORMAL, orientationOfB, alpha, AT, BB, (T)1, CTR );
    }

    basic::internal::LocalGemm
    ( NORMAL, orientationOfB, alpha, AT, BT, (T)0, DTL );

    // TODO: AxpyTrapezoidal?
    DTL.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DTL, CTL );

    basic::internal::LocalGemm
    ( NORMAL, orientationOfB, alpha, AB, BB, (T)0, DBR );

    // TODO: AxpyTrapezoidal?
    DBR.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DBR, CBR );
    //------------------------------------------------------------------------//
#ifndef RELEASE
    PopCallStack();
#endif
}

// Local C := alpha A^{T/H} B + beta C
template<typename T>
inline void
TrrkTNKernel
( UpperOrLower uplo,
  Orientation orientationOfA,
  T alpha, const Matrix<T>& A, const Matrix<T>& B,
  T beta,        Matrix<T>& C )
{
#ifndef RELEASE
    PushCallStack("TrrkTNKernel");
    CheckInputTN( orientationOfA, A, B, C );
#endif
    Matrix<T> AL, AR;
    Matrix<T> BL, BR;
    Matrix<T> CTL, CTR,
              CBL, CBR;
    Matrix<T> DTL, DBR;

    const unsigned half = C.Height()/2;
    C.ScaleTrapezoid( beta, LEFT, uplo );
    LockedPartitionRight( A, AL, AR, half );
    LockedPartitionRight( B, BL, BR, half );
    PartitionDownDiagonal
    ( C, CTL, CTR,
         CBL, CBR, half );

    DTL.ResizeTo( CTL.Height(), CTL.Width() );
    DBR.ResizeTo( CBR.Height(), CBR.Width() );
    //------------------------------------------------------------------------//
    if( uplo == LOWER )
        basic::Gemm( orientationOfA, NORMAL, alpha, AR, BL, (T)1, CBL );
    else
        basic::Gemm( orientationOfA, NORMAL, alpha, AL, BR, (T)1, CTR );

    basic::Gemm( orientationOfA, NORMAL, alpha, AL, BL, (T)0, DTL );
    DTL.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DTL, CTL );

    basic::Gemm( orientationOfA, NORMAL, alpha, AR, BR, (T)0, DBR );
    DBR.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DBR, CBR );
    //------------------------------------------------------------------------//
#ifndef RELEASE
    PopCallStack();
#endif
}

// Distributed C := alpha A^{T/H} B + beta C
template<typename T>
inline void
LocalTrrkKernel
( UpperOrLower uplo,
  Orientation orientationOfA,
  T alpha, const DistMatrix<T,STAR,MC>& A,
           const DistMatrix<T,STAR,MR>& B,
  T beta,        DistMatrix<T,MC,  MR>& C )
{
#ifndef RELEASE
    PushCallStack("LocalTrrkKernel");
    CheckInput( orientationOfA, A, B, C );
#endif
    const Grid& g = C.Grid();

    DistMatrix<T,STAR,MC> AL(g), AR(g);
    DistMatrix<T,STAR,MR> BL(g), BR(g);
    DistMatrix<T,MC,MR> CTL(g), CTR(g),
                        CBL(g), CBR(g);
    DistMatrix<T,MC,MR> DTL(g), DBR(g);

    const unsigned half = C.Height()/2;
    C.ScaleTrapezoid( beta, LEFT, uplo );
    LockedPartitionRight( A, AL, AR, half );
    LockedPartitionRight( B, BL, BR, half );
    PartitionDownDiagonal
    ( C, CTL, CTR,
         CBL, CBR, half );

    DTL.AlignWith( CTL );
    DBR.AlignWith( CBR );
    DTL.ResizeTo( CTL.Height(), CTL.Width() );
    DBR.ResizeTo( CBR.Height(), CBR.Width() );
    //------------------------------------------------------------------------//
    if( uplo == LOWER )
    {
        basic::internal::LocalGemm
        ( orientationOfA, NORMAL, alpha, AR, BL, (T)1, CBL );
    }
    else
    {
        basic::internal::LocalGemm
        ( orientationOfA, NORMAL, alpha, AL, BR, (T)1, CTR );
    }

    basic::internal::LocalGemm
    ( orientationOfA, NORMAL, alpha, AL, BL, (T)0, DTL );

    DTL.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DTL, CTL );

    basic::internal::LocalGemm
    ( orientationOfA, NORMAL, alpha, AR, BR, (T)0, DBR );

    DBR.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DBR, CBR );
    //------------------------------------------------------------------------//
#ifndef RELEASE
    PopCallStack();
#endif
}

// Local C := alpha A^{T/H} B^{T/H} + beta C
template<typename T>
inline void
TrrkTTKernel
( UpperOrLower uplo,
  Orientation orientationOfA,
  Orientation orientationOfB,
  T alpha, const Matrix<T>& A, const Matrix<T>& B,
  T beta,        Matrix<T>& C )
{
#ifndef RELEASE
    PushCallStack("TrrkTTKernel");
    CheckInputTT( orientationOfA, orientationOfB, A, B, C );
#endif
    Matrix<T> AL, AR;
    Matrix<T> BT,
              BB;
    Matrix<T> CTL, CTR,
              CBL, CBR;
    Matrix<T> DTL, DBR;

    const unsigned half = C.Height()/2;
    C.ScaleTrapezoid( beta, LEFT, uplo );
    LockedPartitionRight( A, AL, AR, half );
    LockedPartitionDown
    ( B, BT, 
         BB, half );
    PartitionDownDiagonal
    ( C, CTL, CTR,
         CBL, CBR, half );

    DTL.ResizeTo( CTL.Height(), CTL.Width() );
    DBR.ResizeTo( CBR.Height(), CBR.Width() );
    //------------------------------------------------------------------------//
    if( uplo == LOWER )
        basic::Gemm( orientationOfA, orientationOfB, alpha, AR, BT, (T)1, CBL );
    else
        basic::Gemm( orientationOfA, orientationOfB, alpha, AL, BB, (T)1, CTR );

    basic::Gemm( orientationOfA, orientationOfB, alpha, AL, BT, (T)0, DTL );
    DTL.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DTL, CTL );

    basic::Gemm( orientationOfA, orientationOfB, alpha, AR, BB, (T)0, DBR );
    DBR.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DBR, CBR );
    //------------------------------------------------------------------------//
#ifndef RELEASE
    PopCallStack();
#endif
}

// Distributed C := alpha A^{T/H} B^{T/H} + beta C
template<typename T>
inline void
LocalTrrkKernel
( UpperOrLower uplo,
  Orientation orientationOfA,
  Orientation orientationOfB,
  T alpha, const DistMatrix<T,STAR,MC  >& A,
           const DistMatrix<T,MR,  STAR>& B,
  T beta,        DistMatrix<T,MC,  MR  >& C )
{
#ifndef RELEASE
    PushCallStack("LocalTrrkKernel");
    CheckInput( orientationOfA, orientationOfB, A, B, C );
#endif
    const Grid& g = C.Grid();

    DistMatrix<T,STAR,MC> AL(g), AR(g);
    DistMatrix<T,MR,STAR> BT(g), 
                          BB(g);
    DistMatrix<T,MC,MR> CTL(g), CTR(g),
                        CBL(g), CBR(g);
    DistMatrix<T,MC,MR> DTL(g), DBR(g);

    const unsigned half = C.Height()/2;
    C.ScaleTrapezoid( beta, LEFT, uplo );
    LockedPartitionRight( A, AL, AR, half );
    LockedPartitionDown
    ( B, BT, 
         BB, half );
    PartitionDownDiagonal
    ( C, CTL, CTR,
         CBL, CBR, half );

    DTL.AlignWith( CTL );
    DBR.AlignWith( CBR );
    DTL.ResizeTo( CTL.Height(), CTL.Width() );
    DBR.ResizeTo( CBR.Height(), CBR.Width() );
    //------------------------------------------------------------------------//
    if( uplo == LOWER )
    {
        basic::internal::LocalGemm
        ( orientationOfA, orientationOfB, alpha, AR, BT, (T)1, CBL );
    }
    else
    {
        basic::internal::LocalGemm
        ( orientationOfA, orientationOfB, alpha, AL, BB, (T)1, CTR );
    }

    basic::internal::LocalGemm
    ( orientationOfA, orientationOfB, alpha, AL, BT, (T)0, DTL );

    DTL.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DTL, CTL );

    basic::internal::LocalGemm
    ( orientationOfA, orientationOfB, alpha, AR, BB, (T)0, DBR );

    DBR.MakeTrapezoidal( LEFT, uplo );
    basic::Axpy( (T)1, DBR, CBR );
    //------------------------------------------------------------------------//
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace trrk_util
} // namespace basic
} // namespace elemental

// Local C := alpha A B + beta C
template<typename T>
inline void
elemental::basic::internal::TrrkNN
( UpperOrLower uplo,
  T alpha, const Matrix<T>& A, const Matrix<T>& B,
  T beta,        Matrix<T>& C )
{
    using namespace trrk_util;
#ifndef RELEASE
    PushCallStack("basic::internal::TrrkNN");
    CheckInputNN( A, B, C );
#endif
    if( C.Height() < LocalTrrkBlocksize<T>() )
    {
        TrrkNNKernel( uplo, alpha, A, B, beta, C );
    }
    else
    {
        // Split C in four roughly equal pieces, perform a large gemm on corner
        // and recurse on CTL and CBR.
        Matrix<T> AT,
                  AB;
        Matrix<T> BL, BR;
        Matrix<T> CTL, CTR,
                  CBL, CBR;

        const unsigned half = C.Height() / 2;
        LockedPartitionDown
        ( A, AT,
             AB, half );
        LockedPartitionRight( B, BL, BR, half );
        PartitionDownDiagonal
        ( C, CTL, CTR,
             CBL, CBR, half );

        if( uplo == LOWER )
            basic::Gemm( NORMAL, NORMAL, alpha, AB, BL, beta, CBL );
        else
            basic::Gemm( NORMAL, NORMAL, alpha, AT, BR, beta, CTR );

        // Recurse
        basic::internal::TrrkNN( uplo, alpha, AT, BL, beta, CTL );
        basic::internal::TrrkNN( uplo, alpha, AB, BR, beta, CBR );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Distributed C := alpha A B + beta C
template<typename T>
inline void
elemental::basic::internal::LocalTrrk
( UpperOrLower uplo,
  T alpha, const DistMatrix<T,MC,  STAR>& A,
           const DistMatrix<T,STAR,MR  >& B,
  T beta,        DistMatrix<T,MC,  MR  >& C )
{
    using namespace trrk_util;
#ifndef RELEASE
    PushCallStack("basic::internal::LocalTrrk");
    CheckInput( A, B, C );
#endif
    const Grid& g = C.Grid();

    if( C.Height() < g.Width()*LocalTrrkBlocksize<T>() )
    {
        LocalTrrkKernel( uplo, alpha, A, B, beta, C );
    }
    else
    {
        // Split C in four roughly equal pieces, perform a large gemm on corner
        // and recurse on CTL and CBR.
        DistMatrix<T,MC,STAR> AT(g),
                              AB(g);
        DistMatrix<T,STAR,MR> BL(g), BR(g);
        DistMatrix<T,MC,MR> CTL(g), CTR(g),
                            CBL(g), CBR(g);

        const unsigned half = C.Height() / 2;
        LockedPartitionDown
        ( A, AT,
             AB, half );
        LockedPartitionRight( B, BL, BR, half );
        PartitionDownDiagonal
        ( C, CTL, CTR,
             CBL, CBR, half );

        if( uplo == LOWER )
        { 
            basic::internal::LocalGemm
            ( NORMAL, NORMAL, alpha, AB, BL, beta, CBL );
        }
        else
        {
            basic::internal::LocalGemm
            ( NORMAL, NORMAL, alpha, AT, BR, beta, CTR );
        }

        // Recurse
        basic::internal::LocalTrrk( uplo, alpha, AT, BL, beta, CTL );
        basic::internal::LocalTrrk( uplo, alpha, AB, BR, beta, CBR );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Local C := alpha A B^{T/H} + beta C
template<typename T>
inline void
elemental::basic::internal::TrrkNT
( UpperOrLower uplo,
  Orientation orientationOfB,
  T alpha, const Matrix<T>& A, const Matrix<T>& B,
  T beta,        Matrix<T>& C )
{
    using namespace trrk_util;
#ifndef RELEASE
    PushCallStack("basic::internal::TrrkNT");
    CheckInputNT( orientationOfB, A, B, C );
#endif
    if( C.Height() < LocalTrrkBlocksize<T>() )
    {
        TrrkNTKernel( uplo, orientationOfB, alpha, A, B, beta, C );
    }
    else
    {
        // Split C in four roughly equal pieces, perform a large gemm on corner
        // and recurse on CTL and CBR.
        Matrix<T> AT,
                  AB;
        Matrix<T> BT,
                  BB;
        Matrix<T> CTL, CTR,
                  CBL, CBR;

        const unsigned half = C.Height() / 2;
        LockedPartitionDown
        ( A, AT,
             AB, half );
        LockedPartitionDown
        ( B, BT, 
             BB, half );
        PartitionDownDiagonal
        ( C, CTL, CTR,
             CBL, CBR, half );

        if( uplo == LOWER )
            basic::Gemm( NORMAL, orientationOfB, alpha, AB, BT, beta, CBL );
        else
            basic::Gemm( NORMAL, orientationOfB, alpha, AT, BB, beta, CTR );

        // Recurse
        basic::internal::TrrkNT
        ( uplo, orientationOfB, alpha, AT, BT, beta, CTL );
        basic::internal::TrrkNT
        ( uplo, orientationOfB, alpha, AB, BB, beta, CBR );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Distributed C := alpha A B^{T/H} + beta C
template<typename T>
inline void
elemental::basic::internal::LocalTrrk
( UpperOrLower uplo,
  Orientation orientationOfB,
  T alpha, const DistMatrix<T,MC,STAR>& A,
           const DistMatrix<T,MR,STAR>& B,
  T beta,        DistMatrix<T,MC,MR  >& C )
{
    using namespace trrk_util;
#ifndef RELEASE
    PushCallStack("basic::internal::LocalTrrk");
    CheckInput( orientationOfB, A, B, C );
#endif
    const Grid& g = C.Grid();

    if( C.Height() < g.Width()*LocalTrrkBlocksize<T>() )
    {
        LocalTrrkKernel( uplo, orientationOfB, alpha, A, B, beta, C );
    }
    else
    {
        // Split C in four roughly equal pieces, perform a large gemm on corner
        // and recurse on CTL and CBR.
        DistMatrix<T,MC,STAR> AT(g),
                              AB(g);
        DistMatrix<T,MR,STAR> BT(g), 
                              BB(g);
        DistMatrix<T,MC,MR> CTL(g), CTR(g),
                            CBL(g), CBR(g);

        const unsigned half = C.Height() / 2;
        LockedPartitionDown
        ( A, AT,
             AB, half );
        LockedPartitionDown
        ( B, BT, 
             BB, half );
        PartitionDownDiagonal
        ( C, CTL, CTR,
             CBL, CBR, half );

        if( uplo == LOWER )
        { 
            basic::internal::LocalGemm
            ( NORMAL, orientationOfB, alpha, AB, BT, beta, CBL );
        }
        else
        {
            basic::internal::LocalGemm
            ( NORMAL, orientationOfB, alpha, AT, BB, beta, CTR );
        }

        // Recurse
        basic::internal::LocalTrrk
        ( uplo, orientationOfB, alpha, AT, BT, beta, CTL );
        basic::internal::LocalTrrk
        ( uplo, orientationOfB, alpha, AB, BB, beta, CBR );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Local C := alpha A^{T/H} B + beta C
template<typename T>
inline void
elemental::basic::internal::TrrkTN
( UpperOrLower uplo,
  Orientation orientationOfA,
  T alpha, const Matrix<T>& A, const Matrix<T>& B,
  T beta,        Matrix<T>& C )
{
    using namespace trrk_util;
#ifndef RELEASE
    PushCallStack("basic::internal::TrrkTN");
    CheckInputTN( orientationOfA, A, B, C );
#endif
    if( C.Height() < LocalTrrkBlocksize<T>() )
    {
        TrrkKernelTN( uplo, orientationOfA, alpha, A, B, beta, C );
    }
    else
    {
        // Split C in four roughly equal pieces, perform a large gemm on corner
        // and recurse on CTL and CBR.
        Matrix<T> AL, AR;
        Matrix<T> BL, BR;
        Matrix<T> CTL, CTR,
                  CBL, CBR;

        const unsigned half = C.Height() / 2;
        LockedPartitionRight( A, AL, AR, half );
        LockedPartitionRight( B, BL, BR, half );
        PartitionDownDiagonal
        ( C, CTL, CTR,
             CBL, CBR, half );

        if( uplo == LOWER )
            basic::Gemm( orientationOfA, NORMAL, alpha, AR, BL, beta, CBL );
        else
            basic::Gemm( orientationOfA, NORMAL, alpha, AL, BR, beta, CTR );

        // Recurse
        basic::internal::TrrkTN
        ( uplo, orientationOfA, alpha, AL, BL, beta, CTL );
        basic::internal::TrrkTN
        ( uplo, orientationOfA, alpha, AR, BR, beta, CBR );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Distributed C := alpha A^{T/H} B + beta C
template<typename T>
inline void
elemental::basic::internal::LocalTrrk
( UpperOrLower uplo,
  Orientation orientationOfA,
  T alpha, const DistMatrix<T,STAR,MC>& A,
           const DistMatrix<T,STAR,MR>& B,
  T beta,        DistMatrix<T,MC,  MR>& C )
{
    using namespace trrk_util;
#ifndef RELEASE
    PushCallStack("basic::internal::LocalTrrk");
    CheckInput( orientationOfA, A, B, C );
#endif
    const Grid& g = C.Grid();

    if( C.Height() < g.Width()*LocalTrrkBlocksize<T>() )
    {
        LocalTrrkKernel( uplo, orientationOfA, alpha, A, B, beta, C );
    }
    else
    {
        // Split C in four roughly equal pieces, perform a large gemm on corner
        // and recurse on CTL and CBR.
        DistMatrix<T,STAR,MC> AL(g), AR(g);
        DistMatrix<T,STAR,MR> BL(g), BR(g);
        DistMatrix<T,MC,MR> CTL(g), CTR(g),
                            CBL(g), CBR(g);

        const unsigned half = C.Height() / 2;
        LockedPartitionRight( A, AL, AR, half );
        LockedPartitionRight( B, BL, BR, half );
        PartitionDownDiagonal
        ( C, CTL, CTR,
             CBL, CBR, half );

        if( uplo == LOWER )
        { 
            basic::internal::LocalGemm
            ( orientationOfA, NORMAL, alpha, AR, BL, beta, CBL );
        }
        else
        {
            basic::internal::LocalGemm
            ( orientationOfA, NORMAL, alpha, AL, BR, beta, CTR );
        }

        // Recurse
        basic::internal::LocalTrrk
        ( uplo, orientationOfA, alpha, AL, BL, beta, CTL );
        basic::internal::LocalTrrk
        ( uplo, orientationOfA, alpha, AR, BR, beta, CBR );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Local C := alpha A^{T/H} B^{T/H} + beta C
template<typename T>
inline void
elemental::basic::internal::TrrkTT
( UpperOrLower uplo,
  Orientation orientationOfA, Orientation orientationOfB,
  T alpha, const Matrix<T>& A, const Matrix<T>& B,
  T beta,        Matrix<T>& C )
{
    using namespace trrk_util;
#ifndef RELEASE
    PushCallStack("basic::internal::TrrkTT");
    CheckInputTT( orientationOfA, orientationOfB, A, B, C );
#endif
    if( C.Height() < LocalTrrkBlocksize<T>() )
    {
        TrrkKernelTT
        ( uplo, orientationOfA, orientationOfB, alpha, A, B, beta, C );
    }
    else
    {
        // Split C in four roughly equal pieces, perform a large gemm on corner
        // and recurse on CTL and CBR.
        Matrix<T> AL, AR;
        Matrix<T> BT,
                  BB;
        Matrix<T> CTL, CTR,
                  CBL, CBR;

        const unsigned half = C.Height() / 2;
        LockedPartitionRight( A, AL, AR, half );
        LockedPartitionDown
        ( B, BT, 
             BB, half );
        PartitionDownDiagonal
        ( C, CTL, CTR,
             CBL, CBR, half );

        if( uplo == LOWER )
            basic::Gemm
            ( orientationOfA, orientationOfB, alpha, AR, BT, beta, CBL );
        else
            basic::Gemm
            ( orientationOfA, orientationOfB, alpha, AL, BB, beta, CTR );

        // Recurse
        basic::internal::TrrkTT
        ( uplo, orientationOfA, orientationOfB, alpha, AL, BT, beta, CTL );
        basic::internal::TrrkTT
        ( uplo, orientationOfA, orientationOfB, alpha, AR, BB, beta, CBR );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

// Distributed C := alpha A^{T/H} B^{T/H} + beta C
template<typename T>
inline void
elemental::basic::internal::LocalTrrk
( UpperOrLower uplo,
  Orientation orientationOfA, Orientation orientationOfB,
  T alpha, const DistMatrix<T,STAR,MC  >& A,
           const DistMatrix<T,MR,  STAR>& B,
  T beta,        DistMatrix<T,MC,  MR  >& C )
{
    using namespace trrk_util;
#ifndef RELEASE
    PushCallStack("basic::internal::LocalTrrk");
    CheckInput( orientationOfA, orientationOfB, A, B, C );
#endif
    const Grid& g = C.Grid();

    if( C.Height() < g.Width()*LocalTrrkBlocksize<T>() )
    {
        LocalTrrkKernel
        ( uplo, orientationOfA, orientationOfB, alpha, A, B, beta, C );
    }
    else
    {
        // Split C in four roughly equal pieces, perform a large gemm on corner
        // and recurse on CTL and CBR.
        DistMatrix<T,STAR,MC> AL(g), AR(g);
        DistMatrix<T,MR,STAR> BT(g), 
                              BB(g);
        DistMatrix<T,MC,MR> CTL(g), CTR(g),
                            CBL(g), CBR(g);

        const unsigned half = C.Height() / 2;
        LockedPartitionRight( A, AL, AR, half );
        LockedPartitionDown
        ( B, BT, 
             BB, half );
        PartitionDownDiagonal
        ( C, CTL, CTR,
             CBL, CBR, half );

        if( uplo == LOWER )
        { 
            basic::internal::LocalGemm
            ( orientationOfA, orientationOfB, alpha, AR, BT, beta, CBL );
        }
        else
        {
            basic::internal::LocalGemm
            ( orientationOfA, orientationOfB, alpha, AL, BB, beta, CTR );
        }

        // Recurse
        basic::internal::LocalTrrk
        ( uplo, orientationOfA, orientationOfB, alpha, AL, BT, beta, CTL );
        basic::internal::LocalTrrk
        ( uplo, orientationOfA, orientationOfB, alpha, AR, BB, beta, CBR );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

