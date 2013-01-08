/*
   Copyright (c) 2009-2013, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/

namespace elem {

// Partial specialization to A[* ,MD].
// 
// The rows of these distributed matrices will be distributed like 
// "Matrix Diagonals" (MD). It is important to recognize that the diagonal
// of a sufficiently large distributed matrix is distributed amongst the 
// entire process grid if and only if the dimensions of the process grid
// are coprime.
template<typename T,typename Int>
class DistMatrix<T,STAR,MD,Int> : public AbstractDistMatrix<T,Int>
{
public:
    // Create a 0 x 0 distributed matrix
    DistMatrix( const elem::Grid& g=DefaultGrid() );

    // Create a height x width distributed matrix
    DistMatrix( Int height, Int width, const elem::Grid& g=DefaultGrid() );

    // Create a 0 x 0 distributed matrix with specified alignments
    DistMatrix
    ( bool constrainedRowAlignment,
      Int rowAlignment, const elem::Grid& g );

    // Create a height x width distributed matrix with specified alignments
    DistMatrix
    ( Int height, Int width, bool constrainedRowAlignment, Int rowAlignment,
      const elem::Grid& g );

    // Create a height x width distributed matrix with specified alignments
    // and leading dimension
    DistMatrix
    ( Int height, Int width, bool constrainedRowAlignment, Int rowAlignment,
      Int ldim, const elem::Grid& g );

    // View a constant distributed matrix's buffer
    DistMatrix
    ( Int height, Int width, Int rowAlignment,
      const T* buffer, Int ldim, const elem::Grid& g );

    // View a mutable distributed matrix's buffer
    DistMatrix
    ( Int height, Int width, Int rowAlignment,
      T* buffer, Int ldim, const elem::Grid& g );

    // Create a copy of distributed matrix A
    template<Distribution U,Distribution V>
    DistMatrix( const DistMatrix<T,U,V,Int>& A );

    ~DistMatrix();

    const DistMatrix<T,STAR,MD,Int>& 
    operator=( const DistMatrix<T,MC,MR,Int>& A );

    const DistMatrix<T,STAR,MD,Int>& 
    operator=( const DistMatrix<T,MC,STAR,Int>& A );

    const DistMatrix<T,STAR,MD,Int>& 
    operator=( const DistMatrix<T,STAR,MR,Int>& A );

    const DistMatrix<T,STAR,MD,Int>& 
    operator=( const DistMatrix<T,MD,STAR,Int>& A );

    const DistMatrix<T,STAR,MD,Int>& 
    operator=( const DistMatrix<T,STAR,MD,Int>& A );

    const DistMatrix<T,STAR,MD,Int>& 
    operator=( const DistMatrix<T,MR,MC,Int>& A );

    const DistMatrix<T,STAR,MD,Int>& 
    operator=( const DistMatrix<T,MR,STAR,Int>& A );

    const DistMatrix<T,STAR,MD,Int>& 
    operator=( const DistMatrix<T,STAR,MC,Int>& A );

    const DistMatrix<T,STAR,MD,Int>& 
    operator=( const DistMatrix<T,VC,STAR,Int>& A );

    const DistMatrix<T,STAR,MD,Int>& 
    operator=( const DistMatrix<T,STAR,VC,Int>& A );

    const DistMatrix<T,STAR,MD,Int>& 
    operator=( const DistMatrix<T,VR,STAR,Int>& A );

    const DistMatrix<T,STAR,MD,Int>& 
    operator=( const DistMatrix<T,STAR,VR,Int>& A );

    const DistMatrix<T,STAR,MD,Int>& 
    operator=( const DistMatrix<T,STAR,STAR,Int>& A );

    //------------------------------------------------------------------------//
    // Fulfillments of abstract virtual func's from AbstractDistMatrix        //
    //------------------------------------------------------------------------//

    //
    // Non-collective routines
    //

    virtual Int ColStride() const;
    virtual Int RowStride() const;
    virtual Int ColRank() const;
    virtual Int RowRank() const;

    virtual bool Participating() const;

    //
    // Collective routines
    //

    virtual void SetGrid( const elem::Grid& grid );

    virtual T Get( Int i, Int j ) const;
    virtual void Set( Int i, Int j, T alpha );
    virtual void Update( Int i, Int j, T alpha );

    virtual void ResizeTo( Int height, Int width );

    //
    // Though the following routines are meant for complex data, all but two
    // logically applies to real data.
    //

    virtual typename Base<T>::type GetRealPart( Int i, Int j ) const;
    virtual typename Base<T>::type GetImagPart( Int i, Int j ) const;
    virtual void SetRealPart( Int i, Int j, typename Base<T>::type u );
    // Only valid for complex data
    virtual void SetImagPart( Int i, Int j, typename Base<T>::type u );
    virtual void UpdateRealPart( Int i, Int j, typename Base<T>::type u );
    // Only valid for complex data
    virtual void UpdateImagPart( Int i, Int j, typename Base<T>::type u );

    //------------------------------------------------------------------------//
    // Routines specific to [* ,MD] distribution                              //
    //------------------------------------------------------------------------//

    //
    // Collective routines
    //

    // Set the alignments
    void Align( Int rowAlignment );
    void AlignRows( Int rowAlignment );

    // Aligns all of our DistMatrix's distributions that match a distribution
    // of the argument DistMatrix.
    template<typename S,typename N> 
    void AlignWith( const DistMatrix<S,MD,STAR,N>& A );
    template<typename S,typename N> 
    void AlignWith( const DistMatrix<S,STAR,MD,N>& A );
    template<typename S,typename N> 
    void AlignWith( const DistMatrix<S,STAR,MC,N>& A ) {}
    template<typename S,typename N> 
    void AlignWith( const DistMatrix<S,STAR,MR,N>& A ) {}
    template<typename S,typename N> 
    void AlignWith( const DistMatrix<S,STAR,VC,N>& A ) {}
    template<typename S,typename N> 
    void AlignWith( const DistMatrix<S,STAR,VR,N>& A ) {}
    template<typename S,typename N> 
    void AlignWith( const DistMatrix<S,STAR,STAR,N>& A ) {}
    template<typename S,typename N> 
    void AlignWith( const DistMatrix<S,MC,STAR,N>& A ) {}
    template<typename S,typename N> 
    void AlignWith( const DistMatrix<S,MR,STAR,N>& A ) {}
    template<typename S,typename N> 
    void AlignWith( const DistMatrix<S,VC,STAR,N>& A ) {}
    template<typename S,typename N> 
    void AlignWith( const DistMatrix<S,VR,STAR,N>& A ) {}

    // Aligns our column distribution (i.e., Star) with the matching 
    // distribution of the argument. These are all no-ops and exist solely
    // to allow for templating over distribution parameters.
    template<typename S,typename N>
    void AlignColsWith( const DistMatrix<S,STAR,MC,N>& A ) {}
    template<typename S,typename N>
    void AlignColsWith( const DistMatrix<S,STAR,MR,N>& A ) {}
    template<typename S,typename N>
    void AlignColsWith( const DistMatrix<S,STAR,MD,N>& A ) {}
    template<typename S,typename N>
    void AlignColsWith( const DistMatrix<S,STAR,VC,N>& A ) {}
    template<typename S,typename N>
    void AlignColsWith( const DistMatrix<S,STAR,VR,N>& A ) {}
    template<typename S,typename N>
    void AlignColsWith( const DistMatrix<S,STAR,STAR,N>& A ) {}
    template<typename S,typename N>
    void AlignColsWith( const DistMatrix<S,MC,STAR,N>& A ) {}
    template<typename S,typename N>
    void AlignColsWith( const DistMatrix<S,MR,STAR,N>& A ) {}
    template<typename S,typename N>
    void AlignColsWith( const DistMatrix<S,MD,STAR,N>& A ) {}
    template<typename S,typename N>
    void AlignColsWith( const DistMatrix<S,VC,STAR,N>& A ) {}
    template<typename S,typename N>
    void AlignColsWith( const DistMatrix<S,VR,STAR,N>& A ) {}

    // Aligns our row distribution (i.e., MD) with the matching distribution
    // of the argument. 
    template<typename S,typename N> 
    void AlignRowsWith( const DistMatrix<S,MD,STAR,N>& A );
    template<typename S,typename N> 
    void AlignRowsWith( const DistMatrix<S,STAR,MD,N>& A );

    template<typename S,typename N> 
    bool AlignedWithDiagonal
    ( const DistMatrix<S,MC,MR,N>& A, Int offset=0 ) const;
    template<typename S,typename N> 
    bool AlignedWithDiagonal
    ( const DistMatrix<S,MR,MC,N>& A, Int offset=0 ) const;

    template<typename S,typename N> 
    void AlignWithDiagonal( const DistMatrix<S,MC,MR,N>& A, Int offset=0 );
    template<typename S,typename N> 
    void AlignWithDiagonal( const DistMatrix<S,MR,MC,N>& A, Int offset=0 );

    // (Immutable) view of a distributed matrix's buffer
    void Attach
    ( Int height, Int width, Int rowAlignment,
      T* buffer, Int ldim, const elem::Grid& grid );
    void LockedAttach
    ( Int height, Int width, Int rowAlignment,
      const T* buffer, Int ldim, const elem::Grid& grid );

    Int DiagPath() const;

private:
    Int diagPath_;
    virtual void PrintBase( std::ostream& os, const std::string msg="" ) const;

    template<typename S,Distribution U,Distribution V,typename Ord>
    friend class DistMatrix;
};

} // namespace elem
