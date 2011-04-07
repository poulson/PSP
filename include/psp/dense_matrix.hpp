/*
   Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
   of a sweeping preconditioner for 3d Helmholtz equations.

   Copyright (C) 2011 Jack Poulson, Lexing Ying, and
   The University of Texas at Austin

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef PSP_DENSE_MATRIX_HPP
#define PSP_DENSE_MATRIX_HPP 1

namespace psp {

enum MatrixType { GENERAL, SYMMETRIC /*, HERMITIAN*/ };

// A basic dense matrix representation that is used for storing blocks 
// whose sources and targets are too close to represent as low rank
template<typename Scalar>
class DenseMatrix
{
    int _m; // height of matrix
    int _n; // width of matrix
    int _ldim; // leading dimension of matrix
    bool _viewing;
    bool _lockedView;
    std::vector<Scalar> _memory;
    Scalar* _buffer;
    const Scalar* _lockedBuffer;
    MatrixType _type;

public:
    DenseMatrix
    ( MatrixType type=GENERAL );
    DenseMatrix
    ( int m, int n, MatrixType type=GENERAL );
    DenseMatrix
    ( int m, int n, int ldim, MatrixType type=GENERAL );
    DenseMatrix
    ( Scalar* buffer, int m, int n, int ldim, MatrixType type=GENERAL );
    DenseMatrix
    ( const Scalar* lockedBuffer, int m, int n, int ldim, 
      MatrixType type=GENERAL );
    ~DenseMatrix();

    void SetType( MatrixType type );
    MatrixType Type() const;
    bool General() const;
    bool Symmetric() const;
    /* bool Hermitian() const; */

    int Height() const;
    int Width() const;
    int LDim() const;
    void Resize( int m, int n );
    void Resize( int m, int n, int ldim );

    void Set( int i, int j, Scalar value );
    Scalar Get( int i, int j ) const;

    Scalar* Buffer( int i=0, int j=0 );
    const Scalar* LockedBuffer( int i=0, int j=0 ) const;

    void View( DenseMatrix<Scalar>& A );
    void View( DenseMatrix<Scalar>& A, int i, int j, int m, int n );

    void LockedView( const DenseMatrix<Scalar>& A );
    void LockedView( const DenseMatrix<Scalar>& A, int i, int j, int m, int n );
};

} // namespace psp

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar>
inline
psp::DenseMatrix<Scalar>::DenseMatrix
( MatrixType type )
: _m(0), _n(0), _ldim(1), 
  _viewing(false), _lockedView(false),
  _memory(), _buffer(0), _lockedBuffer(0),
  _type(type)
{ }

template<typename Scalar>
inline
psp::DenseMatrix<Scalar>::DenseMatrix
( int m, int n, MatrixType type )
: _m(m), _n(n), _ldim(std::max(m,1)),
  _viewing(false), _lockedView(false),
  _memory(_ldim*_n), _buffer(&_memory[0]), _lockedBuffer(0),
  _type(type)
{
#ifndef RELEASE
    if( m < 0 || n < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( type == SYMMETRIC && m != n )
        throw std::logic_error("Symmetric matrices must be square");
#endif
}

template<typename Scalar>
inline 
psp::DenseMatrix<Scalar>::DenseMatrix
( int m, int n, int ldim, MatrixType type )
: _m(m), _n(n), _ldim(ldim), 
  _viewing(false), _lockedView(false),
  _memory(_ldim*_n), _buffer(&_memory[0]), _lockedBuffer(0),
  _type(type)
{
#ifndef RELEASE
    if( m < 0 || n < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( type == SYMMETRIC && m != n )
        throw std::logic_error("Symmetric matrices must be square");
    if( ldim <= 0 )
        throw std::logic_error("Leading dimensions must be positive");
#endif
}

template<typename Scalar>
inline 
psp::DenseMatrix<Scalar>::DenseMatrix
( Scalar* buffer, int m, int n, int ldim, MatrixType type )
: _m(m), _n(n), _ldim(ldim), 
  _viewing(true), _lockedView(false),
  _memory(), _buffer(buffer), _lockedBuffer(0),
  _type(type)
{
#ifndef RELEASE
    if( m < 0 || n < 0 )
        throw std::logic_erorr("Height and width must be non-negative");
    if( type == SYMMETRIC && m != n )
        throw std::logic_error("Symmetric matrices must be square");
    if( ldim <= 0 )
        throw std::logic_error("Leading dimensions must be positive");
#endif
}

template<typename Scalar>
inline 
psp::DenseMatrix<Scalar>::DenseMatrix
( const Scalar* lockedBuffer, int m, int n, int ldim, MatrixType type )
: _m(m), _n(n), _ldim(ldim),
  _viewing(true), _lockedView(true),
  _memory(), _buffer(0), _lockedBuffer(lockedBuffer),
  _type(type)
{
#ifndef RELEASE
    if( m < 0 || n < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( type == SYMMETRIC && m != n )
        throw std::logic_error("Symmetric matrices must be square");
    if( ldim <= 0 )
        throw std::logic_error("Leading dimensions must be positive");
#endif
}

template<typename Scalar>
inline 
psp::DenseMatrix<Scalar>::~DenseMatrix()
{ }

template<typename Scalar>
inline void
psp::DenseMatrix<Scalar>::SetType( MatrixType type )
{
#ifndef RELEASE
    if( type == SYMMETRIC && _m != _n )
        throw std::logic_error("Symmetric matrices must be square");
#endif
    _type = type;
}

template<typename Scalar>
inline psp::MatrixType
psp::DenseMatrix<Scalar>::Type() const
{
    return _type;
}

template<typename Scalar>
inline bool
psp::DenseMatrix<Scalar>::General() const
{
    return _type == GENERAL;
}

template<typename Scalar>
inline bool
psp::DenseMatrix<Scalar>::Symmetric() const
{
    return _type == SYMMETRIC;
}

/*
template<typename Scalar>
inline bool
psp::DenseMatrix<Scalar>::Hermitian() const
{
    return _type == HERMITIAN;
}
*/

template<typename Scalar>
inline int
psp::DenseMatrix<Scalar>::Height() const
{
    return _m;
}

template<typename Scalar>
inline int
psp::DenseMatrix<Scalar>::Width() const
{
    return _n;
}

template<typename Scalar>
inline int
psp::DenseMatrix<Scalar>::LDim() const
{
    return _ldim;
}

template<typename Scalar>
inline void
psp::DenseMatrix<Scalar>::Resize( int m, int n )
{
#ifndef RELEASE
    if( _viewing )
        throw std::logic_error("Cannot resize views");
    if( m < 0 || n < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( _type == SYMMETRIC && m != n )
        throw std::logic_error("Destroyed symmetry of symmetric matrix");
#endif
    if( m > _m )
    {
        // We cannot trivially preserve the old contents
        _ldim = std::max( m, 1 );
    }
    _m = m;
    _n = n;
    _memory.resize( _ldim*n );
    _buffer = &_memory[0];
}

template<typename Scalar>
inline void
psp::DenseMatrix<Scalar>::Resize( int m, int n, int ldim )
{
#ifndef RELEASE
    if( _viewing )
        throw std::logic_error("Cannot resize views");
    if( m < 0 || n < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( ldim < m || ldim < 0 )
        throw std::logic_error("LDim must be positive and >= the height");
    if( _type == SYMMETRIC && m != n )
        throw std::logic_error("Destroyed symmetry of symmetric matrix");
#endif
    _m = m;
    _n = n;
    _ldim = ldim;
    _memory.resize( ldim*n );
    _buffer = &_memory[0];
}

template<typename Scalar>
inline void
psp::DenseMatrix<Scalar>::Set( int i, int j, Scalar value )
{
#ifndef RELEASE
    if( _lockedView )
        throw std::logic_error("Cannot change data in a locked view");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i >= m || j >= n )
        throw std::logic_error("Indices are out of bound");
    if( _type == SYMMETRIC && j > i )
        throw std::logic_error("Setting upper entry from symmetric matrix");
#endif
    _buffer[i+j*_ldim] = value;
}

template<typename Scalar>
inline Scalar
psp::DenseMatrix<Scalar>::Get( int i, int j ) const
{
#ifndef RELEASE
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i >= m || j >= n )
        throw std::logic_error("Indices are out of bound");
    if( _type == SYMMETRIC && j > i )
        throw std::logic_error("Retrieving upper entry from symmetric matrix");
#endif
    if( _lockedView )
        return _lockedBuffer[i+j*_ldim];
    else
        return _buffer[i+j*_ldim];
}

template<typename Scalar>
inline Scalar*
psp::DenseMatrix<Scalar>::Buffer( int i, int j )
{
#ifndef RELEASE
    if( _lockedView )
        throw std::logic_error("Cannot modify the buffer from a locked view");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i >= m || j >= n )
        throw std::logic_error("Indices are out of bound");
#endif
    return _buffer;
}

template<typename Scalar>
inline const Scalar*
psp::DenseMatrix<Scalar>::LockedBuffer( int i, int j ) const
{
#ifndef RELEASE
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i >= m || j >= n )
        throw std::logic_error("Indices are out of bound");
#endif
    if( _lockedView )
        return _lockedBuffer;
    else
        return _buffer;
}

template<typename Scalar>
inline void
psp::DenseMatrix<Scalar>::View( DenseMatrix<Scalar>& A )
{
    _m = A.Height();
    _n = A.Width();
    _ldim = A.LDim();
    _viewing = true;
    _lockedView = false;
    _buffer = A.Buffer();
    _type = A.Type();
}

template<typename Scalar>
inline void
psp::DenseMatrix<Scalar>::View
( DenseMatrix<Scalar>& A, int i, int j, int m, int n )
{
#ifndef RELEASE
    if( A.Type() == SYMMETRIC && i != j || m != n )
        throw std::logic_error("Invalid submatrix of symmetric matrix");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i+m > A.Height() || j+n > A.Width() )
        throw std::logic_error("Submatrix out of bound");
#endif
    _m = m;
    _n = n;
    _ldim = A.LDim();
    _viewing = true;
    _lockedView = false;
    _buffer = A.Buffer(i,j);
    _type = A.Type();
}

template<typename Scalar>
inline void
psp::DenseMatrix<Scalar>::LockedView( const DenseMatrix<Scalar>& A )
{
    _m = A.Height();
    _n = A.Width();
    _ldim = A.LDim();
    _viewing = true;
    _lockedView = true;
    _lockedBuffer = A.LockedBuffer();
    _type = A.Type();
}

template<typename Scalar>
inline void
psp::DenseMatrix<Scalar>::LockedView
( const DenseMatrix<Scalar>& A, int i, int j, int m, int n )
{
#ifndef RELEASE
    if( A.Type() == SYMMETRIC && i != j || m != n )
        throw std::logic_error("Invalid submatrix of symmetric matrix");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i+m > A.Height() || j+n > A.Width() )
        throw std::logic_error("Submatrix out of bound");
#endif
    _m = m;
    _n = n;
    _ldim = A.LDim();
    _viewing = true;
    _lockedView = true;
    _lockedBuffer = A.LockedBuffer(i,j);
    _type = A.Type();
}

#endif // PSP_DENSE_MATRIX_HPP
