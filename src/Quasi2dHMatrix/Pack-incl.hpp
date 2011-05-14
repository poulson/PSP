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

//----------------------------------------------------------------------------//
// Private static routines                                                    //
//----------------------------------------------------------------------------//
template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::PackedSizeRecursion
( std::size_t& packedSize, const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    // Make space for the abstract H-matrix information
    packedSize += 6*sizeof(int) + 2*sizeof(bool);

    // Quasi2dHMatrix-specific information
    packedSize += 9*sizeof(int);

    const Shell& shell = H._shell;
    packedSize += sizeof(ShellType);
    switch( shell.type )
    {
    case NODE:
        for( int i=0; i<16; ++i )
            PackedSizeRecursion( packedSize, *shell.data.N->children[i] );
        break;
    case NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
        {
            PackedSizeRecursion
            ( packedSize, *shell.data.NS->children[i] );
        }
        break;
    case LOW_RANK:
    {
        const DenseMatrix<Scalar>& U = shell.data.F->U;
        const DenseMatrix<Scalar>& V = shell.data.F->V;
        const int m = U.Height();
        const int n = V.Height();
        const int r = U.Width();

        // The height and width are already known, we just need the rank
        packedSize += sizeof(int);

        // Make space for U and V
        packedSize += (m+n)*r*sizeof(Scalar);

        break;
    }
    case DENSE:
    {
        const DenseMatrix<Scalar>& D = *shell.data.D;
        const int m = D.Height();
        const int n = D.Width();
        const MatrixType type = D.Type();

        // Make space for the matrix type and data
        packedSize += sizeof(MatrixType);
        if( type == GENERAL )
            packedSize += m*n*sizeof(Scalar);
        else
            packedSize += ((m*m+m)/2)*sizeof(Scalar);
        break;
    }
    }
}

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::PackRecursion
( byte*& head, const Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    // Write out the abstract H-matrix information
    *((int*)head)  = H._height;             head += sizeof(int);
    *((int*)head)  = H._width;              head += sizeof(int);
    *((int*)head)  = H._numLevels;          head += sizeof(int);
    *((int*)head)  = H._maxRank;            head += sizeof(int);
    *((int*)head)  = H._sourceOffset;       head += sizeof(int);
    *((int*)head)  = H._targetOffset;       head += sizeof(int);
    *((bool*)head) = H._symmetric;          head += sizeof(bool);
    *((bool*)head) = H._stronglyAdmissible; head += sizeof(bool);

    // Write out the Quasi2dHMatrix-specific information
    *((int*)head) = H._xSizeSource; head += sizeof(int);
    *((int*)head) = H._xSizeTarget; head += sizeof(int);
    *((int*)head) = H._ySizeSource; head += sizeof(int);
    *((int*)head) = H._ySizeTarget; head += sizeof(int);
    *((int*)head) = H._zSize;       head += sizeof(int);
    *((int*)head) = H._xSource;     head += sizeof(int);
    *((int*)head) = H._xTarget;     head += sizeof(int);
    *((int*)head) = H._ySource;     head += sizeof(int);
    *((int*)head) = H._yTarget;     head += sizeof(int);

    const Shell& shell = H._shell;
    *((ShellType*)head) = shell.type; head += sizeof(ShellType);
    switch( shell.type )
    {
    case NODE:
        for( int i=0; i<16; ++i )
            PackRecursion( head, *shell.data.N->children[i] );
        break;
    case NODE_SYMMETRIC:
        for( int i=0; i<10; ++i )
            PackRecursion( head, *shell.data.NS->children[i] );
        break;
    case LOW_RANK:
    {
        const DenseMatrix<Scalar>& U = shell.data.F->U;
        const DenseMatrix<Scalar>& V = shell.data.F->V;
        const int m = U.Height();
        const int n = V.Height();
        const int r = U.Width();

        // Write out the rank
        *((int*)head) = r; head += sizeof(int);

        // Write out U
        for( int j=0; j<r; ++j )
        {
            std::memcpy( head, U.LockedBuffer(0,j), m*sizeof(Scalar) );
            head += m*sizeof(Scalar);
        }

        // Write out V
        for( int j=0; j<r; ++j )
        {
            std::memcpy( head, V.LockedBuffer(0,j), n*sizeof(Scalar) );
            head += n*sizeof(Scalar);
        }

        break;
    }
    case DENSE:
    {
        const DenseMatrix<Scalar>& D = *shell.data.D;
        const int m = D.Height();
        const int n = D.Width();
        const MatrixType type = D.Type();

        // Write out the matrix type and data
        *((MatrixType*)head) = type; head += sizeof(MatrixType);
        if( type == GENERAL )
        {
            for( int j=0; j<n; ++j )
            {
                std::memcpy( head, D.LockedBuffer(0,j), m*sizeof(Scalar) );
                head += m*sizeof(Scalar);
            }
        }
        else
        {
            for( int j=0; j<n; ++j )
            {
                std::memcpy( head, D.LockedBuffer(j,j), (m-j)*sizeof(Scalar) );
                head += (m-j)*sizeof(Scalar);
            }
        }
        break;
    }
    }
}

//----------------------------------------------------------------------------//
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
psp::Quasi2dHMatrix<Scalar,Conjugated>::Quasi2dHMatrix
( const std::vector<byte>& packedHMatrix )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Quasi2dHMatrix");
#endif
    Unpack( packedHMatrix );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::Quasi2dHMatrix<Scalar,Conjugated>::PackedSize() const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::PackedSize");
#endif
    std::size_t packedSize = 0;
    PackedSizeRecursion( packedSize, *this );
#ifndef RELEASE
    PopCallStack();
#endif
    return packedSize;
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::Quasi2dHMatrix<Scalar,Conjugated>::Pack
( byte* packedHMatrix ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Pack");
#endif
    byte* head = packedHMatrix;
    PackRecursion( head, *this );
#ifndef RELEASE
    PopCallStack();
#endif
    return (head-packedHMatrix);
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::Quasi2dHMatrix<Scalar,Conjugated>::Pack
( std::vector<byte>& packedHMatrix ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Pack");
#endif
    // Create the storage and extract the buffer
    const std::size_t packedSize = PackedSize();
    packedHMatrix.resize( packedSize );
    byte* head = &packedHMatrix[0];

    PackRecursion( head, *this );
#ifndef RELEASE
    PopCallStack();
#endif
    return (head-&packedHMatrix[0]);
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::Quasi2dHMatrix<Scalar,Conjugated>::Unpack
( const byte* packedHMatrix )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Unpack");
#endif
    const byte* head = packedHMatrix;
    UnpackRecursion( head, *this );
#ifndef RELEASE
    PopCallStack();
#endif
    return (head-packedHMatrix);
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::Quasi2dHMatrix<Scalar,Conjugated>::Unpack
( const std::vector<byte>& packedHMatrix )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMatrix::Unpack");
#endif
    const byte* head = &packedHMatrix[0];
    UnpackRecursion( head, *this );
#ifndef RELEASE
    PopCallStack();
#endif
    return (head-&packedHMatrix[0]);
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMatrix<Scalar,Conjugated>::UnpackRecursion
( const byte*& head, Quasi2dHMatrix<Scalar,Conjugated>& H )
{
    // Set the abstract H-matrix data
    H._height             = *((int*)head);  head += sizeof(int);
    H._width              = *((int*)head);  head += sizeof(int);
    H._numLevels          = *((int*)head);  head += sizeof(int);
    H._maxRank            = *((int*)head);  head += sizeof(int);
    H._sourceOffset       = *((int*)head);  head += sizeof(int);
    H._targetOffset       = *((int*)head);  head += sizeof(int);
    H._symmetric          = *((bool*)head); head += sizeof(bool);
    H._stronglyAdmissible = *((bool*)head); head += sizeof(bool);

    // Set the Quasi2dHMatrix-specific information
    H._xSizeSource = *((int*)head); head += sizeof(int);
    H._xSizeTarget = *((int*)head); head += sizeof(int);
    H._ySizeSource = *((int*)head); head += sizeof(int);
    H._ySizeTarget = *((int*)head); head += sizeof(int);
    H._zSize       = *((int*)head); head += sizeof(int);
    H._xSource     = *((int*)head); head += sizeof(int);
    H._xTarget     = *((int*)head); head += sizeof(int);
    H._ySource     = *((int*)head); head += sizeof(int);
    H._yTarget     = *((int*)head); head += sizeof(int);

    // If data has been allocated, delete it
    Shell& shell = H._shell;
    switch( shell.type )
    {
    case NODE:           delete shell.data.N;          break;
    case NODE_SYMMETRIC: delete shell.data.NS; break;
    case LOW_RANK:       delete shell.data.F;             break;
    case DENSE:          delete shell.data.D;             break;
    }

    // Create this layer of the H-matrix from the packed information
    shell.type = *((ShellType*)head); head += sizeof(ShellType);
    switch( shell.type )
    {
    case NODE:
    {
        shell.data.N = 
            new Node
            ( H._xSizeSource, H._xSizeTarget, 
              H._ySizeSource, H._ySizeTarget, H._zSize );
        Node& node = *shell.data.N;
        for( int i=0; i<16; ++i )
        {
            node.children[i] = new Quasi2dHMatrix<Scalar,Conjugated>;
            UnpackRecursion( head, *node.children[i] );
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        shell.data.NS = 
            new NodeSymmetric( H._xSizeSource, H._ySizeSource, H._zSize );
        NodeSymmetric& node = *shell.data.NS;
        for( int i=0; i<10; ++i )
        {
            node.children[i] = new Quasi2dHMatrix<Scalar,Conjugated>;
            UnpackRecursion( head, *node.children[i] );
        }
        break;
    }
    case LOW_RANK:
    {
        shell.data.F = new LowRankMatrix<Scalar,Conjugated>;
        DenseMatrix<Scalar>& U = shell.data.F->U;
        DenseMatrix<Scalar>& V = shell.data.F->V;
        const int m = H._height;
        const int n = H._width;

        // Read in the matrix rank
        const int r = *((int*)head); head += sizeof(int);
        U.SetType( GENERAL ); U.Resize( m, r );
        V.SetType( GENERAL ); V.Resize( n, r );

        // Read in U
        for( int j=0; j<r; ++j )
        {
            std::memcpy( U.Buffer(0,j), head, m*sizeof(Scalar) );
            head += m*sizeof(Scalar);
        }

        // Read in V
        for( int j=0; j<r; ++j )
        {
            std::memcpy( V.Buffer(0,j), head, n*sizeof(Scalar) );
            head += n*sizeof(Scalar);
        }

        break;
    }
    case DENSE:
        shell.data.D = new DenseMatrix<Scalar>;
        DenseMatrix<Scalar>& D = *shell.data.D;
        const int m = H._height;
        const int n = H._width;

        const MatrixType type = *((MatrixType*)head); 
        head += sizeof(MatrixType);
        D.SetType( type ); 
        D.Resize( m, n );

        // Read in the matrix
        if( type == GENERAL )
        {
            for( int j=0; j<n; ++j )
            {
                std::memcpy( D.Buffer(0,j), head, m*sizeof(Scalar) );
                head += m*sizeof(Scalar);
            }
        }
        else
        {
            for( int j=0; j<n; ++j )
            {
                std::memcpy( D.Buffer(j,j), head, (m-j)*sizeof(Scalar) );
                head += (m-j)*sizeof(Scalar);
            }
        }
        break;
    }
}

