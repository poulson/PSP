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
    Write( head, H._height );
    Write( head, H._width );
    Write( head, H._numLevels );
    Write( head, H._maxRank );
    Write( head, H._sourceOffset );
    Write( head, H._targetOffset );
    Write( head, H._symmetric );
    Write( head, H._stronglyAdmissible );

    // Write out the Quasi2dHMatrix-specific information
    Write( head, H._xSizeSource );
    Write( head, H._xSizeTarget );
    Write( head, H._ySizeSource );
    Write( head, H._ySizeTarget );
    Write( head, H._zSize );
    Write( head, H._xSource );
    Write( head, H._xTarget );
    Write( head, H._ySource );
    Write( head, H._yTarget );

    const Shell& shell = H._shell;
    Write( head, shell.type );
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
        Write( head, r );

        // Write out U
        for( int j=0; j<r; ++j )
            Write( head, U.LockedBuffer(0,j), m );

        // Write out V
        for( int j=0; j<r; ++j )
            Write( head, V.LockedBuffer(0,j), n );

        break;
    }
    case DENSE:
    {
        const DenseMatrix<Scalar>& D = *shell.data.D;
        const int m = D.Height();
        const int n = D.Width();
        const MatrixType type = D.Type();

        // Write out the matrix type and data
        Write( head, type );
        if( type == GENERAL )
            for( int j=0; j<n; ++j )
                Write( head, D.LockedBuffer(0,j), m );
        else
            for( int j=0; j<n; ++j )
                Write( head, D.LockedBuffer(j,j), m-j );
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
    H._height             = Read<int>( head );
    H._width              = Read<int>( head );
    H._numLevels          = Read<int>( head );
    H._maxRank            = Read<int>( head );
    H._sourceOffset       = Read<int>( head );
    H._targetOffset       = Read<int>( head );
    H._symmetric          = Read<bool>( head );
    H._stronglyAdmissible = Read<bool>( head );

    // Set the Quasi2dHMatrix-specific information
    H._xSizeSource = Read<int>( head );
    H._xSizeTarget = Read<int>( head );
    H._ySizeSource = Read<int>( head );
    H._ySizeTarget = Read<int>( head );
    H._zSize       = Read<int>( head );
    H._xSource     = Read<int>( head );
    H._xTarget     = Read<int>( head );
    H._ySource     = Read<int>( head );
    H._yTarget     = Read<int>( head );

    // If data has been allocated, delete it
    Shell& shell = H._shell;
    shell.Clear();

    // Create this layer of the H-matrix from the packed information
    shell.type = Read<ShellType>( head );
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
        const int r = Read<int>( head );
        U.SetType( GENERAL ); U.Resize( m, r );
        V.SetType( GENERAL ); V.Resize( n, r );

        // Read in U
        for( int j=0; j<r; ++j )
            Read( U.Buffer(0,j), head, m );

        // Read in V
        for( int j=0; j<r; ++j )
            Read( V.Buffer(0,j), head, n );

        break;
    }
    case DENSE:
        shell.data.D = new DenseMatrix<Scalar>;
        DenseMatrix<Scalar>& D = *shell.data.D;
        const int m = H._height;
        const int n = H._width;

        const MatrixType type = Read<MatrixType>( head );
        D.SetType( type ); 
        D.Resize( m, n );

        // Read in the matrix
        if( type == GENERAL )
            for( int j=0; j<n; ++j )
                Read( D.Buffer(0,j), head, m );
        else
            for( int j=0; j<n; ++j )
                Read( D.Buffer(j,j), head, m-j );
        break;
    }
}

