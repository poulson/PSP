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
// Public non-static routines                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
psp::Quasi2dHMat<Scalar,Conjugated>::Quasi2dHMat
( const std::vector<byte>& packedHMat )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Quasi2dHMat");
#endif
    Unpack( packedHMat );
#ifndef RELEASE
    PopCallStack();
#endif
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::Quasi2dHMat<Scalar,Conjugated>::PackedSize() const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::PackedSize");
#endif
    std::size_t packedSize = 13*sizeof(int) + 2*sizeof(bool);
    PackedSizeRecursion( packedSize );
#ifndef RELEASE
    PopCallStack();
#endif
    return packedSize;
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::Quasi2dHMat<Scalar,Conjugated>::Pack
( byte* packedHMat ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Pack");
#endif
    byte* head = packedHMat;
    
    // Write the header information
    Write( head, _numLevels );
    Write( head, _maxRank );
    Write( head, _sourceOffset );
    Write( head, _targetOffset );
    Write( head, _symmetric );
    Write( head, _stronglyAdmissible );
    Write( head, _xSizeSource );
    Write( head, _xSizeTarget );
    Write( head, _ySizeSource );
    Write( head, _ySizeTarget );
    Write( head, _zSize );
    Write( head, _xSource );
    Write( head, _xTarget );
    Write( head, _ySource );
    Write( head, _yTarget );

    PackRecursion( head );
#ifndef RELEASE
    PopCallStack();
#endif
    return (head-packedHMat);
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::Quasi2dHMat<Scalar,Conjugated>::Pack
( std::vector<byte>& packedHMat ) const
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Pack");
#endif
    // Create the storage and extract the buffer
    const std::size_t packedSize = PackedSize();
    packedHMat.resize( packedSize );
    byte* head = &packedHMat[0];

    // Write the header information
    Write( head, _numLevels );
    Write( head, _maxRank );
    Write( head, _sourceOffset );
    Write( head, _targetOffset );
    Write( head, _symmetric );
    Write( head, _stronglyAdmissible );
    Write( head, _xSizeSource );
    Write( head, _xSizeTarget );
    Write( head, _ySizeSource );
    Write( head, _ySizeTarget );
    Write( head, _zSize );
    Write( head, _xSource );
    Write( head, _xTarget );
    Write( head, _ySource );
    Write( head, _yTarget );

    PackRecursion( head );
#ifndef RELEASE
    PopCallStack();
#endif
    return (head-&packedHMat[0]);
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::Quasi2dHMat<Scalar,Conjugated>::Unpack
( const byte* packedHMat )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Unpack");
#endif
    const byte* head = packedHMat;
    
    // Unpack the top-level header information
    _numLevels          = Read<int>( head );
    _maxRank            = Read<int>( head );
    _sourceOffset       = Read<int>( head );
    _targetOffset       = Read<int>( head );
    _symmetric          = Read<bool>( head );
    _stronglyAdmissible = Read<bool>( head );
    _xSizeSource        = Read<int>( head );
    _xSizeTarget        = Read<int>( head );
    _ySizeSource        = Read<int>( head );
    _ySizeTarget        = Read<int>( head );
    _zSize              = Read<int>( head );
    _xSource            = Read<int>( head );
    _xTarget            = Read<int>( head );
    _ySource            = Read<int>( head );
    _yTarget            = Read<int>( head );

    UnpackRecursion( head );
#ifndef RELEASE
    PopCallStack();
#endif
    return (head-packedHMat);
}

template<typename Scalar,bool Conjugated>
std::size_t
psp::Quasi2dHMat<Scalar,Conjugated>::Unpack
( const std::vector<byte>& packedHMat )
{
#ifndef RELEASE
    PushCallStack("Quasi2dHMat::Unpack");
#endif
    const byte* head = &packedHMat[0];

    // Unpack the top-level header information
    _numLevels          = Read<int>( head );
    _maxRank            = Read<int>( head );
    _sourceOffset       = Read<int>( head );
    _targetOffset       = Read<int>( head );
    _symmetric          = Read<bool>( head );
    _stronglyAdmissible = Read<bool>( head );
    _xSizeSource        = Read<int>( head );
    _xSizeTarget        = Read<int>( head );
    _ySizeSource        = Read<int>( head );
    _ySizeTarget        = Read<int>( head );
    _zSize              = Read<int>( head );
    _xSource            = Read<int>( head );
    _xTarget            = Read<int>( head );
    _ySource            = Read<int>( head );
    _yTarget            = Read<int>( head );
    
    UnpackRecursion( head );
#ifndef RELEASE
    PopCallStack();
#endif
    return (head-&packedHMat[0]);
}

//----------------------------------------------------------------------------//
// Private non-static routines                                                //
//----------------------------------------------------------------------------//

template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::PackedSizeRecursion
( std::size_t& packedSize ) const
{
    packedSize += sizeof(BlockType);
    switch( _block.type )
    {
    case NODE:
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                _block.data.N->Child(t,s).PackedSizeRecursion( packedSize );
        break;
    case NODE_SYMMETRIC:
        for( int t=0; t<4; ++t )
            for( int s=0; s<=t; ++s )
                _block.data.NS->Child(t,s).PackedSizeRecursion( packedSize );
        break;
    case LOW_RANK:
    {
        const Dense<Scalar>& U = _block.data.F->U;
        const Dense<Scalar>& V = _block.data.F->V;
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
        const Dense<Scalar>& D = *_block.data.D;
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
psp::Quasi2dHMat<Scalar,Conjugated>::PackRecursion
( byte*& head ) const
{
    Write( head, _block.type );
    switch( _block.type )
    {
    case NODE:
        for( int t=0; t<4; ++t )
            for( int s=0; s<4; ++s )
                _block.data.N->Child(t,s).PackRecursion( head );
        break;
    case NODE_SYMMETRIC:
        for( int t=0; t<4; ++t )
            for( int s=0; s<=t; ++s )
                _block.data.NS->Child(t,s).PackRecursion( head );
        break;
    case LOW_RANK:
    {
        const Dense<Scalar>& U = _block.data.F->U;
        const Dense<Scalar>& V = _block.data.F->V;
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
        const Dense<Scalar>& D = *_block.data.D;
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


template<typename Scalar,bool Conjugated>
void
psp::Quasi2dHMat<Scalar,Conjugated>::UnpackRecursion
( const byte*& head )
{
    _block.Clear();
    _block.type = Read<BlockType>( head );
    switch( _block.type )
    {
    case NODE:
    {
        _block.data.N = NewNode();
        Node& node = *_block.data.N;
        for( int t=0,tOffset=0; t<4; tOffset+=node.targetSizes[t],++t )
        {
            for( int s=0,sOffset=0; s<4; sOffset+=node.sourceSizes[s],++s )
            {
                node.children[s+4*t] = 
                    new Quasi2dHMat<Scalar,Conjugated>
                    ( _numLevels-1, _maxRank, _symmetric, _stronglyAdmissible,
                      node.xSourceSizes[s&1], node.xTargetSizes[t&1],
                      node.ySourceSizes[s/2], node.yTargetSizes[t/2],
                      _zSize,
                      2*_xSource+(s&1), 2*_xTarget+(t&1),
                      2*_ySource+(s/2), 2*_yTarget+(t/2),
                      sOffset+_sourceOffset, tOffset+_targetOffset );
                node.Child(t,s).UnpackRecursion( head );
            }
        }
        break;
    }
    case NODE_SYMMETRIC:
    {
        _block.data.NS = NewNodeSymmetric();
        NodeSymmetric& node = *_block.data.NS;
        int child = 0;
        for( int t=0,tOffset=0; t<4; tOffset+=node.sizes[t],++t )
        {
            for( int s=0,sOffset=0; s<=t; sOffset+=node.sizes[s],++s )
            {
                node.children[child++] =  
                    new Quasi2dHMat<Scalar,Conjugated>
                    ( _numLevels-1, _maxRank, _symmetric, _stronglyAdmissible,
                      node.xSizes[s&1], node.xSizes[t&1],
                      node.ySizes[s/2], node.ySizes[t/2],
                      _zSize,
                      2*_xSource+(s&1), 2*_xTarget+(t&1),
                      2*_ySource+(s/2), 2*_yTarget+(t/2),
                      sOffset+_targetOffset, tOffset+_targetOffset );
                node.Child(t,s).UnpackRecursion( head );
            }
        }
        break;
    }
    case LOW_RANK:
    {
        _block.data.F = new LowRank<Scalar,Conjugated>;
        Dense<Scalar>& U = _block.data.F->U;
        Dense<Scalar>& V = _block.data.F->V;
        const int m = Height();
        const int n = Width();

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
        _block.data.D = new Dense<Scalar>;
        Dense<Scalar>& D = *_block.data.D;
        const int m = Height();
        const int n = Width();

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

