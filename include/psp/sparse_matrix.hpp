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
#ifndef PSP_SPARSE_MATRIX_HPP
#define PSP_SPARSE_MATRIX_HPP 1

namespace psp {

// A simple Compressed Sparse Row (CSR) data structure
template<typename Scalar>
struct SparseMatrix
{
    bool symmetric;
    int height, width;
    std::vector<Scalar> nonzeros;
    std::vector<int> columnIndices;
    std::vector<int> rowOffsets;

    void Print( const std::string& tag ) const;
    // TODO: Routines for outputting in Matlab and PETSc formats?
};

} // namespace psp

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar>
inline void
psp::SparseMatrix<Scalar>::Print( const std::string& tag ) const
{
    if( symmetric )
        std::cout << tag << "(symmetric)\n";
    else
        std::cout << tag << "\n";

    for( int i=0; i<height; ++i )
    {
        const int numCols = rowOffsets[i+1]-rowOffsets[i];
        const int rowOffset = rowOffsets[i];
        for( int k=0; k<numCols; ++k )
        {
            const int j = columnIndices[rowOffset+k];
            const Scalar alpha = nonzeros[rowOffset+k];
            std::cout << "(" << i << "," << j << "): " 
                      << WrapScalar(alpha) << "\n";
        }
    }
    std::cout << std::endl;
}

#endif // PSP_SPARSE_MATRIX_HPP
