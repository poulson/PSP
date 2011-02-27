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

   Additional permissions under GNU GPL version 3 section 7

   If you modify this Program, or any covered work, by linking or combining it
   with MUMPS and/or ParMetis (or modified versions of those libraries),
   containing parts covered by the terms of the respective licenses of MUMPS
   and ParMetis, the licensors of this Program grant you additional permission
   to convey the resulting work. {Corresponding Source for a non-source form of
   such a combination shall include the source code for the parts of MUMPS and
   ParMetis used as well as that of the covered work.}
*/
#include "psp.hpp"
#include <stdexcept>

// Return the basic information about our finite different discretization.
template<typename R>
void
psp::FiniteDiffInfo
( MPI_Comm comm, const psp::FiniteDiffControl<R>& control, 
  int& firstLocalVertex, int& numLocalVertices, int& numVertices, 
  int& numLocalNonzeros, int& numNonzeros )
{
    int rank, numProcesses;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses );

    numVertices = control.nx*control.ny*control.nz;
    const int vtxChunkSize = 
        static_cast<int>(static_cast<double>(numVertices)/numProcesses);
    firstLocalVertex = vtxChunkSize*rank;
    numLocalVertices = 
        ( rank==numProcesses-1 ? numVertices-firstLocalVertex : vtxChunkSize );

    if( control.stencil == psp::SEVEN_POINT )
    {
        numLocalNonzeros = 0;
        for( int vtx=firstLocalVertex; 
             vtx<firstLocalVertex+numLocalVertices; ++vtx )
        {
            const int i = vtx % control.nx;
            const int j = (vtx/control.nx) % control.ny;
            const int k = vtx/(control.nx*control.ny);

            ++numLocalNonzeros;
            if( i != 0 ) ++numLocalNonzeros;
            if( j != 0 ) ++numLocalNonzeros;
            if( k != 0 ) ++numLocalNonzeros;
        }
    }
    else
    {
        throw std::logic_error("Only 7-point stencils are currently supported");
    }

    MPI_Allreduce
    ( &numLocalNonzeros, &numNonzeros, 1, MPI_INT, MPI_SUM, comm );
}

// Return the 1-indexed row and column indices of each entry of the
// (x,y,z)-ordered finite-diff discretization.
//
// This should almost certainly only be called from the host process 
// just before the MUMPS analysis. Try to free the memory for the 
// row and col indices ASAP.
template<typename R>
void
psp::FiniteDiffConnectivity
( const psp::FiniteDiffControl<R>& control, const int numNonzeros,
  std::vector<int>& rowIndices, std::vector<int>& colIndices )
{
    rowIndices.resize( numNonzeros );    
    colIndices.resize( numNonzeros );

    const int numVertices = control.nx*control.ny*control.nz;

    if( control.stencil == psp::SEVEN_POINT )
    {
        int z = 0;
        for( int vtx=0; vtx<numVertices; ++vtx )
        {
            const int i = vtx % control.nx;
            const int j = (vtx/control.nx) % control.ny;
            const int k = vtx/(control.nx*control.ny);

            // Add the connection to ourself
            rowIndices[z] = vtx+1;
            colIndices[z] = vtx+1;
            ++z;

            // Handle the possible connection in the x direction
            if( i != 0 )
            {
                // Add the connection to (i-1,j,k)
                rowIndices[z] = vtx+1;
                colIndices[z] = vtx;
                ++z;
            }

            // Handle the possible connection in the y direction
            if( j != 0 )
            {
                // Add the connection to (i,j-1,k)
                rowIndices[z] = vtx+1;
                colIndices[z] = vtx+1-control.nx;
                ++z;
            }

            // Handle the possible connection in the z direction
            if( k != 0 )
            {
                // Add the connection to (i,j,k-1)
                rowIndices[z] = vtx+1;
                colIndices[z] = vtx+1-(control.nx*control.ny);
                ++z;
            }
        }
    }
    else
    {
        throw std::logic_error("Only 7-point stencils are currently supported");
    }
}

template struct psp::FiniteDiffControl<float>;
template struct psp::FiniteDiffControl<double>;

template void psp::FiniteDiffInfo
( MPI_Comm comm, const psp::FiniteDiffControl<float>& control, 
  int& firstLocalVertex, int& numLocalVertices, int& numVertices, 
  int& numLocalNonzeros, int& numNonzeros );
template void psp::FiniteDiffInfo
( MPI_Comm comm, const psp::FiniteDiffControl<double>& control, 
  int& firstLocalVertex, int& numLocalVertices, int& numVertices, 
  int& numLocalNonzeros, int& numNonzeros );

template void psp::FiniteDiffConnectivity
( const psp::FiniteDiffControl<float>& control, const int numNonzeros,
  std::vector<int>& rowIndices, std::vector<int>& colIndices );
template void psp::FiniteDiffConnectivity
( const psp::FiniteDiffControl<double>& control, const int numNonzeros,
  std::vector<int>& rowIndices, std::vector<int>& colIndices );

