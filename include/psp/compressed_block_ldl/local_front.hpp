/*
   Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
   of a sweeping preconditioner for 3d Helmholtz equations.

   Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and
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
#ifndef PSP_LOCAL_FRONT_COMPRESSION_HPP
#define PSP_LOCAL_FRONT_COMPRESSION_HPP 1

namespace psp {

template<typename R>
void LocalFrontCompression
( Matrix<Complex<R> >& A, 
  std::vector<Matrix<Complex<R> > >& greens, int depth );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename R> 
inline void LocalFrontCompression
( Matrix<Complex<R> >& A, 
  std::vector<Matrix<Complex<R> > >& greens, int depth )
{
#ifndef RELEASE
    PushCallStack("LocalFrontCompression");
#endif
    // TODO
#ifndef RELEASE
    PopCallStack();
#endif
}

// TODO: Sparse leaf-level B compression

} // namespace psp

#endif // PSP_LOCAL_FRONT_COMPRESSION_HPP