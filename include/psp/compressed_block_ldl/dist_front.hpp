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
#ifndef PSP_DIST_FRONT_COMPRESSION_HPP
#define PSP_DIST_FRONT_COMPRESSION_HPP 1

namespace psp {

template<typename R>
void DistFrontCompression
( DistMatrix<Complex<R> >& A, 
  std::vector<DistMatrix<Complex<R> > >& greens, 
  std::vector<DistMatrix<Complex<R>,STAR,STAR> >& coefficients, 
  int depth, bool useQR=false );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename R> 
inline void DistFrontCompression
( DistMatrix<Complex<R> >& A, 
  std::vector<DistMatrix<Complex<R> > >& greens, 
  std::vector<DistMatrix<Complex<R>,STAR,STAR> >& coefficients,
  int depth, bool useQR=false )
{
#ifndef RELEASE
    PushCallStack("DistFrontCompression");
#endif
    // TODO
#ifndef RELEASE
    PopCallStack();
#endif
}

} // namespace psp

#endif // PSP_DIST_FRONT_COMPRESSION_HPP
