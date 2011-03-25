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
#ifndef PSP_CONFIG_H
#define PSP_CONFIG_H 1

/* Basic variables */
#define PSP_VERSION_MAJOR @PSP_VERSION_MAJOR@
#define PSP_VERSION_MINOR @PSP_VERSION_MINOR@
#cmakedefine RELEASE
#cmakedefine BLAS_POST
#cmakedefine LAPACK_POST
#cmakedefine VIEW_MATRICES
#cmakedefine BUILT_PETSC_WITH_X11

#define RESTRICT @RESTRICT@

#endif /* PSP_CONFIG_H */
