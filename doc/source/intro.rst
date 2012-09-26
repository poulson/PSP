Introduction
************

Overview
========
Parallel Sweeping Preconditioner (PSP) is a prototype implementation of a 
moving-PML preconditioner for large 3D Helmholtz equations with at least one 
PML boundary condition. So far, the main focus has been on its application to
seismic exploration, but the preconditioner works well for most velocity 
models where there are no large-scale internal resonances (e.g., large cavities
are known to be problematic).  

The main motivation behind releasing this software in its current state is 
reproducibility. While the current discretization uses simple seven-point 
finite-difference stencils, which are known to suffer from large amounts of 
pollution when using a fixed number of grid points per wavelength for 
problems spanning many wavelengths, the performance of the preconditioner 
should be qualitatively similar to more sophisticated discretizations 
(e.g., spectral elements, which will be added soon). 

.. note::

   To emphasize the above point about the preconditioner being meant for 
   *large* problems, roughly speaking, if the domain is smaller than about 
   50 or 60 grid points in each direction, then it will be more practical to 
   simply use a sparse-direct solver, ILU preconditioner, or shifted-Laplacian
   preconditioner.

An under-review preprint of the first publication 
`is available here <http://www.ices.utexas.edu/~poulson/pubs/sisc12-sweeping.pdf>`_.

Dependencies
============
PSP is built on top of `Clique <http://github.com/poulson/Clique>`_, which 
is in turn based on `Elemental <http://code.google.com/p/elemental>`_ and 
`ParMETIS <http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview/>`_
(which are both distributed with Clique).

License and copyright
=====================
All files in PSP (other than Elemental and ParMETIS) are made available under
`GPLv3 <http://www.opensource.org/licenses/gpl-3.0>`_. The vast majority of 
files contain the following copyright notice::
    
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
 
Again, Elemental and ParMETIS are not distributed under the GPL; Elemental is 
distributed under the more permissive 
`New BSD License <http://www.opensource.org/licenses/bsd-license.php>`_,
and ParMETIS is distributed under the following license::
    
    The ParMETIS/METIS package is copyrighted by the Regents of the
    University of Minnesota. It can be freely used for educational and
    research purposes by non-profit institutions and US government
    agencies only. Other organizations are allowed to use ParMETIS/METIS
    only for evaluation purposes, and any further uses will require prior
    approval. The software may not be sold or redistributed without prior
    approval. One may make copies of the software for their use provided
    that the copies, are not sold or distributed, are used under the same
    terms and conditions.
     
    As unestablished research software, this code is provided on an
    ``as is'' basis without warranty of any kind, either expressed or
    implied. The downloading, or executing any part of this software
    constitutes an implicit agreement to these terms. These terms and
    conditions are subject to change at any time without prior notice.

