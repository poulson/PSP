The ``DistHelmholtz`` class
===========================
The :cpp:type:`DistHelmholtz\<R>` class is responsible for abstracting the 
formation and usage of the parallel sweeping preconditioner (where `R` is the 
underlying real datatype, usually ``double``). For simplicity, this 
implementation uses a 7-point finite-difference stencil over a three-dimensional
uniform grid. Without loss of generality (at least one face must have PML for 
the sweeping preconditioner to apply), PML is required on the bottom face
of the domain, and the sweeps proceed from the bottom to the top of the domain.
The other five boundary conditions may each be chosen to be either PML or 
zero Dirichlet.

.. cpp:type:: enum Boundary

   * PML
   * DIRICHLET

.. cpp:type:: struct Discretization<R>

   .. cpp:member:: R omega

      The frequency of the Helmholtz problem in 
      :math:`\frac{\mbox{rad}}{\mbox{sec}}`, where

      .. math::

         \left[-\Delta - \frac{\omega^2}{c^2(x)}\right]u = f(x),

      :math:`c(x)` is the spatially varying wave propagation speed (usually 
      in :math:`\frac{\mbox{km}}{\mbox{sec}}` for seismic problems), and 
      :math:`f(x)` is the forcing function (which is implicitly equal to the 
      physical forcing function scaled by the inverse of the spatially varying
      stiffness).

   .. cpp:member:: int nx

      The number of grid points in the :math:`X` direction.

   .. cpp:member:: int ny

      The number of grid points in the :math:`Y` direction.

   .. cpp:member:: int nz

      The number of grid points in the :math:`Z` direction.

   .. cpp:member:: R wx

      The physical length of the domain in the :math:`X` direction (usually in 
      kilometers for seismic problems).

   .. cpp:member:: R wy

      The physical length of the domain in the :math:`Y` direction.

   .. cpp:member:: R wz

      The physical length of the domain in the :math:`Z` direction.

   .. cpp:member:: Boundary frontBC

      Which boundary condition is to be applied on the front face of the 
      domain (i.e., :math:`x=0`).

   .. cpp:member:: Boundary backBC

      Which boundary condition is to be applied on the back face of the 
      domain (i.e., :math:`x=w_x`).

   .. cpp:member:: Boundary leftBC

      Which boundary condition is to be applied on the right face of the 
      domain (i.e., :math:`y=0`).

   .. cpp:member:: Boundary rightBC

      Which boundary condition is to be applied on the right face of the 
      domain (i.e., :math:`y=w_y`).

   .. cpp:member:: Boundary topBC

      Which boundary condition is to be applied to the top face of the 
      domain (i.e., :math:`z=0`).

   .. cpp:member:: int bx

      The number of grid points of PML to be used for each boundary condition
      in the :math:`X` direction.

   .. cpp:member:: int by

      The number of grid points of PML to be used for each boundary condition
      in the :math:`Y` direction.

   .. cpp:member:: int bz

      The number of grid points of PML to be used for each boundary condition
      in the :math:`Z` direction.

   .. cpp:member:: R sigmax

      The maximum imaginary value of the complex coordinate-stretching for 
      applying PML in the :math:`X` direction. As a rule of thumb, 
      :math:`1.5\, \mbox{diam}(\Omega)` is a decent first guess.

   .. cpp:member:: R sigmay

      The maximum imaginary value of the complex coordinate-stretching for 
      applying PML in the :math:`Y` direction. 

   .. cpp:member:: R sigmaz

      The maximum imaginary value of the complex coordinate-stretching for 
      applying PML in the :math:`Y` direction. 

   .. cpp:function:: Discretization( R frequency, int xSize, int ySize, int zSize, R xWidth, R yWidth, R zWidth )

      A constructor which assumes PML on all boundaries and attempts to select 
      decent choices for the PML sizes (5 grid points) and 
      coordinate-stretching magnitudes (:math:`1.5 \max\{w_x,w_y,w_z\}`).

   .. cpp:function:: Discretization( R frequency, int xSize, int ySize, int zSize, R xWidth, R yWidth, R zWidth, Boundary front, Boundary right, Boundary back, Boundary left, Boundary top )

      A constructor which attempts to select decent choices for the PML sizes 
      (5 grid points) and coordinate-stretching magnitudes 
      (:math:`1.5 \max\{w_x,w_y,w_z\}`).

   .. cpp:function:: Discretization( R frequency, int xSize, int ySize, int zSize, R xWidth, R yWidth, R zWidth, Boundary front, Boundary right, Boundary back, Boundary left, Boundary top, int xPMLSize, int yPMLSize, int zPMLSize )

      A constructor which attempts to select decent choices for the complex
      coordinate-stretching magnitudes (:math:`1.5 \max\{w_x,w_y,w_z\}`).

   .. cpp:function:: Discretization( R frequency, int xSize, int ySize, int zSize, R xWidth, R yWidth, R zWidth, Boundary front, Boundary right, Boundary back, Boundary left, Boundary top, int pmlSize, double sigma )

      A constructor which accepts a fixed value for the PML sizes and 
      coordinate-stretching magnitudes 
      (and applies each in all three directions).

.. cpp:type:: enum PanelScheme

   * CLIQUE_LDL_1D: Uses triangular solves with one-dimensional frontal distributions for each of the sparse-direct subdomain solves. This is significantly slower than the next option on parallel machines.
   * CLIQUE_LDL_SELINV_2D: Uses selective inversion (Raghavan et al.) for each of the subdomains in order to allow for fast subdomain solves. This is slightly less stable than ``CLIQUE_LDL_1D``, but it increases the performance of the subdomain solves by orders of magnitude on large numbers of processes (with a small penalty in the setup time).

.. cpp:type:: class DistHelmholtz<R>

   .. cpp:function:: DistHelmholtz( const Discretization<R>& disc, mpi::Comm comm, R damping=7.5, int numPlanesPerPanel=4, int cutoff=12 )

      Constructs a ``DistHelmholtz`` instance using the specified discretization
      scheme, communicator, positive imaginary shift (`damping`), subdomain 
      size (`numPlanesPerPanel`), and maximum two-dimensional subdomain size 
      for the analytical nested dissection (`cutoff`). Keep in mind that 
      modifying `damping` will effect the effectiveness of the preconditioner, 
      and `numPlanesPerPanel` is essentially a memory and performance tuning
      parameter (though it can have a minor effect on convergence rates).

   .. cpp:function:: void Initialize( const DistUniformGrid<R>& velocity, PanelScheme panelScheme=CLIQUE_LDL_SELINV_2D )

      Performs the subdomain factorizations so that the preconditioner can be 
      quickly applied.

   .. cpp:function:: void Solve( DistUniformGrid<Complex<R> >& B, int m=20, R relTol=1e-4, bool viewIterates=false ) const

      Overwrites each of the right-hand sides stored in `B` with their 
      approximate solution via GMRES(`m`) with the specified relative residual
      tolerance. Thus, GMRES will restart every `m` iterations and will 
      continue until

      .. math::

         \|A x_i - b_i\|_2 \le \mbox{relTol}\|b_i\|_2

      for every right-hand side, :math:`b_i`.

   .. cpp:function:: void Finalize() const

      Frees up all significant resources allocated by the class.
