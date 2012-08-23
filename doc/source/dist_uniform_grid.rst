The ``DistUniformGrid`` class
=============================
The purpose of this class is to provide a convenient mechanism for manipulating
and visualizing scalar fields defined on three-dimensional uniform grids 
and distributed over large numbers of processes. The basic strategy is 
straightforward: given :math:`p` processes and a factorization 
:math:`p_x p_y p_z = p`, grid-point :math:`(x,y,z)` is assigned to the process
in the :math:`(x \bmod p_x,y \bmod p_y,z \bmod p_z)` position in the grid.
By default, an XYZ lexicographical ordering is used, but any of the other 
five permutations may be used.

The two visualization subroutines, :cpp:func:`DistUniformGrid\<F>::WritePlane`
and :cpp:func:`DistUniformGrid\<F>::WriteVolume`, output (parallel) VTK image 
files, which the author typically views using 
`ParaView <http://www.paraview.org>`__.

.. cpp:type:: enum GridDataOrder

   Specifies the lexicographical order in which the data is to be stored. 
   The choices are

   * XYZ
   * XZY
   * YXZ
   * YZX
   * ZXY
   * ZYX

.. cpp:type:: enum PlaneType

   Specifies a plane type based upon the dimensions it includes. The choices
   are

   * XY
   * XZ
   * YZ

.. cpp:type:: class DistUniformGrid<F>

   .. cpp:function:: DistUniformGrid( int nx, int ny, int nz, int px, int py, int pz, mpi::Comm comm, int numScalars=1, GridDataOrder order=XYZ ) 

      Generates an :math:`n_x \times n_y \times n_z` grid, where each node 
      contains `numScalars` entries of type `F`, and the grid is distributed
      over a :math:`p_x \times p_y \times p_z` logical grid of processes 
      using the specified communicator.

   .. cpp:function:: DistUniformGrid( int nx, int ny, int nz, mpi::Comm comm, int numScalars=1, GridDataOrder order=XYZ )

      Same as above, but the 3D process grid is determined automatically.

   .. cpp:function:: int XSize() const

      Number of grid points in :math:`X` direction.

   .. cpp:function:: int YSize() const

      Number of grid points in :math:`Y` direction.

   .. cpp:function:: int ZSize() const

      Number of grid points in :math:`Z` direction.

   .. cpp:function:: mpi::Comm Comm() const

      Communicator for the processes sharing the uniform grid.

   .. cpp:function:: int OwningProcess( int naturalIndex ) const

      The rank of the process which will own the grid point :math:`(x,y,z)`,
      where `naturalIndex` equals :math:`x + y n_x + z n_x n_y`.

   .. cpp:function:: int OwningProcess( int x, int y, int z ) const

      The rank of the process which will own the grid point :math:`(x,y,z)`.

   .. cpp:function:: int NumScalars() const

      The number of scalar entries stored at each grid point.

   .. cpp:function:: int LocalIndex( int naturalIndex ) const

      The first local entry (assuming it exists) which stores data at the 
      grid point with the specified natural index.

   .. cpp:function:: int LocalIndex( int x, int y, int z ) const

      The first local entry (assuming it exists) which stores data at the 
      specified grid point.

   .. cpp:function:: int NaturalIndex( int x, int y, int z ) const

      The natural index, :math:`x + y n_x + z n_x n_y`, of the grid point
      :math:`(x,y,z)`.

   .. cpp:function:: F* LocalBuffer()

      A pointer to the locally-stored data.

   .. cpp:function:: const F* LockedLocalBuffer() const

      A const pointer to the locally-stored data.

   .. cpp:function:: int XShift() const

      The first :math:`X`-coordinate which this process can store given the 
      existing :math:`p_x \times p_y \times p_z` process grid.

   .. cpp:function:: int YShift() const

      The first :math:`Y`-coordinate which this process can store given the 
      existing :math:`p_x \times p_y \times p_z` process grid.

   .. cpp:function:: int ZShift() const

      The first :math:`Z`-coordinate which this process can store given the
      existing :math:`p_x \times p_y \times p_z` process grid.

   .. cpp:function:: int XStride() const

      The stride between consecutive :math:`X` grid points which each process
      would store given a sufficiently large uniform grid. This is equal to 
      the number of processes in the :math:`X` dimension, :math:`p_x`, of the 
      three-dimensional process grid.

   .. cpp:function:: int YStride() const

      The stride between consecutive :math:`Y` grid points which this process
      would store given a sufficiently large uniform grid. This is equal to 
      the number of processes in the :math:`Y` dimension, :math:`p_y`, of the 
      three-dimensional process grid.

   .. cpp:function:: int ZStride() const

      The stride between consecutive :math:`Z` grid points which this process
      would store given a sufficiently large uniform grid. This is equal to 
      the number of processes in the :math:`Z` dimension, :math:`p_z`, of the 
      three-dimensional process grid.

   .. cpp:function:: XLocalSize() const

      The number of :math:`X` coordinates assigned to this process given the 
      existing values of :math:`n_x`, the number of grid points in the :math:`X`
      direction, and :math:`p_x`, the number of processes in the :math:`X` 
      direction of the logical process grid. 

   .. cpp:function:: YLocalSize() const

      The number of :math:`Y` coordinates assigned to this process given the 
      existing values of :math:`n_y`, the number of grid points in the :math:`Y`
      direction, and :math:`p_y`, the number of processes in the :math:`Y` 
      direction of the logical process grid. 

   .. cpp:function:: ZLocalSize() const

      The number of :math:`Z` coordinates assigned to this process given the 
      existing values of :math:`n_z`, the number of grid points in the :math:`Z`
      direction, and :math:`p_z`, the number of processes in the :math:`Z` 
      direction of the logical process grid. 

   .. cpp:function:: GridDataOrder Order() const

      The lexicographical ordering imposed on the grid points.

   .. cpp:function:: void SequentialLoad( std::string filename )

      Each process reads in its portion of the data from the sequential 
      data stores in the specified file. Note that the data must use the 
      lexicographical ordering specified by 
      :cpp:func:`DistUniformGrid\<F>::Order`, and the model must be over a
      grid which matches that of the parent class.

   .. cpp:function:: void InterpolateTo( int nx, int ny, int nz )

      The existing grid data is linearly interpolated into a new grid with 
      the specified dimensions. This is particularly useful in conjunction 
      with :cpp:func:`DistUniformGrid\<F>::SequentialLoad`, as a predefined
      velocity model can be loaded and interpolated in parallel with two 
      lines of code.

   .. cpp:function:: void WritePlane( PlaneType planeType, int whichPlane, std::string basename ) const

      Outputs a sequential VTK file for visualizing the grid data in the 
      specified plane (one file for each scalar).

   .. cpp:function:: void WriteVolume( std::string basename ) const

      Outputs a parallel VTK file for visualizing the grid data over the 
      entire volume (one set of files for each scalar).
