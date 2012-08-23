The ``DistUniformGrid`` class
=============================

.. cpp:type:: enum GridDataOrder

   Specifies the lexicographical order in which the data is to be stored:
   * XYZ
   * XZY
   * YXZ
   * YZX
   * ZXY
   * ZYX

.. cpp:type:: enum PlaneType

   Specifies a plane type based upon the dimensions it includes:
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

   .. cpp:function:: int YSize() const

   .. cpp:function:: int ZSize() const

   .. cpp:function:: mpi::Comm Comm() const

   .. cpp:function:: int OwningProcess( int naturalIndex ) const

   .. cpp:function:: int OwningProcess( int x, int y, int z ) const

   .. cpp:function:: int NumScalars() const

   .. cpp:function:: int LocalIndex( int naturalIndex ) const

   .. cpp:function:: int LocalIndex( int x, int y, int z ) const

   .. cpp:function:: int NaturalIndex( int x, int y, int z ) const

   .. cpp:function:: F* LocalBuffer()

   .. cpp:function:: const F* LockedLocalBuffer() const

   .. cpp:function:: int XShift() const

   .. cpp:function:: int YShift() const

   .. cpp:function:: int ZShift() const

   .. cpp:function:: int XStride() const

   .. cpp:function:: int YStride() const

   .. cpp:function:: int ZStride() const

   .. cpp:function:: XLocalSize() const

   .. cpp:function:: YLocalSize() const

   .. cpp:function:: ZLocalSize() const

   .. cpp:function:: GridDataOrder Order() const

   .. cpp:function:: void SequentialLoad( std::string filename )

   .. cpp:function:: void InterpolateTo( int nx, int ny, int nz )

   .. cpp:function:: void WritePlane( PlaneType planeType, int whichPlane, std::string basename ) const

   .. cpp:function:: void WriteVolume( std::string basename ) const
