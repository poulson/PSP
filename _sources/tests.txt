Test drivers
============

Analytical velocity models
--------------------------

Uniform
^^^^^^^
.. image:: velocity-uniform.png

Gaussian perturbation
^^^^^^^^^^^^^^^^^^^^^
.. image:: velocity-gaussian.png

Wave guide
^^^^^^^^^^
.. image:: velocity-guide.png

Two layers
^^^^^^^^^^
.. image:: velocity-twoLayers.png

Cavity
^^^^^^
.. image:: velocity-cavity.png

Reverse cavity
^^^^^^^^^^^^^^
.. image:: velocity-reverseCavity.png

Top half of cavity
^^^^^^^^^^^^^^^^^^
.. image:: velocity-topHalfCavity.png

Bottom half of cavity
^^^^^^^^^^^^^^^^^^^^^
.. image:: velocity-bottomHalfCavity.png

Increasing layers
^^^^^^^^^^^^^^^^^
.. image:: velocity-incLayers.png

Decreasing layers
^^^^^^^^^^^^^^^^^
.. image:: velocity-decLayers.png

Sideways layers
^^^^^^^^^^^^^^^
.. image:: velocity-sidewaysLayers.png

Wedge
^^^^^
.. image:: velocity-wedge.png

Random
^^^^^^
.. image:: velocity-random.png

UnitCube
--------
This section describes the driver 
`tests/UnitCube.cpp <https://github.com/poulson/PSP/blob/master/tests/UnitCube.cpp>`__, which is designed for quickly testing the performance of the sweeping 
preconditioner on a variety of different velocity models.

**TODO**

Interpolate
-----------
The `tests/Interpolate.cpp driver <https://github.com/poulson/PSP/blob/master/tests/Interpolate.cpp>`__ 
is meant to exercise the routine 
:cpp:func:`DistUniformGrid\<F>::InterpolateTo`, which provides a means of 
linearly interpolating a velocity model into a different grid size in parallel.

**TODO**
