Test drivers
============

Analytical velocity models
--------------------------

Uniform
^^^^^^^
A constant unit velocity field. This should be the ideal setting for the 
preconditioner.

.. image:: velocity-uniform.png

Gaussian perturbation
^^^^^^^^^^^^^^^^^^^^^
One of the models tested in Engquist and Ying's *Sweeping preconditioner for 
the Helmholtz equation: moving Perfectly Matched Layers*.

.. image:: velocity-gaussian.png

Wave guide
^^^^^^^^^^
Another model tested in Engquist and Ying's *Sweeping preconditioner for 
the Helmholtz equation: moving Perfectly Matched Layers*.

.. image:: velocity-guide.png

Two layers
^^^^^^^^^^
Two layers stacked on top of each other, where the bottom layer has a velocity 
which is eight times higher than in the top layer.

.. image:: velocity-twoLayers.png

Cavity
^^^^^^
A high-velocity outer layer surrounding a low-velocity interior region.
Interior rays become trapped within the low-velocity region and cause resonance
(resulting in poor performance of the preconditioner).

.. image:: velocity-cavity.png

Reverse cavity
^^^^^^^^^^^^^^
Meant to show that high velocity contrasts are not the problem: since rays
do not become trapped, the preconditioner performs very well on this velocity
model.

.. image:: velocity-reverseCavity.png

Top half of cavity
^^^^^^^^^^^^^^^^^^
Since PSP sweeps upward from the bottom of the domain to the top, this model
should not pose a significant problem since the approximated lower half-space
problems do not replace strong with PML.

.. image:: velocity-topHalfCavity.png

Bottom half of cavity
^^^^^^^^^^^^^^^^^^^^^
On the other hand, if the cavity is on the bottom half of the domain, sweeping
upwards uses approximations which replace the bottom half of the cavity with 
PML, which ignores strong reflections from rays attempting to jump from the 
low to high velocity region.

.. image:: velocity-bottomHalfCavity.png

Increasing layers
^^^^^^^^^^^^^^^^^
Tests the performance of sweeping from high to low velocities. Ideally, 
the sweeping preconditioner should perform worse on this model than the 
next one since the approximated lower half-spaces should have more significant
reflections.

.. image:: velocity-incLayers.png

Decreasing layers
^^^^^^^^^^^^^^^^^
Test the performance of sweeping from low to high velocities.

.. image:: velocity-decLayers.png

Sideways layers
^^^^^^^^^^^^^^^
Tests a model where each panel is equivalent but heterogeneous.

.. image:: velocity-sidewaysLayers.png

Wedge
^^^^^
A 3D version of a standard benchmark problem (**citation needed**).

.. image:: velocity-wedge.png

Random
^^^^^^
Each velocity entry is drawn from a uniform distribution over :math:`[2,3]`.

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
