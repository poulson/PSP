Test drivers
============

Analytical velocity models
--------------------------

0. Uniform
^^^^^^^^^^
A constant unit velocity field. This should be the ideal setting for the 
preconditioner.

.. image:: velocity-uniform.png

1. Converging lens
^^^^^^^^^^^^^^^^^^
One of the models tested in Engquist and Ying's *Sweeping preconditioner for 
the Helmholtz equation: moving Perfectly Matched Layers*.

.. image:: velocity-gaussian.png

2. Wave guide
^^^^^^^^^^^^^
Another model tested in Engquist and Ying's *Sweeping preconditioner for 
the Helmholtz equation: moving Perfectly Matched Layers*.

.. image:: velocity-guide.png

3. Two decreasing layers
^^^^^^^^^^^^^^^^^^^^^^^^
Two layers stacked on top of each other, where the bottom layer has a velocity 
which is eight times higher than in the top layer.

.. image:: velocity-twoDecLayers.png

4. Two increasing layers
^^^^^^^^^^^^^^^^^^^^^^^^
Same as above, but the sweep proceeds from the low to high velocity (instead of 
high to low).

.. image:: velocity-twoIncLayers.png

5. Two sideways layers
^^^^^^^^^^^^^^^^^^^^^^
Same as the above two models, but there is no gradient in the velocity model
in the sweep direction.

.. image:: velocity-twoSidewaysLayers.png

6. Five decreasing layers
^^^^^^^^^^^^^^^^^^^^^^^^^
Test the performance of sweeping from low to high velocities.

.. image:: velocity-fiveDecLayers.png

7. Five increasing layers
^^^^^^^^^^^^^^^^^^^^^^^^^
Tests the performance of sweeping from high to low velocities. Ideally, 
the sweeping preconditioner should perform worse on this model than the 
previous one since the approximated lower half-spaces should have more 
significant reflections.

.. image:: velocity-fiveIncLayers.png

8. Five sideways layers
^^^^^^^^^^^^^^^^^^^^^^^
Tests a five layer model where each panel is equivalent but heterogeneous.

.. image:: velocity-fiveSidewaysLayers.png

9. Wedge
^^^^^^^^
A 3D version of a standard benchmark problem (**citation needed**).

.. image:: velocity-wedge.png

10. Random
^^^^^^^^^^
Each velocity entry is drawn from a uniform distribution over :math:`[2,3]`.

.. image:: velocity-random.png

11. Separator
^^^^^^^^^^^^^
A separator with a velocity ten orders of magnitude higher than the rest 
of the medium is inserted to test the robustness of the preconditioner.
Convergence seems to be unaffected by the separator.

.. image:: velocity-separator.png

12. Cavity
^^^^^^^^^^
A high-velocity outer layer surrounding a low-velocity interior region.
Interior rays become trapped within the low-velocity region and cause resonance
(resulting in poor performance of the preconditioner).

.. image:: velocity-cavity.png

13. Reverse cavity
^^^^^^^^^^^^^^^^^^
Meant to show that high velocity contrasts are not the problem: since rays
do not become trapped, the preconditioner performs very well on this velocity
model.

.. image:: velocity-reverseCavity.png

14. Bottom half of cavity
^^^^^^^^^^^^^^^^^^^^^^^^^
If the cavity is on the bottom half of the domain, sweeping
upwards uses approximations which replace the bottom half of the cavity with 
PML, which ignores strong reflections from rays attempting to jump from the 
low to high velocity region.

.. image:: velocity-bottomHalfCavity.png

15. Top half of cavity
^^^^^^^^^^^^^^^^^^^^^^
Since PSP sweeps upward from the bottom of the domain to the top, this model
should not pose a significant problem since the approximated lower half-space
problems do not replace strong with PML.

.. image:: velocity-topHalfCavity.png


UnitCube
--------
This section describes the driver 
`tests/UnitCube.cpp <https://github.com/poulson/PSP/blob/master/tests/UnitCube.cpp>`__, which is designed for quickly testing the performance of the sweeping 
preconditioner on a variety of different velocity models.

Parameters
^^^^^^^^^^
Usage ::

    UnitCube <velocity> <n> <omega> [pmlOnTop=true] [pmlSize=5] [sigma=1.5] [damping=7] [numPlanesPerPanel=4] [panelScheme=1] [fullViz=1] [factBlocksize=96] [solveBlocksize=64]

* `velocity`: which velocity field to use, see `Analytical velocity models`_
* `n`: size of grid in each dimension
* `omega`: frequency in rad/sec of problem
* `pmlOnTop`: PML if nonzero, Dirichlet otherwise
* `pmlSize`: number of grid points of per PML boundary condition
* `sigma`: magnitude of complex coordinate-stretching for PML
* `damping`: imaginary frequency shift for preconditioner
* `numPlanesPerPanel`: number of planes per subdomain
* `panelScheme`: use traditional scheme if 0, selective inversion if 1
* `fullViz`: full volume visualization if nonzero
* `factBlocksize`: algorithmic blocksize for factorization
* `solveBlocksize`: algorithmic blocksize for solves

For each run of the ``UnitCube`` driver, four different sets of sources are used: 

1. A single localized Gaussian centered at :math:`(0.5,0.5,0.1)`.
2. Three localized Gaussians, centered at :math:`(0.5,0.5,0.1)`, :math:`(0.25,0.25,0.1)`, and :math:`(0.75,0.75,0.5)`.
3. A Gaussian beam centered at :math:`(0.75,0.75,0.5)` and pointed in the direction :math:`(0.57735,0.57735,-0.57735)`.
4. A plane wave pointed in the same direction as the Gaussian beam, but with support in the complement of PML.

Uniform example
^^^^^^^^^^^^^^^
The following results are gathered from running at 314.16 rad/sec over the 
uniform velocity model with a :math:`500 \times 500 \times 500` grid, via the 
command::
    
    UnitCube 0 500 314.16 

which converged to five digits of relative accuracy in 21 iterations of 
GMRES(20) on 256 nodes of TACC's Lonestar.

.. image:: solution-uniform-singleShot-YZ-50.png

The middle YZ plane of the single-shot solution.

.. image:: solution-uniform-threeShots-YZ-50-0.6.png

A slightly off-center YZ plane (x=0.6) of the three-shot solution. 

.. image:: solution-uniform-planeWave-YZ-50.png

The middle YZ plane of the plane wave solution.

Converging lens example
^^^^^^^^^^^^^^^^^^^^^^^
This example used the converging lens velocity model at 235.62 rad/sec over 
another :math:`500 \times 500 \times 500` grid, via the command::
    
    UnitCube 1 500 235.62 1 5 2.0

and converged to five digits of relative accuracy in 43 iterations of GMRES(20)
on 256 nodes of TACC's Lonestar.

.. image:: solution-gaussian-singleShot-YZ-37.5.png

The middle YZ plane of the single-shot solution.

.. image:: solution-gaussian-threeShots-YZ-37.5.png

The middle YZ plane of the three-shot solution.

.. image:: solution-gaussian-planeWave-YZ-37.5.png

The middle YZ plane of the plane wave solution.

Wave guide example
^^^^^^^^^^^^^^^^^^
This third example uses the wave guide velocity model over a 
:math:`500 \times 500 \times 500` grid, again at 235.62 rad/sec. Using five 
grid points of PML, with a coordinate-stretching magnitude of 2.0, via::

    UnitCube 2 500 235.62 1 5 2.0

and all four residuals converged to five digits of relative accuracy in 
51 iterations of GMRES(20) on 256 nodes of TACC's Lonestar. With six grid points
of PML, via::

    UnitCube 2 500 235.62 1 6 2.0

the same model converged in 26 iterations.

.. image:: solution-guide-singleShot-YZ-37.5.png

The middle YZ plane of the single-shot solution.

.. image:: solution-guide-singleShot-YZ-37.5-0.55.png

An off-center YZ plane (x=0.55) of the single-shot solution.

.. image:: solution-guide-threeShots-YZ-37.5.png

The middle YZ plane of the three-shot solution.

.. image:: solution-guide-threeShots-YZ-37.5-0.55.png

An off-center YZ plane (x=0.55) of the three-shot solution.

.. image:: solution-guide-planeWave-YZ-37.5.png

The middle YZ plane of the plane wave solution.

Two sideways layers example
^^^^^^^^^^^^^^^^^^^^^^^^^^^
This example tests a velocity model which is split between a two materials 
with wave speeds which vary by a factor of four. The grid size was 
:math:`500 \times 500 \times 500`, and the frequency was set to 314.16 
radians/second. Using a PML magnitude of 4.0, via the command::
    
    UnitCube 5 500 314.16 1 5 4.0

all four residuals converged to five digits of relative accuracy in 
47 iterations of GMRES(20) on 256 nodes of TACC's Lonestar.

.. image:: solution-twoSidewaysLayers-singleShot-YZ-50.png

The middle YZ plane of the single-shot solution.

.. image:: solution-twoSidewaysLayers-singleShot-YZ-50-0.7.png

An off-center YZ plane (x=0.7) of the single-shot solution.

.. image:: solution-twoSidewaysLayers-threeShots-YZ-50.png

The middle YZ plane of the three-shot solution.

.. image:: solution-twoSidewaysLayers-threeShots-YZ-50-0.7.png

An off-center YZ plane (x=0.7) of the three-shot solution.

Wedge example
^^^^^^^^^^^^^
This example tests a velocity model which is split between a three different 
materials, with the lowest velocity material wedged into the middle. The grid 
size was :math:`500 \times 500 \times 500`, and the frequency was set to 471.25 
radians/second. Using a PML magnitude of 4.0, via the command::
    
    UnitCube 9 500 471.25 1 5 4.0

all four residuals converged to five digits of relative accuracy in 48 
iterations of GMRES(20) on 256 nodes of TACC's Lonestar.

.. image:: solution-wedge-singleShot-YZ-75.png

The center YZ plane of the single-shot solution.

.. image:: solution-wedge-singleShot-YZ-75-0.7.png

An off-center YZ plane (x=0.7) of the single-shot solution.

.. image:: solution-wedge-threeShots-YZ-75-0.7.png

An off-center YZ plane (x=0.7) of the three-shot solution.

Separator
^^^^^^^^^
This example shows that extremely large variations in the velocity field can 
be harmless: even though there is a "separator" with a velocity which is ten
orders of magnitude larger than the background velocity, the sweeping 
preconditioner converged in only 27 iterations for the 50 wavelength 
calculation::
    
    UnitCube 11 500 314.16 1 5 3.0

.. image:: solution-separator-singleShot-YZ-50-0.7.png

An off-center YZ plane (x=0.7) of the single-shot solution.

.. image:: solution-separator-threeShots-YZ-50-0.7.png

An off-center YZ plane (x=0.7) of the three-shot solution.

Interpolate
-----------
The `tests/Interpolate.cpp driver <https://github.com/poulson/PSP/blob/master/tests/Interpolate.cpp>`__ 
is meant to exercise the routine 
:cpp:func:`DistUniformGrid\<F>::InterpolateTo`, which provides a means of 
linearly interpolating a velocity model into a different grid size in parallel.

Parameters
^^^^^^^^^^
Usage ::

    Interpolate <velocity> <m1> <m2> <m3> <n1> <n2> <n3>

* `velocity`: see `Analytical velocity models`_
* `m1`, `m2`, `m3`: original grid size 
* `n1`, `n2`, `n3`: interpolated grid size

**TODO:** Show some results.
