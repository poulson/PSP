Example driver(s)
=================

Overthrust
----------

.. image:: velocity-overthrust.png

**TODO: Describe ``examples/Overthrust.cpp``**

Usage ::
    
    Overthrust <nx> <ny> <nz> <omega> <px> <py> <pz> [pmlOnTop=true] [pmlSize=5] [sigma=24] [damping=7] [numPlanesPerPanel=4] [panelScheme=1] [fullViz=1] [factBlocksize=96] [solveBlocksize=64]

* `nx`, `ny`, `nz`: size of grid in each dimension
* `omega`: frequency (in rad/sec) of the problem
* `px`, `py`, `pz`: 3D process grid dimensions
* `pmlOnTop`: PML if nonzero, Dirichlet otherwise
* `pmlSize`: number of grid points per PML boundary condition
* `sigma`: magnitude of complex coordinate stretching for PML
* `damping`: imaginary frequency shift for preconditioner
* `numPlanesPerPanel`: number of planes per subdomain
* `panelScheme`: traditional approach if zero, selective inversion if 1
* `fullViz`: full volume visualization if nonzero
* `factBlocksize`: algorithmic blocksize for factorization
* `solveBlocksize`: algorithmic blocksize for solves

Eight Hz example
^^^^^^^^^^^^^^^^
This example shows the Overthrust model solved at its native resolution
(:math:`801 \times 801 \times 187`) at a frequency of 8 Hz in order to ensure
that there are slightly more than 10 points per wavelength at the shortest 
wavelength. The command used was::
    
    Overthrust 801 801 187 50.26 16 16 8 1 5 15 7.07 4 1

and all four residuals converged to five digits of relative accuracy in 
77 iterations of GMRES(20) on 256 nodes of Lonestar 
(using eight cores per node, in roughly ten minutes, including the 
preconditioner setup time).

.. image:: solution-overthrust-singleShot-YZ-8.png

The middle YZ plane of the single-shot solution.

.. image:: solution-overthrust-threeShots-YZ-8.png

The middle YZ plane of the three-shot solution.

.. image:: solution-overthrust-threeShots-YZ-8-0.55.png

An off-center YZ plane (x=0.55*20) of the three-shot solution.

.. image:: solution-overthrust-planeWave-YZ-8.png

The middle YZ plane of the plane wave solution.
