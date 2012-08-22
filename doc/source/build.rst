Build system
************
PSP's build system sits on top of 
`Clique's <http://poulson.github.com/Clique/build.html>`_, which in turn 
is a wrapper for 
`Elemental's <http://poulson.github.com/Elemental/build.html>`_, 
and so it would be helpful to get a basic understanding of their build systems
first.

Getting PSP's source
====================
The best way to download PSP's source is to install
`Mercurial <http://mercurial.selenic.com>`_ and run ::

    hg clone http://bitbucket.org/poulson/psp psp

Building PSP
============
PSP can be built in essentially the same way as Clique and Elemental: their
respective build processes are described 
`here <http://poulson.github.com/Elemental/build.html#building-elemental>`__ and
`here <http://poulson.github.com/Clique/build.html#building-clique>`__.
Assuming that all of 
`dependencies <http://poulson.github.com/Elemental/build.html#dependencies>`_ 
have been installed, PSP can often be built and installed using the commands::

    cd psp
    mkdir build
    cd build
    cmake ..
    make
    make install

Note that the default install location is system-wide, e.g., ``/usr/local``.
The installation directory can be changed at any time by running ::

    cmake -D CMAKE_INSTALL_PREFIX=/your/desired/install/path ..
    make install

Testing the installation
========================
Once PSP has been installed, it is easy to test whether or not it is 
functioning. Assuming that PSP's source code sits in the directory 
``/home/username/psp``, and that PSP was installed in ``/usr/local``, then one
can create the following Makefile from any directory::

    include /usr/local/conf/pspvariables

    UnitCube: /home/username/psp/tests/UnitCube.cpp
        ${CXX} ${PSP_COMPILE_FLAGS} $< -o $@ ${PSP_LINK_FLAGS} ${PSP_LIBS}

and then simply running ``make`` should build the test driver.

You can also build a handful of test drivers by using the CMake option::

    -D PSP_TESTS=ON

PSP as a subproject
===================
Adding PSP as a dependency into a project which uses CMake for its build
system is relatively straightforward: simply put an entire copy of the
PSP source tree in a subdirectory of your main project folder, say
``external/psp``, and uncomment out the bottom section of PSP's
``CMakeLists.txt``, i.e., change ::

    ################################################################################
    # Uncomment if including PSP as a subproject in another build system           #
    ################################################################################
    #set(USE_CUSTOM_ALLTOALLV ${USE_CUSTOM_ALLTOALLV} PARENT_SCOPE)
    #set(HAVE_PARMETIS ${HAVE_PARMETIS} PARENT_SCOPE)
    #
    #set(LIBRARY_TYPE ${LIBRARY_TYPE} PARENT_SCOPE)
    #set(MPI_C_COMPILER ${MPI_C_COMPILER} PARENT_SCOPE)
    #set(MPI_C_INCLUDE_PATH ${MPI_C_INCLUDE_PATH} PARENT_SCOPE)
    #set(MPI_C_COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS} PARENT_SCOPE)
    #set(MPI_C_LINK_FLAGS ${MPI_C_LINK_FLAGS} PARENT_SCOPE)
    #set(MPI_C_LIBRARIES ${MPI_C_LIBRARIES} PARENT_SCOPE)
    #set(MPI_CXX_COMPILER ${MPI_CXX_COMPILER} PARENT_SCOPE)
    #set(MPI_CXX_INCLUDE_PATH ${MPI_CXX_INCLUDE_PATH} PARENT_SCOPE)
    #set(MPI_CXX_COMPILE_FLAGS ${MPI_CXX_COMPILE_FLAGS} PARENT_SCOPE)
    #set(MPI_CXX_LINK_FLAGS ${MPI_CXX_LINK_FLAGS} PARENT_SCOPE)
    #set(MPI_CXX_LIBRARIES ${MPI_CXX_LIBRARIES} PARENT_SCOPE)
    #set(MATH_LIBS ${MATH_LIBS} PARENT_SCOPE)
    #set(RESTRICT ${RESTRICT} PARENT_SCOPE)
    #set(RELEASE ${RELEASE} PARENT_SCOPE)
    #set(BLAS_POST ${BLAS_POST} PARENT_SCOPE)
    #set(LAPACK_POST ${LAPACK_POST} PARENT_SCOPE)
    #set(HAVE_OPENMP ${HAVE_OPENMP} PARENT_SCOPE)
    #set(HAVE_F90_INTERFACE ${HAVE_F90_INTERFACE} PARENT_SCOPE)
    #set(WITHOUT_PMRRR ${WITHOUT_PMRRR} PARENT_SCOPE)
    #set(AVOID_COMPLEX_MPI ${AVOID_COMPLEX_MPI} PARENT_SCOPE)
    #set(HAVE_REDUCE_SCATTER_BLOCK ${HAVE_REDUCE_SCATTER_BLOCK} PARENT_SCOPE)
    #set(REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE ${REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE} PARENT_SCOPE)
    #set(USE_BYTE_ALLGATHERS ${USE_BYTE_ALLGATHERS} PARENT_SCOPE)

to ::
    
    ################################################################################
    # Uncomment if including PSP as a subproject in another build system           #
    ################################################################################
    set(USE_CUSTOM_ALLTOALLV ${USE_CUSTOM_ALLTOALLV} PARENT_SCOPE)
    set(HAVE_PARMETIS ${HAVE_PARMETIS} PARENT_SCOPE)
    
    set(LIBRARY_TYPE ${LIBRARY_TYPE} PARENT_SCOPE)
    set(MPI_C_COMPILER ${MPI_C_COMPILER} PARENT_SCOPE)
    set(MPI_C_INCLUDE_PATH ${MPI_C_INCLUDE_PATH} PARENT_SCOPE)
    set(MPI_C_COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS} PARENT_SCOPE)
    set(MPI_C_LINK_FLAGS ${MPI_C_LINK_FLAGS} PARENT_SCOPE)
    set(MPI_C_LIBRARIES ${MPI_C_LIBRARIES} PARENT_SCOPE)
    set(MPI_CXX_COMPILER ${MPI_CXX_COMPILER} PARENT_SCOPE)
    set(MPI_CXX_INCLUDE_PATH ${MPI_CXX_INCLUDE_PATH} PARENT_SCOPE)
    set(MPI_CXX_COMPILE_FLAGS ${MPI_CXX_COMPILE_FLAGS} PARENT_SCOPE)
    set(MPI_CXX_LINK_FLAGS ${MPI_CXX_LINK_FLAGS} PARENT_SCOPE)
    set(MPI_CXX_LIBRARIES ${MPI_CXX_LIBRARIES} PARENT_SCOPE)
    set(MATH_LIBS ${MATH_LIBS} PARENT_SCOPE)
    set(RESTRICT ${RESTRICT} PARENT_SCOPE)
    set(RELEASE ${RELEASE} PARENT_SCOPE)
    set(BLAS_POST ${BLAS_POST} PARENT_SCOPE)
    set(LAPACK_POST ${LAPACK_POST} PARENT_SCOPE)
    set(HAVE_OPENMP ${HAVE_OPENMP} PARENT_SCOPE)
    set(HAVE_F90_INTERFACE ${HAVE_F90_INTERFACE} PARENT_SCOPE)
    set(WITHOUT_PMRRR ${WITHOUT_PMRRR} PARENT_SCOPE)
    set(AVOID_COMPLEX_MPI ${AVOID_COMPLEX_MPI} PARENT_SCOPE)
    set(HAVE_REDUCE_SCATTER_BLOCK ${HAVE_REDUCE_SCATTER_BLOCK} PARENT_SCOPE)
    set(REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE ${REDUCE_SCATTER_BLOCK_VIA_ALLREDUCE} PARENT_SCOPE)
    set(USE_BYTE_ALLGATHERS ${USE_BYTE_ALLGATHERS} PARENT_SCOPE)

Troubleshooting
===============
If you run into problems, please email
`jack.poulson@gmail.com <mailto:jack.poulson@gmail.com>`_. If you are having
build problems, please make sure to attach the file ``include/psp/config.h``,
which should be generated within your build directory.
