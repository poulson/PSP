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
``external/psp``, and then create a ``CMakeLists.txt`` in your main project
folder that builds off of the following snippet::
    
    cmake_minimum_required(VERSION 2.8.5)
    project(Foo)

    add_subdirectory(external/psp)
    include_directories("${PROJECT_BINARY_DIR}/external/psp/include")
    include_directories("${PROJECT_BINARY_DIR}/external/psp/external/clique/include")
    if(HAVE_PARMETIS)
      include_directories(
        "${PROJECT_SOURCE_DIR}/external/psp/external/clique/external/parmetis/include"
      )
      include_directories(
        "${PROJECT_SOURCE_DIR}/external/psp/external/clique/external/parmetis/metis/include"
      )
    endif()
    include_directories(
      "${PROJECT_BINARY_DIR}/external/psp/external/clique/external/elemental/include")
    )
    include_directories(${MPI_CXX_INCLUDE_PATH})
     
    # Build your project here
    # e.g.,
    #   add_library(foo ${LIBRARY_TYPE} ${FOO_SRC})
    #   target_link_libraries(foo psp)

Troubleshooting
===============
If you run into problems, please email
`jack.poulson@gmail.com <mailto:jack.poulson@gmail.com>`_. If you are having
build problems, please make sure to attach the file ``include/psp/config.h``,
which should be generated within your build directory.
