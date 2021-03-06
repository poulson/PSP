/*
   Copyright (C) 2011-2014 Jack Poulson, Lexing Ying, 
   The University of Texas at Austin, and the Georgia Institute of Technology
 
   This file is part of Parallel Sweeping Preconditioner (PSP) and is under the
   GNU General Public License, which can be found in the LICENSE file in the 
   root directory, or at <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef PSP_ENVIRONMENT_HPP
#define PSP_ENVIRONMENT_HPP

#include "clique.hpp"
#include ELEM_MAXNORM_INC
#include ELEM_SVD_INC

namespace psp {

bool Initialized();
void Initialize( int& argc, char**& argv );
void Finalize();

#ifndef RELEASE
void PushCallStack( std::string s );
void PopCallStack();
void DumpCallStack();

class CallStackEntry
{
public:
    CallStackEntry( std::string s ) 
    { 
        if( !std::uncaught_exception() )
            PushCallStack(s); 
    }
    ~CallStackEntry() 
    { 
        if( !std::uncaught_exception() )
            PopCallStack(); 
    }
};
#endif

// Pull in some of Elemental's imported libraries
namespace blas = elem::blas;
namespace lapack = elem::lapack;
namespace mpi = elem::mpi;

// Pull in a number of useful enums from Elemental
using namespace elem::unit_or_non_unit_wrapper;
using namespace elem::distribution_wrapper;
using namespace elem::orientation_wrapper;
using namespace elem::upper_or_lower_wrapper;
using namespace elem::left_or_right_wrapper;
using namespace elem::vertical_or_horizontal_wrapper;
using namespace elem::forward_or_backward_wrapper;
using namespace elem::conjugation_wrapper;

// Pull in a few classes from Elemental
using elem::Complex;
using elem::Matrix;
using elem::Grid;
using elem::DistMatrix;

// Pull in a few indexing functions
using elem::Length;
using elem::Shift;

using elem::SwapClear;
using elem::LogicError;
using elem::RuntimeError;

// Pull in a few scalar math functions
using elem::Abs;
using elem::Conj;
using elem::Exp;

using elem::Base;

using elem::Input;
using elem::ProcessInput;
using elem::ArgException;

#ifdef HAVE_MKDIR
void EnsureDirExists( const char* path, mode_t mode=0777 );
#endif

// Some simple utility functions useful for running several GMRES(k) 
// instances simultaneously

template<typename T>
bool CheckForNaN( T alpha );
template<typename T>
bool CheckForNaN( const std::vector<T>& alphaList );
template<typename T>
bool CheckForZero( const std::vector<T>& alphaList );

template<typename F>
void Norms
( const Matrix<F>& xList, std::vector<Base<F>>& normList, mpi::Comm comm );

template<typename F>
void InnerProducts
( const Matrix<F>& xList, const Matrix<F>& yList,
  std::vector<F>& alphaList, mpi::Comm comm );

template<typename R>
void DivideColumns
( Matrix<Complex<R> >& xList, const std::vector<R>& deltaList );
template<typename T>
void MultiplyColumns( Matrix<T>& xList, const std::vector<T>& deltaList );
template<typename T>
void SubtractScaledColumns
( const std::vector<T>& deltaList, const Matrix<T>& xList, Matrix<T>& yList );

// For use in Dist(Spectral)Helmholtz::FillPanelLocalElimTree
struct Box
{
    int parentIndex, nx, ny, xOffset, yOffset;
    bool leftChild;
};

} // namespace psp

#include "psp/config.h"

#ifdef HAVE_MKDIR
#include <sys/stat.h>
#endif

namespace psp {

#ifdef HAVE_MKDIR
inline void  EnsureDirExists( const char* path, mode_t mode=0777 )
{
    struct stat s;
    if( stat( path, &s ) != 0 )
    {
        // The directory doesn't yet exist, so try to create it
        if( mkdir( path, mode ) != 0 )
            RuntimeError("Could not create directory");
    }
    else if( !S_ISDIR( s.st_mode ) )
        RuntimeError("Invalid directory");
}
#endif

template<typename T>
inline bool CheckForNaN( T alpha )
{
    return alpha != alpha; // Hopefully this is not optimized away
}

template<typename T>
inline bool CheckForNaN( const std::vector<T>& alphaList ) 
{
    bool foundNaN = false;
    const int numAlphas = alphaList.size();
    for( int k=0; k<numAlphas; ++k )
        if( CheckForNaN(alphaList[k]) )
            foundNaN = true;
    return foundNaN;
}

template<typename T>
inline bool CheckForZero( const std::vector<T>& alphaList )
{
    // TODO: Think about generalizing to use a tolerance
    bool foundZero = false;
    const int numAlphas = alphaList.size();
    for( int k=0; k<numAlphas; ++k )
        if( alphaList[k] == 0 )
            foundZero = true;
    return foundZero;
}

template<typename F>
void Norms
( const Matrix<F>& xList, std::vector<Base<F>>& normList, mpi::Comm comm )
{
    typedef Base<F> Real;

    const int numCols = xList.Width();
    const int localHeight = xList.Height();
    const int commSize = mpi::CommSize( comm );
    std::vector<Real> localNorms( numCols );
    for( int j=0; j<numCols; ++j )
        localNorms[j] = blas::Nrm2( localHeight, xList.LockedBuffer(0,j), 1 );
    std::vector<Real> allLocalNorms( numCols*commSize );
    mpi::AllGather( &localNorms[0], numCols, &allLocalNorms[0], numCols, comm );
    normList.resize( numCols );
    for( int j=0; j<numCols; ++j )
        normList[j] = blas::Nrm2( commSize, &allLocalNorms[j], numCols );
}

template<typename F>
void InnerProducts
( const Matrix<F>& xList, const Matrix<F>& yList,
  std::vector<F>& alphaList, mpi::Comm comm )
{
    const int numCols = xList.Width();
    const int localHeight = xList.Height();
    std::vector<F> localAlphaList( numCols );
    for( int j=0; j<numCols; ++j )
        localAlphaList[j] =
            blas::Dot
            ( localHeight, xList.LockedBuffer(0,j), 1,
                           yList.LockedBuffer(0,j), 1 );
    alphaList.resize( numCols );
    mpi::AllReduce( &localAlphaList[0], &alphaList[0], numCols, MPI_SUM, comm );
}

template<typename R>
void DivideColumns
( Matrix<Complex<R> >& xList, const std::vector<R>& deltaList )
{
    const R one = 1;
    const int numCols = xList.Width();
    const int localHeight = xList.Height();
    for( int j=0; j<numCols; ++j )
    {
        const R invDelta = one/deltaList[j];
        Complex<R>* x = xList.Buffer(0,j);
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
            x[iLocal] *= invDelta;
    }
}

template<typename T>
void MultiplyColumns( Matrix<T>& xList, const std::vector<T>& deltaList )
{
    const int numCols = xList.Width();
    const int localHeight = xList.Height();
    for( int j=0;j<numCols; ++j )
    {
        const T delta = deltaList[j];
        T* x = xList.Buffer(0,j);
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
            x[iLocal] *= delta;
    }
}

template<typename T>
void SubtractScaledColumns
( const std::vector<T>& deltaList, const Matrix<T>& xList, Matrix<T>& yList )
{
    const int numCols = xList.Width();
    const int localHeight = xList.Height();
    for( int j=0; j<numCols; ++j )
    {
        const T delta = deltaList[j];
        const T* x = xList.LockedBuffer(0,j);
        T* y = yList.Buffer(0,j);
        for( int iLocal=0; iLocal<localHeight; ++iLocal )
            y[iLocal] -= delta*x[iLocal];
    }
}

} // namespace psp

#endif // PSP_ENVIRONMENT_HPP
