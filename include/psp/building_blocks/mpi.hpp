/*
   Parallel Sweeping Preconditioner (PSP): a distributed-memory implementation
   of a sweeping preconditioner for 3d Helmholtz equations.

   Copyright (C) 2011 Jack Poulson, Lexing Ying, and
   The University of Texas at Austin

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef PSP_MPI_HPP
#define PSP_MPI_HPP 1

#include "psp/config.h"
#include <complex>
#include <sstream>
#include <stdexcept>

#include "mpi.h"

namespace psp {
namespace mpi {

void SafeMpi( int mpiError );

void Allgather
( const unsigned* sendBuf, int sendCount,
        unsigned* recvBuf, int recvCount, MPI_Comm comm );
void AllGather
( const int* sendBuf, int sendCount,
        int* recvBuf, int recvCount, MPI_Comm comm );
void AllGather
( const float* sendBuf, int sendCount,
        float* recvBuf, int recvCount, MPI_Comm comm );
void AllGather
( const double* sendBuf, int sendCount,
        double* recvBuf, int recvCount, MPI_Comm comm );
void AllGather
( const std::complex<float>* sendBuf, int sendCount,
        std::complex<float>* recvBuf, int recvCount, MPI_Comm comm );
void AllGather
( const std::complex<double>* sendBuf, int sendCount,
        std::complex<double>* recvBuf, int recvCount, MPI_Comm comm );

void AllReduce
( const int* sendBuf, int* recvBuf, int count, 
  MPI_Op op, MPI_Comm comm );
void AllReduce
( const float* sendBuf, float* recvBuf, int count, 
  MPI_Op op, MPI_Comm comm );
void AllReduce
( const double* sendBuf, double* recvBuf, int count, 
  MPI_Op op, MPI_Comm comm );
void AllReduce
( const std::complex<float>* sendBuf, std::complex<float>* recvBuf, int count,
  MPI_Op op, MPI_Comm comm );
void AllReduce
( const std::complex<double>* sendBuf, std::complex<double>* recvBuf, int count,
  MPI_Op op, MPI_Comm comm );

void AllToAll
( const byte* sendBuf, int sendCount,
        byte* recvBuf, int recvCount, MPI_Comm comm );
void AllToAll
( const int* sendBuf, int sendCount,
        int* recvBuf, int recvCount, MPI_Comm comm );
void AllToAll
( const float* sendBuf, int sendCount,
        float* recvBuf, int recvCount, MPI_Comm comm );
void AllToAll
( const double* sendBuf, int sendCount,
        double* recvBuf, int recvCount, MPI_Comm comm );
void AllToAll
( const std::complex<float>* sendBuf, int sendCount,
        std::complex<float>* recvBuf, int recvCount, MPI_Comm comm );
void AllToAll
( const std::complex<double>* sendBuf, int sendCount,
        std::complex<double>* recvBuf, int recvCount, MPI_Comm comm );

void AllToAllV
( const byte* sendBuf, const int* sendCounts, const int* sendDispls,
        byte* recvBuf, const int* recvCounts, const int* recvDispls, 
  MPI_Comm comm );
void AllToAllV
( const int* sendBuf, const int* sendCounts, const int* sendDispls,
        int* recvBuf, const int* recvCounts, const int* recvDispls, 
  MPI_Comm comm );
void AllToAllV
( const float* sendBuf, const int* sendCounts, const int* sendDispls,
        float* recvBuf, const int* recvCounts, const int* recvDispls, 
  MPI_Comm comm );
void AllToAllV
( const double* sendBuf, const int* sendCounts, const int* sendDispls,
        double* recvBuf, const int* recvCounts, const int* recvDispls, 
  MPI_Comm comm );
void AllToAllV
( const std::complex<float>* sendBuf, 
  const int* sendCounts, const int* sendDispls,
        std::complex<float>* recvBuf, 
  const int* recvCounts, const int* recvDispls, 
  MPI_Comm comm );
void AllToAllV
( const std::complex<double>* sendBuf, 
  const int* sendCounts, const int* sendDispls,
        std::complex<double>* recvBuf, 
  const int* recvCounts, const int* recvDispls, 
  MPI_Comm comm );

void Barrier( MPI_Comm comm );

void Broadcast( int* buf, int count, int root, MPI_Comm comm );
void Broadcast( float* buf, int count, int root, MPI_Comm comm );
void Broadcast( double* buf, int count, int root, MPI_Comm comm );
void Broadcast( std::complex<float>* buf, int count, int root, MPI_Comm comm );
void Broadcast( std::complex<double>* buf, int count, int root, MPI_Comm comm );

void CommDup( MPI_Comm comm, MPI_Comm& commDup );

void CommFree( MPI_Comm& comm );

int CommRank( MPI_Comm comm );

int CommSize( MPI_Comm comm );

void CommSplit( MPI_Comm comm, int color, int key, MPI_Comm& newComm );

void Gather
( const int* sendBuf, int sendCount,
        int* recvBuf, int recvCount, int root, MPI_Comm comm );
void Gather
( const float* sendBuf, int sendCount,
        float* recvBuf, int recvCount, int root, MPI_Comm comm );
void Gather
( const double* sendBuf, int sendCount,
        double* recvBuf, int recvCount, int root, MPI_Comm comm );
void Gather
( const std::complex<float>* sendBuf, int sendCount,
        std::complex<float>* recvBuf, int recvCount, int root, MPI_Comm comm );
void Gather
( const std::complex<double>* sendBuf, int sendCount,
        std::complex<double>* recvBuf, int recvCount, 
  int root, MPI_Comm comm );

void Send
( const byte* buf, 
  int count, int dest, int tag, MPI_Comm comm );
void Send
( const int* buf, 
  int count, int dest, int tag, MPI_Comm comm );
void Send
( const float* buf, 
  int count, int dest, int tag, MPI_Comm comm );
void Send
( const double* buf, 
  int count, int dest, int tag, MPI_Comm comm );
void Send
( const std::complex<float>* buf, 
  int count, int dest, int tag, MPI_Comm comm );
void Send
( const std::complex<double>* buf, 
  int count, int dest, int tag, MPI_Comm comm );

void Recv
( byte* buf, int count, int source, int tag, MPI_Comm comm );
void Recv
( int* buf, int count, int source, int tag, MPI_Comm comm );
void Recv
( float* buf, int count, int source, int tag, MPI_Comm comm );
void Recv
( double* buf, int count, int source, int tag, MPI_Comm comm );
void Recv
( std::complex<float>* buf, int count, int source, int tag, MPI_Comm comm );
void Recv
( std::complex<double>* buf, int count, int source, int tag, MPI_Comm comm );

void SendRecv
( const byte* sendBuf, int sendCount, int dest,   int sendTag,
        byte* recvBuf, int recvCount, int source, int recvTag, 
  MPI_Comm comm );
void SendRecv
( const int* sendBuf, int sendCount, int dest,   int sendTag,
        int* recvBuf, int recvCount, int source, int recvTag, 
  MPI_Comm comm );
void SendRecv
( const float* sendBuf, int sendCount, int dest,   int sendTag,
        float* recvBuf, int recvCount, int source, int recvTag, 
  MPI_Comm comm );
void SendRecv
( const double* sendBuf, int sendCount, int dest,   int sendTag,
        double* recvBuf, int recvCount, int source, int recvTag, 
  MPI_Comm comm );
void SendRecv
( const std::complex<float>* sendBuf, int sendCount, int dest,   int sendTag,
        std::complex<float>* recvBuf, int recvCount, int source, int recvTag, 
  MPI_Comm comm );
void SendRecv
( const std::complex<double>* sendBuf, int sendCount, int dest,   int sendTag,
        std::complex<double>* recvBuf, int recvCount, int source, int recvTag, 
  MPI_Comm comm );

void Reduce
( const int* sendBuf, int* recvBuf, 
  int count, int root, MPI_Op op, MPI_Comm comm );
void Reduce
( const float* sendBuf, float* recvBuf, 
  int count, int root, MPI_Op op, MPI_Comm comm );
void Reduce
( const double* sendBuf, double* recvBuf, 
  int count, int root, MPI_Op op, MPI_Comm comm );
void Reduce
( const std::complex<float>* sendBuf, std::complex<float>* recvBuf, 
  int count, int root, MPI_Op op, MPI_Comm comm );
void Reduce
( const std::complex<double>* sendBuf, std::complex<double>* recvBuf, 
  int count, int root, MPI_Op op, MPI_Comm comm );

void ReduceScatter
( const int* sendBuf, int* recvBuf, const int* recvCounts, 
  MPI_Op op, MPI_Comm comm );
void ReduceScatter
( const float* sendBuf, float* recvBuf, const int* recvCounts, 
  MPI_Op op, MPI_Comm comm );
void ReduceScatter
( const double* sendBuf, double* recvBuf, const int* recvCounts, 
  MPI_Op op, MPI_Comm comm );
void ReduceScatter
( const std::complex<float>* sendBuf, std::complex<float>* recvBuf, 
  const int* recvCounts, MPI_Op op, MPI_Comm comm );
void ReduceScatter
( const std::complex<double>* sendBuf, std::complex<double>* recvBuf, 
  const int* recvCounts, MPI_Op op, MPI_Comm comm );

double WallTime();

} // namespace mpi
} // namespace psp

//----------------------------------------------------------------------------//
// Inlined implementations                                                    //
//----------------------------------------------------------------------------//

namespace psp {
namespace mpi {

inline void
SafeMpi( int mpiError )
{
#ifndef RELEASE
    if( mpiError != MPI_SUCCESS )
    {
        char errorString[200];
        int lengthOfErrorString;
        MPI_Error_string( mpiError, errorString, &lengthOfErrorString );
        throw std::logic_error( errorString );
    }
#endif
}

inline void AllGather
( const unsigned* sendBuf, int sendCount,
        unsigned* recvBuf, int recvCount, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllGather");
#endif
    SafeMpi(
        MPI_Allgather
        ( const_cast<unsigned*>(sendBuf), sendCount, MPI_UNSIGNED,
          recvBuf,                        recvCount, MPI_UNSIGNED, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllGather
( const int* sendBuf, int sendCount,
        int* recvBuf, int recvCount, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllGather");
#endif
    SafeMpi(
        MPI_Allgather
        ( const_cast<int*>(sendBuf), sendCount, MPI_INT,
          recvBuf,                   recvCount, MPI_INT, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllGather
( const float* sendBuf, int sendCount,
        float* recvBuf, int recvCount, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllGather");
#endif
    SafeMpi(
        MPI_Allgather
        ( const_cast<float*>(sendBuf), sendCount, MPI_FLOAT,
          recvBuf,                     recvCount, MPI_FLOAT, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllGather
( const double* sendBuf, int sendCount,
        double* recvBuf, int recvCount, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllGather");
#endif
    SafeMpi(
        MPI_Allgather
        ( const_cast<double*>(sendBuf), sendCount, MPI_DOUBLE,
          recvBuf,                      recvCount, MPI_DOUBLE, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllGather
( const std::complex<float>* sendBuf, int sendCount,
        std::complex<float>* recvBuf, int recvCount, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllGather");
#endif
    SafeMpi(
        MPI_Allgather
        ( const_cast<std::complex<float>*>(sendBuf), 
          sendCount, MPI_COMPLEX,
          recvBuf,  
          recvCount, MPI_COMPLEX, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllGather
( const std::complex<double>* sendBuf, int sendCount,
        std::complex<double>* recvBuf, int recvCount, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllGather");
#endif
    SafeMpi(
        MPI_Allgather
        ( const_cast<std::complex<double>*>(sendBuf), 
          sendCount, MPI_DOUBLE_COMPLEX,
          recvBuf,  
          recvCount, MPI_DOUBLE_COMPLEX, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllReduce
( const int* sendBuf, int* recvBuf, int count, MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllReduce");
#endif
    SafeMpi( 
        MPI_Allreduce
        ( const_cast<int*>(sendBuf), recvBuf, count, 
          MPI_INT, op, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllReduce
( const float* sendBuf, float* recvBuf, int count, MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllReduce");
#endif
    SafeMpi( 
        MPI_Allreduce
        ( const_cast<float*>(sendBuf), recvBuf, count, 
          MPI_FLOAT, op, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllReduce
( const double* sendBuf, double* recvBuf, int count, 
  MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllReduce");
#endif
    SafeMpi( 
        MPI_Allreduce
        ( const_cast<double*>(sendBuf), recvBuf, count, 
          MPI_DOUBLE, op, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllReduce
( const std::complex<float>* sendBuf, std::complex<float>* recvBuf, int count,
  MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllReduce");
#endif
    if( op == MPI_SUM )
    {
        SafeMpi(
            MPI_Allreduce
            ( const_cast<std::complex<float>*>(sendBuf), recvBuf, 2*count, 
              MPI_FLOAT, op, comm )
        );
    }
    else
    {
        SafeMpi(
            MPI_Allreduce
            ( const_cast<std::complex<float>*>(sendBuf), recvBuf, count,
              MPI_COMPLEX, op, comm )
        );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllReduce
( const std::complex<double>* sendBuf, std::complex<double>* recvBuf, int count,
  MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllReduce");
#endif
    if( op == MPI_SUM )
    {
        SafeMpi(
            MPI_Allreduce
            ( const_cast<std::complex<double>*>(sendBuf), recvBuf, 2*count, 
              MPI_DOUBLE, op, comm )
        );
    }
    else
    {
        SafeMpi(
            MPI_Allreduce
            ( const_cast<std::complex<double>*>(sendBuf), recvBuf, count,
              MPI_DOUBLE_COMPLEX, op, comm )
        );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllToAll
( const byte* sendBuf, int sendCount,
        byte* recvBuf, int recvCount, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllToAll");
#endif
    SafeMpi( 
        MPI_Alltoall
        ( const_cast<byte*>(sendBuf), sendCount, MPI_UNSIGNED_CHAR,
                            recvBuf,  recvCount, MPI_UNSIGNED_CHAR, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllToAll
( const int* sendBuf, int sendCount,
        int* recvBuf, int recvCount, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllToAll");
#endif
    SafeMpi( 
        MPI_Alltoall
        ( const_cast<int*>(sendBuf), sendCount, MPI_INT,
                           recvBuf,  recvCount, MPI_INT, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllToAll
( const float* sendBuf, int sendCount,
        float* recvBuf, int recvCount, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllToAll");
#endif
    SafeMpi( 
        MPI_Alltoall
        ( const_cast<float*>(sendBuf), sendCount, MPI_FLOAT,
                             recvBuf,  recvCount, MPI_FLOAT, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllToAll
( const double* sendBuf, int sendCount,
        double* recvBuf, int recvCount, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllToAll");
#endif
    SafeMpi( 
        MPI_Alltoall
        ( const_cast<double*>(sendBuf), sendCount, MPI_DOUBLE,
                              recvBuf,  recvCount, MPI_DOUBLE, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllToAll
( const std::complex<float>* sendBuf, int sendCount,
        std::complex<float>* recvBuf, int recvCount, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllToAll");
#endif
    /*
    SafeMpi( 
        MPI_Alltoall
        ( const_cast<std::complex<float>*>(sendBuf), sendCount, MPI_COMPLEX,
          recvBuf, recvCount, MPI_COMPLEX, comm )
    );
    */
    SafeMpi(
        MPI_Alltoall
        ( const_cast<std::complex<float>*>(sendBuf), 2*sendCount, MPI_FLOAT,
          recvBuf, 2*recvCount, MPI_FLOAT, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllToAll
( const std::complex<double>* sendBuf, int sendCount,
        std::complex<double>* recvBuf, int recvCount, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllToAll");
#endif
    /*
    SafeMpi( 
        MPI_Alltoall
        ( const_cast<std::complex<double>*>(sendBuf), 
          sendCount, MPI_DOUBLE_COMPLEX,
          recvBuf, recvCount, MPI_DOUBLE_COMPLEX, comm )
    );
    */
    SafeMpi(
        MPI_Alltoall
        ( const_cast<std::complex<double>*>(sendBuf),
          2*sendCount, MPI_DOUBLE,
          recvBuf, 2*recvCount, MPI_DOUBLE, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllToAllV
( const byte* sendBuf, const int* sendCounts, const int* sendDispls,
        byte* recvBuf, const int* recvCounts, const int* recvDispls, 
  MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllToAllV");
#endif
    SafeMpi(
        MPI_Alltoallv
        ( const_cast<byte*>(sendBuf), 
          const_cast<int*>(sendCounts), 
          const_cast<int*>(sendDispls), 
          MPI_UNSIGNED_CHAR,
          recvBuf,                    
          const_cast<int*>(recvCounts), 
          const_cast<int*>(recvDispls), 
          MPI_UNSIGNED_CHAR,
          comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}
    
inline void AllToAllV
( const int* sendBuf, const int* sendCounts, const int* sendDispls,
        int* recvBuf, const int* recvCounts, const int* recvDispls, 
  MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllToAllV");
#endif
    SafeMpi(
        MPI_Alltoallv
        ( const_cast<int*>(sendBuf), 
          const_cast<int*>(sendCounts), 
          const_cast<int*>(sendDispls), 
          MPI_INT,
          recvBuf,                   
          const_cast<int*>(recvCounts), 
          const_cast<int*>(recvDispls), 
          MPI_INT,
          comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}
    
inline void AllToAllV
( const float* sendBuf, const int* sendCounts, const int* sendDispls,
        float* recvBuf, const int* recvCounts, const int* recvDispls, 
  MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllToAllV");
#endif
    SafeMpi(
        MPI_Alltoallv
        ( const_cast<float*>(sendBuf), 
          const_cast<int*>(sendCounts), 
          const_cast<int*>(sendDispls), 
          MPI_FLOAT,
          recvBuf,                     
          const_cast<int*>(recvCounts), 
          const_cast<int*>(recvDispls), 
          MPI_FLOAT,
          comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}
    
inline void AllToAllV
( const double* sendBuf, const int* sendCounts, const int* sendDispls,
        double* recvBuf, const int* recvCounts, const int* recvDispls, 
  MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllToAllV");
#endif
    SafeMpi(
        MPI_Alltoallv
        ( const_cast<double*>(sendBuf), 
          const_cast<int*>(sendCounts), 
          const_cast<int*>(sendDispls), 
          MPI_DOUBLE,
          recvBuf,                      
          const_cast<int*>(recvCounts), 
          const_cast<int*>(recvDispls), 
          MPI_DOUBLE,
          comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void AllToAllV
( const std::complex<float>* sendBuf, 
  const int* sendCounts, const int* sendDispls,
        std::complex<float>* recvBuf, 
  const int* recvCounts, const int* recvDispls, 
  MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllToAllV");
#endif
    /*
    SafeMpi(
        MPI_Alltoallv
        ( const_cast<std::complex<float>*>(sendBuf), 
          const_cast<int*>(sendCounts), 
          const_cast<int*>(sendDispls), 
          MPI_COMPLEX,
          recvBuf,                      
          const_cast<int*>(recvCounts), 
          const_cast<int*>(recvDispls), 
          MPI_COMPLEX,
          comm )
    );
    */
    const int commSize = mpi::CommSize( comm );
    std::vector<int> doubledSendCounts( commSize ), 
                     doubledSendDispls( commSize ),
                     doubledRecvCounts( commSize ),
                     doubledRecvDispls( commSize );
    for( int i=0; i<commSize; ++i )
        doubledSendCounts[i] = 2*sendCounts[i];
    for( int i=0; i<commSize; ++i )
        doubledSendDispls[i] = 2*sendDispls[i];
    for( int i=0; i<commSize; ++i )
        doubledRecvCounts[i] = 2*recvCounts[i];
    for( int i=0; i<commSize; ++i )
        doubledRecvDispls[i] = 2*recvDispls[i];
    SafeMpi(
        MPI_Alltoallv
        ( const_cast<std::complex<float>*>(sendBuf),
          &doubledSendCounts[0], &doubledSendDispls[0], MPI_FLOAT,
          recvBuf, 
          &doubledRecvCounts[0], &doubledRecvDispls[0], MPI_FLOAT, 
          comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}
    
inline void AllToAllV
( const std::complex<double>* sendBuf, 
  const int* sendCounts, const int* sendDispls,
        std::complex<double>* recvBuf, 
  const int* recvCounts, const int* recvDispls, 
  MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::AllToAllV");
#endif
    /*
    SafeMpi(
        MPI_Alltoallv
        ( const_cast<std::complex<double>*>(sendBuf), 
          const_cast<int*>(sendCounts), 
          const_cast<int*>(sendDispls), 
          MPI_DOUBLE_COMPLEX,
          recvBuf,                      
          const_cast<int*>(recvCounts), 
          const_cast<int*>(recvDispls), 
          MPI_DOUBLE_COMPLEX,
          comm )
    );
    */
    const int commSize = mpi::CommSize( comm );
    std::vector<int> doubledSendCounts( commSize ), 
                     doubledSendDispls( commSize ),
                     doubledRecvCounts( commSize ),
                     doubledRecvDispls( commSize );
    for( int i=0; i<commSize; ++i )
        doubledSendCounts[i] = 2*sendCounts[i];
    for( int i=0; i<commSize; ++i )
        doubledSendDispls[i] = 2*sendDispls[i];
    for( int i=0; i<commSize; ++i )
        doubledRecvCounts[i] = 2*recvCounts[i];
    for( int i=0; i<commSize; ++i )
        doubledRecvDispls[i] = 2*recvDispls[i];
    SafeMpi(
        MPI_Alltoallv
        ( const_cast<std::complex<double>*>(sendBuf),
          &doubledSendCounts[0], &doubledSendDispls[0], MPI_DOUBLE,
          recvBuf, 
          &doubledRecvCounts[0], &doubledRecvDispls[0], MPI_DOUBLE, 
          comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Barrier( MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Barrier");
#endif
    SafeMpi( MPI_Barrier( comm ) );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Broadcast
( int* buf, int count, int root, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Broadcast");
#endif
    SafeMpi(
        MPI_Bcast( buf, count, MPI_INT, root, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Broadcast
( float* buf, int count, int root, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Broadcast");
#endif
    SafeMpi(
        MPI_Bcast( buf, count, MPI_FLOAT, root, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Broadcast
( double* buf, int count, int root, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Broadcast");
#endif
    SafeMpi(
        MPI_Bcast( buf, count, MPI_DOUBLE, root, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Broadcast
( std::complex<float>* buf, int count, int root, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Broadcast");
#endif
    /*
    SafeMpi(
        MPI_Bcast( buf, count, MPI_COMPLEX, root, comm )
    );
    */
    SafeMpi(
        MPI_Bcast( buf, 2*count, MPI_FLOAT, root, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Broadcast
( std::complex<double>* buf, int count, int root, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Broadcast");
#endif
    /*
    SafeMpi(
        MPI_Bcast( buf, count, MPI_DOUBLE_COMPLEX, root, comm )
    );
    */
    SafeMpi(
        MPI_Bcast( buf, 2*count, MPI_DOUBLE, root, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void CommDup( MPI_Comm comm, MPI_Comm& commDup )
{
#ifndef RELEASE
    PushCallStack("mpi::CommDup");
#endif
    SafeMpi( MPI_Comm_dup( comm, &commDup ) );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void CommFree( MPI_Comm& comm )
{
#ifndef RELEASE
    PushCallStack("mpi::CommFree");
#endif
    SafeMpi( MPI_Comm_free( &comm ) );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline int CommRank( MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::CommRank");
#endif
    int rank;
    SafeMpi( MPI_Comm_rank( comm, &rank ) );
#ifndef RELEASE
    PopCallStack();
#endif
    return rank;
}

inline int CommSize( MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::CommSize");
#endif
    int size;
    SafeMpi( MPI_Comm_size( comm, &size ) );
#ifndef RELEASE
    PopCallStack();
#endif
    return size;
}

inline void CommSplit
( MPI_Comm comm, int color, int key, MPI_Comm& newComm )
{
#ifndef RELEASE
    PushCallStack("mpi::CommSplit");
#endif
    SafeMpi( MPI_Comm_split( comm, color, key, &newComm ) );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Gather
( const int* sendBuf, int sendCount,
        int* recvBuf, int recvCount, int root, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Gather");
#endif
    SafeMpi(
        MPI_Gather
        ( const_cast<int*>(sendBuf), sendCount, MPI_INT,
          recvBuf,                   recvCount, MPI_INT, root, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Gather
( const float* sendBuf, int sendCount,
        float* recvBuf, int recvCount, int root, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Gather");
#endif
    SafeMpi(
        MPI_Gather
        ( const_cast<float*>(sendBuf), sendCount, MPI_FLOAT,
          recvBuf,                     recvCount, MPI_FLOAT, root, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Gather
( const double* sendBuf, int sendCount,
        double* recvBuf, int recvCount, int root, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Gather");
#endif
    SafeMpi(
        MPI_Gather
        ( const_cast<double*>(sendBuf), sendCount, MPI_DOUBLE,
          recvBuf,                      recvCount, MPI_DOUBLE, root, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Gather
( const std::complex<float>* sendBuf, int sendCount,
        std::complex<float>* recvBuf, int recvCount, int root, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Gather");
#endif
    SafeMpi(
        MPI_Gather
        ( const_cast<std::complex<float>*>(sendBuf), 
          sendCount, MPI_COMPLEX,
          recvBuf,  
          recvCount, MPI_COMPLEX, root, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Gather
( const std::complex<double>* sendBuf, int sendCount,
        std::complex<double>* recvBuf, int recvCount, 
  int root, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Gather");
#endif
    SafeMpi(
        MPI_Gather
        ( const_cast<std::complex<double>*>(sendBuf), 
          sendCount, MPI_DOUBLE_COMPLEX,
          recvBuf,  
          recvCount, MPI_DOUBLE_COMPLEX, root, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Send
( const byte* buf, int count, int dest, int tag, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Send");
#endif
    SafeMpi(
        MPI_Send
        ( const_cast<byte*>(buf), count, MPI_UNSIGNED_CHAR, dest, tag, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Send
( const int* buf, int count, int dest, int tag, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Send");
#endif
    SafeMpi(
        MPI_Send
        ( const_cast<int*>(buf), count, MPI_INT, dest, tag, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Send
( const float* buf, int count, int dest, int tag, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Send");
#endif
    SafeMpi(
        MPI_Send
        ( const_cast<float*>(buf), count, MPI_FLOAT, dest, tag, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Send
( const double* buf, int count, int dest, int tag, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Send");
#endif
    SafeMpi(
        MPI_Send
        ( const_cast<double*>(buf), count, MPI_DOUBLE, dest, tag, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Send
( const std::complex<float>* buf, int count, int dest, int tag, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Send");
#endif
    /*
    SafeMpi(
        MPI_Send
        ( const_cast<std::complex<float>*>(buf), 
          count, MPI_COMPLEX, dest, tag, comm )
    );
    */
    SafeMpi(
        MPI_Send
        ( const_cast<std::complex<float>*>(buf),
          2*count, MPI_FLOAT, dest, tag, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Send
( const std::complex<double>* buf, int count, int dest, int tag, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Send");
#endif
    /*
    SafeMpi(
        MPI_Send
        ( const_cast<std::complex<double>*>(buf), 
          count, MPI_DOUBLE_COMPLEX, dest, tag, comm )
    );
    */
    SafeMpi(
        MPI_Send
        ( const_cast<std::complex<double>*>(buf),
          2*count, MPI_DOUBLE, dest, tag, comm )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Recv
( byte* buf, int count, int source, int tag, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Recv");
#endif
    MPI_Status status;
    SafeMpi(
        MPI_Recv
        ( buf, count, MPI_UNSIGNED_CHAR, source, tag, comm, &status )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Recv
( int* buf, int count, int source, int tag, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Recv");
#endif
    MPI_Status status;
    SafeMpi(
        MPI_Recv
        ( buf, count, MPI_INT, source, tag, comm, &status )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Recv
( float* buf, int count, int source, int tag, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Recv");
#endif
    MPI_Status status;
    SafeMpi(
        MPI_Recv
        ( buf, count, MPI_FLOAT, source, tag, comm, &status )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Recv
( double* buf, int count, int source, int tag, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Recv");
#endif
    MPI_Status status;
    SafeMpi(
        MPI_Recv
        ( buf, count, MPI_DOUBLE, source, tag, comm, &status )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Recv
( std::complex<float>* buf, int count, int source, int tag, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Recv");
#endif
    MPI_Status status;
    /*
    SafeMpi(
        MPI_Recv
        ( buf, count, MPI_COMPLEX, source, tag, comm, &status )
    );
    */
    SafeMpi(
        MPI_Recv
        ( buf, 2*count, MPI_FLOAT, source, tag, comm, &status )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Recv
( std::complex<double>* buf, int count, int source, int tag, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Recv");
#endif
    MPI_Status status;
    /*
    SafeMpi(
        MPI_Recv
        ( buf, count, MPI_DOUBLE_COMPLEX, source, tag, comm, &status )
    );
    */
    SafeMpi(
        MPI_Recv
        ( buf, 2*count, MPI_DOUBLE, source, tag, comm, &status )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void SendRecv
( const byte* sendBuf, int sendCount, int dest,   int sendTag,
        byte* recvBuf, int recvCount, int source, int recvTag, 
  MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::SendRecv");
#endif
    MPI_Status status;
    SafeMpi(
        MPI_Sendrecv
        ( const_cast<byte*>(sendBuf), 
          sendCount, MPI_UNSIGNED_CHAR, dest, sendTag,
          recvBuf,
          recvCount, MPI_UNSIGNED_CHAR, source, recvTag,
          comm, &status )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void SendRecv
( const int* sendBuf, int sendCount, int dest,   int sendTag,
        int* recvBuf, int recvCount, int source, int recvTag, 
  MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::SendRecv");
#endif
    MPI_Status status;
    SafeMpi(
        MPI_Sendrecv
        ( const_cast<int*>(sendBuf), 
          sendCount, MPI_INT, dest, sendTag,
          recvBuf,
          recvCount, MPI_INT, source, recvTag, 
          comm, &status )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void SendRecv
( const float* sendBuf, int sendCount, int dest,   int sendTag,
        float* recvBuf, int recvCount, int source, int recvTag, 
  MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::SendRecv");
#endif
    MPI_Status status;
    SafeMpi(
        MPI_Sendrecv
        ( const_cast<float*>(sendBuf), 
          sendCount, MPI_FLOAT, dest, sendTag, 
          recvBuf,
          recvCount, MPI_FLOAT, source, recvTag,
          comm, &status )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void SendRecv
( const double* sendBuf, int sendCount, int dest,   int sendTag,
        double* recvBuf, int recvCount, int source, int recvTag, 
  MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::SendRecv");
#endif
    MPI_Status status;
    SafeMpi(
        MPI_Sendrecv
        ( const_cast<double*>(sendBuf), 
          sendCount, MPI_DOUBLE, dest, sendTag,
          recvBuf,
          recvCount, MPI_DOUBLE, source, recvTag,
          comm, &status )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void SendRecv
( const std::complex<float>* sendBuf, int sendCount, int dest,   int sendTag,
        std::complex<float>* recvBuf, int recvCount, int source, int recvTag, 
  MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::SendRecv");
#endif
    MPI_Status status;
    SafeMpi(
        MPI_Sendrecv
        ( const_cast<std::complex<float>*>(sendBuf), 
          2*sendCount, MPI_FLOAT, dest, sendTag,
          recvBuf,
          2*recvCount, MPI_FLOAT, source, recvTag,
          comm, &status )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void SendRecv
( const std::complex<double>* sendBuf, int sendCount, int dest,   int sendTag,
        std::complex<double>* recvBuf, int recvCount, int source, int recvTag, 
  MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::SendRecv");
#endif
    MPI_Status status;
    SafeMpi(
        MPI_Sendrecv
        ( const_cast<std::complex<double>*>(sendBuf), 
          2*sendCount, MPI_DOUBLE, dest, sendTag,
          recvBuf,
          2*recvCount, MPI_DOUBLE, source, recvTag,
          comm, &status )
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Reduce
( const int* sendBuf, int* recvBuf, 
  int count, int root, MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Reduce");
#endif
    SafeMpi( 
        MPI_Reduce
        ( const_cast<int*>(sendBuf), recvBuf, count, 
          MPI_INT, op, root, comm ) 
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Reduce
( const float* sendBuf, float* recvBuf, 
  int count, int root, MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Reduce");
#endif
    SafeMpi( 
        MPI_Reduce
        ( const_cast<float*>(sendBuf), recvBuf, count, 
          MPI_FLOAT, op, root, comm ) 
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Reduce
( const double* sendBuf, double* recvBuf, 
  int count, int root, MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Reduce");
#endif
    SafeMpi( 
        MPI_Reduce
        ( const_cast<double*>(sendBuf), recvBuf, count, 
          MPI_DOUBLE, op, root, comm ) 
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Reduce
( const std::complex<float>* sendBuf, std::complex<float>* recvBuf, 
  int count, int root, MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Reduce");
#endif
    if( op == MPI_SUM )
    {
        SafeMpi(
            MPI_Reduce
            ( const_cast<std::complex<float>*>(sendBuf), recvBuf, 2*count,
              MPI_FLOAT, op, root, comm )
        );
    }
    else
    {
        SafeMpi( 
            MPI_Reduce
            ( const_cast<std::complex<float>*>(sendBuf), recvBuf, count, 
              MPI_COMPLEX, op, root, comm ) 
        );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void Reduce
( const std::complex<double>* sendBuf, std::complex<double>* recvBuf, 
  int count, int root, MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::Reduce");
#endif
    if( op == MPI_SUM )
    {
        SafeMpi(
            MPI_Reduce
            ( const_cast<std::complex<double>*>(sendBuf), recvBuf, 2*count,
              MPI_DOUBLE, op, root, comm )
        );
    }
    else
    {
        SafeMpi( 
            MPI_Reduce
            ( const_cast<std::complex<double>*>(sendBuf), recvBuf, count, 
              MPI_DOUBLE_COMPLEX, op, root, comm ) 
        );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void ReduceScatter
( const int* sendBuf, int* recvBuf, const int* recvCounts, 
  MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::ReduceScatter");
#endif
    SafeMpi( 
        MPI_Reduce_scatter
        ( const_cast<int*>(sendBuf), 
          recvBuf, 
          const_cast<int*>(recvCounts), 
          MPI_INT, 
          op, 
          comm ) 
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void ReduceScatter
( const float* sendBuf, float* recvBuf, const int* recvCounts, 
  MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::ReduceScatter");
#endif
    SafeMpi( 
        MPI_Reduce_scatter
        ( const_cast<float*>(sendBuf), 
          recvBuf, 
          const_cast<int*>(recvCounts),
          MPI_FLOAT, 
          op, 
          comm ) 
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void ReduceScatter
( const double* sendBuf, double* recvBuf, const int* recvCounts, 
  MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::ReduceScatter");
#endif
    SafeMpi( 
        MPI_Reduce_scatter
        ( const_cast<double*>(sendBuf), 
          recvBuf, 
          const_cast<int*>(recvCounts), 
          MPI_DOUBLE, 
          op, 
          comm ) 
    );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void ReduceScatter
( const std::complex<float>* sendBuf, std::complex<float>* recvBuf, 
  const int* recvCounts, MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::ReduceScatter");
#endif
    if( op == MPI_SUM )
    {
        const int p = mpi::CommSize( comm );
        std::vector<int> recvCountsDoubled(p);
        for( int i=0; i<p; ++i )
            recvCountsDoubled[i] = 2*recvCounts[i];
        SafeMpi(
            MPI_Reduce_scatter
            ( const_cast<std::complex<float>*>(sendBuf), recvBuf, 
              &recvCountsDoubled[0], MPI_FLOAT, op, comm )
        );
    }
    else
    {
        SafeMpi( 
            MPI_Reduce_scatter
            ( const_cast<std::complex<float>*>(sendBuf), 
              recvBuf, 
              const_cast<int*>(recvCounts), 
              MPI_COMPLEX, 
              op, 
              comm ) 
        );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void ReduceScatter
( const std::complex<double>* sendBuf, std::complex<double>* recvBuf, 
  const int* recvCounts, MPI_Op op, MPI_Comm comm )
{
#ifndef RELEASE
    PushCallStack("mpi::ReduceScatter");
#endif
    if( op == MPI_SUM )
    {
        const int p = mpi::CommSize( comm );
        std::vector<int> recvCountsDoubled(p);
        for( int i=0; i<p; ++i )
            recvCountsDoubled[i] = 2*recvCounts[i];
        SafeMpi(
            MPI_Reduce_scatter
            ( const_cast<std::complex<double>*>(sendBuf), recvBuf, 
              &recvCountsDoubled[0], MPI_DOUBLE, op, comm )
        );
    }
    else
    {
        SafeMpi( 
            MPI_Reduce_scatter
            ( const_cast<std::complex<double>*>(sendBuf), 
              recvBuf, 
              const_cast<int*>(recvCounts), 
              MPI_DOUBLE_COMPLEX, 
              op, 
              comm ) 
        );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

inline double WallTime()
{ 
    return MPI_Wtime();
}

} // namespace mpi
} // namespace psp

#endif // PSP_MPI_HPP
