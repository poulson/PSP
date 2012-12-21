/*
   Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and 
   The University of Texas at Austin
 
   This file is part of Clique and is under the GNU General Public License,
   which can be found in the LICENSE file in the root directory, or at 
   <http://www.gnu.org/licenses/>.
*/
#include "clique.hpp"
using namespace cliq;

int
main( int argc, char* argv[] )
{
    cliq::Initialize( argc, argv );
    mpi::Comm comm = mpi::COMM_WORLD;
    const int commRank = mpi::CommRank( comm );

    try
    {
        const int n = Input("--n","size of n x n x n grid",30);
        const bool sequential = Input
            ("--sequential","sequential partitions?",true);
        const int numDistSeps = Input
            ("--numDistSeps",
             "number of separators to try per distributed partition",1);
        const int numSeqSeps = Input
            ("--numSeqSeps",
             "number of separators to try per sequential partition",1);
        const int cutoff = Input("--cutoff","cutoff for nested dissection",128);
        ProcessInput();

        const int numVertices = n*n*n;
        DistGraph graph( numVertices, comm );

        const int firstLocalSource = graph.FirstLocalSource();
        const int numLocalSources = graph.NumLocalSources();

        // Fill our portion of the graph of a 3D n x n x n 7-point stencil
        // in natural ordering: (x,y,z) at x + y*n + z*n*n
        if( commRank == 0 )
        {
            std::cout << "Filling local portion of graph...";
            std::cout.flush();
        }
        graph.StartAssembly();
        graph.Reserve( 7*numLocalSources );
        for( int iLocal=0; iLocal<numLocalSources; ++iLocal )
        {
            const int i = firstLocalSource + iLocal;
            const int x = i % n;
            const int y = (i/n) % n;
            const int z = i/(n*n);

            graph.Insert( i, i );
            if( x != 0 )
                graph.Insert( i, i-1 );
            if( x != n-1 )
                graph.Insert( i, i+1 );
            if( y != 0 )
                graph.Insert( i, i-n );
            if( y != n-1 )
                graph.Insert( i, i+n );
            if( z != 0 )
                graph.Insert( i, i-n*n );
            if( z != n-1 )
                graph.Insert( i, i+n*n );
        }
        graph.StopAssembly();
        mpi::Barrier( comm );
        if( commRank == 0 )
            std::cout << "done" << std::endl;

        if( commRank == 0 )
        {
            std::cout << "Running nested dissection...";
            std::cout.flush();
        }
        DistSymmInfo info;
        DistSeparatorTree sepTree;
        DistMap map;
        NestedDissection
        ( graph, map, sepTree, info, 
          sequential, numDistSeps, numSeqSeps, cutoff );
        mpi::Barrier( comm );
        if( commRank == 0 )
            std::cout << "done" << std::endl;

        if( commRank == 0 )
        {
            const int numDistNodes = info.distNodes.size();
            const int numLocalNodes = info.localNodes.size(); 
            const int rootSepSize = info.distNodes.back().size;
            std::cout << "\n"
                      << "On the root process:\n"
                      << "-----------------------------------------\n"
                      << numLocalNodes << " local nodes\n"
                      << numDistNodes  << " distributed nodes\n"
                      << rootSepSize << " vertices in root separator\n"
                      << "\n";
            for( int s=0; s<rootSepSize; ++s )
            {
                const int i = 
                    ( numDistNodes > 1 ? 
                      sepTree.distSeps.back().indices[s] :
                      sepTree.localSepsAndLeaves.back()->indices[s] );
                const int x = i % n;
                const int y = (i/n) % n;
                const int z = i/(n*n);
                std::cout << "rootSep[" << s << "]: " << i << ", ("
                          << x << "," << y << "," << z << ")\n";
            }
            std::cout << std::endl;
        }
    }
    catch( ArgException& e ) { }
    catch( std::exception& e )
    {
        std::ostringstream msg;
        msg << "Process " << commRank << " caught message:\n"
            << e.what() << std::endl;
        std::cerr << msg.str();
#ifndef RELEASE
        elem::DumpCallStack();
        cliq::DumpCallStack();
#endif
    }

    cliq::Finalize();
    return 0;
}
