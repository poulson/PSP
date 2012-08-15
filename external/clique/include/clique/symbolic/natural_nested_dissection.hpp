/*
   Clique: a scalable implementation of the multifrontal algorithm

   Copyright (C) 2011-2012 Jack Poulson, Lexing Ying, and 
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

namespace cliq {

void NaturalNestedDissection
(       int nx, 
        int ny, 
        int nz,
  const DistGraph& graph, 
        DistMap& map,
        DistSeparatorTree& sepTree, 
        DistSymmInfo& info,
        int cutoff=128, 
        bool storeFactRecvIndices=false );

int NaturalBisect
(       int nx, 
        int ny, 
        int nz, 
  const Graph& graph, 
        int& nxLeft, 
        int& nyLeft, 
        int& nzLeft,
        Graph& leftChild, 
        int& nxRight, 
        int& nyRight, 
        int& nzRight,
        Graph& rightChild, 
        std::vector<int>& perm );

// NOTE: for two or more processes
int NaturalBisect
(       int nx, 
        int ny, 
        int nz,
  const DistGraph& graph, 
        int& nxChild, 
        int& nyChild, 
        int& nzChild,
        DistGraph& child, 
        DistMap& perm,
        bool& onLeft );

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

inline void
NaturalNestedDissectionRecursion
(       int nx,
        int ny,
        int nz,
  const Graph& graph, 
  const std::vector<int>& perm,
        DistSeparatorTree& sepTree, 
        DistSymmElimTree& eTree,
        int parent, 
        int offset, 
        int cutoff=128 )
{
#ifndef RELEASE
    PushCallStack("NaturalNestedDissectionRecursion");
#endif
    if( graph.NumSources() <= cutoff )
    {
        // Fill in this node of the local separator tree
        const int numSources = graph.NumSources();
        sepTree.localSepsAndLeaves.push_back( new LocalSepOrLeaf );
        LocalSepOrLeaf& leaf = *sepTree.localSepsAndLeaves.back();
        leaf.parent = parent;
        leaf.offset = offset;
        leaf.indices = perm;

        // Fill in this node of the local elimination tree
        eTree.localNodes.push_back( new LocalSymmNode );
        LocalSymmNode& node = *eTree.localNodes.back();
        node.size = numSources;
        node.offset = offset;
        node.parent = parent;
        node.children.clear();
        std::set<int> connectedAncestors;
        for( int s=0; s<node.size; ++s )
        {
            const int numConnections = graph.NumConnections( s );
            const int edgeOffset = graph.EdgeOffset( s );
            for( int t=0; t<numConnections; ++t )
            {
                const int target = graph.Target( edgeOffset+t );
                if( target >= numSources )
                    connectedAncestors.insert( offset+target );
            }
        }
        node.lowerStruct.resize( connectedAncestors.size() );
        std::copy
        ( connectedAncestors.begin(), connectedAncestors.end(), 
          node.lowerStruct.begin() );
    }
    else
    {
        // Partition the graph and construct the inverse map
        int nxLeft, nyLeft, nzLeft, nxRight, nyRight, nzRight;
        Graph leftChild, rightChild;
        std::vector<int> map;
        const int sepSize = 
            NaturalBisect
            ( nx, ny, nz, graph, 
              nxLeft, nyLeft, nzLeft, leftChild, 
              nxRight, nyRight, nzRight, rightChild, map );
        const int numSources = graph.NumSources();
        std::vector<int> inverseMap( numSources );
        for( int s=0; s<numSources; ++s )
            inverseMap[map[s]] = s;

        // Mostly compute this node of the local separator tree
        // (we will finish computing the separator indices soon)
        sepTree.localSepsAndLeaves.push_back( new LocalSepOrLeaf );
        LocalSepOrLeaf& sep = *sepTree.localSepsAndLeaves.back();
        sep.parent = parent;
        sep.offset = offset + (numSources-sepSize);
        sep.indices.resize( sepSize );
        for( int s=0; s<sepSize; ++s )
        {
            const int mappedSource = s + (numSources-sepSize);
            sep.indices[s] = inverseMap[mappedSource];
        }
    
        // Fill in this node in the local elimination tree
        eTree.localNodes.push_back( new LocalSymmNode );
        LocalSymmNode& node = *eTree.localNodes.back();
        node.size = sepSize;
        node.offset = sep.offset;
        node.parent = parent;
        node.children.resize( 2 );
        std::set<int> connectedAncestors;
        for( int s=0; s<sepSize; ++s )
        {
            const int source = sep.indices[s];
            const int numConnections = graph.NumConnections( source );
            const int edgeOffset = graph.EdgeOffset( source );
            for( int t=0; t<numConnections; ++t )
            {
                const int target = graph.Target( edgeOffset+t );
                if( target >= numSources )
                    connectedAncestors.insert( offset+target );
            }
        }
        node.lowerStruct.resize( connectedAncestors.size() );
        std::copy
        ( connectedAncestors.begin(), connectedAncestors.end(), 
          node.lowerStruct.begin() );

        // Finish computing the separator indices
        for( int s=0; s<sepSize; ++s )
            sep.indices[s] = perm[sep.indices[s]];

        // Construct the inverse maps from the child indices to the original
        // degrees of freedom
        const int leftChildSize = leftChild.NumSources();
        std::vector<int> leftPerm( leftChildSize );
        for( int s=0; s<leftChildSize; ++s )
            leftPerm[s] = perm[inverseMap[s]];
        const int rightChildSize = rightChild.NumSources();
        std::vector<int> rightPerm( rightChildSize );
        for( int s=0; s<rightChildSize; ++s )
            rightPerm[s] = perm[inverseMap[s+leftChildSize]];

        // Update right then left so that, once we later reverse the order 
        // of the nodes, the left node will be ordered first
        const int parent = eTree.localNodes.size()-1;
        node.children[1] = eTree.localNodes.size();
        NaturalNestedDissectionRecursion
        ( nxRight, nyRight, nzRight, rightChild, rightPerm, sepTree, eTree, 
          parent, offset+leftChildSize, cutoff );
        node.children[0] = eTree.localNodes.size();
        NaturalNestedDissectionRecursion
        ( nxLeft, nyLeft, nzLeft, leftChild, leftPerm, sepTree, eTree, 
          parent, offset, cutoff );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void
NaturalNestedDissectionRecursion
(       int nx,
        int ny,
        int nz,
  const DistGraph& graph, 
  const DistMap& perm,
        DistSeparatorTree& sepTree, 
        DistSymmElimTree& eTree,
        int depth, 
        int offset, 
        bool onLeft,
        int cutoff=128 )
{
#ifndef RELEASE
    PushCallStack("NaturalNestedDissectionRecursion");
#endif
    const int distDepth = sepTree.distSeps.size();
    mpi::Comm comm = graph.Comm();
    if( distDepth - depth > 0 )
    {
        // Partition the graph and construct the inverse map
        int nxChild, nyChild, nzChild;
        DistGraph child;
        bool childIsOnLeft;
        DistMap map;
        const int sepSize = 
            NaturalBisect
            ( nx, ny, nz, graph, nxChild, nyChild, nzChild, child, 
              map, childIsOnLeft );
        const int numSources = graph.NumSources();
        const int childSize = child.NumSources();
        const int leftChildSize = 
            ( childIsOnLeft ? childSize : numSources-sepSize-childSize );

        DistMap inverseMap;
        map.FormInverse( inverseMap );

        // Mostly fill this node of the DistSeparatorTree
        // (we will finish computing the separator indices at the end)
        DistSeparator& sep = sepTree.distSeps[distDepth-1-depth];
        mpi::CommDup( comm, sep.comm );
        sep.offset = offset + (numSources-sepSize);
        sep.indices.resize( sepSize );
        for( int s=0; s<sepSize; ++s )
            sep.indices[s] = s + (numSources-sepSize);
        inverseMap.Translate( sep.indices );

        // Fill in this node of the DistSymmElimTree
        DistSymmNode& node = eTree.distNodes[distDepth-depth];
        node.size = sepSize;
        node.offset = sep.offset;
        node.onLeft = onLeft;
        mpi::CommDup( comm, node.comm );
        const int numLocalSources = graph.NumLocalSources();
        const int firstLocalSource = graph.FirstLocalSource();
        std::set<int> localConnectedAncestors;
        for( int s=0; s<sepSize; ++s )
        {
            const int source = sep.indices[s];
            if( source >= firstLocalSource && 
                source < firstLocalSource+numLocalSources )
            {
                const int localSource = source - firstLocalSource;
                const int numConnections = graph.NumConnections( localSource );
                const int localOffset = graph.LocalEdgeOffset( localSource );
                for( int t=0; t<numConnections; ++t )
                {
                    const int target = graph.Target( localOffset+t );
                    if( target >= numSources )
                        localConnectedAncestors.insert( offset+target );
                }
            }
        }
        const int numLocalConnected = localConnectedAncestors.size();
        const int commSize = mpi::CommSize( comm );
        std::vector<int> localConnectedSizes( commSize );
        mpi::AllGather
        ( &numLocalConnected, 1, &localConnectedSizes[0], 1, comm );
        std::vector<int> localConnectedVector( numLocalConnected );
        std::copy
        ( localConnectedAncestors.begin(), localConnectedAncestors.end(), 
          localConnectedVector.begin() );
        int sumOfLocalConnectedSizes=0;
        std::vector<int> localConnectedOffsets( commSize );
        for( int q=0; q<commSize; ++q )
        {
            localConnectedOffsets[q] = sumOfLocalConnectedSizes;
            sumOfLocalConnectedSizes += localConnectedSizes[q];
        }
        std::vector<int> localConnections( sumOfLocalConnectedSizes );
        mpi::AllGather
        ( &localConnectedVector[0], numLocalConnected,
          &localConnections[0], 
          &localConnectedSizes[0], &localConnectedOffsets[0], comm );
        std::set<int> connectedAncestors
        ( localConnections.begin(), localConnections.end() );
        node.lowerStruct.resize( connectedAncestors.size() );
        std::copy
        ( connectedAncestors.begin(), connectedAncestors.end(), 
          node.lowerStruct.begin() );

        // Finish computing the separator indices
        perm.Translate( sep.indices );

        // Construct map from child indices to the original ordering
        DistMap newPerm( child.NumSources(), child.Comm() );
        const int localChildSize = child.NumLocalSources();
        const int firstLocalChildSource = child.FirstLocalSource();
        if( childIsOnLeft )
            for( int s=0; s<localChildSize; ++s )
                newPerm.SetLocal( s, s+firstLocalChildSource );
        else
            for( int s=0; s<localChildSize; ++s )
                newPerm.SetLocal( s, s+firstLocalChildSource+leftChildSize );
        inverseMap.Extend( newPerm );
        perm.Extend( newPerm );

        // Recurse
        const int newOffset = ( childIsOnLeft ? offset : offset+leftChildSize );
        NaturalNestedDissectionRecursion
        ( nxChild, nyChild, nzChild, child, newPerm, sepTree, eTree, depth+1, 
          newOffset, childIsOnLeft, cutoff );
    }
    else if( graph.NumSources() <= cutoff )
    {
        // Convert to a sequential graph
        const int numSources = graph.NumSources();
        Graph seqGraph( graph );

        // Fill in this node of the local separator tree
        sepTree.localSepsAndLeaves.push_back( new LocalSepOrLeaf );
        LocalSepOrLeaf& leaf = *sepTree.localSepsAndLeaves.back();
        leaf.parent = -1;
        leaf.offset = offset;
        leaf.indices = perm.LocalMap();

        // Fill in this node of the local and distributed parts of the 
        // elimination tree
        eTree.localNodes.push_back( new LocalSymmNode );
        LocalSymmNode& localNode = *eTree.localNodes.back();
        DistSymmNode& distNode = eTree.distNodes[0];
        mpi::CommDup( comm, distNode.comm );
        distNode.onLeft = onLeft;
        distNode.size = localNode.size = numSources;
        distNode.offset = localNode.offset = offset;
        localNode.parent = -1;
        localNode.children.clear();
        std::set<int> connectedAncestors;
        for( int s=0; s<numSources; ++s )
        {
            const int numConnections = seqGraph.NumConnections( s );
            const int edgeOffset = seqGraph.EdgeOffset( s );
            for( int t=0; t<numConnections; ++t )
            {
                const int target = seqGraph.Target( edgeOffset+t );
                if( target >= numSources )
                    connectedAncestors.insert( offset+target );
            }
        }
        localNode.lowerStruct.resize( connectedAncestors.size() );
        std::copy
        ( connectedAncestors.begin(), connectedAncestors.end(), 
          localNode.lowerStruct.begin() );    
        distNode.lowerStruct = localNode.lowerStruct;
    }
    else
    {
        // Convert to a sequential graph
        Graph seqGraph( graph );

        // Partition the graph and construct the inverse map
        int nxLeft, nyLeft, nzLeft, nxRight, nyRight, nzRight;
        Graph leftChild, rightChild;
        std::vector<int> map;
        const int sepSize = 
            NaturalBisect
            ( nx, ny, nz, seqGraph, 
              nxLeft, nyLeft, nzLeft, leftChild, 
              nxRight, nyRight, nzRight, rightChild, map );
        const int numSources = graph.NumSources();
        std::vector<int> inverseMap( numSources );
        for( int s=0; s<numSources; ++s )
            inverseMap[map[s]] = s;

        // Mostly compute this node of the local separator tree
        // (we will finish computing the separator indices soon)
        sepTree.localSepsAndLeaves.push_back( new LocalSepOrLeaf );
        LocalSepOrLeaf& sep = *sepTree.localSepsAndLeaves.back();
        sep.parent = -1;
        sep.offset = offset + (numSources-sepSize);
        sep.indices.resize( sepSize );
        for( int s=0; s<sepSize; ++s )
        {
            const int mappedSource = s + (numSources-sepSize);
            sep.indices[s] = inverseMap[mappedSource];
        }
        
        // Fill in this node in both the local and distributed parts of 
        // the elimination tree
        eTree.localNodes.push_back( new LocalSymmNode );
        LocalSymmNode& localNode = *eTree.localNodes.back();
        DistSymmNode& distNode = eTree.distNodes[0];
        mpi::CommDup( comm, distNode.comm );
        distNode.onLeft = onLeft;
        distNode.size = localNode.size = sepSize;
        distNode.offset = localNode.offset = sep.offset;
        localNode.parent = -1;
        localNode.children.resize( 2 );
        std::set<int> connectedAncestors;
        for( int s=0; s<sepSize; ++s )
        {
            const int source = sep.indices[s];
            const int numConnections = seqGraph.NumConnections( source );
            const int edgeOffset = seqGraph.EdgeOffset( source );
            for( int t=0; t<numConnections; ++t )
            {
                const int target = seqGraph.Target( edgeOffset+t );
                if( target >= numSources )
                    connectedAncestors.insert( offset+target );
            }
        }
        localNode.lowerStruct.resize( connectedAncestors.size() );
        std::copy
        ( connectedAncestors.begin(), connectedAncestors.end(), 
          localNode.lowerStruct.begin() );
        distNode.lowerStruct = localNode.lowerStruct;

        // Finish computing the separator indices
        // (This is a faster version of the Translate member function)
        for( int s=0; s<sepSize; ++s )
            sep.indices[s] = perm.GetLocal( sep.indices[s] );

        // Construct the inverse maps from the child indices to the original
        // degrees of freedom
        const int leftChildSize = leftChild.NumSources();
        std::vector<int> leftPerm( leftChildSize );
        for( int s=0; s<leftChildSize; ++s )
            leftPerm[s] = perm.GetLocal( inverseMap[s] );
        const int rightChildSize = rightChild.NumSources();
        std::vector<int> rightPerm( rightChildSize );
        for( int s=0; s<rightChildSize; ++s )
            rightPerm[s] = perm.GetLocal( inverseMap[s+leftChildSize] );

        // Update right then left so that, once we later reverse the order 
        // of the nodes, the left node will be ordered first
        const int parent=0;
        localNode.children[1] = eTree.localNodes.size();
        NaturalNestedDissectionRecursion
        ( nxRight, nyRight, nzRight, rightChild, rightPerm, sepTree, eTree, 
          parent, offset+leftChildSize, cutoff );
        localNode.children[0] = eTree.localNodes.size();
        NaturalNestedDissectionRecursion
        ( nxLeft, nyLeft, nzLeft, leftChild, leftPerm, sepTree, eTree, 
          parent, offset, cutoff );
    }
#ifndef RELEASE
    PopCallStack();
#endif
}

inline void 
NaturalNestedDissection
(       int nx,
        int ny,
        int nz,
  const DistGraph& graph, 
        DistMap& map,
        DistSeparatorTree& sepTree, 
        DistSymmInfo& info,
        int cutoff, 
        bool storeFactRecvIndices )
{
#ifndef RELEASE
    PushCallStack("NaturalNestedDissection");
#endif
    // NOTE: There is a potential memory leak here if these data structures 
    //       are reused. Their destructors should call a member function which
    //       we can simply call here to clear the data
    DistSymmElimTree eTree;
    eTree.localNodes.clear();
    sepTree.localSepsAndLeaves.clear();

    mpi::Comm comm = graph.Comm();
    const int distDepth = DistributedDepth( comm );
    eTree.distNodes.resize( distDepth+1 );
    sepTree.distSeps.resize( distDepth );

    DistMap perm( graph.NumSources(), graph.Comm() );
    const int firstLocalSource = perm.FirstLocalSource();
    const int numLocalSources = perm.NumLocalSources();
    for( int s=0; s<numLocalSources; ++s )
        perm.SetLocal( s, s+firstLocalSource );
    NaturalNestedDissectionRecursion
    ( nx, ny, nz, graph, perm, sepTree, eTree, 0, 0, false, cutoff );

    ReverseOrder( sepTree, eTree );

    // Construct the distributed reordering    
    BuildMap( graph, sepTree, map );
#ifndef RELEASE
    EnsurePermutation( map );
#endif

    // Run the symbolic analysis
    SymmetricAnalysis( eTree, info, storeFactRecvIndices );
#ifndef RELEASE
    PopCallStack();
#endif
}

inline int 
NaturalBisect
(       int nx, 
        int ny, 
        int nz,
  const Graph& graph, 
        int& nxLeft,
        int& nyLeft,
        int& nzLeft,
        Graph& leftChild, 
        int& nxRight,
        int& nyRight,
        int& nzRight,
        Graph& rightChild,
        std::vector<int>& perm )
{
#ifndef RELEASE
    PushCallStack("NaturalBisect");
#endif
    const int numSources = graph.NumSources();
    if( numSources == 0 )
        throw std::logic_error
        ("There is no reason to bisect an empty sequential graph");

    int leftChildSize, rightChildSize, sepSize;
    perm.resize( numSources );
    if( nx >= ny && nx >= nz )
    {
        nxLeft = (nx-1)/2;
        nyLeft = ny;
        nzLeft = nz;
        leftChildSize = nxLeft*nyLeft*nzLeft;

        nxRight = nx-1-nxLeft;
        nyRight = ny;
        nzRight = nz;
        rightChildSize = nxRight*nyRight*nzRight;

        sepSize = ny*nz;

        // Fill the left side
        int offset=0;
        for( int z=0; z<nz; ++z )
            for( int y=0; y<ny; ++y )
                for( int x=0; x<nxLeft; ++x )
                    perm[x+y*nx+z*nx*ny] = offset++;

        // Fill the right side
        offset = leftChildSize;
        for( int z=0; z<nz; ++z )
            for( int y=0; y<ny; ++y )
                for( int x=nxLeft+1; x<nx; ++x )
                    perm[x+y*nx+z*nx*ny] = offset++;

        // Fill the separator
        offset=leftChildSize+rightChildSize;
        for( int z=0; z<nz; ++z )
            for( int y=0; y<ny; ++y )
                perm[nxLeft+y*nx+z*nx*ny] = offset++;
    }
    else if( ny >= nx && ny >= nz )
    {
        nxLeft = nx;
        nyLeft = (ny-1)/2;
        nzLeft = nz;
        leftChildSize = nxLeft*nyLeft*nzLeft;

        nxRight = nx;
        nyRight = ny-1-nyLeft;
        nzRight = nz;
        rightChildSize = nxRight*nyRight*nzRight;

        sepSize = nx*nz;

        // Fill the left side
        int offset=0;
        for( int z=0; z<nz; ++z )
            for( int y=0; y<nyLeft; ++y )
                for( int x=0; x<nx; ++x )
                    perm[x+y*nx+z*nx*ny] = offset++;

        // Fill the right side
        offset = leftChildSize;
        for( int z=0; z<nz; ++z )
            for( int y=nyLeft+1; y<ny; ++y )
                for( int x=0; x<nx; ++x )
                    perm[x+y*nx+z*nx*ny] = offset++;

        // Fill the separator
        offset=leftChildSize+rightChildSize;
        for( int z=0; z<nz; ++z )
            for( int x=0; x<nx; ++x )
                perm[x+nyLeft*nx+z*nx*ny] = offset++;
    }
    else
    {
        nxLeft = nx;
        nyLeft = ny;
        nzLeft = (nz-1)/2;
        leftChildSize = nxLeft*nyLeft*nzLeft;

        nxRight = nx;
        nyRight = ny;
        nzRight = nz-1-nzLeft;
        rightChildSize = nxRight*nyRight*nzRight;

        sepSize = nx*ny;

        // Fill the left side
        int offset=0;
        for( int z=0; z<nzLeft; ++z )
            for( int y=0; y<ny; ++y )
                for( int x=0; x<nx; ++x )
                    perm[x+y*nx+z*nx*ny] = offset++;

        // Fill the right side
        offset = leftChildSize;
        for( int z=nzLeft+1; z<nz; ++z )
            for( int y=0; y<ny; ++y )
                for( int x=0; x<nx; ++x )
                    perm[x+y*nx+z*nx*ny] = offset++;

        // Fill the separator
        offset=leftChildSize+rightChildSize;
        for( int y=0; y<ny; ++y )
            for( int x=0; x<nx; ++x )
                perm[x+y*nx+nzLeft*nx*ny] = offset++;
    }
#ifndef RELEASE
    EnsurePermutation( perm );
#endif

    BuildChildrenFromPerm
    ( graph, perm, leftChildSize, leftChild, rightChildSize, rightChild );
#ifndef RELEASE
    PopCallStack();
#endif
    return sepSize;
}

inline int 
NaturalBisect
(       int nx,
        int ny,
        int nz,
  const DistGraph& graph, 
        int& nxChild,
        int& nyChild,
        int& nzChild,
        DistGraph& child, 
        DistMap& perm,
        bool& onLeft )
{
#ifndef RELEASE
    PushCallStack("NaturalBisect");
#endif
    const int numSources = graph.NumSources();
    const int firstLocalSource = graph.FirstLocalSource();
    const int numLocalSources = graph.NumLocalSources();
    mpi::Comm comm = graph.Comm();
    const int commSize = mpi::CommSize( comm );
    const int commRank = mpi::CommRank( comm );
    if( commSize == 1 )
        throw std::logic_error
        ("This routine assumes at least two processes are used, "
         "otherwise one child will be lost");

    int leftChildSize, rightChildSize, sepSize;
    int nxLeft, nyLeft, nzLeft, nxRight, nyRight, nzRight;
    perm.SetComm( comm );
    perm.ResizeTo( numSources );
    if( nx != 0 && ny != 0 && nz != 0 )
    {
        if( nx >= ny && nx >= nz )
        {
            nxLeft = (nx-1)/2;
            nyLeft = ny;
            nzLeft = nz;
            leftChildSize = nxLeft*nyLeft*nzLeft;

            nxRight = nx-1-nxLeft;
            nyRight = ny;
            nzRight = nz;
            rightChildSize = nxRight*nyRight*nzRight;

            sepSize = ny*nz;

            const int rightOffset=leftChildSize, 
                      sepOffset=leftChildSize+rightChildSize;
            for( int iLocal=0; iLocal<numLocalSources; ++iLocal )
            {
                const int i = iLocal + firstLocalSource;
                const int x = i % nx;
                const int y = (i/nx) % ny;
                const int z = i/(nx*ny);
                if( x < nxLeft )
                {
                    const int xLeft = x;
                    const int leftIndex = xLeft + y*nxLeft + z*nxLeft*ny;
                    perm.SetLocal( iLocal, leftIndex );
                }
                else if( x > nxLeft )
                {
                    const int xRight = x-(nxLeft+1);
                    const int rightIndex = xRight + y*nxRight + z*nxRight*ny;
                    perm.SetLocal( iLocal, rightOffset+rightIndex );
                }
                else
                {
                    const int sepIndex = y + z*ny;
                    perm.SetLocal( iLocal, sepOffset+sepIndex );
                }
            }
        }
        else if( ny >= nx && ny >= nz )
        {
            nxLeft = nx;
            nyLeft = (ny-1)/2;
            nzLeft = nz;
            leftChildSize = nxLeft*nyLeft*nzLeft;

            nxRight = nx;
            nyRight = ny-1-nyLeft;
            nzRight = nz;
            rightChildSize = nxRight*nyRight*nzRight;

            sepSize = nx*nz;

            const int rightOffset=leftChildSize, 
                      sepOffset=leftChildSize+rightChildSize;
            for( int iLocal=0; iLocal<numLocalSources; ++iLocal )
            {
                const int i = iLocal + firstLocalSource;
                const int x = i % nx;
                const int y = (i/nx) % ny;
                const int z = i/(nx*ny);
                if( y < nyLeft )
                {
                    const int yLeft = y;
                    const int leftIndex = x + yLeft*nx + z*nx*nyLeft;
                    perm.SetLocal( iLocal, leftIndex );
                }
                else if( y > nyLeft )
                {
                    const int yRight = y - (nyLeft+1);
                    const int rightIndex = x + yRight*nx + z*nx*nyRight;
                    perm.SetLocal( iLocal, rightOffset+rightIndex );
                }
                else
                {
                    const int sepIndex = x + z*nx;
                    perm.SetLocal( iLocal, sepOffset+sepIndex );
                }
            }
        }
        else
        {
            nxLeft = nx;
            nyLeft = ny;
            nzLeft = (nz-1)/2;
            leftChildSize = nxLeft*nyLeft*nzLeft;

            nxRight = nx;
            nyRight = ny;
            nzRight = nz-1-nzLeft;
            rightChildSize = nxRight*nyRight*nzRight;

            sepSize = nx*ny;

            const int rightOffset=leftChildSize, 
                      sepOffset=leftChildSize+rightChildSize;
            for( int iLocal=0; iLocal<numLocalSources; ++iLocal )
            {
                const int i = iLocal + firstLocalSource;
                const int x = i % nx;
                const int y = (i/nx) % ny;
                const int z = i/(nx*ny);
                if( z < nzLeft )
                {
                    const int zLeft = z;
                    const int leftIndex = x + y*nx + zLeft*nx*ny;
                    perm.SetLocal( iLocal, leftIndex );
                }
                else if( z > nzLeft )
                {
                    const int zRight = z - (nzLeft+1);
                    const int rightIndex = x + y*nx + zRight*nx*ny;
                    perm.SetLocal( iLocal, rightOffset+rightIndex );
                }
                else
                {
                    const int sepIndex = x + y*nx;
                    perm.SetLocal( iLocal, sepOffset+sepIndex );
                }
            }
        }
    }
    else
    {
        leftChildSize = rightChildSize = sepSize = 0;
        nxLeft = nx;
        nyLeft = ny;
        nzLeft = nz;
        nxRight = nx;
        nyRight = ny;
        nzRight = nz;
    }
#ifndef RELEASE
    EnsurePermutation( perm );
#endif

    BuildChildFromPerm
    ( graph, perm, leftChildSize, rightChildSize, onLeft, child );

    if( onLeft )
    {
        nxChild = nxLeft;
        nyChild = nyLeft;
        nzChild = nzLeft;
    }
    else
    {
        nxChild = nxRight;
        nyChild = nyRight;
        nzChild = nzRight;
    }
#ifndef RELEASE
    PopCallStack();
#endif
    return sepSize;
}

} // namespace cliq
