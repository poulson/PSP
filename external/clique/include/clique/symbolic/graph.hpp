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
#ifndef CLIQUE_GRAPH_HPP
#define CLIQUE_GRAPH_HPP 1

namespace cliq {

class Graph
{
public:
    Graph();
    Graph( int numVertices );
    Graph( int numSources, int numTargets );
    Graph( const Graph& graph );
    // NOTE: This requires the DistGraph to be over a single process
    Graph( const DistGraph& graph );
    ~Graph();

    int NumSources() const;
    int NumTargets() const;

    int NumEdges() const;
    int Capacity() const;

    int Source( int edge ) const;
    int Target( int edge ) const;

    int EdgeOffset( int source ) const;
    int NumConnections( int source ) const;

    void StartAssembly();
    void StopAssembly();
    void Reserve( int numEdges );
    void PushBack( int source, int target );

    void Empty();
    void ResizeTo( int numVertices );
    void ResizeTo( int numSources, int numTargets );

    const Graph& operator=( const Graph& graph );
    // NOTE: This requires the DistGraph to be over a single process
    const Graph& operator=( const DistGraph& graph );

private:
    int numSources_, numTargets_;
    std::vector<int> sources_, targets_;

    // Helpers for local indexing
    bool assembling_, sorted_;
    std::vector<int> edgeOffsets_;
    void ComputeEdgeOffsets();

    static bool ComparePairs
    ( const std::pair<int,int>& a, const std::pair<int,int>& b );

    void EnsureNotAssembling() const;
    void EnsureConsistentSizes() const;
    void EnsureConsistentCapacities() const;

    friend class DistGraph;
    template<typename F> friend class SparseMatrix;
};

} // namespace cliq

#endif // CLIQUE_GRAPH_HPP
