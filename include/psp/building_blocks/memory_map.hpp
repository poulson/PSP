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
#ifndef PSP_MEMORY_MAP_HPP
#define PSP_MEMORY_MAP_HPP 1

namespace psp {

template<typename T1,typename T2> 
class MemoryMap 
{   
private:
    std::map<T1,T2*> _baseMap;
public:
    // NOTE: Insertion with the same key without manual deletion
    //       will cause a memory leak.
    T2*& operator[]( T1 key ) { return _baseMap[key]; }
    
    void Clear()
    {
        typename std::map<T1,T2*>::iterator it; 
        for( it=_baseMap.begin(); it!=_baseMap.end(); it++ )
        {
            delete (*it).second;
            (*it).second = 0;
        }
        _baseMap.clear();
    }

    ~MemoryMap() { Clear(); }
};  

} // namespace psp

#endif // PSP_MEMORY_MAP_HPP
