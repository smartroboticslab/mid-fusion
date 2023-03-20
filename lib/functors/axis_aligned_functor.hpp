/*

Copyright 2016 Emanuele Vespa, Imperial College London 

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

*/

#ifndef AA_FUNCTOR_HPP
#define AA_FUNCTOR_HPP
#include <functional>
#include <vector>

#include "math_utils.h"
#include "algorithms/filter.hpp"
#include "node.hpp"
#include "functors/data_handler.hpp"
#include "geometry/aabb_collision.hpp"

namespace iterators {
  namespace functor {

  template <typename FieldType, template <typename FieldT> class MapT, 
            typename UpdateF>

    class axis_aligned {
      public:
      axis_aligned(MapT<FieldType>& map, UpdateF f) : _map(map), _function(f),
      _min(make_int3(0)), _max(make_int3(map.size())){ }

      axis_aligned(MapT<FieldType>& map, UpdateF f, const int3 min,
          const int3 max) : _map(map), _function(f),
      _min(min), _max(max){ }

      void update_block(VoxelBlock<FieldType> * block) {
        int3 blockCoord = block->coordinates();
        unsigned int y, z, x; 
        int3 blockSide = make_int3(VoxelBlock<FieldType>::side);
        int3 start = max(blockCoord, _min);
        int3 last = min(blockCoord + blockSide, _max);

        for(z = start.z; z < last.z; ++z) {
          for (y = start.y; y < last.y; ++y) {
            for (x = start.x; x < last.x; ++x) {
              int3 vox = make_int3(x, y, z);
              VoxelBlockHandler<FieldType> handler = {block, vox};
              _function(handler, vox);
            }
          }
        }
      }

      void update_node(Node<FieldType> * node) { 
        int3 voxel = make_int3(unpack_morton(node->code));
#pragma omp simd
        for(int i = 0; i < 8; ++i) {
          const int3 dir =  make_int3((i & 1) > 0, (i & 2) > 0, (i & 4) > 0);
          voxel = voxel + (dir * (node->side/2));
          if(!(in(voxel.x, _min.x, _max.x) && in(voxel.y, _min.y, _max.y) 
                && in(voxel.z, _min.z, _max.z))) continue;
          NodeHandler<FieldType> handler = {node, i};
          _function(handler, voxel);
        }
      }

      void apply() {

        auto& block_list = _map.getBlockBuffer();
        size_t list_size = block_list.size();
#pragma omp parallel for
        for(unsigned int i = 0; i < list_size; ++i){
          update_block(block_list[i]);
        }

        auto& nodes_list = _map.getNodesBuffer();
        list_size = nodes_list.size();
#pragma omp parallel for
        for(unsigned int i = 0; i < list_size; ++i){
          update_node(nodes_list[i]);
        }
      }

    private:
      MapT<FieldType>& _map; 
      UpdateF _function; 
      int3 _min;
      int3 _max;
    };
  }
}
#endif
