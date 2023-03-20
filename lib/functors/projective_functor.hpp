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


#ifndef PROJECTIVE_FUNCTOR_HPP
#define PROJECTIVE_FUNCTOR_HPP
#include <functional>
#include <vector>

#include "math_utils.h"
#include "algorithms/filter.hpp"
#include "node.hpp"
#include "functors/data_handler.hpp"

namespace iterators {
  template <typename FieldType, template <typename FieldT> class MapT, 
            typename UpdateF>
  class projective_functor {

    public:
      projective_functor(MapT<FieldType>& map, UpdateF f, const Matrix4& Tcw, 
          const Matrix4& K, const int2 framesize) : 
        _map(map), _function(f), _Tcw(Tcw), _K(K), _frame_size(framesize) {
      } 

      void build_active_list() {
        using namespace std::placeholders;
        /* Retrieve the active list */ 
        const MemoryPool<VoxelBlock<FieldType> >& block_array = 
          _map.getBlockBuffer();

        /* Predicates definition */
        const float voxel_size = _map.dim()/_map.size();
        auto in_frustum_predicate = 
          std::bind(algorithms::in_frustum<VoxelBlock<FieldType>>, _1, 
              voxel_size, _K*_Tcw, _frame_size); 
        auto is_active_predicate = [](const VoxelBlock<FieldType>* b) {
          return b->active();
        };

        algorithms::filter(_active_list, block_array, is_active_predicate,
            in_frustum_predicate);
      }

      void update_block(VoxelBlock<FieldType> * block, const float voxel_size) {

        const int3 blockCoord = block->coordinates();
        const float3 delta = rotate(_Tcw, make_float3(voxel_size, 0, 0));
        const float3 cameraDelta = rotate(_K, delta);
        bool is_visible = false;

        unsigned int y, z, blockSide; 
        blockSide = VoxelBlock<FieldType>::side;
        unsigned int ylast = blockCoord.y + blockSide;
        unsigned int zlast = blockCoord.z + blockSide;

        for(z = blockCoord.z; z < zlast; ++z)
          for (y = blockCoord.y; y < ylast; ++y){
            int3 pix = make_int3(blockCoord.x, y, z);
            float3 start = _Tcw * make_float3((pix.x) * voxel_size, 
                (pix.y) * voxel_size, (pix.z) * voxel_size);
            float3 camerastart = _K * start;
#pragma omp simd
            for (unsigned int x = 0; x < blockSide; ++x){
              pix.x = x + blockCoord.x; 
              const float3 camera_voxel = camerastart + (x*cameraDelta);
              const float3 pos = start + (x*delta);
              if (pos.z < 0.0001f) continue;

              const float inverse_depth = 1.f / camera_voxel.z;
              const float2 pixel = make_float2(
                  camera_voxel.x * inverse_depth + 0.5f,
                  camera_voxel.y * inverse_depth + 0.5f);
              if (pixel.x < 0.5f || pixel.x > _frame_size.x - 1.5f || 
                  pixel.y < 0.5f || pixel.y > _frame_size.y - 1.5f) continue;
              is_visible = true;

              VoxelBlockHandler<FieldType> handler = {block, pix};
              _function(handler, pix, pos, pixel);
            }
          }
        block->active(is_visible);
      }

      void update_node(Node<FieldType> * node, const float voxel_size) { 
        const int3 voxel = make_int3(unpack_morton(node->code));
        const float3 delta = rotate(_Tcw, make_float3(0.5f * voxel_size * node->side));
        const float3 delta_c = rotate(_K, delta);
        float3 base_cam = _Tcw * (make_float3(voxel) * voxel_size);
        float3 basepix_hom = _K * base_cam;

#pragma omp simd
        for(int i = 0; i < 8; ++i) {
          const int3 dir =  make_int3((i & 1) > 0, (i & 2) > 0, (i & 4) > 0);
          const float3 vox_cam = base_cam + make_float3(dir) * delta; 
          const float3 pix_hom = basepix_hom + make_float3(dir) * delta_c; 

          if (vox_cam.z < 0.0001f) continue;
          const float inverse_depth = 1.f / pix_hom.z;
          const float2 pixel = make_float2(
              pix_hom.x * inverse_depth + 0.5f,
              pix_hom.y * inverse_depth + 0.5f);
          if (pixel.x < 0.5f || pixel.x > _frame_size.x - 1.5f || 
              pixel.y < 0.5f || pixel.y > _frame_size.y - 1.5f) continue;

          NodeHandler<FieldType> handler = {node, i};
          _function(handler, voxel + dir, vox_cam, pixel);
        }
      }

      void apply() {

        build_active_list();
        const float voxel_size = _map.dim() / _map.size();
        size_t list_size = _active_list.size();
#pragma omp parallel for
        for(unsigned int i = 0; i < list_size; ++i){
          update_block(_active_list[i], voxel_size);
        }
        _active_list.clear();

        auto& nodes_list = _map.getNodesBuffer();
        list_size = nodes_list.size();
#pragma omp parallel for
          for(unsigned int i = 0; i < list_size; ++i){
            update_node(nodes_list[i], voxel_size);
         }
      }

    private:
      MapT<FieldType>& _map; 
      UpdateF _function; 
      Matrix4 _Tcw;
      Matrix4 _K;
      int2 _frame_size;
      std::vector<VoxelBlock<FieldType>*> _active_list;
  };
}
#endif
