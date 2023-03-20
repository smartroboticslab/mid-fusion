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

#ifndef INTERP_GATHER_H
#define INTERP_GATHER_H
#include "node.hpp"

/*
 * Interpolation's point gather offsets
 */

static constexpr const int3 interp_offsets[8] = 
  {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, 
   {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

template <typename FieldType, typename FieldSelector>
inline void gather_local(const VoxelBlock<FieldType>* block, const int3 base, 
    FieldSelector select, float points[8]) {

  if(!block) {
    points[0] = select(VoxelBlock<FieldType>::empty());
    points[1] = select(VoxelBlock<FieldType>::empty());
    points[2] = select(VoxelBlock<FieldType>::empty());
    points[3] = select(VoxelBlock<FieldType>::empty());
    points[4] = select(VoxelBlock<FieldType>::empty());
    points[5] = select(VoxelBlock<FieldType>::empty());
    points[6] = select(VoxelBlock<FieldType>::empty());
    points[7] = select(VoxelBlock<FieldType>::empty());
    return;
  }

  points[0] = select(block->data(base + interp_offsets[0]));
  points[1] = select(block->data(base + interp_offsets[1]));
  points[2] = select(block->data(base + interp_offsets[2]));
  points[3] = select(block->data(base + interp_offsets[3]));
  points[4] = select(block->data(base + interp_offsets[4]));
  points[5] = select(block->data(base + interp_offsets[5]));
  points[6] = select(block->data(base + interp_offsets[6]));
  points[7] = select(block->data(base + interp_offsets[7]));
  return;
}

template <typename FieldType, typename FieldSelector>
inline void gather_4(const VoxelBlock<FieldType>* block, const int3 base, 
    FieldSelector select, const unsigned int offsets[4], float points[8]) {

  if(!block) {
    points[offsets[0]] = select(VoxelBlock<FieldType>::empty());
    points[offsets[1]] = select(VoxelBlock<FieldType>::empty());
    points[offsets[2]] = select(VoxelBlock<FieldType>::empty());
    points[offsets[3]] = select(VoxelBlock<FieldType>::empty());
    return;
  }

  points[offsets[0]] = select(block->data(base + interp_offsets[offsets[0]]));
  points[offsets[1]] = select(block->data(base + interp_offsets[offsets[1]]));
  points[offsets[2]] = select(block->data(base + interp_offsets[offsets[2]]));
  points[offsets[3]] = select(block->data(base + interp_offsets[offsets[3]]));
  return;
}

template <typename FieldType, typename FieldSelector>
inline void gather_2(const VoxelBlock<FieldType>* block, const int3 base, 
    FieldSelector select, const unsigned int offsets[2], float points[8]) {

  if(!block) {
    points[offsets[0]] = select(VoxelBlock<FieldType>::empty());
    points[offsets[1]] = select(VoxelBlock<FieldType>::empty());
    return;
  }

  points[offsets[0]] = select(block->data(base + interp_offsets[offsets[0]]));
  points[offsets[1]] = select(block->data(base + interp_offsets[offsets[1]]));
  return;
}

template <typename FieldType, template<typename FieldT> class MapIndex,
         class FieldSelector>
inline void gather_points(const MapIndex<FieldType>& fetcher, const int3 base, 
    FieldSelector select, float points[8]) {
 
  unsigned int blockSize =  VoxelBlock<FieldType>::side;
  unsigned int crossmask = ((base.x % blockSize == blockSize - 1) << 2) | 
                           ((base.y % blockSize == blockSize - 1) << 1) |
                           ((base.z % blockSize) == blockSize - 1);

  switch(crossmask) {
    case 0: /* all local */
      {
        VoxelBlock<FieldType> * block = fetcher.fetch(base.x, base.y, base.z);
        gather_local(block, base, select, points);
      }
      break;
    case 1: /* z crosses */
      {
        const unsigned int offs1[4] = {0, 1, 2, 3};
        const unsigned int offs2[4] = {4, 5, 6, 7};
        VoxelBlock<FieldType> * block = fetcher.fetch(base.x, base.y, base.z);
        gather_4(block, base, select, offs1, points);
        const int3 base1 = base + interp_offsets[offs2[0]];
        block = fetcher.fetch(base1.x, base1.y, base1.z);
        gather_4(block, base, select, offs2, points);
      }
      break;
    case 2: /* y crosses */ 
      {
        const unsigned int offs1[4] = {0, 1, 4, 5};
        const unsigned int offs2[4] = {2, 3, 6, 7};
        VoxelBlock<FieldType> * block = fetcher.fetch(base.x, base.y, base.z);
        gather_4(block, base, select, offs1, points);
        const int3 base1 = base + interp_offsets[offs2[0]];
        block = fetcher.fetch(base1.x, base1.y, base1.z);
        gather_4(block, base, select, offs2, points);
      }
      break;
    case 3: /* y, z cross */ 
      {
        const unsigned int offs1[2] = {0, 1};
        const unsigned int offs2[2] = {2, 3};
        const unsigned int offs3[2] = {4, 5};
        const unsigned int offs4[2] = {6, 7};
        const int3 base2 = base + interp_offsets[offs2[0]];
        const int3 base3 = base + interp_offsets[offs3[0]];
        const int3 base4 = base + interp_offsets[offs4[0]];
        VoxelBlock<FieldType> * block = fetcher.fetch(base.x, base.y, base.z);
        gather_2(block, base, select, offs1, points);
        block = fetcher.fetch(base2.x, base2.y, base2.z);
        gather_2(block, base, select, offs2, points);
        block = fetcher.fetch(base3.x, base3.y, base3.z);
        gather_2(block, base, select, offs3, points);
        block = fetcher.fetch(base4.x, base4.y, base4.z);
        gather_2(block, base, select, offs4, points);
      }
      break;
    case 4: /* x crosses */ 
      {
        const unsigned int offs1[4] = {0, 2, 4, 6};
        const unsigned int offs2[4] = {1, 3, 5, 7};
        VoxelBlock<FieldType> * block = fetcher.fetch(base.x, base.y, base.z);
        gather_4(block, base, select, offs1, points);
        const int3 base1 = base + interp_offsets[offs2[0]];
        block = fetcher.fetch(base1.x, base1.y, base1.z);
        gather_4(block, base, select, offs2, points);
      }
      break;
    case 5: /* x,z cross */ 
      {
        const unsigned int offs1[2] = {0, 2};
        const unsigned int offs2[2] = {1, 3};
        const unsigned int offs3[2] = {4, 6};
        const unsigned int offs4[2] = {5, 7};
        const int3 base2 = base + interp_offsets[offs2[0]];
        const int3 base3 = base + interp_offsets[offs3[0]];
        const int3 base4 = base + interp_offsets[offs4[0]];
        VoxelBlock<FieldType> * block = fetcher.fetch(base.x, base.y, base.z);
        gather_2(block, base, select, offs1, points);
        block = fetcher.fetch(base2.x, base2.y, base2.z);
        gather_2(block, base, select, offs2, points);
        block = fetcher.fetch(base3.x, base3.y, base3.z);
        gather_2(block, base, select, offs3, points);
        block = fetcher.fetch(base4.x, base4.y, base4.z);
        gather_2(block, base, select, offs4, points);
      }
      break;
    case 6: /* x,y cross */ 
      {
        const unsigned int offs1[2] = {0, 4};
        const unsigned int offs2[2] = {1, 5};
        const unsigned int offs3[2] = {2, 6};
        const unsigned int offs4[2] = {3, 7};
        const int3 base2 = base + interp_offsets[offs2[0]];
        const int3 base3 = base + interp_offsets[offs3[0]];
        const int3 base4 = base + interp_offsets[offs4[0]];
        VoxelBlock<FieldType> * block = fetcher.fetch(base.x, base.y, base.z);
        gather_2(block, base, select, offs1, points);
        block = fetcher.fetch(base2.x, base2.y, base2.z);
        gather_2(block, base, select, offs2, points);
        block = fetcher.fetch(base3.x, base3.y, base3.z);
        gather_2(block, base, select, offs3, points);
        block = fetcher.fetch(base4.x, base4.y, base4.z);
        gather_2(block, base, select, offs4, points);
      }
      break;

    case 7:
      {
        int3 vox[8];
        vox[0] = base + interp_offsets[0];
        vox[1] = base + interp_offsets[1];
        vox[2] = base + interp_offsets[2];
        vox[3] = base + interp_offsets[3];
        vox[4] = base + interp_offsets[4];
        vox[5] = base + interp_offsets[5];
        vox[6] = base + interp_offsets[6];
        vox[7] = base + interp_offsets[7];

        points[0] = select(fetcher.get_fine(vox[0].x, vox[0].y, vox[0].z));
        points[1] = select(fetcher.get_fine(vox[1].x, vox[1].y, vox[1].z));
        points[2] = select(fetcher.get_fine(vox[2].x, vox[2].y, vox[2].z));
        points[3] = select(fetcher.get_fine(vox[3].x, vox[3].y, vox[3].z));
        points[4] = select(fetcher.get_fine(vox[4].x, vox[4].y, vox[4].z));
        points[5] = select(fetcher.get_fine(vox[5].x, vox[5].y, vox[5].z));
        points[6] = select(fetcher.get_fine(vox[6].x, vox[6].y, vox[6].z));
        points[7] = select(fetcher.get_fine(vox[7].x, vox[7].y, vox[7].z));
      }
      break;
  }
}
#endif
