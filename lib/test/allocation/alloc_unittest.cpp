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

#include <random>
#include "octree.hpp"
#include "math_utils.h"
#include "utils/morton_utils.hpp"
#include "gtest/gtest.h"

template <>
struct voxel_traits<float> {
  typedef float ComputeType;
  typedef float StoredType;
  static inline ComputeType empty(){ return 0.f; }
  static inline ComputeType initValue(){ return 0.f; }
  static inline StoredType translate(const ComputeType value) {
     return value;
  }
};

TEST(AllocationTest, EmptySingleVoxel) {
  typedef Octree<float> OctreeF;
  OctreeF oct;
  oct.init(256, 5);
  const int3 vox = {25, 65, 127};
  const octlib::key_t code = oct.hash(vox.x, vox.y, vox.z); 
  octlib::key_t allocList[1] = {code};
  const float val = oct.get(vox.x, vox.y, vox.z);
  EXPECT_EQ(val, voxel_traits<float>::empty());
}

TEST(AllocationTest, SetSingleVoxel) {
  typedef Octree<float> OctreeF;
  OctreeF oct;
  oct.init(256, 5);
  const int3 vox = {25, 65, 127};
  const octlib::key_t code = oct.hash(vox.x, vox.y, vox.z); 
  octlib::key_t allocList[1] = {code};
  oct.allocate(allocList, 1);

  VoxelBlock<float> * block = oct.fetch(vox.x, vox.y, vox.z);
  float written_val = 2.f;
  block->data(vox, written_val);

  const float read_val = oct.get(vox.x, vox.y, vox.z);
  EXPECT_EQ(written_val, read_val);
}

TEST(AllocationTest, FetchOctant) {
  typedef Octree<float> OctreeF;
  OctreeF oct;
  oct.init(256, 5);
  const int3 vox = {25, 65, 127};
  const uint code = oct.hash(vox.x, vox.y, vox.z); 
  octlib::key_t allocList[1] = {code};
  oct.allocate(allocList, 1);

  const int depth = 3; /* 32 voxels per side */
  Node<float> * node = oct.fetch_octant(vox.x, vox.y, vox.z, 3);

  EXPECT_NE(node, nullptr);
}

TEST(AllocationTest, MortonPrefixMask) {

  const unsigned int max_bits = 21; 
  const unsigned int block_side = 8;
  const unsigned int size = std::pow(2, max_bits);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> dis(0, size);

  constexpr int num_samples = 10;
  octlib::key_t keys[num_samples];
  octlib::key_t tempkeys[num_samples];
  int3 coordinates[num_samples];

  for(int i = 0; i < num_samples; ++i) {
    const uint3 vox = {dis(gen), dis(gen), dis(gen)};
    coordinates[i] = make_int3(vox);
    const octlib::key_t code = compute_morton(vox.x, vox.y, vox.z);
    keys[i] = code;
  }

  const int max_level = log2(size);
  const int leaf_level = max_level - log2(block_side);
  const unsigned int shift = max_bits - max_level;
  int edge = size/2;
  for (int level = 0; level <= leaf_level; level++){
    const octlib::key_t mask = MASK[level + shift];
    compute_prefix(keys, tempkeys, num_samples, mask);
    for(int i = 0; i < num_samples; ++i) {
      const uint3 masked_vox = unpack_morton(tempkeys[i]);
      ASSERT_EQ(masked_vox.x % edge, 0);
      ASSERT_EQ(masked_vox.y % edge, 0);
      ASSERT_EQ(masked_vox.z % edge, 0);
      const int3 vox = coordinates[i];
      // printf("vox: %d, %d, %d\n", vox.x, vox.y, vox.z);
      // printf("masked level %d: %d, %d, %d\n", level, masked_vox.x, masked_vox.y, masked_vox.z );
    }
    edge = edge/2;
  }
}
