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

#include "math_utils.h"
#include "geometry/octree_collision.hpp"
#include "geometry/aabb_collision.hpp"
#include "algorithms/mapping.hpp"
#include "utils/morton_utils.hpp"
#include "octree.hpp"
#include "gtest/gtest.h"

typedef float testT;

template <>
struct voxel_traits<testT> {
  typedef float ComputeType;
  typedef float StoredType;
  static inline ComputeType empty(){ return 0.f; }
  static inline ComputeType initValue(){ return 1.f; }
  static inline StoredType translate(const ComputeType value) {
     return value;
  }
};

collision_status test_voxel(const voxel_traits<testT>::ComputeType & val) {
  if(val == voxel_traits<testT>::initValue()) return collision_status::unseen;
  if(val == 10.f) return collision_status::empty;
  return collision_status::occupied;
};

class OctreeCollisionTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      oct_.init(256, 5);
      const int3 blocks[1] = {{56, 12, 254}};
      octlib::key_t alloc_list[1];
      alloc_list[0] = oct_.hash(blocks[0].x, blocks[0].y, blocks[0].z);

      auto update = [](Node<testT> * n){
        uint3 coords = unpack_morton(n->code);
        /* Empty for coords above the below values, 
         * except where leaves are allocated.
         */
        if(coords.x >= 48 && coords.y >= 0 && coords.z >= 240) {
          n->value_[0] = 10.f;
        }
      };
      oct_.alloc_update(alloc_list, 1);
      algorithms::integratePass(oct_.getNodesBuffer(), oct_.getNodesBuffer().size(),
          update);
    }

  typedef Octree<testT> OctreeF;
  OctreeF oct_;
};

TEST_F(OctreeCollisionTest, TotallyUnseen) {

  leaf_iterator<testT> it(oct_);
  typedef std::tuple<int3, int, typename Octree<testT>::compute_type> it_result;
  it_result node = it.next();
  for(int i = 128; std::get<1>(node) > 0; node = it.next(), i /= 2){
    const int3 coords = std::get<0>(node);
    const int side = std::get<1>(node);
    const Octree<testT>::compute_type val = std::get<2>(node);
    printf("Node's coordinates: (%d, %d, %d), side %d, value %.2f\n", 
        coords.x, coords.y, coords.z, side, val);
    EXPECT_EQ(side, i);
  }

  const int3 test_bbox = {23, 0, 100};
  const int3 width = {2, 2, 2};

  const collision_status collides = collides_with(oct_, test_bbox, width, 
      test_voxel);
  ASSERT_EQ(collides, collision_status::unseen);
}

TEST_F(OctreeCollisionTest, PartiallyUnseen) {
  const int3 test_bbox = {47, 0, 239};
  const int3 width = {6, 6, 6};
  const collision_status collides = collides_with(oct_, test_bbox, width, 
      test_voxel);
  ASSERT_EQ(collides, collision_status::unseen);
}

TEST_F(OctreeCollisionTest, Empty) {
  const int3 test_bbox = {49, 1, 242};
  const int3 width = {1, 1, 1};
  const collision_status collides = collides_with(oct_, test_bbox, width, 
      test_voxel);
  ASSERT_EQ(collides, collision_status::empty);
}

TEST_F(OctreeCollisionTest, CollisionPlausible){
  const int3 test_bbox = {54, 10, 249};
  const int3 width = {5, 5, 3};

  const collision_status collides = collides_with(oct_, test_bbox, width, 
      test_voxel);
  ASSERT_EQ(collides, collision_status::unseen);
}

TEST_F(OctreeCollisionTest, Collision){
  const int3 test_bbox = {54, 10, 249};
  const int3 width = {5, 5, 3};
  /* Update leaves as occupied node */
  auto update = [](VoxelBlock<testT> * block){
    const int3 blockCoord = block->coordinates();
    int x, y, z, blockSide; 
    blockSide = (int) VoxelBlock<testT>::side;
    int xlast = blockCoord.x + blockSide;
    int ylast = blockCoord.y + blockSide;
    int zlast = blockCoord.z + blockSide;
    for(z = blockCoord.z; z < zlast; ++z){
      for (y = blockCoord.y; y < ylast; ++y){
        for (x = blockCoord.x; x < xlast; ++x){
          block->data(make_int3(x, y, z), 2.f);
        }
      }
    }
  };
 
  algorithms::integratePass(oct_.getBlockBuffer(), oct_.getBlockBuffer().size(),
      update);

  const collision_status collides = collides_with(oct_, test_bbox, width, 
      test_voxel);
  ASSERT_EQ(collides, collision_status::occupied);
}

TEST_F(OctreeCollisionTest, CollisionFreeLeaf){
  // Allocated block: {56, 8, 248};
  const int3 test_bbox = {61, 13, 253};
  const int3 width = {2, 2, 2};
  /* Update leaves as occupied node */
  auto update = [](VoxelBlock<testT> * block){
    const int3 blockCoord = block->coordinates();
    int x, y, z, blockSide; 
    blockSide = (int) VoxelBlock<testT>::side;
    int xlast = blockCoord.x + blockSide/2;
    int ylast = blockCoord.y + blockSide/2;
    int zlast = blockCoord.z + blockSide/2;
    for(z = blockCoord.z; z < zlast; ++z){
      for (y = blockCoord.y; y < ylast; ++y){
        for (x = blockCoord.x; x < xlast; ++x){
          block->data(make_int3(x, y, z), 2.f);
        }
      }
    }
    for(z = zlast; z < zlast + blockSide/2; ++z){
      for (y = ylast; y < ylast + blockSide/2; ++y){
        for (x = xlast; x < xlast + blockSide/2; ++x){
          block->data(make_int3(x, y, z), 10.f);
        }
      }
    }
  };
 
  algorithms::integratePass(oct_.getBlockBuffer(), oct_.getBlockBuffer().size(),
      update);

  const collision_status collides = collides_with(oct_, test_bbox, width, 
      test_voxel);
  ASSERT_EQ(collides, collision_status::empty);
}



