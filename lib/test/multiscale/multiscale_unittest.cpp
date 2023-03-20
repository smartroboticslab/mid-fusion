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

#include "octree.hpp"
#include "math_utils.h"
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

class MultiscaleTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      oct_.init(512, 5);

    }

  typedef Octree<testT> OctreeF;
  OctreeF oct_;
};

TEST_F(MultiscaleTest, Init) {
  EXPECT_EQ(oct_.get(137, 138, 130), voxel_traits<testT>::initValue());
}

TEST_F(MultiscaleTest, PlainAlloc) {
  const int3 blocks[2] = {{56, 12, 254}, {87, 32, 423}};
  octlib::key_t alloc_list[2];
  for(int i = 0; i < 2; ++i) {
    alloc_list[i] = oct_.hash(blocks[i].x, blocks[i].y, blocks[i].z);
  }
  oct_.allocate(alloc_list, 2);

  oct_.set(56, 12, 254, 3.f);

  EXPECT_EQ(oct_.get(56, 12, 254), 3.f);
  EXPECT_EQ(oct_.get(106, 12, 254), voxel_traits<testT>::initValue());
  EXPECT_NE(oct_.get(106, 12, 254), 3.f);
}

TEST_F(MultiscaleTest, ScaledAlloc) {
  const int3 blocks[2] = {{200, 12, 25}, {87, 32, 423}};
  octlib::key_t alloc_list[2];
  for(int i = 0; i < 2; ++i) {
    alloc_list[i] = oct_.hash(blocks[i].x, blocks[i].y, blocks[i].z, 5);
  }

  oct_.alloc_update(alloc_list, 2);
  Node<testT>* n = oct_.fetch_octant(87, 32, 420, 5);
  ASSERT_TRUE(n != NULL);
  n->value_[0] = 10.f;
  EXPECT_EQ(oct_.get(87, 32, 420), 10.f);
}

TEST_F(MultiscaleTest, Iterator) {
  const int3 blocks[1] = {{56, 12, 254}};
  octlib::key_t alloc_list[1];
  alloc_list[0] = oct_.hash(blocks[0].x, blocks[0].y, blocks[0].z);

  oct_.alloc_update(alloc_list, 1);
  leaf_iterator<testT> it(oct_);

  typedef std::tuple<int3, int, typename Octree<testT>::compute_type> it_result;
  it_result node = it.next();
  for(int i = 256; std::get<1>(node) > 0; node = it.next(), i /= 2){
    const int3 coords = std::get<0>(node);
    const int side = std::get<1>(node);
    const Octree<testT>::compute_type val = std::get<2>(node);
    EXPECT_EQ(side, i);
  }
}

TEST_F(MultiscaleTest, ChildrenMaskTest) {
  const int3 blocks[10] = {{56, 12, 254}, {87, 32, 423}, {128, 128, 128},
    {136, 128, 128}, {128, 136, 128}, {136, 136, 128}, 
    {128, 128, 136}, {136, 128, 136}, {128, 136, 136}, {136, 136, 136}};
  octlib::key_t alloc_list[10];
  for(int i = 0; i < 10; ++i) {
    alloc_list[i] = oct_.hash(blocks[i].x, blocks[i].y, blocks[i].z, 5);
  }

  oct_.alloc_update(alloc_list, 10);
  const MemoryPool<Node<testT> >& nodes = oct_.getNodesBuffer();
  const size_t num_nodes = nodes.size();
  for(size_t i = 0; i < num_nodes; ++i) {
    Node<testT>* n = nodes[i];
    for(int c = 0; c < 8; ++c) {
      if(n->child(c)) {
        ASSERT_TRUE(n->children_mask_ & (1 << c));
      }
    }
  } 
}

TEST_F(MultiscaleTest, OctantAlloc) {
  const int3 blocks[10] = {{56, 12, 254}, {87, 32, 423}, {128, 128, 128},
    {136, 128, 128}, {128, 136, 128}, {136, 136, 128}, 
    {128, 128, 136}, {136, 128, 136}, {128, 136, 136}, {136, 136, 136}};
  octlib::key_t alloc_list[10];
  for(int i = 0; i < 10; ++i) {
    alloc_list[i] = oct_.hash(blocks[i].x, blocks[i].y, blocks[i].z);
  }

  alloc_list[2] = alloc_list[2] | 3;
  alloc_list[9] = alloc_list[2] | 5;
  oct_.alloc_update(alloc_list, 10);
  Node<testT> * octant = oct_.fetch_octant(blocks[4].x, blocks[4].y,
      blocks[4].z, 3);
  ASSERT_TRUE(octant != NULL);
  octant = oct_.fetch_octant(blocks[9].x, blocks[9].y,
      blocks[9].z, 6);
  ASSERT_TRUE(octant == NULL);
}
