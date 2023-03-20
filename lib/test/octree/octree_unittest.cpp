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
#include "gtest/gtest.h"
#include "octant_ops.hpp"
#include <bitset>

TEST(Octree, OctantFaceNeighbours) {
  const uint3 octant = {112, 80, 160};
  const unsigned int max_depth = 8;
  const unsigned int leaves_depth = 5;
  const octlib::key_t code = 
    octlib::keyops::encode(octant.x, octant.y, octant.z, leaves_depth, max_depth);
  const unsigned int side = 8;
  const int3 faces[6] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, 
    {0, 0, -1}, {0, 0, 1}};
  for(int i = 0; i < 6; ++i) {
    const int3 neighbour = make_int3(octant) + side * faces[i];
    const uint3 computed = face_neighbour(code, i, leaves_depth, max_depth); 
    ASSERT_EQ(neighbour.x, computed.x); 
    ASSERT_EQ(neighbour.y, computed.y); 
    ASSERT_EQ(neighbour.z, computed.z); 
  }
}

TEST(Octree, OctantDescendant) {
  const unsigned max_depth = 8;
  uint3 octant = {110, 80, 159};
  octlib::key_t code = 
    octlib::keyops::encode(octant.x, octant.y, octant.z, 5, max_depth);
  octlib::key_t ancestor = 
    octlib::keyops::encode(96, 64, 128, 3, max_depth);
  ASSERT_EQ(true , descendant(code, ancestor, max_depth)); 

  ancestor = octlib::keyops::encode(128, 64, 64, 3, max_depth);
  ASSERT_FALSE(descendant(code, ancestor, max_depth)); 
}

TEST(Octree, OctantParent) {
  const int max_depth = 8;
  uint3 octant = {112, 80, 160};
  octlib::key_t code = 
    octlib::keyops::encode(octant.x, octant.y, octant.z, 5, max_depth);
  octlib::key_t p = parent(code, max_depth);
  ASSERT_EQ(octlib::keyops::code(code), octlib::keyops::code(p));
  ASSERT_EQ(4, p & SCALE_MASK);

  code = p;
  p = parent(code, max_depth); 
  ASSERT_EQ(3, octlib::keyops::level(p));
  ASSERT_EQ(p, octlib::keyops::encode(96, 64, 160, 3, max_depth));

  code = p;
  p = parent(code, max_depth); 
  ASSERT_EQ(2, octlib::keyops::level(p));
  ASSERT_EQ(p, octlib::keyops::encode(64, 64, 128, 2, max_depth));
}

TEST(Octree, FarCorner) {
  /*
   * The far corner should always be "exterior", meaning that moving one step
   * in the outward direction (i.e. away from the center) in *any* direction 
   * implies leaving the parent octant. For simplicity here the corners
   * individually, but this can be done programmatically testing this property.
   * TODO: change this test case to be more exhaustive.
   */

  const int max_depth = 5;
  const int level = 2;

  /* First child */
  const octlib::key_t cell0 = 
    octlib::keyops::encode(16, 16, 16, level, max_depth);
  const int3 fc0 = far_corner(cell0, level, max_depth);
  ASSERT_EQ(fc0.x, 16);
  ASSERT_EQ(fc0.y, 16);
  ASSERT_EQ(fc0.z, 16);

  /* Second child */
  const octlib::key_t cell1 = octlib::keyops::encode(24, 16, 16, level, max_depth);
  const int3 fc1 = far_corner(cell1, level, max_depth);
  ASSERT_EQ(fc1.x, 32);
  ASSERT_EQ(fc1.y, 16);
  ASSERT_EQ(fc1.z, 16);

  /* Third child */
  const octlib::key_t cell2 = octlib::keyops::encode(16, 24, 16, level, max_depth);
  const int3 fc2 = far_corner(cell2, level, max_depth);
  ASSERT_EQ(fc2.x, 16);
  ASSERT_EQ(fc2.y, 32);
  ASSERT_EQ(fc2.z, 16);

  /* Fourth child */
  const octlib::key_t cell3 = octlib::keyops::encode(24, 24, 16, level, max_depth);
  const int3 fc3 = far_corner(cell3, level, max_depth);
  ASSERT_EQ(fc3.x, 32);
  ASSERT_EQ(fc3.y, 32);
  ASSERT_EQ(fc3.z, 16);

  /* Fifth child */
  const octlib::key_t cell4 = octlib::keyops::encode(24, 24, 16, level, max_depth);
  const int3 fc4 = far_corner(cell4, level, max_depth);
  ASSERT_EQ(fc4.x, 32);
  ASSERT_EQ(fc4.y, 32);
  ASSERT_EQ(fc4.z, 16);

  /* sixth child */
  const octlib::key_t cell5 = octlib::keyops::encode(16, 16, 24, level, max_depth);
  const int3 fc5 = far_corner(cell5, level, max_depth);
  ASSERT_EQ(fc5.x, 16);
  ASSERT_EQ(fc5.y, 16);
  ASSERT_EQ(fc5.z, 32);

  /* seventh child */
  const octlib::key_t cell6 = octlib::keyops::encode(24, 16, 24, level, max_depth);
  const int3 fc6 = far_corner(cell6, level, max_depth);
  ASSERT_EQ(fc6.x, 32);
  ASSERT_EQ(fc6.y, 16);
  ASSERT_EQ(fc6.z, 32);

  /* eight child */
  const octlib::key_t cell7 = octlib::keyops::encode(24, 24, 24, level, max_depth);
  const int3 fc7 = far_corner(cell7, level, max_depth);
  ASSERT_EQ(fc7.x, 32);
  ASSERT_EQ(fc7.y, 32);
  ASSERT_EQ(fc7.z, 32);
}

TEST(Octree, InnerOctantExteriorNeighbours) {
  const int max_depth = 5;
  const int level = 2;
  const int side = 1 << (max_depth - level);
  const octlib::key_t cell = octlib::keyops::encode(16, 16, 16, level, max_depth);
  octlib::key_t N[7];
  exterior_neighbours(N, cell, level, max_depth);
  const octlib::key_t p = parent(cell, max_depth);
  
  const octlib::key_t neighbours_gt[7] = 
    {octlib::keyops::encode(15, 16, 16, level, max_depth),
     octlib::keyops::encode(16, 15, 16, level, max_depth),
     octlib::keyops::encode(15, 15, 16, level, max_depth),
     octlib::keyops::encode(16, 16, 15, level, max_depth),
     octlib::keyops::encode(15, 16, 15, level, max_depth),
     octlib::keyops::encode(16, 15, 15, level, max_depth),
     octlib::keyops::encode(15, 15, 15, level, max_depth)};
  for(int i = 0; i < 7; ++i) {
    // std::bitset<64> c(N[i]);
    // std::bitset<64> a(p);
    // std::cout << a << std::endl;
    // std::cout << c << std::endl << std::endl;
    ASSERT_EQ(neighbours_gt[i], N[i]);
    ASSERT_FALSE(parent(N[i], max_depth) == p);
    // std::cout << (unpack_morton(N[i] & ~SCALE_MASK)) << std::endl;
  }
}

TEST(Octree, EdgeOctantExteriorNeighbours) {
  const int max_depth = 5;
  const uint size = std::pow(2, 5);
  const int level = 2;
  const octlib::key_t cell = octlib::keyops::encode(0, 16, 16, level, max_depth);
  octlib::key_t N[7];
  exterior_neighbours(N, cell, level, max_depth);
  const octlib::key_t p = parent(cell, max_depth);
  
  for(int i = 0; i < 7; ++i) {
    const uint3 corner = unpack_morton(N[i] & ~SCALE_MASK);
    ASSERT_TRUE(in(corner.x, 0u, size - 1) && 
                in(corner.y, 0u, size - 1) &&
                in(corner.z, 0u, size - 1));
  }
}

TEST(Octree, OctantSiblings) {
  const int max_depth = 5;
  const uint size = std::pow(2, 5);
  const int level = 2;
  const octlib::key_t cell = octlib::keyops::encode(16, 16, 16, level, max_depth);
  octlib::key_t s[8];
  siblings(s, cell, max_depth);

  const int childidx = child_id(cell, level, max_depth);
  ASSERT_EQ(s[childidx], cell);
  
  for(int i = 0; i < 8; ++i) {
    // std::cout << (unpack_morton(s[i] & ~SCALE_MASK)) << std::endl;
    ASSERT_TRUE(parent(s[i], max_depth) == parent(cell, max_depth));
  }
}
