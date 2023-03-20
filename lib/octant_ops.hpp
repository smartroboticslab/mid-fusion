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


#ifndef OCTANT_OPS_HPP
#define OCTANT_OPS_HPP
#include "utils/morton_utils.hpp"
#include "octree_defines.h"
#include <iostream>
#include <bitset>

namespace octlib {
  namespace keyops {

    inline octlib::key_t code(const octlib::key_t key) {
      return key & ~SCALE_MASK;
    }

    inline int level(const octlib::key_t key) {
      return key & SCALE_MASK;
}

    inline octlib::key_t encode(const int x, const int y, const int z, 
        const int level, const int max_depth) {
      const int offset = MAX_BITS - max_depth + level - 1;
      return (compute_morton(x, y, z) & MASK[offset]) | level;
    }

    inline uint3 decode(const octlib::key_t key) {
      return unpack_morton(key & ~SCALE_MASK);
    }
  }
}

/*
 * Algorithm 5 of p4est paper: https://epubs.siam.org/doi/abs/10.1137/100791634
 */
inline uint3 face_neighbour(const octlib::key_t o, 
    const unsigned int face, const unsigned int l, 
    const unsigned int max_depth) {
  uint3 coords = octlib::keyops::decode(o);
  const unsigned int side = 1 << (max_depth - l); 
  coords.x = coords.x + ((face == 0) ? -side : (face == 1) ? side : 0);
  coords.y = coords.y + ((face == 2) ? -side : (face == 3) ? side : 0);
  coords.z = coords.z + ((face == 4) ? -side : (face == 5) ? side : 0);
  return {coords.x, coords.y, coords.z};
}

/*
 * \brief Return true if octant is a descendant of ancestor
 * \param octant 
 * \param ancestor 
 */
inline bool descendant(octlib::key_t octant, 
    octlib::key_t ancestor, const int max_depth) {
  const int level = octlib::keyops::level(ancestor);
  const int idx = MAX_BITS - max_depth - 1 + level;
  octant = octant & MASK[idx];
  ancestor = octlib::keyops::code(ancestor);
  return (ancestor ^ octant) == 0;
}

/*
 * \brief Computes the parent's morton code of a given octant
 * \param octant
 * \param max_depth max depth of the tree on which the octant lives
 */
inline octlib::key_t parent(const octlib::key_t& octant, const int max_depth) {
  const int level = octlib::keyops::level(octant) - 1;
  const int idx = MAX_BITS - max_depth + level - 1;
  return (octant & MASK[idx]) | level;
}

/*
 * \brief Computes the octants's id in its local brotherhood
 * \param octant
 * \param level of octant 
 * \param max_depth max depth of the tree on which the octant lives
 */
inline int child_id(octlib::key_t octant, const int level, 
    const int max_depth) {
  int shift = max_depth - level;
  octant = octlib::keyops::code(octant) >> shift*3;
  int idx = (octant & 0x01) | (octant & 0x02) | (octant & 0x04);
  return idx;
}

/*
 * \brief Computes the octants's corner which is not shared with its siblings
 * \param octant
 * \param level of octant 
 * \param max_depth max depth of the tree on which the octant lives
 */
inline int3 far_corner(const octlib::key_t octant, const int level, 
    const int max_depth) {
  const unsigned int side = 1 << (max_depth - level); 
  const int idx = child_id(octant, level, max_depth);
  const uint3 coordinates = octlib::keyops::decode(octant);
  return make_int3(coordinates.x + (idx & 1) * side,
                   coordinates.y + ((idx & 2) >> 1) * side,
                   coordinates.z + ((idx & 4) >> 2) * side);
}

/*
 * \brief Computes the non-sibling neighbourhood around an octants. In the
 * special case in which the octant lies on an edge, neighbour are duplicated 
 * as movement outside the enclosing cube is forbidden.
 * \param result 7-vector containing the neighbours
 * \param octant
 * \param level of octant 
 * \param max_depth max depth of the tree on which the octant lives
 */
inline void exterior_neighbours(octlib::key_t result[7], 
    const octlib::key_t octant, const int level, const int max_depth) {

  const int idx = child_id(octant, level, max_depth);
  int3 dir = make_int3((idx & 1) ? 1 : -1,
                       (idx & 2) ? 1 : -1,
                       (idx & 4) ? 1 : -1);
  int3 base = far_corner(octant, level, max_depth);
  dir.x = in(base.x + dir.x , 0, std::pow(2, max_depth) - 1) ? dir.x : 0;
  dir.y = in(base.y + dir.y , 0, std::pow(2, max_depth) - 1) ? dir.y : 0;
  dir.z = in(base.z + dir.z , 0, std::pow(2, max_depth) - 1) ? dir.z : 0;

 result[0] = octlib::keyops::encode(base.x + dir.x, base.y + 0, base.z + 0, 
     level, max_depth);
 result[1] = octlib::keyops::encode(base.x + 0, base.y + dir.y, base.z + 0, 
     level, max_depth); 
 result[2] = octlib::keyops::encode(base.x + dir.x, base.y + dir.y, base.z + 0, 
     level, max_depth); 
 result[3] = octlib::keyops::encode(base.x + 0, base.y + 0, base.z + dir.z, 
     level, max_depth); 
 result[4] = octlib::keyops::encode(base.x + dir.x, base.y + 0, base.z + dir.z, 
     level, max_depth); 
 result[5] = octlib::keyops::encode(base.x + 0, base.y + dir.y, base.z + dir.z, 
     level, max_depth); 
 result[6] = octlib::keyops::encode(base.x + dir.x, base.y + dir.y, 
     base.z + dir.z, level, max_depth); 
}

/*
 * \brief Computes the morton number of all siblings around an octant,
 * including itself.
 * \param result 8-vector containing the neighbours
 * \param octant
 * \param max_depth max depth of the tree on which the octant lives
 */
inline void siblings(octlib::key_t result[8], 
    const octlib::key_t octant, const int max_depth) {
  const int level = (octant & SCALE_MASK);
  const int shift = 3*(max_depth - level);
  const octlib::key_t p = parent(octant, max_depth) + 1; // set-up next level
  for(int i = 0; i < 8; ++i) {
    result[i] = p | (i << shift);
  }
}
#endif
