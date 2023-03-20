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
#include "geometry/aabb_collision.hpp"
#include "gtest/gtest.h"

TEST(AABBAABBTest, SquareOverlap) {

  const int3 a = make_int3(5, 6, 9);
  const int3 a_edge = make_int3(3);
  const int3 b = make_int3(4, 4, 8);
  const int3 b_edge = make_int3(3);
  int overlaps = geometry::aabb_aabb_collision(a, a_edge, b, b_edge);
  ASSERT_EQ(overlaps, 1);
}

TEST(AABBAABBTest, SquareDisjoint1Axis) {

  const int3 a = make_int3(5, 6, 9);
  const int3 a_edge = make_int3(3);
  const int3 b = make_int3(4, 4, 1);
  const int3 b_edge = make_int3(3);
  int overlaps = geometry::aabb_aabb_collision(a, a_edge, b, b_edge);
  ASSERT_EQ(overlaps, 0);
}

TEST(AABBAABBTest, SquareDisjoint2Axis) {
  /* Disjoint on y and z */
  const int3 a = make_int3(5, 6, 9);
  const int3 a_edge = make_int3(3);
  const int3 b = make_int3(6, 22, 13);
  const int3 b_edge = make_int3(10);

  int overlapx = geometry::axis_overlap(a.x, a_edge.x, b.x, b_edge.x);
  int overlapy = geometry::axis_overlap(a.y, a_edge.y, b.y, b_edge.y);
  int overlapz = geometry::axis_overlap(a.z, a_edge.z, b.z, b_edge.z);

  ASSERT_EQ(overlapx, 1);
  ASSERT_EQ(overlapy, 0);
  ASSERT_EQ(overlapz, 0);

  int overlaps = geometry::aabb_aabb_collision(a, a_edge, b, b_edge);
  ASSERT_EQ(overlaps, 0);
}

TEST(AABBAABBTest, SquareDisjoint) {
  /* Disjoint on x, y and z */
  const int3 a = make_int3(5, 6, 9);
  const int3 a_edge = make_int3(4);
  const int3 b = make_int3(12, 22, 43);
  const int3 b_edge = make_int3(10);

  int overlapx = geometry::axis_overlap(a.x, a_edge.x, b.x, b_edge.x);
  int overlapy = geometry::axis_overlap(a.y, a_edge.y, b.y, b_edge.y);
  int overlapz = geometry::axis_overlap(a.z, a_edge.z, b.z, b_edge.z);

  ASSERT_EQ(overlapx, 0);
  ASSERT_EQ(overlapy, 0);
  ASSERT_EQ(overlapz, 0);

  int overlaps = geometry::aabb_aabb_collision(a, a_edge, b, b_edge);
  ASSERT_EQ(overlaps, 0);
}

TEST(AABBAABBTest, SquareEnclosed) {
  /* Disjoint on x, y and z */
  const int3 a = make_int3(5, 6, 9);
  const int3 a_edge = make_int3(10);
  const int3 b = make_int3(6, 7, 10);
  const int3 b_edge = make_int3(2);

  int overlapx = geometry::axis_overlap(a.x, a_edge.x, b.x, b_edge.x);
  int overlapy = geometry::axis_overlap(a.y, a_edge.y, b.y, b_edge.y);
  int overlapz = geometry::axis_overlap(a.z, a_edge.z, b.z, b_edge.z);

  ASSERT_EQ(overlapx, 1);
  ASSERT_EQ(overlapy, 1);
  ASSERT_EQ(overlapz, 1);

  int overlaps = geometry::aabb_aabb_collision(a, a_edge, b, b_edge);
  ASSERT_EQ(overlaps, 1);
}

TEST(AABBAABBTest, Inclusion) {
  const int3 a = make_int3(2, 1, 3); 
  const int3 a_edge = make_int3(10);
  const int3 b = make_int3(3, 4, 5);
  const int3 b_edge = make_int3(2);
  int included = geometry::aabb_aabb_inclusion(a, a_edge, b, b_edge);
  ASSERT_EQ(included, 1);
}
