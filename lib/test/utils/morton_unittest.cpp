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
#include "math_utils.h"
#include "utils/morton_utils.hpp"
#include "octree_defines.h"
#include "gtest/gtest.h"

TEST(MortonCoding, RandomInts) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> dis(0, 4096);

  for(int i = 0; i < 1000; ++i) {
    const uint3 vox = {dis(gen), dis(gen), dis(gen)};
    const octlib::key_t code = compute_morton(vox.x, vox.y, vox.z);
    const uint3 decoded = unpack_morton(code);
    ASSERT_EQ(decoded.x, vox.x);
    ASSERT_EQ(decoded.y, vox.y);
    ASSERT_EQ(decoded.z, vox.z);
  }

}

TEST(MortonCoding, ExhaustiveTest) {

  for(unsigned int z = 2048; z < 4096; ++z)
    for(unsigned int y = 2048; y < 2050; ++y)
      for(unsigned int x = 0; x < 4096; ++x){
        const uint3 vox = {x, y, z};
        const octlib::key_t code = compute_morton(vox.x, vox.y, vox.z);
        const uint3 decoded = unpack_morton(code);
        ASSERT_EQ(decoded.x, vox.x);
        ASSERT_EQ(decoded.y, vox.y);
        ASSERT_EQ(decoded.z, vox.z);
  }
}

