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

#ifndef NODE_H
#define NODE_H

#include <time.h>
#include <math_utils.h>
#include <atomic>
#include <octree_defines.h>
#include <utils/memory_pool.hpp>

template <typename T>
class Node {

public:
  typedef voxel_traits<T> traits_type;
  typedef typename traits_type::ComputeType compute_type;
  typedef typename traits_type::StoredType stored_type;
  compute_type empty() const { return traits_type::empty(); }
  compute_type init_val() const { return traits_type::initValue(); }

  compute_type value_[8];
  octlib::key_t code;
  unsigned int side;
  unsigned char children_mask_;

  Node(){
    code = 0;
    side = 0;
    children_mask_ = 0;
    for (unsigned int i = 0; i < 8; i++){
      value_[i]     = init_val();
      child_ptr_[i] = NULL;
    }
  }

    virtual ~Node(){};

    Node *& child(const int x, const int y, 
        const int z) {
      return child_ptr_[x + y*2 + z*4];
    };

    Node *& child(const int offset ){
      return child_ptr_[offset];
    }

    virtual bool isLeaf(){ return false; }

protected:
    Node *child_ptr_[8];
};

template <typename T>
class VoxelBlock: public Node<T> {

  public:

    typedef voxel_traits<T> traits_type;
    typedef typename traits_type::ComputeType compute_type;
    typedef typename traits_type::StoredType stored_type;
    static constexpr unsigned int side = BLOCK_SIDE;
    static constexpr unsigned int sideSq = side*side;

    static constexpr compute_type empty() { 
      return traits_type::empty(); 
    }
    static constexpr stored_type initValue() { 
      return traits_type::initValue();
    }
    stored_type translate(const compute_type value) { 
      return traits_type::translate(value); 
    }
    compute_type translate(const stored_type value) const {
      return traits_type::translate(value);
    }

    VoxelBlock(){
      timestamp_ = 0;
      coordinates_ = make_int3(0);
      for (unsigned int i = 0; i < side*sideSq; i++)
        voxel_block_[i] = initValue();
    }

    bool isLeaf(){ return true; }

    int3 coordinates() const { return coordinates_; }
    void coordinates(const int3 c){ coordinates_ = c; }

    compute_type data(const int3 pos) const;
    void data(const int3 pos, const compute_type& value);

    void timestamp(const time_t frame){ 
      timestamp_ = frame;
    }

    int timestamp() const {
      return timestamp_;
    }

    void active(const bool a){ active_ = a; }
    bool active() const { return active_; }

    stored_type * getBlockRawPtr(){ return voxel_block_; }
    static constexpr int size(){ return sizeof(VoxelBlock<T>); }

  private:
    VoxelBlock(const VoxelBlock&) = delete;
    int3 coordinates_;
    stored_type voxel_block_[side*sideSq]; // Brick of data.
    time_t timestamp_;
    bool active_;
};

template <typename T>
inline typename VoxelBlock<T>::compute_type VoxelBlock<T>::data(const int3 pos) const {
  int3 offset = pos - coordinates_;
  const stored_type& data = voxel_block_[offset.x + offset.y*side + 
                                         offset.z*sideSq];
  return translate(data);
}

template <typename T>
inline void VoxelBlock<T>::data(const int3 pos, 
                                const compute_type &value){
  int3 offset = pos - coordinates_;
  voxel_block_[offset.x + offset.y*side + offset.z*sideSq] = translate(value);
}
#endif
