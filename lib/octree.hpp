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

#ifndef OCTREE_H
#define OCTREE_H

#include <cstring>
#include <math_utils.h>
#include <octree_defines.h>
#include <utils/morton_utils.hpp>
#include "octant_ops.hpp"
#include <algorithm>
#include <tuple>
#include <parallel/algorithm>
#include <node.hpp>
#include <utils/memory_pool.hpp>
#include <algorithms/unique.hpp>
#include <geometry/aabb_collision.hpp>
#include <interpolation/interp_gather.hpp>


inline int __float_as_int(float value){

  union float_as_int {
    float f;
    int i;
  };

  float_as_int u;
  u.f = value;
  return u.i;
}

inline float __int_as_float(int value){

  union int_as_float {
    int i;
    float f;
  };

  int_as_float u;
  u.i = value;
  return u.f;
}

template <typename T>
class ray_iterator;
template <typename T>
class leaf_iterator;

template <typename T>
class Octree
{

public:

  typedef voxel_traits<T> traits_type;
  typedef typename traits_type::ComputeType compute_type;
  typedef typename traits_type::StoredType stored_type;
  compute_type empty() const { return traits_type::empty(); }
  compute_type init_val() const { return traits_type::initValue(); }

  // Compile-time constant expressions
  // # of voxels per side in a voxel block
  static constexpr unsigned int blockSide = BLOCK_SIDE;
  // maximum tree depth in bits
  static constexpr unsigned int max_depth = ((sizeof(octlib::key_t)*8)/3);
  // Tree depth at which blocks are found
  static constexpr unsigned int block_depth = max_depth - log2_const(BLOCK_SIDE);


  Octree(){
  };

  ~Octree(){
  }

  /*! \brief Initialises the octree attributes
   * \param size number of voxels per side of the cube
   * \param dim cube extension per side, in meter
   */
  void init(int size, float dim);

  inline int size() const { return size_; }
  inline float dim() const { return dim_; }
  inline Node<T>* root() const { return root_; }

  /*! \brief Retrieves voxel value at coordinates (x,y,z), if not present it 
   * allocates it. This method is not thread safe.
   * \param x x coordinate in interval [0, size]
   * \param y y coordinate in interval [0, size]
   * \param z z coordinate in interval [0, size]
   */
  void set(const int x, const int y, const int z, const compute_type val);

  /*! \brief Retrieves voxel value at coordinates (x,y,z)
   * \param x x coordinate in interval [0, size]
   * \param y y coordinate in interval [0, size]
   * \param z z coordinate in interval [0, size]
   */
  compute_type get(const int x, const int y, const int z) const;
  compute_type get_fine(const int x, const int y, const int z) const;

  /*! \brief Fetch the voxel block at which contains voxel  (x,y,z)
   * \param x x coordinate in interval [0, size]
   * \param y y coordinate in interval [0, size]
   * \param z z coordinate in interval [0, size]
   */
  VoxelBlock<T> * fetch(const int x, const int y, const int z) const;

  /*! \brief Fetch the voxel block at which contains voxel  (x,y,z)
   * \param x x coordinate in interval [0, size]
   * \param y y coordinate in interval [0, size]
   * \param z z coordinate in interval [0, size]
   * \param depth maximum depth to be searched 
   */
  Node<T> * fetch_octant(const int x, const int y, const int z, 
      const int depth) const;

  /*! \brief Interp voxel value at voxel position  (x,y,z)
   * \param pos three-dimensional coordinates in which each component belongs 
   * to the interval [0, size]
   * \return signed distance function value at voxel position (x, y, z)
   */

  template <typename FieldSelect>
  float interp(const float3 pos, FieldSelect f) const;

  /*! \brief Compute the gradient at voxel position  (x,y,z)
   * \param pos three-dimensional coordinates in which each component belongs 
   * to the interval [0, size]
   * \return gradient at voxel position pos
   */
  float3 grad(const float3 pos) const;

  template <typename FieldSelect>
  float3 grad(const float3 pos, FieldSelect selector) const;

  /*! \brief Cast a ray to find the closest signed distance function 
   * intersection.
   * \param pos camera pixel from which the ray is originating 
   * \param view camera SE3 pose 
   * \param nearPlane search range (in meters) 
   * \param farPlane maximum search range (in meters) 
   * \param mu signed distance function truncation value (in meters) 
   * \param step ray-casting step size (usually proportional to 
   * the voxel spacing 
   * \param largeStep larger step size (usually a fraction of mu)
   * \return float4 with the first three coordinates indicating the 
   * intersection position and the last one the distance travelled 
   * (in metric units). If no intersection is found the last coordinate is set
   * to 0.
   */

  float4 raycast(const uint2 pos, const Matrix4 view,
      const float nearPlane, const float farPlane, const float mu,
      const float step, const float largeStep) const;

  /*! \brief Get the list of allocated block. If the active switch is set to
   * true then only the visible blocks are retrieved.
   * \param blocklist output vector of allocated blocks
   * \param active boolean switch. Set to true to retrieve visible, allocated 
   * blocks, false to retrieve all allocated blocks.
   */
  void getBlockList(std::vector<VoxelBlock<T> *>& blocklist, bool active);
  MemoryPool<VoxelBlock<T> >& getBlockBuffer(){ return block_memory_; };
  MemoryPool<Node<T> >& getNodesBuffer(){ return nodes_buffer_; };
  /*! \brief Computes the morton code of the block containing voxel 
   * at coordinates (x,y,z)
   * \param x x coordinate in interval [0, size]
   * \param y y coordinate in interval [0, size]
   * \param z z coordinate in interval [0, size]
   */
  octlib::key_t hash(const int x, const int y, const int z) {
    const int scale = max_level_ - log2_const(blockSide); // depth of blocks
    return octlib::keyops::encode(x, y, z, scale, max_level_);   
  }

  octlib::key_t hash(const int x, const int y, const int z, octlib::key_t scale) {
    return octlib::keyops::encode(x, y, z, scale, max_level_); 
  }

  /*! \brief allocate a set of voxel blocks via their positional key  
   * \param keys collection of voxel block keys to be allocated (i.e. their 
   * morton number)
   * \param number of keys in the keys array
   */
  bool allocate(octlib::key_t *keys, int num_elem);
  bool alloc_update(octlib::key_t *keys, int num_elem);

  /*! \brief Counts the number of blocks allocated
   * \return number of voxel blocks allocated
   */
  int leavesCount();

  /*! \brief Counts the number of internal nodes
   * \return number of internal nodes
   */
  int nodeCount();

  void printMemStats(){
    // memory.printStats();
  };

private:

  struct stack_entry {
    int scale;
    Node<T> * parent;
    float t_max;
  };

  Node<T> * root_;
  int size_;
  float dim_;
  int max_level_;
  MemoryPool<VoxelBlock<T> > block_memory_;
  MemoryPool<Node<T> > nodes_buffer_;

  friend class ray_iterator<T>;
  friend class leaf_iterator<T>;

  // Allocation specific variables
  octlib::key_t* keys_at_level_;
  int reserved_;

  // Private implementation of cached methods
  compute_type get(const int x, const int y, const int z, VoxelBlock<T>* cached) const;
  compute_type get(const float3 pos, VoxelBlock<T>* cached) const;

  // Parallel allocation of a given tree level for a set of input keys.
  // Pre: levels above target_level must have been already allocated
  bool allocateLevel(octlib::key_t * keys, int num_tasks, int target_level);
  bool updateLevel(octlib::key_t * keys, int num_tasks, int target_level, 
      const octlib::key_t scale_mask = 0x0);

  void reserveBuffers(const int n);

  // General helpers

  int leavesCountRecursive(Node<T> *);
  int nodeCountRecursive(Node<T> *);
  void getActiveBlockList(Node<T> *, std::vector<VoxelBlock<T> *>& blocklist);
  void getAllocatedBlockList(Node<T> *, std::vector<VoxelBlock<T> *>& blocklist);

  void deleteNode(Node<T> ** node);
  void deallocateTree(){ deleteNode(&root_); }
};


template <typename T>
inline typename Octree<T>::compute_type Octree<T>::get(const float3 p, 
    VoxelBlock<T>* cached) const {

  const int3 pos = make_int3((p.x * size_ / dim_),
      (p.y * size_ / dim_),
      (p.z * size_ / dim_));

  if(cached != NULL){
    int3 lower = cached->coordinates();
    int3 upper = lower + (blockSide-1);
    if(in(pos.x, lower.x, upper.x) && in(pos.y, lower.y, upper.y) &&
       in(pos.z, lower.z, upper.z)){
      return cached->data(pos);
    }
  }

  Node<T> * n = root_;
  if(!n) {
    return empty();
  }

  // Get the block.

  uint edge = size_ >> 1;
  for(; edge >= blockSide; edge = edge >> 1){
    n = n->child((pos.x & edge) > 0, (pos.y & edge) > 0, (pos.z & edge) > 0);
    if(!n){
    return empty();
    }
  }

  // Get the element in the voxel block
  return static_cast<VoxelBlock<T>*>(n)->data(pos);
}

template <typename T>
inline void  Octree<T>::set(const int x,
    const int y, const int z, const compute_type val) {

  Node<T> * n = root_;
  if(!n) {
    return;
  }

  uint edge = size_ >> 1;
  for(; edge >= blockSide; edge = edge >> 1){
    Node<T>* tmp = n->child((x & edge) > 0, (y & edge) > 0, (z & edge) > 0);
    if(!tmp){
      return;
    }
    n = tmp;
  }

  static_cast<VoxelBlock<T> *>(n)->data(make_int3(x, y, z), val);
}


template <typename T>
inline typename Octree<T>::compute_type Octree<T>::get(const int x,
    const int y, const int z) const {

  Node<T> * n = root_;
  if(!n) {
    return init_val();
  }

  uint edge = size_ >> 1;
  for(; edge >= blockSide; edge = edge >> 1){
    const int childid = ((x & edge) > 0) +  2 * ((y & edge) > 0) +  4*((z & edge) > 0);
    Node<T>* tmp = n->child(childid);
    if(!tmp){
      return n->value_[childid];
    }
    n = tmp;
  }

  return static_cast<VoxelBlock<T> *>(n)->data(make_int3(x, y, z));
}

template <typename T>
inline typename Octree<T>::compute_type Octree<T>::get_fine(const int x,
    const int y, const int z) const {

  Node<T> * n = root_;
  if(!n) {
    return init_val();
  }

  uint edge = size_ >> 1;
  for(; edge >= blockSide; edge = edge >> 1){
    const int childid = ((x & edge) > 0) +  2 * ((y & edge) > 0) +  4*((z & edge) > 0);
    Node<T>* tmp = n->child(childid);
    if(!tmp){
      return init_val();
    }
    n = tmp;
  }

  return static_cast<VoxelBlock<T> *>(n)->data(make_int3(x, y, z));
}

template <typename T>
inline typename Octree<T>::compute_type Octree<T>::get(const int x,
   const int y, const int z, VoxelBlock<T>* cached) const {

  if(cached != NULL){
    const int3 lower = cached->coordinates();
    const int3 upper = lower + (blockSide-1);
    if(in(x, lower.x, upper.x) && in(y, lower.y, upper.y) &&
       in(z, lower.z, upper.z)){
      return cached->data(make_int3(x, y, z));
    }
  }

  Node<T> * n = root_;
  if(!n) {
    return init_val();
  }

  uint edge = size_ >> 1;
  for(; edge >= blockSide; edge = edge >> 1){
    n = n->child((x & edge) > 0, (y & edge) > 0, (z & edge) > 0);
    if(!n){
      return init_val();
    }
  }

  return static_cast<VoxelBlock<T> *>(n)->data(make_int3(x, y, z));
}

template <typename T>
void Octree<T>::deleteNode(Node<T> **node){

  if(*node){
    for (int i = 0; i < 8; i++) {
      if((*node)->child(i)){
        deleteNode(&(*node)->child(i));
      }
    }
    if(!(*node)->isLeaf()){
      delete *node;
      *node = NULL;
    }
  }
}


template <typename T>
void Octree<T>::init(int size, float dim) {
  size_ = size;
  dim_ = dim;
  max_level_ = log2(size);
  root_ = new Node<T>();
  root_->side = size;
  reserved_ = 1024;
  keys_at_level_ = new octlib::key_t[reserved_];
  std::memset(keys_at_level_, 0, reserved_);
}

template <typename T>
inline VoxelBlock<T> * Octree<T>::fetch(const int x, const int y, 
   const int z) const {

  Node<T> * n = root_;
  if(!n) {
    return NULL;
  }

  // Get the block.
  uint edge = size_ / 2;
  for(; edge >= blockSide; edge /= 2){
    n = n->child((x & edge) > 0u, (y & edge) > 0u, (z & edge) > 0u);
    if(!n){
      return NULL;
    }
  }
  return static_cast<VoxelBlock<T>* > (n);
}

template <typename T>
inline Node<T> * Octree<T>::fetch_octant(const int x, const int y, 
   const int z, const int depth) const {

  Node<T> * n = root_;
  if(!n) {
    return NULL;
  }

  // Get the block.
  uint edge = size_ / 2;
  for(int d = 1; edge >= blockSide && d <= depth; edge /= 2, ++d){
    n = n->child((x & edge) > 0u, (y & edge) > 0u, (z & edge) > 0u);
    if(!n){
      return NULL;
    }
  }
  return n;
}

template <typename T>
template <typename FieldSelector>
float Octree<T>::interp(float3 pos, FieldSelector select) const {
  
  const int3 base = make_int3(floorf(pos));
  const float3 factor = fracf(pos);
  const int3 lower = max(base, make_int3(0));

  float points[8];
  gather_points(*this, lower, select, points);

  return (((points[0] * (1 - factor.x)
          + points[1] * factor.x) * (1 - factor.y)
          + (points[2] * (1 - factor.x)
          + points[3] * factor.x) * factor.y)
          * (1 - factor.z)
          + ((points[4] * (1 - factor.x)
          + points[5] * factor.x)
          * (1 - factor.y)
          + (points[6] * (1 - factor.x)
          + points[7] * factor.x)
          * factor.y) * factor.z);
}


template <typename T>
float3 Octree<T>::grad(const float3 pos) const {

   int3 base = make_int3(floorf(pos));
   float3 factor = fracf(pos);
   int3 lower_lower = max(base - make_int3(1), make_int3(0));
   int3 lower_upper = max(base, make_int3(0));
   int3 upper_lower = min(base + make_int3(1),
      make_int3(size_) - make_int3(1));
   int3 upper_upper = min(base + make_int3(2),
      make_int3(size_) - make_int3(1));
   int3 & lower = lower_upper;
   int3 & upper = upper_lower;

  float3 gradient;

  VoxelBlock<T> * n = fetch(base.x, base.y, base.z);
  gradient.x = (((get(upper_lower.x, lower.y, lower.z, n).x
          - get(lower_lower.x, lower.y, lower.z, n).x) * (1 - factor.x)
        + (get(upper_upper.x, lower.y, lower.z, n).x
          - get(lower_upper.x, lower.y, lower.z, n).x) * factor.x)
      * (1 - factor.y)
      + ((get(upper_lower.x, upper.y, lower.z, n).x
          - get(lower_lower.x, upper.y, lower.z, n).x) * (1 - factor.x)
        + (get(upper_upper.x, upper.y, lower.z, n).x
          - get(lower_upper.x, upper.y, lower.z, n).x)
        * factor.x) * factor.y) * (1 - factor.z)
    + (((get(upper_lower.x, lower.y, upper.z, n).x
            - get(lower_lower.x, lower.y, upper.z, n).x) * (1 - factor.x)
          + (get(upper_upper.x, lower.y, upper.z, n).x
            - get(lower_upper.x, lower.y, upper.z, n).x)
          * factor.x) * (1 - factor.y)
        + ((get(upper_lower.x, upper.y, upper.z, n).x
            - get(lower_lower.x, upper.y, upper.z, n).x)
          * (1 - factor.x)
          + (get(upper_upper.x, upper.y, upper.z, n).x
            - get(lower_upper.x, upper.y, upper.z, n).x)
          * factor.x) * factor.y) * factor.z;

  gradient.y = (((get(lower.x, upper_lower.y, lower.z, n).x
          - get(lower.x, lower_lower.y, lower.z, n).x) * (1 - factor.x)
        + (get(upper.x, upper_lower.y, lower.z, n).x
          - get(upper.x, lower_lower.y, lower.z, n).x) * factor.x)
      * (1 - factor.y)
      + ((get(lower.x, upper_upper.y, lower.z, n).x
          - get(lower.x, lower_upper.y, lower.z, n).x) * (1 - factor.x)
        + (get(upper.x, upper_upper.y, lower.z, n).x
          - get(upper.x, lower_upper.y, lower.z, n).x)
        * factor.x) * factor.y) * (1 - factor.z)
    + (((get(lower.x, upper_lower.y, upper.z, n).x
            - get(lower.x, lower_lower.y, upper.z, n).x) * (1 - factor.x)
          + (get(upper.x, upper_lower.y, upper.z, n).x
            - get(upper.x, lower_lower.y, upper.z, n).x)
          * factor.x) * (1 - factor.y)
        + ((get(lower.x, upper_upper.y, upper.z, n).x
            - get(lower.x, lower_upper.y, upper.z, n).x)
          * (1 - factor.x)
          + (get(upper.x, upper_upper.y, upper.z, n).x
            - get(upper.x, lower_upper.y, upper.z, n).x)
          * factor.x) * factor.y) * factor.z;

  gradient.z = (((get(lower.x, lower.y, upper_lower.z, n).x
          - get(lower.x, lower.y, lower_lower.z, n).x) * (1 - factor.x)
        + (get(upper.x, lower.y, upper_lower.z, n).x
          - get(upper.x, lower.y, lower_lower.z, n).x) * factor.x)
      * (1 - factor.y)
      + ((get(lower.x, upper.y, upper_lower.z, n).x
          - get(lower.x, upper.y, lower_lower.z, n).x) * (1 - factor.x)
        + (get(upper.x, upper.y, upper_lower.z, n).x
          - get(upper.x, upper.y, lower_lower.z, n).x)
        * factor.x) * factor.y) * (1 - factor.z)
    + (((get(lower.x, lower.y, upper_upper.z, n).x
            - get(lower.x, lower.y, lower_upper.z, n).x) * (1 - factor.x)
          + (get(upper.x, lower.y, upper_upper.z, n).x
            - get(upper.x, lower.y, lower_upper.z, n).x)
          * factor.x) * (1 - factor.y)
        + ((get(lower.x, upper.y, upper_upper.z, n).x
            - get(lower.x, upper.y, lower_upper.z, n).x)
          * (1 - factor.x)
          + (get(upper.x, upper.y, upper_upper.z, n).x
            - get(upper.x, upper.y, lower_upper.z, n).x)
          * factor.x) * factor.y) * factor.z;

  return gradient
    * make_float3(dim_ / size_, dim_ / size_, dim_ / size_)
    * (0.5f);

}

template <typename T>
template <typename FieldSelector>
float3 Octree<T>::grad(const float3 pos, FieldSelector select) const {

   int3 base = make_int3(floorf(pos));
   float3 factor = fracf(pos);
   int3 lower_lower = max(base - make_int3(1), make_int3(0));
   int3 lower_upper = max(base, make_int3(0));
   int3 upper_lower = min(base + make_int3(1),
      make_int3(size_) - make_int3(1));
   int3 upper_upper = min(base + make_int3(2),
      make_int3(size_) - make_int3(1));
   int3 & lower = lower_upper;
   int3 & upper = upper_lower;

  float3 gradient;

  VoxelBlock<T> * n = fetch(base.x, base.y, base.z);
  gradient.x = (((select(get(upper_lower.x, lower.y, lower.z, n))
          - select(get(lower_lower.x, lower.y, lower.z, n))) * (1 - factor.x)
        + (select(get(upper_upper.x, lower.y, lower.z, n))
          - select(get(lower_upper.x, lower.y, lower.z, n))) * factor.x)
      * (1 - factor.y)
      + ((select(get(upper_lower.x, upper.y, lower.z, n))
          - select(get(lower_lower.x, upper.y, lower.z, n))) * (1 - factor.x)
        + (select(get(upper_upper.x, upper.y, lower.z, n))
          - select(get(lower_upper.x, upper.y, lower.z, n)))
        * factor.x) * factor.y) * (1 - factor.z)
    + (((select(get(upper_lower.x, lower.y, upper.z, n))
            - select(get(lower_lower.x, lower.y, upper.z, n))) * (1 - factor.x)
          + (select(get(upper_upper.x, lower.y, upper.z, n))
            - select(get(lower_upper.x, lower.y, upper.z, n)))
          * factor.x) * (1 - factor.y)
        + ((select(get(upper_lower.x, upper.y, upper.z, n))
            - select(get(lower_lower.x, upper.y, upper.z, n)))
          * (1 - factor.x)
          + (select(get(upper_upper.x, upper.y, upper.z, n))
            - select(get(lower_upper.x, upper.y, upper.z, n)))
          * factor.x) * factor.y) * factor.z;

  gradient.y = (((select(get(lower.x, upper_lower.y, lower.z, n))
          - select(get(lower.x, lower_lower.y, lower.z, n))) * (1 - factor.x)
        + (select(get(upper.x, upper_lower.y, lower.z, n))
          - select(get(upper.x, lower_lower.y, lower.z, n))) * factor.x)
      * (1 - factor.y)
      + ((select(get(lower.x, upper_upper.y, lower.z, n))
          - select(get(lower.x, lower_upper.y, lower.z, n))) * (1 - factor.x)
        + (select(get(upper.x, upper_upper.y, lower.z, n))
          - select(get(upper.x, lower_upper.y, lower.z, n)))
        * factor.x) * factor.y) * (1 - factor.z)
    + (((select(get(lower.x, upper_lower.y, upper.z, n))
            - select(get(lower.x, lower_lower.y, upper.z, n))) * (1 - factor.x)
          + (select(get(upper.x, upper_lower.y, upper.z, n))
            - select(get(upper.x, lower_lower.y, upper.z, n)))
          * factor.x) * (1 - factor.y)
        + ((select(get(lower.x, upper_upper.y, upper.z, n))
            - select(get(lower.x, lower_upper.y, upper.z, n)))
          * (1 - factor.x)
          + (select(get(upper.x, upper_upper.y, upper.z, n))
            - select(get(upper.x, lower_upper.y, upper.z, n)))
          * factor.x) * factor.y) * factor.z;

  gradient.z = (((select(get(lower.x, lower.y, upper_lower.z, n))
          - select(get(lower.x, lower.y, lower_lower.z, n))) * (1 - factor.x)
        + (select(get(upper.x, lower.y, upper_lower.z, n))
          - select(get(upper.x, lower.y, lower_lower.z, n))) * factor.x)
      * (1 - factor.y)
      + ((select(get(lower.x, upper.y, upper_lower.z, n))
          - select(get(lower.x, upper.y, lower_lower.z, n))) * (1 - factor.x)
        + (select(get(upper.x, upper.y, upper_lower.z, n))
          - select(get(upper.x, upper.y, lower_lower.z, n)))
        * factor.x) * factor.y) * (1 - factor.z)
    + (((select(get(lower.x, lower.y, upper_upper.z, n))
            - select(get(lower.x, lower.y, lower_upper.z, n))) * (1 - factor.x)
          + (select(get(upper.x, lower.y, upper_upper.z, n))
            - select(get(upper.x, lower.y, lower_upper.z, n)))
          * factor.x) * (1 - factor.y)
        + ((select(get(lower.x, upper.y, upper_upper.z, n))
            - select(get(lower.x, upper.y, lower_upper.z, n)))
          * (1 - factor.x)
          + (select(get(upper.x, upper.y, upper_upper.z, n))
            - select(get(upper.x, upper.y, lower_upper.z, n)))
          * factor.x) * factor.y) * factor.z;

  return gradient
    * make_float3(dim_ / size_, dim_ / size_, dim_ / size_)
    * (0.5f);
}

template <typename T>
int Octree<T>::leavesCount(){
  return leavesCountRecursive(root_);
}

template <typename T>
int Octree<T>::leavesCountRecursive(Node<T> * n){

  if(!n) return 0;

  if(n->isLeaf()){
    return 1;
  }

  int sum = 0;

  for (int i = 0; i < 8; i++){
    sum += leavesCountRecursive(n->child(i));
  }

  return sum;
}

template <typename T>
int Octree<T>::nodeCount(){
  return nodeCountRecursive(root_);
}

template <typename T>
int Octree<T>::nodeCountRecursive(Node<T> * node){
  if (!node) {
    return 0;
  }

  int n = 1;
  for (int i = 0; i < 8; ++i) {
    n += (n ? nodeCountRecursive((node)->child(i)) : 0);
  }
  return n;
}

template <typename T>
void Octree<T>::reserveBuffers(const int n){

  if(n > reserved_){
    // std::cout << "Reserving " << n << " entries in allocation buffers" << std::endl;
    delete[] keys_at_level_;
    keys_at_level_ = new octlib::key_t[n];
    reserved_ = n;
  }
  block_memory_.reserve(n);
}

template <typename T>
bool Octree<T>::allocate(octlib::key_t *keys, int num_elem){

#ifdef _OPENMP
  __gnu_parallel::sort(keys, keys+num_elem);
#else
std::sort(keys, keys+num_elem);
#endif

  num_elem = algorithms::unique(keys, num_elem);
  reserveBuffers(num_elem);

  int last_elem = 0;
  bool success = false;

  const int leaf_level = max_level_ - log2(blockSide);
  const unsigned int shift = MAX_BITS - max_level_ - 1;
  for (int level = 1; level <= leaf_level; level++){
    const octlib::key_t mask = MASK[level + shift];
    compute_prefix(keys, keys_at_level_, num_elem, mask);
    last_elem = algorithms::unique(keys_at_level_, num_elem);
    success = allocateLevel(keys_at_level_, last_elem, level);
  }
  return success;
}

template <typename T>
bool Octree<T>::alloc_update(octlib::key_t *keys, int num_elem){

#ifdef _OPENMP
  __gnu_parallel::sort(keys, keys+num_elem);
#else
std::sort(keys, keys+num_elem);
#endif

  num_elem = algorithms::unique_multiscale(keys, num_elem, SCALE_MASK);
  reserveBuffers(num_elem);

  int last_elem = 0;
  bool success = false;

  const int leaves_level = max_level_ - log2(blockSide);
  const unsigned int shift = MAX_BITS - max_level_ - 1;
  for (int level = 1; level <= leaves_level; level++){
    const octlib::key_t mask = MASK[level + shift] | SCALE_MASK;
    compute_prefix(keys, keys_at_level_, num_elem, mask);
    last_elem = algorithms::unique_multiscale(keys_at_level_, num_elem, 
        SCALE_MASK, level);
    success = updateLevel(keys_at_level_, last_elem, level, SCALE_MASK);
  }
  return success;
}

template <typename T>
bool Octree<T>::allocateLevel(octlib::key_t* keys, int num_tasks, int target_level){

  int leaves_level = max_level_ - log2(blockSide);
  nodes_buffer_.reserve(num_tasks);

#pragma omp parallel for
  for (int i = 0; i < num_tasks; i++){
    Node<T> ** n = &root_;
    octlib::key_t myKey = keys[i];
    int edge = size_/2;

    for (int level = 1; level <= target_level; ++level){
      int index = child_id(myKey, level, max_level_); 
      Node<T> * parent = *n;
      n = &(*n)->child(index);

      if(!(*n)){
        if(level == leaves_level){
          *n = block_memory_.acquire_block();
          static_cast<VoxelBlock<T> *>(*n)->coordinates(make_int3(unpack_morton(myKey)));
          static_cast<VoxelBlock<T> *>(*n)->active(true);
          (*n)->code = myKey;
          parent->children_mask_ = parent->children_mask_ | (1 << index);
        }
        else {
          *n = nodes_buffer_.acquire_block();;
          (*n)->code = myKey;
          (*n)->side = edge;
          parent->children_mask_ = parent->children_mask_ | (1 << index);
        }
      }
      edge /= 2;
    }
  }
  return true;
}

template <typename T>
bool Octree<T>::updateLevel(octlib::key_t* keys, int num_tasks, int target_level,
    const octlib::key_t scale_mask){

  int leaves_level = max_level_ - log2(blockSide);
  nodes_buffer_.reserve(num_tasks);

#pragma omp parallel for
  for (int i = 0; i < num_tasks; i++){
    Node<T> ** n = &root_;
    octlib::key_t myKey = keys[i] & (~scale_mask);
    int edge = size_/2;

    for (int level = 1; level <= target_level; ++level){
      int index = child_id(myKey, level, max_level_); 
      Node<T> * parent = *n;
      n = &(*n)->child(index);

      if(!(*n)){
        if(level == leaves_level){
          *n = block_memory_.acquire_block();
          static_cast<VoxelBlock<T> *>(*n)->coordinates(make_int3(unpack_morton(myKey)));
          static_cast<VoxelBlock<T> *>(*n)->active(true);
          static_cast<VoxelBlock<T> *>(*n)->code = myKey | level;
          parent->children_mask_ = parent->children_mask_ | (1 << index);
        }
        else  {
          *n = nodes_buffer_.acquire_block();;
          (*n)->code = myKey | level;
          (*n)->side = edge;
          parent->children_mask_ = parent->children_mask_ | (1 << index);
        }
      }
      edge /= 2;
    }
  }
  return true;
}

template <typename T>
void Octree<T>::getBlockList(std::vector<VoxelBlock<T>*>& blocklist, bool active){
  Node<T> * n = root_;
  if(!n) return;
  if(active) getActiveBlockList(n, blocklist);
  else getAllocatedBlockList(n, blocklist);
}

template <typename T>
void Octree<T>::getActiveBlockList(Node<T> *n,
    std::vector<VoxelBlock<T>*>& blocklist){
  using tNode = Node<T>;
  if(!n) return;
  std::queue<tNode *> q;
  q.push(n);
  while(!q.empty()){
    tNode* node = q.front();
    q.pop();

    if(node->isLeaf()){
      VoxelBlock<T>* block = static_cast<VoxelBlock<T> *>(node);
      if(block->active()) blocklist.push_back(block);
      continue;
    }

    for(int i = 0; i < 8; ++i){
      if(node->child(i)) q.push(node->child(i));
    }
  }
}

template <typename T>
void Octree<T>::getAllocatedBlockList(Node<T> *,
    std::vector<VoxelBlock<T>*>& blocklist){
  for(unsigned int i = 0; i < block_memory_.size(); ++i) {
      blocklist.push_back(block_memory_[i]);
    }
  }


/*****************************************************************************
 *
 *
 * Ray iterator definition
 *
 * A modified version of the ray-caster introduced in the paper:
 * https://research.nvidia.com/publication/efficient-sparse-voxel-octrees
 *
 * Original code available at:
 * https://code.google.com/p/efficient-sparse-voxel-octrees/
 *
 * 
*****************************************************************************/

template <typename T>
class ray_iterator {

  public:
    ray_iterator(const Octree<T>& m, const float3 origin, 
        const float3 direction, float nearPlane, float farPlane) : map_(m) {

      pos_ = make_float3(1.0f, 1.0f, 1.0f);
      idx_ = 0;
      parent_ = map_.root_;
      child_ = NULL;
      scale_exp2_ = 0.5f;
      scale_ = CAST_STACK_DEPTH-1;
      min_scale_ = CAST_STACK_DEPTH - log2(m.size_/Octree<T>::blockSide);
      static const float epsilon = exp2f(-log2(map_.size_));
      voxelSize_ = map_.dim_/map_.size_;
      state_ = INIT; 

      direction_.x = fabsf(direction.x) < epsilon ? 
        copysignf(epsilon, direction.x) : direction.x;
      direction_.y = fabsf(direction.y) < epsilon ? 
        copysignf(epsilon, direction.y) : direction.y;
      direction_.z = fabsf(direction.z) < epsilon ? 
        copysignf(epsilon, direction.z) : direction.z;

      /* Scaling the origin to resides between coordinates [1,2] */
      const float3 scaled_origin = origin/map_.dim_ + 1.f;

      /* Precomputing the ray coefficients */
      t_coef_ = -1.f/fabs(direction_);
      t_bias_ = t_coef_ * scaled_origin;


      /* Build the octrant mask to to mirror the coordinate system such that
       * each ray component points in negative coordinates. The octree is 
       * assumed to reside at coordinates [1, 2]
       * 
       */
      octant_mask_ = 7;
      if(direction_.x > 0.0f) octant_mask_ ^=1, t_bias_.x = 3.0f * t_coef_.x - t_bias_.x;
      if(direction_.y > 0.0f) octant_mask_ ^=2, t_bias_.y = 3.0f * t_coef_.y - t_bias_.y;
      if(direction_.z > 0.0f) octant_mask_ ^=4, t_bias_.z = 3.0f * t_coef_.z - t_bias_.z;

      /* Find the min-max t ranges. */
      t_min_ = fmaxf(
          fmaxf(2.0f * t_coef_.x - t_bias_.x, 2.0f * t_coef_.y - t_bias_.y), 2.0f * t_coef_.z - t_bias_.z);
      t_max_ = fminf(fminf(t_coef_.x - t_bias_.x, t_coef_.y - t_bias_.y), t_coef_.z - t_bias_.z);
      h_ = t_max_;
      t_min_ = fmaxf(t_min_, nearPlane/map_.dim_);
      t_max_ = fminf(t_max_, farPlane/map_.dim_);

      /*
       * Initialise the ray position
       */
      if (1.5f * t_coef_.x - t_bias_.x > t_min_) idx_^= 1, pos_.x = 1.5f;
      if (1.5f * t_coef_.y - t_bias_.y > t_min_) idx_^= 2, pos_.y = 1.5f;
      if (1.5f * t_coef_.z - t_bias_.z > t_min_) idx_^= 4, pos_.z = 1.5f;

    };

    /* 
     * Advance the ray.
     */
    inline void advance_ray() {

      int step_mask = 0;
      
      step_mask = (t_corner_.x <= tc_max_) | 
          ((t_corner_.y <= tc_max_) << 1) | ((t_corner_.z <= tc_max_) << 2);
      pos_.x -= scale_exp2_ * bool(step_mask & 1);
      pos_.y -= scale_exp2_ * bool(step_mask & 2);
      pos_.z -= scale_exp2_ * bool(step_mask & 4);

      t_min_ = tc_max_;
      idx_ ^= step_mask;

      // POP if bits flips disagree with ray direction

      if ((idx_ & step_mask) != 0) {

        // Get the different bits for each component.
        // This is done by xoring the bit patterns of the new and old pos
        // (float_as_int reinterprets a floating point number as int,
        // it is a sort of reinterpret_cast). This work because the volume has
        // been scaled between [1, 2]. Still digging why this is the case. 

        unsigned int differing_bits = 0;
        if ((step_mask & 1) != 0) differing_bits |= __float_as_int(pos_.x) ^ __float_as_int(pos_.x + scale_exp2_);
        if ((step_mask & 2) != 0) differing_bits |= __float_as_int(pos_.y) ^ __float_as_int(pos_.y + scale_exp2_);
        if ((step_mask & 4) != 0) differing_bits |= __float_as_int(pos_.z) ^ __float_as_int(pos_.z + scale_exp2_);

        // Get the scale at which the two differs. Here's there are different subtlelties related to how fp are stored.
        // MIND BLOWN: differing bit (i.e. the MSB) extracted using the 
        // exponent part of the fp representation. 
        scale_ = (__float_as_int((float)differing_bits) >> 23) - 127; // position of the highest bit
        scale_exp2_ = __int_as_float((scale_ - CAST_STACK_DEPTH + 127) << 23); // exp2f(scale - s_max)
        struct stack_entry&  e = stack[scale_];
        parent_ = e.parent;
        t_max_ = e.t_max;

        // Round cube position and extract child slot index.

        int shx = __float_as_int(pos_.x) >> scale_;
        int shy = __float_as_int(pos_.y) >> scale_;
        int shz = __float_as_int(pos_.z) >> scale_;
        pos_.x = __int_as_float(shx << scale_);
        pos_.y = __int_as_float(shy << scale_);
        pos_.z = __int_as_float(shz << scale_);
        idx_  = (shx & 1) | ((shy & 1) << 1) | ((shz & 1) << 2);

        h_ = 0.0f;
        child_ = NULL;

      }
    }

    /* 
     * Descend the hiararchy and compute the next child position.
     */
    inline void descend() {
      float tv_max = fminf(t_max_, tc_max_);
      float half = scale_exp2_ * 0.5f;
      float3 t_center = half * t_coef_ + t_corner_;

      // Descend to the first child if the resulting t-span is non-empty.

      if (tc_max_ < h_) {
        stack[scale_] = {scale_, parent_, t_max_};
      }

      h_ = tc_max_;
      parent_ = child_;

      idx_ = 0;
      scale_--;
      scale_exp2_ = half;
      idx_ = (t_center.x > t_min_) | 
          ((t_center.y > t_min_) << 1) | ((t_center.z > t_min_) << 2);

      pos_.x += scale_exp2_ * bool(idx_ & 1);
      pos_.y += scale_exp2_ * bool(idx_ & 2);
      pos_.z += scale_exp2_ * bool(idx_ & 4);

      t_max_ = tv_max;
      child_ = NULL;
    }

    /*
     * Returns the next leaf along the ray direction.
     */

    std::tuple<float, float, float> next() {

      if(state_ == ADVANCE) advance_ray();
      else if (state_ == FINISHED) return std::make_tuple(-1.f, -1.f, -1.f);

      while (scale_ < CAST_STACK_DEPTH) {
        t_corner_ = pos_ * t_coef_ - t_bias_;
        tc_max_ = fminf(fminf(t_corner_.x, t_corner_.y), t_corner_.z);

        child_ = parent_->child(idx_ ^ octant_mask_ ^ 7);

        if (scale_ == min_scale_ && child_ != NULL){
          state_ = ADVANCE;
          return std::make_tuple(t_min_ * map_.dim_ /voxelSize_, 
              t_min_ * map_.dim_ /voxelSize_,
              stack[CAST_STACK_DEPTH-1].t_max * map_.dim_ /voxelSize_);
        } else if (child_ != NULL && t_min_ <= t_max_){  // If the child is valid, descend the tree hierarchy.
          descend();
          continue;
        }
        advance_ray();
      }
      return std::make_tuple(-1.f, -1.f, -1.f);
    }

  private:
    struct stack_entry {
      int scale;
      Node<T> * parent;
      float t_max;
    };
    
    typedef enum STATE {
      INIT,
      ADVANCE,
      FINISHED
    } STATE;

    const Octree<T>& map_;
    float voxelSize_; 
    float3 origin_;
    float3 direction_;
    float3 t_coef_;
    float3 t_bias_;
    struct stack_entry stack[CAST_STACK_DEPTH];
    Node<T> * parent_;
    Node<T> * child_;
    int idx_;
    float3 pos_;
    int scale;
    int min_scale_;
    float scale_exp2_;
    float t_min_;
    float t_max_;
    float tc_max_;
    float3 t_corner_;
    float h_;
    int octant_mask_;
    int scale_;
    STATE state_;
};

template <typename T>
class leaf_iterator {

  public:

  leaf_iterator(const Octree<T>& m): map_(m){
    state_ = BRANCH_NODES;
    last = 0;
  };

  std::tuple<int3, int, typename Octree<T>::compute_type> next() {
    switch(state_) {
      case BRANCH_NODES:
        if(last < map_.nodes_buffer_.size()) {
          Node<T>* n = map_.nodes_buffer_[last++];
          return std::make_tuple(make_int3(unpack_morton(n->code)), 
                                 n->side, n->value_[0]);
        } else {
          last = 0;
          state_ = LEAF_NODES; 
          return next();
        }
        break;
      case LEAF_NODES:
        if(last < map_.block_memory_.size()) {
          VoxelBlock<T>* n = map_.block_memory_[last++];
          return std::make_tuple(make_int3(unpack_morton(n->code)), 
              int(VoxelBlock<T>::side), n->value_[0]); 
              /* the above int init required due to odr-use of static member */
        } else {
          last = 0;
          state_ = FINISHED; 
          return std::make_tuple(make_int3(-1), -1, Octree<T>::traits_type::empty());
        }
        break;
      case FINISHED:
          break;
    }
  }

  private:
  typedef enum ITER_STATE {
    BRANCH_NODES,
    LEAF_NODES,
    FINISHED
  } ITER_STATE;

  const Octree<T>& map_;
  ITER_STATE state_;
  size_t last;
};
#endif // OCTREE_H
