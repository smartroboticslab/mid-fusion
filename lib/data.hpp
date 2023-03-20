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

#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <vector>
#include <atomic>
#include <mutex>
#include <math_utils.h>
#include <voxel_traits.hpp>

class StaticStorage;
class DynamicStorage;

template <typename FieldType, typename StorageType>
class Data {};

template <typename FieldType>
class Data<FieldType, StaticStorage> {

  public: 
    typedef voxel_traits<FieldType> traits_type;
    typedef typename traits_type::StoredType stored_type;

    stored_type& operator()(const int i) {
      return _data[i];
    };  

    const stored_type& operator()(const int i) const {
      return _data[i];
    };  

    void init(std::size_t size) {
 		  _data = (stored_type *) malloc(size * sizeof(stored_type));
    };
    void release() { 
      free(_data); 
    };

    stored_type * raw_data() { return _data; };

  private:
  stored_type * _data;

};

template <typename FieldType>
class Data<FieldType, DynamicStorage> {

  public:
    typedef voxel_traits<FieldType> traits_type;
    typedef typename traits_type::StoredType stored_type;
    stored_type& operator()(const int pos, const int i);  
    const stored_type& operator()(const int pos, const int i) const;  

  private:
  stored_type * _data;
};
#endif
