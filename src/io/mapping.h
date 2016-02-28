#ifndef MAPPING_H
#define MAPPING_H

#include <unordered_map>

class Mapping {
 public:
  std::unordered_map<long long, long long> wid_map;
  std::unordered_map<long long, long long> wid_reverse_map;
  std::unordered_map<long long, long long> vid_map;
  std::unordered_map<long long, long long> vid_reverse_map;

  Mapping(){

  };
};

#endif