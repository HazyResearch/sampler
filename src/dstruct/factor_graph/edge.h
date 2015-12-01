#ifndef _EDGE_H_
#define _EDGE_H_

namespace dd{

  /**
   * Encapsulates a edge for factor graph. 
   */
  class Edge {
  public:
    long long variable_id;
    long long factor_id;
    long long position;
    bool ispositive;
    long long equal_predicate;

    Edge(const long long & variable_id,
      const long long & factor_id,
      const long long & position,
      const bool & ispositive,
      const long long & equal_predicate);

    Edge();

  };
}

#endif  