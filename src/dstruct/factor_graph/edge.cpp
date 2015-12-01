#include "dstruct/factor_graph/edge.h"

dd::Edge::Edge(const long long & _variable_id,
      const long long & _factor_id,
      const long long & _position,
      const bool & _ispositive,
      const long long & _equal_predicate):
  variable_id(_variable_id),
  factor_id(_factor_id),
  position(_position),
  ispositive(_ispositive),
  equal_predicate(_equal_predicate){}

dd::Edge::Edge(){}