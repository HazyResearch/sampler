
#include "gibbs.h"
#include "gtest/gtest.h"
#include "io/binary_parser.h"
#include "dstruct/factor_graph/factor_graph.h"
#include "dstruct/factor_graph/factor.h"
#include "io/partition.h"
#include <fstream>

using namespace dd;

TEST(PartitionTest, variables) {
	dd::FactorGraph fg(200, 200, 200, 1);
	Partition partition(3, "./test/partition/graph.partition.variables", 
		"./test/partition/graph.factors");
	partition.partition_variables("./test/partition/graph.variables");
	read_variables("./test/partition/graph.variables.part0", fg);
	read_variables("./test/partition/graph.variables.part1", fg);
	read_variables("./test/partition/graph.variables.part2", fg);
}