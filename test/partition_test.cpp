
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
	Partition partition(3, 200, "./test/partition/graph.partition.variables", 
		"./test/partition/graph.factors");

	partition.partition_variables("./test/partition/graph.variables");
	read_variables("./test/partition/graph.variables.part0", fg);
	read_variables("./test/partition/graph.variables.part1", fg);
	read_variables("./test/partition/graph.variables.part2", fg);

	partition.partition_factors("./test/partition/graph.factors");
	read_factors("./test/partition/graph.factors.part0", fg);
	read_factors("./test/partition/graph.factors.part1", fg);
	read_factors("./test/partition/graph.factors.part2", fg);

	partition.partition_edges("./test/partition/graph.edges");
	read_edges("./test/partition/graph.edges.part0", fg);
	read_edges("./test/partition/graph.edges.part1", fg);
	read_edges("./test/partition/graph.edges.part2", fg);

	for (int i = 0; i < 3; i++) {
		std::cout << partition.metas[i].num_variables << " " << partition.metas[i].num_factors << " "
			<< partition.metas[i].num_edges << std::endl;
	}
}