
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
	std::cout << "haha" << std::endl;

}

// TEST(PartitionTest, integration) {

// 	const char* argv[23] = {
// 		"dw", "gibbs", "-w", "./test/coin/graph.weights", "-v", "./test/coin/graph.variables", 
// 		"-f", "./test/coin/graph.factors", "-e", "./test/coin/graph.edges", "-m", "./test/coin/graph.meta",
// 		"-o", ".", "-l", "300", "-i", "300", "-s", "1", "--alpha", "0.1", "--partition",
// 		"", "", "", ""
// 	};

// 	dd::CmdParser cmd_parser = parse_input(23, (char **)argv);
// 	gibbs(cmd_parser);

// 	std::ifstream fin("./inference_result.out.text");
// 	int nvar = 0;
// 	int id, e;
// 	double prob;
// 	while(fin >> id >> e >> prob){
// 		EXPECT_NEAR(prob, 0.89, 0.1);
// 		nvar ++;
// 	}
// 	EXPECT_EQ(nvar, 9);

// 	std::ifstream fin_weight("./inference_result.out.weights.text");
// 	int nweight = 0;
// 	double weight;
// 	while(fin_weight >> id >> weight){
// 		EXPECT_NEAR(weight, 2.0, 0.3);
// 		nweight ++;
// 	}
// 	EXPECT_EQ(nweight, 1);

// }

