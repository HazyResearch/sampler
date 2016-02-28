/**
 * Integration test for binary biased coin
 *
 * Author: Ce Zhang
 */

#include <limits.h>
#include "gibbs.h"
#include "gtest/gtest.h"
#include <fstream>

using namespace dd;

// the factor graph used for test is from biased coin, which contains 18
// variables,
// 1 weight, 18 factors, and 18 edges. Variables of id 0-8 are evidence: id 0-7
// positive and id 8 negative.
TEST(PartitionTest, PartitionTest) {
  const char *argv[24] = {"dw",
                          "gibbs",
                          "-w",
                          "./test/coin/graph.weights",
                          "-v",
                          "./test/coin/graph.variables",
                          "-f",
                          "./test/coin/graph.factors",
                          "-m",
                          "./test/coin/graph.meta",
                          "-o",
                          ".",
                          "-l",
                          "10",
                          "-i",
                          "0",
                          "-s",
                          "1",
                          "--alpha",
                          "0.1",
                          "--diminish 1",
                          "--reg_param 0",
                          "--assignments ./assignments.txt",
                          "--weights_binary ./weights_binary.txt"};
  dd::CmdParser cmd_parser = parse_input(24, (char **)argv);
  for (int i = 0; i < 100; i++) {
    gibbs(cmd_parser);
  }
  std::ifstream fin_weight("./weights_binary.txt", ios::in | ios::binary);
  long wid;
  double weight;
  fin_weight.read((char *)&wid, sizeof(long));
  fin_weight.read((char *)&weight, sizeof(double));
  EXPECT_NEAR(weight, 2.1, 0.3);
  fin_weight.close();
}
