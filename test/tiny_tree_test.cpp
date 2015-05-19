/**
 * Integration test for inference on a tiny tree
 * The factor graph contains a chain A-B. The factor between A, B is equal. 
 * A and B are connected to two isTrue factors with weight 1 and -2, respectively.
 */

#include <limits.h>
#include "gibbs.h"
#include "gtest/gtest.h"
#include <fstream>

using namespace dd;

TEST(TinyTreeTest, INFERENCE) {

  const char* argv[23] = {
    "dw", "gibbs", "-w", "./test/tree/graph.weights", "-v", "./test/tree/graph.variables", 
    "-f", "./test/tree/graph.factors", "-e", "./test/tree/graph.edges", "-m", "./test/tree/graph.meta",
    "-o", ".", "-l", "0", "-i", "1000", "-s", "1", "--alpha", "0.1", ""
  };

  dd::CmdParser cmd_parser = parse_input(23, (char **)argv);
  gibbs(cmd_parser);

  int id;
  int e;
  double prob;
  std::ifstream fin;
  fin.open("./inference_result.out.text");
  while (fin >> id >> e >> prob){
    EXPECT_NEAR(prob, 0.15, 0.03);
  }
  fin.close();

  const char* argv2[23] = {
    "dw", "bp", "-w", "./test/tree/graph.weights", "-v", "./test/tree/graph.variables", 
    "-f", "./test/tree/graph.factors", "-e", "./test/tree/graph.edges", "-m", "./test/tree/graph.meta",
    "-o", ".", "-l", "0", "-i", "1000", "-s", "1", "--alpha", "0.1", ""
  };

  cmd_parser = parse_input(23, (char **)argv2);
  bp(cmd_parser);

  fin.open("./inference_result.out.text");
  while (fin >> id >> e >> prob){
    EXPECT_NEAR(prob, 0.15, 0.03);
  }
  fin.close();

}

