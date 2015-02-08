
#include "gibbs.h"
#include "io/partition.h"
#include "dstruct/factor_graph/inference_result.h"
#include <fstream>
#include <thread>

/*
 * Parse input arguments
 */
dd::CmdParser parse_input(int argc, char** argv){

  std::vector<std::string> new_args;
  if (argc < 2 || strcmp(argv[1], "gibbs") != 0) {
    new_args.push_back(std::string(argv[0]) + " " + "gibbs");
    new_args.push_back("-h");
  } else {
    new_args.push_back(std::string(argv[0]) + " " + std::string(argv[1]));
    for(int i=2;i<argc;i++){
      new_args.push_back(std::string(argv[i]));
    }
  }
  char ** new_argv = new char*[new_args.size()];
  for(size_t i=0;i<new_args.size();i++){
    new_argv[i] = new char[new_args[i].length() + 1];
    std::copy(new_args[i].begin(), new_args[i].end(), new_argv[i]);
    new_argv[i][new_args[i].length()] = '\0';
  }
  dd::CmdParser cmd_parser("gibbs");
  cmd_parser.parse(new_args.size(), new_argv);
  return cmd_parser;
}

void swap_ptr(dd::FactorGraph *&a, dd::FactorGraph *&b) {
  dd::FactorGraph *tmp = a;
  a = b;
  b = tmp;
}

void gibbs(dd::CmdParser & cmd_parser){

  // number of NUMA nodes
  int n_numa_node = numa_max_node() + 1;
  // number of max threads per NUMA node
  int n_thread_per_numa = (sysconf(_SC_NPROCESSORS_CONF))/(n_numa_node);

  // get command line arguments
  std::string fg_file = cmd_parser.fg_file->getValue();

  std::string weight_file = cmd_parser.weight_file->getValue();
  std::string variable_file = cmd_parser.variable_file->getValue();
  std::string factor_file = cmd_parser.factor_file->getValue();
  std::string edge_file = cmd_parser.edge_file->getValue();

  std::string output_folder = cmd_parser.output_folder->getValue();

  int n_learning_epoch = cmd_parser.n_learning_epoch->getValue();
  int n_samples_per_learning_epoch = cmd_parser.n_samples_per_learning_epoch->getValue();
  int n_inference_epoch = cmd_parser.n_inference_epoch->getValue();

  double stepsize = cmd_parser.stepsize->getValue();
  double stepsize2 = cmd_parser.stepsize2->getValue();
  // hack to support two parameters to specify step size
  if (stepsize == 0.01) stepsize = stepsize2;
  double decay = cmd_parser.decay->getValue();

  int n_datacopy = cmd_parser.n_datacopy->getValue();
  double reg_param = cmd_parser.reg_param->getValue();
  bool is_quiet = cmd_parser.quiet->getValue();
  int num_partitions = cmd_parser.num_partitions->getValue();

  Meta meta = read_meta(fg_file); 

  if (is_quiet) {
    std::cout << "Running in quiet mode..." << std::endl;
  } else {
    std::cout << std::endl;
    std::cout << "#################MACHINE CONFIG#################" << std::endl;
    std::cout << "# # NUMA Node        : " << n_numa_node << std::endl;
    std::cout << "# # Thread/NUMA Node : " << n_thread_per_numa << std::endl;
    std::cout << "################################################" << std::endl;
    std::cout << std::endl;
    std::cout << "#################GIBBS SAMPLING#################" << std::endl;
    std::cout << "# fg_file            : " << fg_file << std::endl;
    std::cout << "# edge_file          : " << edge_file << std::endl;
    std::cout << "# weight_file        : " << weight_file << std::endl;
    std::cout << "# variable_file      : " << variable_file << std::endl;
    std::cout << "# factor_file        : " << factor_file << std::endl;
    std::cout << "# output_folder      : " << output_folder << std::endl;
    std::cout << "# n_learning_epoch   : " << n_learning_epoch << std::endl;
    std::cout << "# n_samples/l. epoch : " << n_samples_per_learning_epoch << std::endl;
    std::cout << "# n_inference_epoch  : " << n_inference_epoch << std::endl;
    std::cout << "# stepsize           : " << stepsize << std::endl;
    std::cout << "# decay              : " << decay << std::endl;
    std::cout << "# regularization     : " << reg_param << std::endl;
    std::cout << "# num_partitions     : " << num_partitions << std::endl;
    std::cout << "################################################" << std::endl;
    std::cout << "# IGNORE -s (n_samples/l. epoch). ALWAYS -s 1. #" << std::endl;
    std::cout << "# IGNORE -t (threads). ALWAYS USE ALL THREADS. #" << std::endl;
    std::cout << "################################################" << std::endl;


    std::cout << "# nvar               : " << meta.num_variables << std::endl;
    std::cout << "# nfac               : " << meta.num_factors << std::endl;
    std::cout << "# nweight            : " << meta.num_weights << std::endl;
    std::cout << "# nedge              : " << meta.num_edges << std::endl;
    std::cout << "################################################" << std::endl;
  }

  // run on NUMA node 0
  numa_run_on_node(0);
  numa_set_localalloc();

  // number of learning epochs
  // the factor graph is copied on each NUMA node, so the total epochs =
  // epochs specified / number of NUMA nodes
  int numa_aware_n_learning_epoch = (int)(n_learning_epoch/n_numa_node) + 
    (n_learning_epoch%n_numa_node==0?0:1);
  // number of inference epochs
  int numa_aware_n_epoch = (int)(n_inference_epoch/n_numa_node) + 
    (n_inference_epoch%n_numa_node==0?0:1);

  if (num_partitions == 0) {
    dd::FactorGraph fg(meta.num_variables, meta.num_factors, meta.num_weights, meta.num_edges);
    fg.load(cmd_parser, is_quiet);
    dd::GibbsSampling gibbs(&fg, &cmd_parser, n_datacopy);

    // learning
    gibbs.learn(numa_aware_n_learning_epoch, n_samples_per_learning_epoch, 
      stepsize, decay, reg_param, is_quiet);

    // dump weights
    gibbs.dump_weights(is_quiet);

    // inference
    gibbs.inference(numa_aware_n_epoch, is_quiet);
    gibbs.aggregate_results_and_dump(is_quiet);

  } else {

    // partition
    n_datacopy = 1;
    std::string partition_variableids_file = cmd_parser.partition_variableids_file->getValue();
    std::string partition_factorids_file = cmd_parser.partition_factorids_file->getValue();
    Partition partition(num_partitions, meta.num_weights, partition_variableids_file, partition_factorids_file);
    std::cerr << "Partitioning factor graph..." << std::endl;
    partition.partition_factor_graph(variable_file, factor_file, edge_file);

    // id offsets
    // variable and factor ids are mapped to a continous range starting from 0
    // each partition has an id offset
    std::vector<long> vid_offsets, tally_offsets;
    long vid_offset = 0;
    for (int i = 0; i < num_partitions; i++) {
      vid_offsets.push_back(vid_offset);
      vid_offset += partition.metas[i].num_variables;
    }
    
    // inference result
    dd::InferenceResult infrs(meta.num_variables, meta.num_weights);

    meta = partition.metas[0];
    dd::FactorGraph fg(meta.num_variables, meta.num_factors, meta.num_weights, meta.num_edges);
    dd::FactorGraph fg_buffer(meta.num_variables, meta.num_factors, meta.num_weights, meta.num_edges);
    // first pass over data, read variables and weights to init infrs
    // load weights
    fg.load_weights(cmd_parser, is_quiet);
    infrs.init_weights(fg.weights);
    fg_buffer.load_weights(cmd_parser, is_quiet);

    // fg working and fg loading 
    dd::FactorGraph *fg_work_ptr = &fg;
    dd::FactorGraph *fg_load_ptr = &fg_buffer;

    // load variables, init variable assignments in infrs
    // variable ids are mapped to a continous range starting from 0
    // variable assignments in each partition are continous in infrs 
    std::cerr << "Loading variables..." << std::endl;
    int tally_offset = 0;
    for (int i = 0; i < num_partitions; i++) {
      long ntallies = fg.reload_variables(partition.metas[i].num_variables, cmd_parser,
        partition.numbers[i], partition.vid_maps[i], partition.vid_reverse_maps[i]);
      tally_offsets.push_back(tally_offset);
      tally_offset += ntallies;
      infrs.init_variables(fg.variables, partition.metas[i].num_variables, vid_offsets[i]);
    }
    infrs.init_tallies(tally_offset);

    std::thread work_thread, load_thread;

    // load the first partition
    meta = partition.metas[0];
    fg_work_ptr->reload(meta.num_variables, meta.num_factors, meta.num_edges, cmd_parser,
      partition.numbers[0], &infrs, vid_offsets[0], tally_offsets[0],
      partition.vid_maps[0], partition.fid_maps[0], partition.vid_reverse_maps[0]);

    // for each partition, load factor graph and sample it
    for (int i = 0; i < numa_aware_n_learning_epoch; i++) {
      for (int j = 0; j < num_partitions; j++) {
        int k = (j + 1) % num_partitions;
        meta = partition.metas[k];
        // reload factor graph
        // variable and factor ids are mapped to a continous range starting from 0
        load_thread = std::thread(&dd::FactorGraph::reload, fg_load_ptr, meta.num_variables, meta.num_factors, meta.num_edges, cmd_parser,
          partition.numbers[k], &infrs, vid_offsets[k], tally_offsets[k],
          partition.vid_maps[k], partition.fid_maps[k], partition.vid_reverse_maps[k]);
        dd::GibbsSampling gibbs(fg_work_ptr, &cmd_parser, n_datacopy);
        // learn
        work_thread = std::thread(&dd::GibbsSampling::learn, &gibbs, 1, 
          n_samples_per_learning_epoch, stepsize, decay, reg_param, is_quiet, i);
        work_thread.join();
        load_thread.join();
        swap_ptr(fg_work_ptr, fg_load_ptr);
      }
      stepsize *= decay;
    }
    dd::GibbsSampling gibbs(&fg, &cmd_parser, n_datacopy);
    gibbs.dump_weights(is_quiet);
    meta = partition.metas[0];
    fg_work_ptr->reload(meta.num_variables, meta.num_factors, meta.num_edges, cmd_parser,
      partition.numbers[0], &infrs, vid_offsets[0], tally_offsets[0],
      partition.vid_maps[0], partition.fid_maps[0], partition.vid_reverse_maps[0]);

    // for each partition, load factor graph and sample it
    for (int i = 0; i < numa_aware_n_epoch; i++) {
      for (int j = 0; j < num_partitions; j++) {
        int k = (j + 1) % num_partitions;
        meta = partition.metas[k];
        // reload factor graph
        // variable and factor ids are mapped to a continous range starting from 0
        load_thread = std::thread(&dd::FactorGraph::reload, fg_load_ptr, meta.num_variables, meta.num_factors, meta.num_edges, cmd_parser,
          partition.numbers[k], &infrs, vid_offsets[k], tally_offsets[k],
          partition.vid_maps[k], partition.fid_maps[k], partition.vid_reverse_maps[k]);
        // reload factor graph
        dd::GibbsSampling gibbs(fg_work_ptr, &cmd_parser, n_datacopy);
        // inference
        work_thread = std::thread(&dd::GibbsSampling::inference, &gibbs, 1, is_quiet, i);
        load_thread.join();
        work_thread.join();
      }
    }
    // dump results
    std::string filename_text = cmd_parser.output_folder->getValue() + "/inference_result.out.text";
    std::ofstream fout_text(filename_text.c_str());
    fout_text.close();

    // load each partition and dump
    for (int j = 0; j < num_partitions; j++) {
      meta = partition.metas[j];
      // reload factor graph
      fg.reload(meta.num_variables, meta.num_factors, meta.num_edges, cmd_parser,
        partition.numbers[j], &infrs, vid_offsets[j], tally_offsets[j],
        partition.vid_maps[j], partition.fid_maps[j], partition.vid_reverse_maps[j]);
      dd::GibbsSampling gibbs(&fg, &cmd_parser, n_datacopy);
      gibbs.aggregate_results_and_dump(is_quiet, partition.vid_reverse_maps[j]);
    }

  } // end if partition

}