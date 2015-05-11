#include <unistd.h>
#include <fstream>
#include <memory>
#include "app/bp/belief_propagation.h"
#include "app/bp/single_thread_bp.h"
#include "common.h"
#include "timer.h"

void single_thread_bp_task(FactorGraph * const fg, int i_worker, int n_worker, bool vtof) {
  SingleThreadBP worker = SingleThreadBP(fg);
  worker.send_message_block(i_worker, n_worker, vtof);
}

BeliefPropagation::BeliefPropagation(FactorGraph * const fg, 
  CmdParser * const cmd_parser, int n_threads) 
  : fg(fg), cmd_parser(cmd_parser), n_threads(n_threads) {
}

void BeliefPropagation::init_messages() {
  for (long i = 0; i < fg->n_var; i++) {
    fg->variables[i].init_messages();
  }
  for (long i = 0; i < fg->n_factor; i++) {
    fg->factors[i].init_messages();
  }
}

void BeliefPropagation::inference(int n_epoch) {
  Timer t_total;
  Timer t;

  // initialize
  init_messages();

  // inference epochs
  for (int i_epoch = 0; i_epoch < n_epoch; i_epoch++) {

    if (!is_quiet) {
      std::cout << std::setprecision(2) << "INFERENCE EPOCH " << i_epoch <<  "~" 
        << (i_epoch + 1) << "...." << std::flush;
    }

    // restart timer
    t.restart();

    // send messages
    // v to f
    threads.clear();
    for (int i = 0; i < n_threads; i++) {
      threads.push_back(std::thread(single_thread_bp_task, fg, i, n_threads, true));
    }

    for (int i = 0; i < n_threads; i++) {
      threads[i].join();
    }
    // single_thread_bp_task(fg, 0, n_threads, true);
    // printf("***********\n");
    // single_thread_bp_task(fg, 0, n_threads, false);
    // printf("~~~~~~~~~~~\n");

    // // f to v
    threads.clear();
    for (int i = 0; i < n_threads; i++) {
      threads.push_back(std::thread(single_thread_bp_task, fg, i, n_threads, false));
    }

    for (int i = 0; i < n_threads; i++) {
      threads[i].join();
    }

    double elapsed = t.elapsed();
    if (!is_quiet) {
      std::cout << elapsed << " sec." ;
      std::cout << ","  << fg->n_var / elapsed << " vars/sec" << std::endl;
    }
  }

  double elapsed = t_total.elapsed();
  std::cout << "TOTAL INFERENCE TIME: " << elapsed << " sec." << std::endl;

  // compute marginals for each variable
  marginals = std::vector<double>(fg->n_var, 0);
  double marginal0;
  double marginal1;
  for (long i = 0; i < fg->n_var; i++) {
    Variable &variable = fg->variables[i];
    marginal0 = 0;
    marginal1 = 0;
    for (int j = 0; j < variable.n_factors; j++) {
      // messages in log domain
      marginal0 += variable.messages0[j]; 
      marginal1 += variable.messages1[j];
      // if (DEBUG && i == 58) printf("messages0 = %f, messages1 = %f\n", 
      //   variable.messages0[j], variable.messages1[j]);
    }
    // if (DEBUG && i == 58) printf("marginal0 = %f, marginal1 = %f\n", marginal0, marginal1);
    marginals[i] = exp(marginal1) / (exp(marginal0) + exp(marginal1));
    // if (DEBUG && i == 58) printf("vid = %ld, marginal = %f\n", i, marginals[i]);
  }
}

void BeliefPropagation::dump_inference_result() {
  std::string filename = cmd_parser->output_folder->getValue() + 
    "/inference_result.out.text";
  std::cout << "DUMPING... INFERENCE RESULT    : " << filename << std::endl;
  std::ofstream fout(filename.c_str());
  for (long i = 0; i < fg->n_var; i++) {
    if (fg->variables[i].is_evid) continue;
    fout << i << " " << 1 << " " << marginals[i] << std::endl;
  }
  fout.close();
}
