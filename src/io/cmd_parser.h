
#include <string>
#include <iostream>
#include <algorithm>
#include <assert.h>
#include <tclap/CmdLine.h>

#ifndef _CMD_PARSER_H_
#define _CMD_PARSER_H_

namespace dd{

    /**
     * Command line argument parser
     */
  class CmdParser{
  public:

    // all the arguments are defined in cmd_parser.cpp
    std::string app_name;

    TCLAP::ValueArg<std::string> * fg_file;

    TCLAP::ValueArg<std::string> * edge_file;
    TCLAP::ValueArg<std::string> * weight_file;
    TCLAP::ValueArg<std::string> * variable_file;
    TCLAP::ValueArg<std::string> * factor_file;
    TCLAP::ValueArg<std::string> * output_folder;

    TCLAP::ValueArg<int> * n_learning_epoch;
    TCLAP::ValueArg<int> * n_samples_per_learning_epoch;
    TCLAP::ValueArg<int> * n_inference_epoch;

    TCLAP::ValueArg<int> * n_thread;

    TCLAP::ValueArg<double> * stepsize;
    TCLAP::ValueArg<double> * stepsize2;
    TCLAP::ValueArg<double> * decay;

    TCLAP::ValueArg<int> * n_datacopy;
    TCLAP::ValueArg<double> * reg_param;
    TCLAP::SwitchArg * quiet; 
    TCLAP::ValueArg<int> * num_partitions;
    TCLAP::ValueArg<std::string> * partition_variableids_file;
    TCLAP::ValueArg<std::string> * partition_factorids_file;

    TCLAP::CmdLine * cmd;

    CmdParser(std::string _app_name);

    /**
     * parses the given command line arguments
     */
    void parse(int argc, char** argv);

  };

}


#endif