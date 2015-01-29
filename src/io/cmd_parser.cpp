
#include "io/cmd_parser.h"

namespace dd{

    CmdParser::CmdParser(std::string _app_name){

      app_name = _app_name;

      if(app_name == "gibbs"){
        cmd = new TCLAP::CmdLine("DimmWitted GIBBS", ' ', "0.01");

        fg_file = new TCLAP::ValueArg<std::string>("m","fg_meta","factor graph metadata file",true,"","string"); 
        edge_file = new TCLAP::ValueArg<std::string>("e","edges","edges file",true,"","string"); 
        weight_file = new TCLAP::ValueArg<std::string>("w","weights","weights file",true,"","string"); 
        variable_file = new TCLAP::ValueArg<std::string>("v","variables","variables file",true,"","string"); 
        factor_file = new TCLAP::ValueArg<std::string>("f","factors","factors file",true,"","string"); 
        output_folder = new TCLAP::ValueArg<std::string>("o","outputFile","Output Folder",true,"","string");
        
        n_learning_epoch = new TCLAP::ValueArg<int>("l","n_learning_epoch","Number of Learning Epochs",true,-1,"int");
        n_samples_per_learning_epoch = new TCLAP::ValueArg<int>("s","n_samples_per_learning_epoch","Number of Samples per Leraning Epoch",true,-1,"int");
        n_inference_epoch = new TCLAP::ValueArg<int>("i","n_inference_epoch","Number of Samples for Inference",true,-1,"int");

        stepsize = new TCLAP::ValueArg<double>("a","alpha","Stepsize",false,0.01,"double");
        stepsize2 = new TCLAP::ValueArg<double>("p","stepsize","Stepsize",false,0.01,"double");
        decay = new TCLAP::ValueArg<double>("d","diminish","Decay of stepsize per epoch",false,0.95,"double");

        n_thread = new TCLAP::ValueArg<int>("t","threads","This setting is no longer supported and will be ignored.",false,-1,"int");
        n_datacopy = new TCLAP::ValueArg<int>("c","n_datacopy","Number of factor graph copies",false,0,"int");
        reg_param = new TCLAP::ValueArg<double>("b","reg_param","l2 regularization parameter",false,0.01,"double");
        quiet = new TCLAP::SwitchArg("q", "quiet", "quiet output", false);
        partition = new TCLAP::SwitchArg("", "partition", "partition mode", false);

        cmd->add(*fg_file);
        
        cmd->add(*edge_file);
        cmd->add(*weight_file);
        cmd->add(*variable_file);
        cmd->add(*factor_file);
        cmd->add(*output_folder);

        cmd->add(*n_learning_epoch);
        cmd->add(*n_samples_per_learning_epoch);
        cmd->add(*n_inference_epoch);

        cmd->add(*stepsize);
        cmd->add(*stepsize2);
        cmd->add(*decay);
        cmd->add(*n_thread);

        cmd->add(*n_datacopy);
        cmd->add(*reg_param);
        cmd->add(*quiet);
        cmd->add(*partition);
      }else{
        std::cout << "ERROR: UNKNOWN APP NAME " << app_name << std::endl;
        std::cout << "AVAILABLE APP {gibbs}" << app_name << std::endl;
        assert(false);
      }
    }

    void CmdParser::parse(int argc, char** argv){
      cmd->parse(argc, argv);
    }
}
