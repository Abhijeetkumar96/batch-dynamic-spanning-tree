#ifndef COMMAND_LINE_PARSER_H
#define COMMAND_LINE_PARSER_H

#include <iostream>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h> // Ensure this is included if you're calling cudaGetDeviceCount

class CommandLineParser {
public:
    enum Rep_Algorithm {
        SUPER_GRAPH,
        HOOKING_SHORTCUTTING,
        UNKNOWN_REP_ALGORITHM // Specific unknown enum for replacement algorithms
    };

    enum PR_Algorithm {
        EULERIAN_TOUR,
        PATH_REVERSAL,
        UNKNOWN_PR_ALGORITHM // Specific unknown enum for path reversal algorithms
    };

    struct CommandLineArgs {
        std::string inputFile;
        std::string batchInputFile;
        Rep_Algorithm rep_algorithm = UNKNOWN_REP_ALGORITHM;
        PR_Algorithm pr_algorithm   = UNKNOWN_PR_ALGORITHM;
        int cudaDevice = 0;
        bool verbose = false;
        bool checkerMode = false;
        bool error = false;
        bool print_stat = false;
        bool testgen = false;
    };

    CommandLineParser(int argc, char* argv[]) {
        parseArguments(argc, argv);
    }

    const CommandLineArgs& getArgs() const {
        return args;
    }

    static bool isValidCudaDevice(int device) {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        return device >= 0 && device < deviceCount;
    }

    static const std::string help_msg;

private:
    CommandLineArgs args;

    void parseArguments(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help") {
                std::cout << help_msg << std::endl;
                exit(0);
            } else if (arg == "-i" && i + 1 < argc) {
                args.inputFile = argv[++i];
            } else if (arg == "-b" && i + 1 < argc) {
                args.batchInputFile = argv[++i];
            } else if (arg == "-r" && i + 1 < argc) {
                std::string rep_alg = argv[++i];
                if (rep_alg == "SG") {
                    args.rep_algorithm = SUPER_GRAPH;
                } else if (rep_alg == "HS") {
                    args.rep_algorithm = HOOKING_SHORTCUTTING;
                } else {
                    std::cerr << "Unrecognized replacement edge algorithm\n";
                    args.error = true;
                }
            } else if (arg == "-p" && i + 1 < argc) {
                std::string pr_alg = argv[++i];
                if (pr_alg == "ET") {
                    args.pr_algorithm = EULERIAN_TOUR;
                } else if (pr_alg == "PR") {
                    args.pr_algorithm = PATH_REVERSAL;
                } else {
                    std::cerr << "Unrecognized path reversal algorithm\n";
                    args.error = true;
                }
            } else if (arg == "-d" && i + 1 < argc) {
                int device = std::atoi(argv[++i]);
                if (!isValidCudaDevice(device)) {
                    std::cerr << "Error: Invalid CUDA device number." << std::endl;
                    args.error = true;
                } else {
                    args.cudaDevice = device;
                }
            } else if (arg == "-s") {
                args.print_stat = true;

            } else if (arg == "-v") {
                args.verbose = true;
            } else if (arg == "-c") {
                args.checkerMode = true;
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                args.error = true;
            }
        }
        if (args.inputFile.empty()) {
            std::cerr << "Error: Please provide an input file." << std::endl;
            args.error = true;
        }

        if (args.batchInputFile.empty()) {
            args.testgen = true;
        }
    }
};

const std::string CommandLineParser::help_msg =
    "Command line arguments:\n"
    "  -h, --help           Print this help message and exit.\n"
    "  -i $input            Set the input file path to $input. Supported extensions: txt & binary.\n"
    "  -b $batch_input      Set the delete/insert batch file path to $batch_input.\n"
    "  -r $algorithm        Select the algorithm for obtaining replacement edges:\n"
    "       SG              Use the 'SuperGraph' algorithm to compute replacement edges.\n"
    "       HS              Use the 'Hooking and Shortcutting' method for replacement edges.\n"
    "  -p $path_reversal    Choose the path reversal algorithm to be applied:\n"
    "       ET              Employ the 'Eulerian Tour' method for path reversal.\n"
    "       PR              Apply the 'Path Reversal' method for path reversal.\n"
    "  -d $device           Set the CUDA device to $device (default: 0).\n"
    "  -s                   Print stats of the input graph.\n"
    "  -v                   Enable verbose output.\n"
    "  -c                   Enable checker mode for verification.\n"
    "\n"
    "Usage Examples:\n"
    "  Run with specific input and batch files, using SuperGraph for edge replacement and Eulerian Tour for path reversal:\n"
    "    ./dynamic_spanning_tree -i path/to/input.txt -b path/to/batch.txt -r SG -p ET\n"
    "  Generate statistics for a given graph:\n"
    "    ./dynamic_spanning_tree -i path/to/graph.txt -s\n"
    "  Run on a specific CUDA device with verbose output, using Hooking and Shortcutting and Path Reversal:\n"
    "    ./dynamic_spanning_tree -i path/to/input.bin -d 1 -v -r HS -p PR\n"
    "\n";

#endif // COMMAND_LINE_PARSER_H
