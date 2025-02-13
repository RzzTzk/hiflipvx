/**
 *  Copyright <2024> <Lester Kalms>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
 * “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
 * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Additional restriction: The Software and its derivatives may not be used for, or in support of, any military purposes.
 *
 * @file    hvx_vitis.h
 * @author  Lester Kalms <lester.kalms@tu-dresden.de>
 * @version 4.0
 * @brief Description:\n
 *
 */

#ifndef HVX_VITIS_H_
#define HVX_VITIS_H_

#include <array>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

/******************************************************************************************************************************************/
class VitisParam {
private:

    // configuration
    bool do_c_sim;
    bool do_c_synth;
    bool do_rtl_synth;
    std::string part;
    std::string period;
    std::string vitis_path;
    std::string hvx_path;

    //
    const std::string out_path           = "out/";
    const std::string tcl_file           = "tests/hvx_synth.tcl";
    const std::string csim_prefix        = "csim_";
    const std::string csim_res_file_name = "csim.txt";
    std::string csyn_prefix              = "csyn_";
    std::string syn_prefix               = "syn_";

    // RTL synthesis
    std::string syn_res_header_line       = "name;LUT;FF;DSP;BRAM;URAM;SRL;period\n";
    std::vector<std::string> syn_keywords = {"<LUT>", "<FF>", "<DSP>", "<BRAM>", "<URAM>", "<SRL>", "<AchievedClockPeriod>"};

    // C synthesis
    std::string csyn_res_header_line       = "name;LUT;FF;DSP;BRAM;URAM;latency;period\n";
    std::vector<std::string> csyn_keywords = {
        "<LUT>", "<FF>", "<DSP>", "<BRAM_18K>", "<URAM>", "<Average-caseLatency>", "<EstimatedClockPeriod>"};

    // semaphore
    std::mutex mutex;
    std::condition_variable cv;
    int32_t count = 0;

    void SemaphoreNotify() {
        std::unique_lock<std::mutex> lock(mutex);
        ++count;
        cv.notify_one();
    }

    void SemaphoreWait() {
        std::unique_lock<std::mutex> lock(mutex);
        while (count == 0) {
            cv.wait(lock);
        }
        --count;
    }

    auto CopyFile(const std::string& sourcePath, const std::string& destinationPath) noexcept -> void {
        std::ifstream source(sourcePath, std::ios::binary);
        std::ofstream destination(destinationPath, std::ios::binary);

        if (source && destination) {
            destination << source.rdbuf();
            std::cout << "File copied successfully.\n";
        } else {
            std::cerr << "Error: Unable to open the files for copying.\n";
        }
    }

    /*!
     * @brief
     */
    template<int64_t node_num_>
    auto CsimResults(std::array<std::array<const char*, 3>, node_num_>& names) noexcept -> void {
        std::string project_path = hvx_path + out_path;

        // Open the output file
        std::ofstream dst_file(project_path + csim_res_file_name);
        if (!dst_file.is_open()) {
            std::cerr << "Error opening the output file:" << (project_path + csim_res_file_name) << ".\n";
            return;
        }

        // Process each node file
        for (int64_t i = 0; i < node_num_; ++i) {
            const auto file_name = project_path + csim_prefix + names.at(i).at(0) + ".txt";

            // Open the input file for the current node
            std::ifstream src_file(file_name);
            if (!src_file.is_open()) {
                std::cerr << "Error opening input file: " << file_name << std::endl;
                continue;
            }

            // Append all files from the src to the dst and close src
            dst_file << src_file.rdbuf() << "\n";
            src_file.close();

            // Remove the input file (delete it)
            if (std::remove(file_name.c_str()) != 0)
                std::cerr << "Could not remove file: " << file_name << std::endl;
        }

        // close dst
        dst_file.close();
    }

    /*!
     * @brief
     */
    template<int64_t node_num_>
    auto SynResults(std::array<std::array<const char*, 3>, node_num_>& names,
                    std::string& prefix,
                    std::string& res_file_name,
                    std::string& res_header_line,
                    std::vector<std::string>& keywords) noexcept -> void {
        // Construct the full project path
        std::string project_path = hvx_path + out_path;

        // Open the output file
        std::ofstream dst_file(project_path + res_file_name);
        if (!dst_file.is_open()) {
            std::cerr << "Error opening the output file: " << (project_path + res_file_name) << ".\n";
            return;
        }

        // Write the header line to the output file
        dst_file << res_header_line;

        // Process each node file
        for (int64_t i = 0; i < node_num_; ++i) {
            const auto file_name = project_path + prefix + names[i][0] + ".xml";

            // Open the input file for the current node
            std::ifstream src_file(file_name);
            if (!src_file.is_open()) {
                std::cerr << "Error opening input file: " << file_name << std::endl;
                continue;
            }

            // Write node name to the output file
            dst_file << names[i][0] << ";";

            // Process each keyword for the current node
            for (const auto& keyword: keywords) {
                // Go back to the beginning of the file
                src_file.seekg(0);

                std::string line;
                while (std::getline(src_file, line)) {
                    size_t found = line.find(keyword);
                    if (found != std::string::npos) {
                        // If keyword is found, extract the number following it
                        std::istringstream iss(line.substr(found + keyword.size()));
                        float number;
                        if (iss >> number) {
                            dst_file << " " << number << " ;";
                            break; // Exit the loop once the number is extracted
                        } else {
                            std::cerr << "Error: Unable to parse number for " << keyword << std::endl;
                        }
                    }
                }
            }

            // Write newline character to separate node entries
            dst_file << "\n";

            // Close the input file
            src_file.close();

            // Remove the input file
            if (std::remove(file_name.c_str()) != 0) {
                std::cerr << "Could not remove file: " << file_name << std::endl;
            }
        }

        // Close the output file
        dst_file.close();
    }

    /*!
     * @brief Synthesizes one node and stores its results in "result_path"
     */
    auto VitisRun(const char* solution, const char* file, const char* top_function) noexcept -> void {
        // Construct the source file path
        std::string src_path     = hvx_path + file;
        std::string tcl_path     = hvx_path + tcl_file;
        std::string project_path = hvx_path + out_path;

        SemaphoreWait();

        // Construct the command for Vitis HLS
        std::stringstream com;
        com << vitis_path << " -f " << tcl_path << " -tclargs ";
        com << project_path << " ";                   // [2] result path
        com << solution << ".csv ";                   // [3] result file
        com << project_path << " ";                   // [4] project path
        com << solution << " ";                       // [5] solution name
        com << part << " ";                           // [6] part
        com << period << " ";                         // [7] clock period
        com << top_function << " ";                   // [8] set top function name
        com << std::boolalpha << do_c_sim << " ";     // [9] c simulation
        com << std::boolalpha << do_c_synth << " ";   // [10] c synthesis
        com << std::boolalpha << do_rtl_synth << " "; // [11] rtl synthesis (IP-core generation)
        com << src_path << " ";                       // [12] include src file synthesis

        // Open Vitis HLS and create IP-cores
        std::system(com.str().c_str()); // NOLINT

        // C simulation
        if (do_c_sim) {
            const std::string csim_src = project_path + solution + "/" + solution + "/csim/report/" + top_function + "_csim.log";
            const std::string csim_dst = project_path + csim_prefix + solution + ".txt";
            CopyFile(csim_src, csim_dst);
        }

        // C synthesis
        if (do_c_synth) {
            const std::string csynth_src = project_path + solution + "/" + solution + "/syn/report/" + top_function + "_csynth.xml";
            const std::string csynth_dst = project_path + csyn_prefix + solution + ".xml";
            CopyFile(csynth_src, csynth_dst);
        }

        // RTL synthesis
        if (do_rtl_synth) {
            const std::string rtlsynth_src = project_path + solution + "/" + solution + "/impl/report/vhdl/export_syn.xml";
            const std::string rtlsynth_dst = project_path + syn_prefix + solution + ".xml";
            CopyFile(rtlsynth_src, rtlsynth_dst);
        }

        SemaphoreNotify();
    }

public:

    VitisParam(bool do_c_sim,
               bool do_c_synth,
               bool do_rtl_synth,
               std::string part,
               std::string period,
               std::string vitis_path,
               std::string hvx_path):
        do_c_sim(do_c_sim),
        do_c_synth(do_c_synth),
        do_rtl_synth(do_rtl_synth),
        part(part),
        period(period),
        vitis_path(vitis_path),
        hvx_path(hvx_path) {
    }

    template<int64_t node_num_, int64_t thread_num_>
    auto Compute(std::array<std::array<const char*, 3>, node_num_> names, std::string csyn_res_file_name, std::string syn_res_file_name)
        -> void {
        // Container to hold threads
        std::vector<std::thread> threads;
        threads.reserve(node_num_);
        count = thread_num_;
        // Semaphore semaphore(thread_num_);

        // Launch threads for Synthesis
        for (int64_t i = 0; i < node_num_; ++i)
            threads.emplace_back(&VitisParam::VitisRun, this, names.at(i).at(0), names.at(i).at(1), names.at(i).at(2));

        // Wait for all threads to finish
        for (auto& thread: threads)
            thread.join();

        // Call Results functions
        if (do_c_sim)
            CsimResults<node_num_>(names);
        if (do_c_synth)
            SynResults<node_num_>(names, csyn_prefix, csyn_res_file_name, csyn_res_header_line, csyn_keywords);
        if (do_rtl_synth)
            SynResults<node_num_>(names, syn_prefix, syn_res_file_name, syn_res_header_line, syn_keywords);
    }
};

/******************************************************************************************************************************************/

#endif
