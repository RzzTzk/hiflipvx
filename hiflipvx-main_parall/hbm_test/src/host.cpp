/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

/********************************************************************************************
 * Description:
 * This is host application to test HBM Full bandwidth.
 * Design contains 8 compute units of Kernel. Each compute unit has full access
 *to all HBM
 * memory (0 to 31). Host application allocate buffers into all 32 HBM Banks(16
 *Input buffers
 * and 16 output buffers). Host application runs all 8 compute units together
 *and measure
 * the overall HBM bandwidth.
 *
 ******************************************************************************************/

#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "xcl2.hpp"

#define NUM_KERNEL 1

// HBM Pseudo-channel(PC) requirements
#define MAX_HBM_PC_COUNT 32
#define PC_NAME(n) n | XCL_MEM_TOPOLOGY
const int pc[MAX_HBM_PC_COUNT] = {
    PC_NAME(0),  PC_NAME(1),  PC_NAME(2),  PC_NAME(3),  PC_NAME(4),  PC_NAME(5),  PC_NAME(6),  PC_NAME(7),
    PC_NAME(8),  PC_NAME(9),  PC_NAME(10), PC_NAME(11), PC_NAME(12), PC_NAME(13), PC_NAME(14), PC_NAME(15),
    PC_NAME(16), PC_NAME(17), PC_NAME(18), PC_NAME(19), PC_NAME(20), PC_NAME(21), PC_NAME(22), PC_NAME(23),
    PC_NAME(24), PC_NAME(25), PC_NAME(26), PC_NAME(27), PC_NAME(28), PC_NAME(29), PC_NAME(30), PC_NAME(31)};

// Function for verifying results
bool verify(std::vector<int, aligned_allocator<int> >& source_sw_results,
            std::vector<int, aligned_allocator<int> >& source_hw_results1,
            std::vector<int, aligned_allocator<int> >& source_hw_results2,
            unsigned int size) {
    bool check = true;
    for (size_t i = 0; i < size; i++) {
        if (source_hw_results1[i] != source_sw_results[i]) {
            std::cout << "Error: Result mismatch in Addition Operation" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_results[i]
                      << " Device result = " << source_hw_results1[i] << std::endl;
            check = false;
            break;
        }
        if (source_hw_results2[i] != source_sw_results[i]) {
            std::cout << "Error: Result mismatch in Addition Operation" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_results[i]
                      << " Device result = " << source_hw_results2[i] << std::endl;
            check = false;
            break;
        }        
    }

    return check;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <XCLBIN> \n", argv[0]);
        return -1;
    }

    unsigned int in_dataSize = 512; // taking maximum possible data size value for an HBM bank
    unsigned int out_dataSize = 128;
    unsigned int num_times = 1;            // num_times specify, number of times a kernel
                                              // will execute the same operation. This is
                                              // needed
    // to keep the kernel busy to test the actual bandwidth of all banks running
    // concurrently.

    // reducing the test data capacity to run faster in emulation mode
    if (xcl::is_emulation()) {
        in_dataSize = 512;
        out_dataSize = 128;
        num_times = 1;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::CommandQueue q;
    std::string krnl_name = "krnl_TestHw";
    std::vector<cl::Kernel> krnls(NUM_KERNEL);
    cl::Context context;
    std::vector<int, aligned_allocator<int> > src1(in_dataSize);
    std::vector<int, aligned_allocator<int> > src_sw_results(out_dataSize);

    std::vector<int, aligned_allocator<int> > dst1[NUM_KERNEL];
    std::vector<int, aligned_allocator<int> > dst2[NUM_KERNEL];
    for (int i = 0; i < NUM_KERNEL; i++) {
        dst1[i].resize(out_dataSize);
        dst2[i].resize(out_dataSize);
    }

    // Create the test data
    std::generate(src1.begin(), src1.end(), std::rand);

    // for (size_t i = 0; i < dataSize; i++) {
    //     src_sw_results[i] = src[i];
    // }
    for (int64_t batch = 0; batch < 1; ++batch) {
        for (int64_t chnl = 0; chnl < 8; ++chnl) {
            for (int64_t dst_row = 0; dst_row < 4; ++dst_row) {
                for (int64_t dst_col = 0; dst_col < 4; ++dst_col) {
                
                    int64_t result = 0;
                    // compute pooling
                    for (int64_t knl_row = 0; knl_row < 2; ++knl_row) {
                        for (int64_t knl_col = 0; knl_col < 2; ++knl_col) {

                            // read input
                            int64_t ptr_src = batch * 8 * 4 * 4 * 4 + chnl * 4 * 4 * 4 + dst_row * 4 * 4 + dst_col * 4 + knl_row * 2 + knl_col;
                            int64_t data = src1[ptr_src];

                            // update max or average pooling
                            result = result + data;
                        }
                    }

                    // compute average pooling
                    result = result / static_cast<float>(4);

                    // write output
                    int64_t ptr_dst = batch * 8 * 4 * 4 + chnl * 4 * 4 + dst_row * 4 + dst_col;
                    src_sw_results[ptr_dst] = result; // NOLINT
                }
            }
        }
    }    
    

    // Initializing output vectors to zero
    for (size_t i = 0; i < NUM_KERNEL; i++) {
        std::fill(dst1[i].begin(), dst1[i].end(), 0);
        std::fill(dst2[i].begin(), dst2[i].end(), 0);
    }

    // OPENCL HOST CODE AREA START
    // The get_xil_devices will return vector of Xilinx Devices
    auto devices = xcl::get_xil_devices();

    // read_binary_file() command will find the OpenCL binary file created using
    // the
    // V++ compiler load into OpenCL Binary and return pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);

    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            // Creating Kernel object using Compute unit names

            for (int i = 0; i < NUM_KERNEL; i++) {
                std::string cu_id = std::to_string(i + 1);
                std::string krnl_name_full = krnl_name + ":{" + "krnl_TestHw_" + cu_id + "}";

                printf("Creating a kernel [%s] for CU(%d)\n", krnl_name_full.c_str(), i + 1);

                // Here Kernel object is created by specifying kernel name along with
                // compute unit.
                // For such case, this kernel object can only access the specific
                // Compute unit

                OCL_CHECK(err, krnls[i] = cl::Kernel(program, krnl_name_full.c_str(), &err));
            }
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    std::vector<cl_mem_ext_ptr_t> inBufExt1(NUM_KERNEL);
    std::vector<cl_mem_ext_ptr_t> inBufExt2(NUM_KERNEL);    
    std::vector<cl_mem_ext_ptr_t> outBufExt1(NUM_KERNEL);
    std::vector<cl_mem_ext_ptr_t> outBufExt2(NUM_KERNEL);
    std::vector<cl::Buffer> buffer_src1(NUM_KERNEL);
    std::vector<cl::Buffer> buffer_src2(NUM_KERNEL);    
    std::vector<cl::Buffer> buffer_output1(NUM_KERNEL);
    std::vector<cl::Buffer> buffer_output2(NUM_KERNEL);
    // For Allocating Buffer to specific Global Memory PC, user has to use
    // cl_mem_ext_ptr_t
    // and provide the PCs
    for (int i = 0; i < NUM_KERNEL; i++) {
        inBufExt1[i].obj = src1.data();
        inBufExt1[i].param = 0;
        inBufExt1[i].flags = pc[i * 4];

        inBufExt2[i].obj = src1.data();
        inBufExt2[i].param = 0;
        inBufExt2[i].flags = pc[i * 4 + 1];        

        outBufExt1[i].obj = dst1[i].data();
        outBufExt1[i].param = 0;
        outBufExt1[i].flags = pc[(i * 4) + 2];

        outBufExt2[i].obj = dst2[i].data();
        outBufExt2[i].param = 0;
        outBufExt2[i].flags = pc[(i * 4) + 3];        
    }

    // These commands will allocate memory on the FPGA. The cl::Buffer objects can
    // be used to reference the memory locations on the device.
    // Creating Buffers
    for (int i = 0; i < NUM_KERNEL; i++) {
        OCL_CHECK(err,
                  buffer_src1[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                                sizeof(uint32_t) * in_dataSize, &inBufExt1[i], &err));
        OCL_CHECK(err,
                  buffer_src2[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                                sizeof(uint32_t) * in_dataSize, &inBufExt2[i], &err));                                                
        OCL_CHECK(err, buffer_output1[i] =
                           cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                      sizeof(uint32_t) * out_dataSize, &outBufExt1[i], &err));
        OCL_CHECK(err, buffer_output2[i] =
                           cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                      sizeof(uint32_t) * out_dataSize, &outBufExt2[i], &err));
    }

    // Copy input data to Device Global Memory
    for (int i = 0; i < NUM_KERNEL; i++) {
        OCL_CHECK(err,
                  err = q.enqueueMigrateMemObjects({buffer_src1[i], buffer_src2[i]}, 0 /* 0 means from host*/));
    }
    q.finish();

    double kernel_time_in_sec = 0, result = 0;

    std::chrono::duration<double> kernel_time(0);

    auto kernel_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_KERNEL; i++) {
        // Setting the k_vadd Arguments
        OCL_CHECK(err, err = krnls[i].setArg(0, buffer_src1[i]));
        OCL_CHECK(err, err = krnls[i].setArg(1, buffer_src2[i]));
        OCL_CHECK(err, err = krnls[i].setArg(2, buffer_output1[i]));
        OCL_CHECK(err, err = krnls[i].setArg(3, buffer_output2[i]));
        // Invoking the kernel
        OCL_CHECK(err, err = q.enqueueTask(krnls[i]));
    }
    q.finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();

    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);

    kernel_time_in_sec = kernel_time.count();
    kernel_time_in_sec /= NUM_KERNEL;

    // Copy Result from Device Global Memory to Host Local Memory
    for (int i = 0; i < NUM_KERNEL; i++) {
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output1[i], buffer_output2[i]},
                                                        CL_MIGRATE_MEM_OBJECT_HOST));
    }
    q.finish();

    bool match = true;

    for (int i = 0; i < NUM_KERNEL; i++) {
        match = verify(src_sw_results, dst1[i], dst2[i],
                       out_dataSize);
    }

    // Multiplying the actual data size by 4 because four buffers are being used.
    result = 2 * (float)in_dataSize * num_times * sizeof(uint32_t);
    result /= 1000;               // to KB
    result /= 1000;               // to MB
    result /= 1000;               // to GB
    result /= kernel_time_in_sec; // to GBps

    std::cout << "THROUGHPUT = " << result << " GB/s" << std::endl;
    // OPENCL HOST CODE AREA ENDS

    std::cout << (match ? "TEST PASSED" : "TEST FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}





// #include <iostream>
// #include <vector>
// #include "xcl2.hpp" // Xilinx OpenCL Helper Library

// #define DATA_SIZE 32768 // Adjust based on the actual size requirements

// #define NUM_KERNEL 1

// // HBM Pseudo-channel(PC) requirements
// #define MAX_HBM_PC_COUNT 32
// #define PC_NAME(n) n | XCL_MEM_TOPOLOGY
// const int pc[MAX_HBM_PC_COUNT] = {
//     PC_NAME(0),  PC_NAME(1),  PC_NAME(2),  PC_NAME(3),  PC_NAME(4),  PC_NAME(5),  PC_NAME(6),  PC_NAME(7),
//     PC_NAME(8),  PC_NAME(9),  PC_NAME(10), PC_NAME(11), PC_NAME(12), PC_NAME(13), PC_NAME(14), PC_NAME(15),
//     PC_NAME(16), PC_NAME(17), PC_NAME(18), PC_NAME(19), PC_NAME(20), PC_NAME(21), PC_NAME(22), PC_NAME(23),
//     PC_NAME(24), PC_NAME(25), PC_NAME(26), PC_NAME(27), PC_NAME(28), PC_NAME(29), PC_NAME(30), PC_NAME(31)};
// using src_type = int16_t; // Adjust if needed
// using dst_type = int16_t;

// int main(int argc, char** argv) {
//     // Check input arguments
//     if (argc != 2) {
//         std::cerr << "Usage: " << argv[0] << " <XCLBIN File>\n";
//         return EXIT_FAILURE;
//     }

//     std::string binaryFile = argv[1];
//     cl_int err;
//     cl::Context context;
//     cl::CommandQueue q;
//     cl::Program program;
//     cl::Kernel kernel;
//     //xxx
//     std::vector<int, aligned_allocator<int> > source_in1(dataSize);
//     std::vector<int, aligned_allocator<int> > source_in2(dataSize);
//     std::vector<int, aligned_allocator<int> > source_sw_add_results(dataSize);
//     std::vector<int, aligned_allocator<int> > source_sw_mul_results(dataSize);

//     std::vector<int, aligned_allocator<int> > source_hw_add_results[NUM_KERNEL];
//     std::vector<int, aligned_allocator<int> > source_hw_mul_results[NUM_KERNEL];

//     for (int i = 0; i < NUM_KERNEL; i++) {
//         source_hw_add_results[i].resize(dataSize);
//         source_hw_mul_results[i].resize(dataSize);
//     }

//     // Create the test data
//     std::generate(source_in1.begin(), source_in1.end(), std::rand);
//     std::generate(source_in2.begin(), source_in2.end(), std::rand);
//     for (size_t i = 0; i < dataSize; i++) {
//         source_sw_add_results[i] = source_in1[i] + source_in2[i];
//         source_sw_mul_results[i] = source_in1[i] * source_in2[i];
//     }

//     // Initializing output vectors to zero
//     for (size_t i = 0; i < NUM_KERNEL; i++) {
//         std::fill(source_hw_add_results[i].begin(), source_hw_add_results[i].end(), 0);
//         std::fill(source_hw_mul_results[i].begin(), source_hw_mul_results[i].end(), 0);
//     }

//     // Load Xilinx OpenCL binary
//     auto devices = xcl::get_xil_devices();
//     if (devices.empty()) {
//         std::cerr << "Error: No OpenCL devices found!" << std::endl;
//         return EXIT_FAILURE;
//     }    
//     auto fileBuf = xcl::read_binary_file(binaryFile);
//     if (fileBuf.size() == 0) {
//         std::cerr << "Error: Failed to read XCLBIN file!" << std::endl;
//         return EXIT_FAILURE;
//     }
//     cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

//     bool valid_device = false;
//     // for (size_t i = 0; i < devices.size(); i++) {
//     //     try {
//     //         context = cl::Context(devices[i], nullptr, nullptr, nullptr, &err);
//     //         q = cl::CommandQueue(context, devices[i], CL_QUEUE_PROFILING_ENABLE, &err);
//     //         program = cl::Program(context, {devices[i]}, bins, nullptr, &err);
//     //         kernel = cl::Kernel(program, "krnl_TestHw", &err);
//     //         valid_device = true;
//     //         break;
//     //     } catch (...) {
//     //         std::cerr << "Failed to program device " << i << " with XCLBIN file." << std::endl;
//     //     }
//     // }
//     // if (!valid_device) {
//     //     std::cerr << "No valid devices found. Exiting...\n";
//     //     return EXIT_FAILURE;
//     // }
//     for (unsigned int i = 0; i < devices.size(); i++) {
//         auto device = devices[i];
//         // Creating Context and Command Queue for selected Device
//         OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
//         OCL_CHECK(err, q = cl::CommandQueue(context, device,
//                                             CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err));

//         std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
//         cl::Program program(context, {device}, bins, nullptr, &err);
//         if (err != CL_SUCCESS) {
//             std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
//         } else {
//             std::cout << "Device[" << i << "]: program successful!\n";
//             // Creating Kernel object using Compute unit names

//             kernel = cl::Kernel(program, "krnl_TestHw", &err);
//             valid_device = true;
//             break; // we break because we found a valid device
//         }
//     }
//     if (!valid_device) {
//         std::cout << "Failed to program any device found, exit!\n";
//         exit(EXIT_FAILURE);
//     }


//     // Allocate input and output buffers in HBM
//     // std::vector<src_type> src(DATA_SIZE, 1); // Example initialization
//     // std::vector<dst_type> dst(DATA_SIZE, 0);

    
//     // cl::Buffer buffer_src(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_EXT_PTR_XILINX, sizeof(src_type) * DATA_SIZE, src.data(), &err);
//     // cl::Buffer buffer_dst(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, sizeof(dst_type) * DATA_SIZE, nullptr, &err);
//     std::vector<cl_mem_ext_ptr_t> inBufExt1(NUM_KERNEL);
//     std::vector<cl_mem_ext_ptr_t> inBufExt2(NUM_KERNEL);
//     std::vector<cl_mem_ext_ptr_t> outAddBufExt(NUM_KERNEL);
//     std::vector<cl_mem_ext_ptr_t> outMulBufExt(NUM_KERNEL);

//     std::vector<cl::Buffer> buffer_input1(NUM_KERNEL);
//     std::vector<cl::Buffer> buffer_input2(NUM_KERNEL);
//     std::vector<cl::Buffer> buffer_output_add(NUM_KERNEL);
//     std::vector<cl::Buffer> buffer_output_mul(NUM_KERNEL);

//     // For Allocating Buffer to specific Global Memory PC, user has to use
//     // cl_mem_ext_ptr_t
//     // and provide the PCs
//     for (int i = 0; i < NUM_KERNEL; i++) {
//         inBufExt1[i].obj = source_in1.data();
//         inBufExt1[i].param = 0;
//         inBufExt1[i].flags = pc[i * 4];

//         inBufExt2[i].obj = source_in2.data();
//         inBufExt2[i].param = 0;
//         inBufExt2[i].flags = pc[(i * 4) + 1];

//         outAddBufExt[i].obj = source_hw_add_results[i].data();
//         outAddBufExt[i].param = 0;
//         outAddBufExt[i].flags = pc[(i * 4) + 2];

//         outMulBufExt[i].obj = source_hw_mul_results[i].data();
//         outMulBufExt[i].param = 0;
//         outMulBufExt[i].flags = pc[(i * 4) + 3];
//     }
    
//     // Set HBM-specific memory banks
//     // cl_mem_ext_ptr_t ext_src, ext_dst;
//     // ext_src.flags = XCL_MEM_TOPOLOGY | 0; // Assign to HBM Bank 0
//     // ext_src.obj = src.data();
//     // ext_src.param = 0;
    
//     // ext_dst.flags = XCL_MEM_TOPOLOGY | 1; // Assign to HBM Bank 1
//     // ext_dst.obj = dst.data();
//     // ext_dst.param = 0;
    
//     // buffer_src = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, sizeof(src_type) * DATA_SIZE, &ext_src, &err);
//     // buffer_dst = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, sizeof(dst_type) * DATA_SIZE, &ext_dst, &err);
//     for (int i = 0; i < NUM_KERNEL; i++) {
//         OCL_CHECK(err,
//                   buffer_input1[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
//                                                 sizeof(uint32_t) * dataSize, &inBufExt1[i], &err));
//         OCL_CHECK(err,
//                   buffer_input2[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
//                                                 sizeof(uint32_t) * dataSize, &inBufExt2[i], &err));
//         OCL_CHECK(err, buffer_output_add[i] =
//                            cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
//                                       sizeof(uint32_t) * dataSize, &outAddBufExt[i], &err));
//         OCL_CHECK(err, buffer_output_mul[i] =
//                            cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
//                                       sizeof(uint32_t) * dataSize, &outMulBufExt[i], &err));  
//     }


//     if (err != CL_SUCCESS) {

//         std::cerr << "Error: Failed to create buffer_src! Error Code: " << err << std::endl;
//         return EXIT_FAILURE;
//     }    
//     // Set kernel arguments
//     // kernel.setArg(0, buffer_src);
//     // kernel.setArg(1, buffer_dst);
//     for (int i = 0; i < NUM_KERNEL; i++) {
//         // Setting the k_vadd Arguments
//         OCL_CHECK(err, err = krnls[i].setArg(0, buffer_input1[i]));
//         OCL_CHECK(err, err = krnls[i].setArg(1, buffer_input2[i]));
//         OCL_CHECK(err, err = krnls[i].setArg(2, buffer_output_add[i]));
//         OCL_CHECK(err, err = krnls[i].setArg(3, buffer_output_mul[i]));
//         OCL_CHECK(err, err = krnls[i].setArg(4, dataSize));
//         OCL_CHECK(err, err = krnls[i].setArg(5, num_times));

//         // Invoking the kernel
//         OCL_CHECK(err, err = q.enqueueTask(krnls[i]));
//     }
//     // // Copy data to device
//     // q.enqueueWriteBuffer(buffer_src, CL_TRUE, 0, sizeof(src_type) * DATA_SIZE, src.data());

//     // // Execute kernel
//     // q.enqueueTask(kernel);
//     q.finish();
//     // Copy Result from Device Global Memory to Host Local Memory
//     for (int i = 0; i < NUM_KERNEL; i++) {
//         OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_add[i], buffer_output_mul[i]},
//                                                         CL_MIGRATE_MEM_OBJECT_HOST));
//     }
//     q.finish();    

//     // Copy result back to host
//     q.enqueueReadBuffer(buffer_dst, CL_TRUE, 0, sizeof(dst_type) * DATA_SIZE, dst.data());

//     // Validate results (basic check)
//     for (size_t i = 0; i < 10; i++) {
//         std::cout << "dst[" << i << "] = " << dst[i] << std::endl;
//     }

//     std::cout << "Kernel execution completed successfully!" << std::endl;
//     return EXIT_SUCCESS;
// }
