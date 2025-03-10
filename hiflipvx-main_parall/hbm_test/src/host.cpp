#include <iostream>
#include <vector>
#include "xcl2.hpp" // Xilinx OpenCL Helper Library

#define DATA_SIZE 32768 // Adjust based on the actual size requirements

using src_type = int16_t; // Adjust if needed
using dst_type = int16_t;

int main(int argc, char** argv) {
    // Check input arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <XCLBIN File>\n";
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Program program;
    cl::Kernel kernel;

    // Load Xilinx OpenCL binary
    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

    bool valid_device = false;
    for (size_t i = 0; i < devices.size(); i++) {
        try {
            context = cl::Context(devices[i], nullptr, nullptr, nullptr, &err);
            q = cl::CommandQueue(context, devices[i], CL_QUEUE_PROFILING_ENABLE, &err);
            program = cl::Program(context, {devices[i]}, bins, nullptr, &err);
            kernel = cl::Kernel(program, "krnl_TestHw", &err);
            valid_device = true;
            break;
        } catch (...) {
            std::cerr << "Failed to program device " << i << " with XCLBIN file." << std::endl;
        }
    }
    if (!valid_device) {
        std::cerr << "No valid devices found. Exiting...\n";
        return EXIT_FAILURE;
    }

    // Allocate input and output buffers in HBM
    std::vector<src_type> src(DATA_SIZE, 1); // Example initialization
    std::vector<dst_type> dst(DATA_SIZE, 0);
    
    cl::Buffer buffer_src(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_EXT_PTR_XILINX, sizeof(src_type) * DATA_SIZE, src.data(), &err);
    cl::Buffer buffer_dst(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, sizeof(dst_type) * DATA_SIZE, nullptr, &err);

    // Set HBM-specific memory banks
    cl_mem_ext_ptr_t ext_src, ext_dst;
    ext_src.flags = XCL_MEM_TOPOLOGY | 0; // Assign to HBM Bank 0
    ext_src.obj = src.data();
    ext_src.param = 0;
    
    ext_dst.flags = XCL_MEM_TOPOLOGY | 1; // Assign to HBM Bank 1
    ext_dst.obj = dst.data();
    ext_dst.param = 0;
    
    buffer_src = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, sizeof(src_type) * DATA_SIZE, &ext_src, &err);
    buffer_dst = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, sizeof(dst_type) * DATA_SIZE, &ext_dst, &err);

    // Set kernel arguments
    kernel.setArg(0, buffer_src);
    kernel.setArg(1, buffer_dst);

    // Copy data to device
    q.enqueueWriteBuffer(buffer_src, CL_TRUE, 0, sizeof(src_type) * DATA_SIZE, src.data());

    // Execute kernel
    q.enqueueTask(kernel);
    q.finish();

    // Copy result back to host
    q.enqueueReadBuffer(buffer_dst, CL_TRUE, 0, sizeof(dst_type) * DATA_SIZE, dst.data());

    // Validate results (basic check)
    for (size_t i = 0; i < 10; i++) {
        std::cout << "dst[" << i << "] = " << dst[i] << std::endl;
    }

    std::cout << "Kernel execution completed successfully!" << std::endl;
    return EXIT_SUCCESS;
}
