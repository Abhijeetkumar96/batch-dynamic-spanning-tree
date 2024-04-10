// nvcc -std=c++17 -O3 cub_flagged.cu -o cub_flagged

#include <stdio.h>
#include <random>
#include <limits.h>
#include <cuda_runtime.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_select.cuh>

using namespace cub;

bool g_verbose = true;  // Whether to display input/output to console
CachingDeviceAllocator g_allocator(true);  // Caching allocator for device memory

void RandomBits(unsigned short &value) {
    static std::random_device rd;     // Obtain a random number from hardware
    static std::mt19937 eng(rd());    // Seed the generator
    static std::uniform_int_distribution<> distr(0, USHRT_MAX); // Define the range

    value = static_cast<unsigned short>(distr(eng));
}

void DisplayUint64Array(uint64_t* arr, int num_items) {
    std::cout << "h_in Array: ";
    for(int i = 0; i < num_items; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void DisplayUCharArray(unsigned char* arr, int num_items) {
    std::cout << "h_flags Array: ";
    for(int i = 0; i < num_items; ++i) {
        std::cout << static_cast<int>(arr[i]) << " "; // Cast to int for clearer output
    }
    std::cout << std::endl;
}


template <typename T>
int CompareDeviceResults(T *host_data, T *device_data, int num_items, bool verbose, bool) {
    T *device_data_host = new T[num_items];
    cudaMemcpy(device_data_host, device_data, num_items * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_items; i++) {
        if (host_data[i] != device_data_host[i]) {
            if (verbose) {
                printf("Mismatch at %d: Host: %llu, Device: %llu\n", i, (unsigned long long)host_data[i], (unsigned long long)device_data_host[i]);
            }
            delete[] device_data_host;
            return 1;
        }
    }

    delete[] device_data_host;
    return 0;
}

void AssertEquals(int expected, int actual) {
    if (expected != actual) {
        printf("Assertion failed: expected %d, got %d\n", expected, actual);
        exit(1); // Or handle the error as you see fit
    }
}

template <typename T>
void DisplayResults(T* arr, int num_items) {
    for(int i = 0; i < num_items; ++i) {
        printf("%llu ", (unsigned long long)arr[i]);
    }
    printf("\n");
}

void Initialize(uint64_t *h_in, unsigned char *h_flags, int num_items, int max_segment) {
    unsigned short max_short = (unsigned short) -1;

    int key = 0;
    int i = 0;
    while (i < num_items) {
        unsigned short repeat;
        RandomBits(repeat);
        repeat = (unsigned short) ((float(repeat) * (float(max_segment) / float(max_short))));
        repeat = CUB_MAX(1, repeat);

        int j = i;
        while (j < CUB_MIN(i + repeat, num_items)) {
            h_flags[j] = 0;
            h_in[j] = key;
            j++;
        }

        h_flags[i] = 1;
        i = j;
        key++;
    }

    if (g_verbose) {
        printf("Input:\n");
        DisplayResults(h_in, num_items);
        printf("Flags:\n");
        DisplayResults(h_flags, num_items);
        printf("\n\n");
    }
}

int Solve(uint64_t *h_in, unsigned char *h_flags, uint64_t *h_reference, int num_items) {
    int num_selected = 0;
    for (int i = 0; i < num_items; ++i) {
        if (h_flags[i]) {
            h_reference[num_selected] = h_in[i];
            num_selected++;
        }
    }

    return num_selected;
}

int main(int argc, char** argv) {
    int num_items = 10; // Reduced the number for demonstration purposes
    int max_segment = 4;

    // Allocate host arrays
    uint64_t *h_in = new uint64_t[num_items];
    uint64_t *h_reference = new uint64_t[num_items];
    unsigned char *h_flags = new unsigned char[num_items];

    // Initialize problem and solution
    Initialize(h_in, h_flags, num_items, max_segment);
    int num_selected = Solve(h_in, h_flags, h_reference, num_items);

    printf("Selected %d items from %d\n", num_selected, num_items);

    printf("cub::DeviceSelect::Flagged %d items, %d selected (avg distance %d), %ld-byte elements\n",
        num_items, num_selected, (num_selected > 0) ? num_items / num_selected : 0, (uint64_t) sizeof(uint64_t));
    fflush(stdout);

    // Allocate problem device arrays
    uint64_t        *d_in = NULL;
    unsigned char   *d_flags = NULL;

    g_allocator.DeviceAllocate((void**)&d_in, sizeof(uint64_t) * num_items);
    g_allocator.DeviceAllocate((void**)&d_flags, sizeof(unsigned char) * num_items);

    // Display the contents of the arrays
    DisplayUint64Array(h_in, num_items);
    DisplayUCharArray(h_flags, num_items);
    
    // Initialize device input
    cudaMemcpy(d_in, h_in, sizeof(uint64_t) * num_items, cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, h_flags, sizeof(unsigned char) * num_items, cudaMemcpyHostToDevice);

    // Allocate device output array and num selected
    uint64_t     *d_out            = NULL;
    int     *d_num_selected_out   = NULL;
    g_allocator.DeviceAllocate((void**)&d_out, sizeof(uint64_t) * num_items);
    g_allocator.DeviceAllocate((void**)&d_num_selected_out, sizeof(int));

    // Allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);
    g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes);

    // Run
    DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);

    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(h_reference, d_out, num_selected, true, g_verbose);
    printf("\t Data %s ", compare ? "FAIL" : "PASS");
    compare |= CompareDeviceResults(&num_selected, d_num_selected_out, 1, true, g_verbose);
    printf("\t Count %s ", compare ? "FAIL" : "PASS");
    AssertEquals(0, compare);

    int h_num;
    cudaMemcpy(&h_num, d_num_selected_out, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "\nh_num: " <<  h_num << std::endl;

    // Copy output data back to host
    uint64_t* h_out = new uint64_t[num_items];
    cudaMemcpy(h_out, d_out, sizeof(uint64_t) * num_items, cudaMemcpyDeviceToHost);

    // Print output data
    printf("\nOutput Data (h_out):\n");
    DisplayResults(h_out, num_selected); // Print only the selected elements

    // Cleanup
    if (h_in) 
        delete[] h_in;
    if (h_reference) 
        delete[] h_reference;
    if (d_out) 
        g_allocator.DeviceFree(d_out);
    if (d_num_selected_out) 
        g_allocator.DeviceFree(d_num_selected_out);
    if (d_temp_storage) 
        g_allocator.DeviceFree(d_temp_storage);
    if (d_in) 
        g_allocator.DeviceFree(d_in);
    if (d_flags) 
        g_allocator.DeviceFree(d_flags);

    printf("\n\n");

    return 0;
}
