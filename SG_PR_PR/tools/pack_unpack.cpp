#include <iostream>
#include <cstdint>

void unpack(uint64_t packed, uint32_t &x, uint32_t &y) {
    // Extract the upper 32 bits by right-shifting 32 bits
    x = packed >> 32;

    // Extract the lower 32 bits by applying a bitmask
    y = packed & 0xFFFFFFFF;
}

int main() {
    // Example packed value
    uint32_t i = 10;
    uint32_t j = 20;
    uint64_t packed = ((uint64_t)(i) << 32 | j);

    // Variables to hold the unpacked values
    uint32_t x, y;
    unpack(packed, x, y);

    // Output the unpacked values
    std::cout << "Unpacked x: " << x << std::endl;
    std::cout << "Unpacked y: " << y << std::endl;

    return 0;
}
