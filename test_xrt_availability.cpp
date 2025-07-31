#include <iostream>
#include <dlfcn.h>

int main() {
    // Test if XRT libraries can be loaded dynamically
    std::cout << "Testing XRT library availability...\n";
    
    // Try to load xrt++
    void* xrtpp_handle = dlopen("libxrt++.so", RTLD_LAZY);
    if (!xrtpp_handle) {
        std::cout << "❌ Failed to load libxrt++.so: " << dlerror() << "\n";
    } else {
        std::cout << "✅ libxrt++.so loaded successfully\n";
        dlclose(xrtpp_handle);
    }
    
    // Try to load xrt_core
    void* xrt_core_handle = dlopen("libxrt_core.so", RTLD_LAZY);
    if (!xrt_core_handle) {
        std::cout << "❌ Failed to load libxrt_core.so: " << dlerror() << "\n";
    } else {
        std::cout << "✅ libxrt_core.so loaded successfully\n";
        dlclose(xrt_core_handle);
    }
    
    // Try to load xrt_coreutil
    void* xrt_coreutil_handle = dlopen("libxrt_coreutil.so", RTLD_LAZY);
    if (!xrt_coreutil_handle) {
        std::cout << "❌ Failed to load libxrt_coreutil.so: " << dlerror() << "\n";
    } else {
        std::cout << "✅ libxrt_coreutil.so loaded successfully\n";
        dlclose(xrt_coreutil_handle);
    }
    
    std::cout << "\nSummary: ";
    if (xrtpp_handle && xrt_core_handle && xrt_coreutil_handle) {
        std::cout << "All XRT libraries are available! ✅\n";
        return 0;
    } else {
        std::cout << "Some XRT libraries are missing ❌\n";
        return 1;
    }
}