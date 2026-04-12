#include "ggml-vulkan.h"

#include "ggml-backend-impl.h"
#include "ggml-backend-dl.h"

#include <cstring>
#include <mutex>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>

namespace fs = std::filesystem;

static std::once_flag s_vk_init_once;
static dl_handle_ptr s_vk_handle;
static std::vector<dl_handle_ptr> s_vk_dep_handles;
static ggml_backend_reg_t s_vk_reg = nullptr;

static bool vk_bridge_debug_enabled() {
    const char * v = std::getenv("QWEN3_VK_BRIDGE_DEBUG");
    return v && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

static void vk_bridge_debug(const char * fmt, ...) {
    if (!vk_bridge_debug_enabled()) {
        return;
    }
    std::va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "ggml-vulkan-bridge: ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}

static fs::path vk_backend_default_dir() {
#ifdef _WIN32
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(NULL, path.data(), (DWORD) path.size());
    if (len == 0) {
        return {};
    }
    std::wstring exe(path.data(), len);
    auto slash = exe.find_last_of(L'\\');
    if (slash == std::wstring::npos) {
        return {};
    }
    return fs::path(exe.substr(0, slash));
#else
    return fs::current_path();
#endif
}

static ggml_backend_reg_t try_load_vk_backend(const fs::path & dll_path) {
    if (dll_path.empty() || !fs::exists(dll_path)) {
        vk_bridge_debug("skip path (not found): %s", dll_path.string().c_str());
        return nullptr;
    }

    const fs::path dep_dir = dll_path.parent_path();
    if (!dep_dir.empty()) {
        const char * deps[] = {
            "ggml-base.dll",
            "ggml.dll",
            "ggml-cpu.dll",
        };
        for (const char * dep : deps) {
            const fs::path dep_path = dep_dir / dep;
            if (!fs::exists(dep_path)) {
                continue;
            }
            dl_handle_ptr dep_handle{ dl_load_library(dep_path) };
            if (dep_handle) {
                vk_bridge_debug("preloaded dependency: %s", dep_path.string().c_str());
                s_vk_dep_handles.push_back(std::move(dep_handle));
            } else {
                vk_bridge_debug("failed to preload dependency: %s", dep_path.string().c_str());
            }
        }
    }

    vk_bridge_debug("trying path: %s", dll_path.string().c_str());
    dl_handle_ptr handle{ dl_load_library(dll_path) };
    if (!handle) {
        vk_bridge_debug("LoadLibrary failed: %s", dll_path.string().c_str());
        return nullptr;
    }

    ggml_backend_reg_t reg = nullptr;
    auto init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
    if (init_fn) {
        reg = init_fn();
    } else {
        auto reg_fn = (ggml_backend_reg_t (*)()) dl_get_sym(handle.get(), "ggml_backend_vk_reg");
        if (!reg_fn) {
            vk_bridge_debug("missing symbols ggml_backend_init/ggml_backend_vk_reg: %s", dll_path.string().c_str());
            return nullptr;
        }
        reg = reg_fn();
    }

    if (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
        vk_bridge_debug("invalid reg or api mismatch at %s (reg=%p, api=%d, expected=%d)",
            dll_path.string().c_str(),
            (void *) reg,
            reg ? reg->api_version : -1,
            GGML_BACKEND_API_VERSION);
        return nullptr;
    }

    s_vk_handle = std::move(handle);
    vk_bridge_debug("loaded Vulkan backend from: %s", dll_path.string().c_str());
    return reg;
}

static void init_vk_reg_once() {
    const char * env_path = std::getenv("QWEN3_GGML_VULKAN_DLL");
    if (env_path && env_path[0] != '\0') {
        s_vk_reg = try_load_vk_backend(fs::u8path(env_path));
        if (s_vk_reg) {
            return;
        }
    }

    const fs::path exe_dir = vk_backend_default_dir();
    if (!exe_dir.empty()) {
        s_vk_reg = try_load_vk_backend(exe_dir / "ggml-vulkan.dll");
        if (s_vk_reg) {
            return;
        }
    }

    const fs::path known_paths[] = {
        fs::path("C:/Apps/llama.cpp/build-vulkan-llm-groups/bin/Release/ggml-vulkan.dll"),
        fs::path("C:/Apps/llama.cpp/build-vulkan/bin/Release/ggml-vulkan.dll"),
        fs::path("C:/Apps/llama.cpp/build-vulkan-default-after/bin/Release/ggml-vulkan.dll"),
        fs::path("C:/Apps/llama.cpp/build-vulkan-trim-safe/bin/Release/ggml-vulkan.dll"),
    };

    for (const fs::path & p : known_paths) {
        s_vk_reg = try_load_vk_backend(p);
        if (s_vk_reg) {
            return;
        }
    }
}

ggml_backend_reg_t ggml_backend_vk_reg(void) {
    std::call_once(s_vk_init_once, init_vk_reg_once);
    return s_vk_reg;
}

int ggml_backend_vk_get_device_count(void) {
    ggml_backend_reg_t reg = ggml_backend_vk_reg();
    if (!reg) {
        return 0;
    }
    return (int) ggml_backend_reg_dev_count(reg);
}

ggml_backend_t ggml_backend_vk_init(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_vk_reg();
    if (!reg) {
        return nullptr;
    }
    if (dev_num >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, dev_num);
    return ggml_backend_dev_init(dev, nullptr);
}

bool ggml_backend_is_vk(ggml_backend_t backend) {
    if (!backend) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        return false;
    }
    return ggml_backend_dev_backend_reg(dev) == ggml_backend_vk_reg();
}

void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size) {
    if (!description || description_size == 0) {
        return;
    }

    description[0] = '\0';
    ggml_backend_reg_t reg = ggml_backend_vk_reg();
    if (!reg || device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        const char * msg = "Vulkan backend unavailable";
        std::strncpy(description, msg, description_size - 1);
        description[description_size - 1] = '\0';
        return;
    }

    const char * src = ggml_backend_dev_description(ggml_backend_reg_dev_get(reg, (size_t) device));
    if (!src) {
        src = "Vulkan device";
    }
    std::strncpy(description, src, description_size - 1);
    description[description_size - 1] = '\0';
}

void ggml_backend_vk_get_device_memory(int device, size_t * free_mem, size_t * total_mem) {
    if (free_mem) {
        *free_mem = 0;
    }
    if (total_mem) {
        *total_mem = 0;
    }

    ggml_backend_reg_t reg = ggml_backend_vk_reg();
    if (!reg || device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        return;
    }

    ggml_backend_dev_memory(ggml_backend_reg_dev_get(reg, (size_t) device), free_mem, total_mem);
}

ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_vk_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, dev_num));
}

ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void) {
    ggml_backend_reg_t reg = ggml_backend_vk_reg();
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        return nullptr;
    }
    return ggml_backend_dev_host_buffer_type(ggml_backend_reg_dev_get(reg, 0));
}

GGML_BACKEND_DL_IMPL(ggml_backend_vk_reg)
