#include <executorch/backends/vulkan/runtime/api/ShaderPackLoader.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

namespace {

vkcompute::vkapi::ShaderInfo make_shader(
    const std::string& kernel_name,
    const uint32_t* bin,
    uint32_t size_bytes) {
  return vkcompute::vkapi::ShaderInfo(
      kernel_name,
      bin,
      size_bytes,
      {},
      {1u, 1u, 1u},
      false,
      false,
      false,
      false,
      false,
      false);
}

} // namespace

int main() {
  using vkcompute::api::DispatchKey;
  using vkcompute::api::ShaderPackKernelEntry;
  using vkcompute::api::ShaderPackLoader;
  using vkcompute::api::ShaderRegistry;
  using vkcompute::api::shader_registry;

  ShaderRegistry& registry = shader_registry();
  registry.clear_dynamic_overlay();

  static const uint32_t kAotBin[] = {0x07230203u};
  auto aot_shader = make_shader("binary_add_buffer_float", kAotBin, 4u);
  registry.register_shader(std::move(aot_shader));
  registry.register_op_dispatch(
      "aten.add.Tensor", DispatchKey::CATCHALL, "binary_add_buffer_float");

  {
    auto shader = registry.get_shader_info("binary_add_buffer_float");
    assert(shader.src_code.size == 4u);
  }

  std::string resolved_name;
  bool resolved =
      registry.try_get_op_dispatch("aten.add.Tensor", DispatchKey::ADRENO, &resolved_name);
  assert(resolved);
  assert(resolved_name == "binary_add_buffer_float");

  ShaderPackLoader loader;
  ShaderPackKernelEntry entry;
  entry.shader_info = make_shader("binary_add_buffer_float", nullptr, 0u);
  entry.spirv_words = {0x07230203u, 0x00010000u};
  entry.register_for_op = "aten.add.Tensor";
  entry.dispatch_keys = {DispatchKey::CATCHALL};
  std::vector<ShaderPackKernelEntry> entries;
  entries.emplace_back(std::move(entry));
  loader.load(std::move(entries), /*clear_existing_overlay=*/false);

  {
    auto shader = registry.get_shader_info("binary_add_buffer_float");
    assert(shader.src_code.size == 8u);
  }

  // Verify manifest-based loading path.
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / "vk_shader_registry_smoke";
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path spv_path = tmp_dir / "override.spv";
  const std::filesystem::path manifest_path = tmp_dir / "manifest.txt";

  {
    std::ofstream spv_out(spv_path, std::ios::binary);
    const uint32_t words[] = {0x07230203u, 0x00010000u, 0x00020000u};
    spv_out.write(
        reinterpret_cast<const char*>(words),
        static_cast<std::streamsize>(sizeof(words)));
  }
  {
    std::ofstream manifest_out(manifest_path);
    manifest_out << "binary_add_buffer_float=override.spv\n";
  }

  vkcompute::api::load_shader_override_manifest(
      manifest_path.string(), /*clear_existing_overlay=*/false);
  {
    auto shader = registry.get_shader_info("binary_add_buffer_float");
    assert(shader.src_code.size == 12u);
  }

  loader.clear();

  {
    auto shader = registry.get_shader_info("binary_add_buffer_float");
    assert(shader.src_code.size == 4u);
  }

  std::filesystem::remove_all(tmp_dir);

  std::cout << "shader_registry_smoke: OK" << std::endl;
  return 0;
}
