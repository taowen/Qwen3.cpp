/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/ShaderPackLoader.h>

#include <filesystem>
#include <fstream>
#include <utility>
#include <cstring>

namespace vkcompute {
namespace api {
namespace {

std::string trim_ascii(const std::string& value) {
  const char* whitespace = " \t\r\n";
  const size_t begin = value.find_first_not_of(whitespace);
  if (begin == std::string::npos) {
    return "";
  }
  const size_t end = value.find_last_not_of(whitespace);
  return value.substr(begin, end - begin + 1);
}

std::vector<uint32_t> read_spv_words(const std::filesystem::path& spv_path) {
  std::ifstream in(spv_path, std::ios::binary);
  VK_CHECK_COND(in.good(), "Failed to open SPIR-V file ", spv_path.string());

  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
  VK_CHECK_COND(
      !bytes.empty(),
      "SPIR-V file is empty: ",
      spv_path.string());
  VK_CHECK_COND(
      (bytes.size() % sizeof(uint32_t)) == 0,
      "SPIR-V byte size must be divisible by 4: ",
      spv_path.string());

  const size_t words_len = bytes.size() / sizeof(uint32_t);
  std::vector<uint32_t> words(words_len);
  memcpy(words.data(), bytes.data(), bytes.size());
  return words;
}

} // namespace

void ShaderPackLoader::load(
    std::vector<ShaderPackKernelEntry>&& entries,
    bool clear_existing_overlay) {
  ShaderRegistry& registry = shader_registry();

  if (clear_existing_overlay) {
    registry.clear_dynamic_overlay();
  }

  for (ShaderPackKernelEntry& entry : entries) {
    VK_CHECK_COND(
        !entry.spirv_words.empty(),
        "Shader pack entry ",
        entry.shader_info.kernel_name,
        " has empty SPIR-V.");

    const std::string kernel_name = entry.shader_info.kernel_name;
    registry.upsert_shader_with_owned_binary(
        std::move(entry.shader_info), std::move(entry.spirv_words));

    if (!entry.register_for_op.empty()) {
      for (const DispatchKey key : entry.dispatch_keys) {
        registry.upsert_op_dispatch(entry.register_for_op, key, kernel_name);
      }
    }
  }
}

void ShaderPackLoader::load_override_manifest(
    const std::string& manifest_path,
    bool clear_existing_overlay) {
  const std::filesystem::path manifest = manifest_path;
  std::ifstream in(manifest);
  VK_CHECK_COND(
      in.good(),
      "Failed to open shader override manifest: ",
      manifest.string());

  std::vector<ShaderPackKernelEntry> entries;
  std::string line;
  size_t line_no = 0u;
  while (std::getline(in, line)) {
    line_no++;
    std::string t = trim_ascii(line);
    if (t.empty() || t[0] == '#') {
      continue;
    }

    const size_t eq = t.find('=');
    VK_CHECK_COND(
        eq != std::string::npos,
        "Invalid manifest line ",
        line_no,
        " in ",
        manifest.string(),
        " (expected kernel=path).");

    const std::string kernel_name = trim_ascii(t.substr(0, eq));
    const std::string spv_value = trim_ascii(t.substr(eq + 1));
    VK_CHECK_COND(
        !kernel_name.empty(),
        "Empty kernel name on line ",
        line_no,
        " in ",
        manifest.string());
    VK_CHECK_COND(
        !spv_value.empty(),
        "Empty SPIR-V path on line ",
        line_no,
        " in ",
        manifest.string());

    std::filesystem::path spv_path = spv_value;
    if (!spv_path.is_absolute()) {
      spv_path = manifest.parent_path() / spv_path;
    }

    ShaderPackKernelEntry entry;
    entry.shader_info = shader_registry().get_shader_info(kernel_name);
    entry.shader_info.kernel_name = kernel_name;
    entry.spirv_words = read_spv_words(spv_path);
    entries.emplace_back(std::move(entry));
  }

  VK_CHECK_COND(
      !entries.empty(),
      "Shader override manifest has no valid entries: ",
      manifest.string());

  load(std::move(entries), clear_existing_overlay);
}

void ShaderPackLoader::clear() {
  shader_registry().clear_dynamic_overlay();
}

void load_shader_pack(
    std::vector<ShaderPackKernelEntry>&& entries,
    bool clear_existing_overlay) {
  ShaderPackLoader loader;
  loader.load(std::move(entries), clear_existing_overlay);
}

void load_shader_override_manifest(
    const std::string& manifest_path,
    bool clear_existing_overlay) {
  ShaderPackLoader loader;
  loader.load_override_manifest(manifest_path, clear_existing_overlay);
}

void clear_shader_pack_overlay() {
  ShaderPackLoader loader;
  loader.clear();
}

} // namespace api
} // namespace vkcompute
