/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/ShaderRegistry.h>

#include <string>
#include <vector>

namespace vkcompute {
namespace api {

struct ShaderPackKernelEntry final {
  vkapi::ShaderInfo shader_info;
  std::vector<uint32_t> spirv_words;

  // Optional op dispatch registrations for this shader.
  std::string register_for_op;
  std::vector<DispatchKey> dispatch_keys;
};

class ShaderPackLoader final {
 public:
  ShaderPackLoader() = default;
  ~ShaderPackLoader() = default;

  /*
   * Load a shader pack into dynamic overlay namespace.
   * If clear_existing_overlay is true, existing dynamic entries are removed.
   */
  void load(
      std::vector<ShaderPackKernelEntry>&& entries,
      bool clear_existing_overlay = true);

  /*
   * Load dynamic shader overrides from a simple manifest file.
   *
   * Manifest format:
   *   - UTF-8 text
   *   - Empty lines and lines starting with '#' are ignored
   *   - Each non-empty line is: <kernel_name>=<spv_path>
   *   - spv_path can be absolute, or relative to manifest directory
   *
   * The loader clones the currently registered ShaderInfo metadata for
   * <kernel_name> and only replaces src_code with the dynamic SPIR-V.
   */
  void load_override_manifest(
      const std::string& manifest_path,
      bool clear_existing_overlay = true);

  /*
   * Remove all dynamic overlay registrations.
   */
  void clear();
};

/*
 * Convenience helpers for bindings/callers that don't need a loader instance.
 */
void load_shader_pack(
    std::vector<ShaderPackKernelEntry>&& entries,
    bool clear_existing_overlay = true);

void load_shader_override_manifest(
    const std::string& manifest_path,
    bool clear_existing_overlay = true);

void clear_shader_pack_overlay();

} // namespace api
} // namespace vkcompute
