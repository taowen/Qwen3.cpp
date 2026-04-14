/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/vk_api/Shader.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define VK_KERNEL(shader_name) \
  ::vkcompute::api::shader_registry().get_shader_info(#shader_name)

#define VK_KERNEL_FROM_STR(shader_name_str) \
  ::vkcompute::api::shader_registry().get_shader_info(shader_name_str)

namespace vkcompute {
namespace api {

enum class DispatchKey : int8_t {
  CATCHALL,
  ADRENO,
  MALI,
  OVERRIDE,
};

class ShaderRegistry final {
  using ShaderListing = std::unordered_map<std::string, vkapi::ShaderInfo>;
  using Dispatcher = std::unordered_map<DispatchKey, std::string>;
  using Registry = std::unordered_map<std::string, Dispatcher>;

  mutable std::mutex mutex_;

  ShaderListing aot_listings_;
  ShaderListing dynamic_listings_;
  // Owns dynamic shader binaries. Keep append-only so old ShaderInfo copies
  // remain valid even after upsert of the same kernel name.
  std::vector<std::vector<uint32_t>> dynamic_binary_pool_;

  Registry aot_registry_;
  Registry dynamic_registry_;

  static void upsert_dispatch_entry(
      Registry& registry,
      const std::string& op_name,
      const DispatchKey key,
      const std::string& shader_name);

 public:
  /*
   * Check if the registry has a shader registered under the given name
   */
  bool has_shader(const std::string& shader_name);

  /*
   * Check if the registry has a dispatch registered under the given name
   */
  bool has_dispatch(const std::string& op_name);

  /*
   * Register a ShaderInfo to a given shader name
   */
  void register_shader(vkapi::ShaderInfo&& shader_info);

  /*
   * Register or replace a shader in dynamic overlay namespace.
   * Should be used before building/running compute graphs that consume it.
   */
  void upsert_shader(vkapi::ShaderInfo&& shader_info);

  /*
   * Register or replace a shader in dynamic overlay namespace, owning SPIR-V.
   * The shader_info src_code pointer/size will be rewritten from `spv_words`.
   */
  void upsert_shader_with_owned_binary(
      vkapi::ShaderInfo&& shader_info,
      std::vector<uint32_t>&& spv_words);

  /*
   * Register a dispatch entry to the given op name
   */
  void register_op_dispatch(
      const std::string& op_name,
      const DispatchKey key,
      const std::string& shader_name);

  /*
   * Register or replace a dispatch entry in dynamic overlay namespace.
   */
  void upsert_op_dispatch(
      const std::string& op_name,
      const DispatchKey key,
      const std::string& shader_name);

  /*
   * Resolve op dispatch to shader name. Dynamic overlay takes precedence.
   * Returns true when resolved.
   */
  bool try_get_op_dispatch(
      const std::string& op_name,
      const DispatchKey key,
      std::string* out_shader_name);

  /*
   * Remove all dynamic overlay registrations.
   * Existing graph objects that copied dynamic ShaderInfo may become invalid.
   */
  void clear_dynamic_overlay();

  /*
   * Given a shader name, return the ShaderInfo which contains the SPIRV binary
   */
  vkapi::ShaderInfo get_shader_info(const std::string& shader_name);
};

class ShaderRegisterInit final {
  using InitFn = void();

 public:
  ShaderRegisterInit(InitFn* init_fn) {
    init_fn();
  };
};

// The global shader registry is retrieved using this function, where it is
// declared as a static local variable.
ShaderRegistry& shader_registry();

} // namespace api
} // namespace vkcompute
