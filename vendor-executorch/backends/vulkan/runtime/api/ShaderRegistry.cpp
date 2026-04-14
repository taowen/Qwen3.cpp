/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/ShaderRegistry.h>

#include <utility>

namespace vkcompute {
namespace api {

void ShaderRegistry::upsert_dispatch_entry(
    Registry& registry,
    const std::string& op_name,
    const DispatchKey key,
    const std::string& shader_name) {
  auto it = registry.find(op_name);
  if (it == registry.end()) {
    it = registry.emplace(op_name, Dispatcher()).first;
  }
  it->second[key] = shader_name;
}

bool ShaderRegistry::has_shader(const std::string& shader_name) {
  std::lock_guard<std::mutex> guard(mutex_);
  return dynamic_listings_.find(shader_name) != dynamic_listings_.end() ||
      aot_listings_.find(shader_name) != aot_listings_.end();
}

bool ShaderRegistry::has_dispatch(const std::string& op_name) {
  std::lock_guard<std::mutex> guard(mutex_);
  return dynamic_registry_.find(op_name) != dynamic_registry_.end() ||
      aot_registry_.find(op_name) != aot_registry_.end();
}

void ShaderRegistry::register_shader(vkapi::ShaderInfo&& shader_info) {
  std::lock_guard<std::mutex> guard(mutex_);
  const std::string shader_name = shader_info.kernel_name;
  if (
      aot_listings_.find(shader_name) != aot_listings_.end() ||
      dynamic_listings_.find(shader_name) != dynamic_listings_.end()) {
    VK_THROW(
        "Shader with name ", shader_name, "already registered");
  }
  aot_listings_.emplace(shader_name, std::move(shader_info));
}

void ShaderRegistry::upsert_shader(vkapi::ShaderInfo&& shader_info) {
  std::lock_guard<std::mutex> guard(mutex_);
  const std::string shader_name = shader_info.kernel_name;
  dynamic_listings_[shader_name] = std::move(shader_info);
}

void ShaderRegistry::upsert_shader_with_owned_binary(
    vkapi::ShaderInfo&& shader_info,
    std::vector<uint32_t>&& spv_words) {
  std::lock_guard<std::mutex> guard(mutex_);

  VK_CHECK_COND(
      !spv_words.empty(),
      "Cannot register dynamic shader ",
      shader_info.kernel_name,
      " with empty SPIR-V.");

  const std::string shader_name = shader_info.kernel_name;
  dynamic_binary_pool_.emplace_back(std::move(spv_words));
  const std::vector<uint32_t>& owned_words = dynamic_binary_pool_.back();
  shader_info.src_code.bin = owned_words.data();
  shader_info.src_code.size = static_cast<uint32_t>(4u * owned_words.size());

  dynamic_listings_[shader_name] = std::move(shader_info);
}

void ShaderRegistry::register_op_dispatch(
    const std::string& op_name,
    const DispatchKey key,
    const std::string& shader_name) {
  std::lock_guard<std::mutex> guard(mutex_);
  upsert_dispatch_entry(aot_registry_, op_name, key, shader_name);
}

void ShaderRegistry::upsert_op_dispatch(
    const std::string& op_name,
    const DispatchKey key,
    const std::string& shader_name) {
  std::lock_guard<std::mutex> guard(mutex_);
  upsert_dispatch_entry(dynamic_registry_, op_name, key, shader_name);
}

bool ShaderRegistry::try_get_op_dispatch(
    const std::string& op_name,
    const DispatchKey key,
    std::string* out_shader_name) {
  VK_CHECK_COND(out_shader_name != nullptr, "out_shader_name must not be null.");
  std::lock_guard<std::mutex> guard(mutex_);

  auto lookup_registry =
      [&](const Registry& registry,
          const std::string& op,
          const DispatchKey dispatch_key,
          std::string* out) -> bool {
    auto op_it = registry.find(op);
    if (op_it == registry.end()) {
      return false;
    }

    const Dispatcher& dispatch = op_it->second;
    auto key_it = dispatch.find(dispatch_key);
    if (key_it != dispatch.end()) {
      *out = key_it->second;
      return true;
    }

    auto catchall_it = dispatch.find(DispatchKey::CATCHALL);
    if (catchall_it != dispatch.end()) {
      *out = catchall_it->second;
      return true;
    }
    return false;
  };

  if (lookup_registry(dynamic_registry_, op_name, key, out_shader_name)) {
    return true;
  }
  return lookup_registry(aot_registry_, op_name, key, out_shader_name);
}

void ShaderRegistry::clear_dynamic_overlay() {
  std::lock_guard<std::mutex> guard(mutex_);
  dynamic_registry_.clear();
  dynamic_listings_.clear();
  // Intentionally keep dynamic_binary_pool_ allocated to avoid invalidating
  // src_code pointers that may still be referenced by already-built graphs.
}

vkapi::ShaderInfo ShaderRegistry::get_shader_info(
    const std::string& shader_name) {
  std::lock_guard<std::mutex> guard(mutex_);

  const ShaderListing::const_iterator dynamic_it =
      dynamic_listings_.find(shader_name);
  if (dynamic_it != dynamic_listings_.end()) {
    return dynamic_it->second;
  }

  const ShaderListing::const_iterator aot_it = aot_listings_.find(shader_name);
  VK_CHECK_COND(
      aot_it != aot_listings_.end(),
      "Could not find ShaderInfo with name ",
      shader_name);

  return aot_it->second;
}

ShaderRegistry& shader_registry() {
  static ShaderRegistry registry;
  return registry;
}

} // namespace api
} // namespace vkcompute
