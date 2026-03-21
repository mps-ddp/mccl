/**
 * MPS dispatch key registration for c10d collective ops.
 *
 * PyTorch 2.10 routes all c10d ops through the c10 Dispatcher, keyed by the
 * device type of the tensor arguments. Out of the box only CPU, CUDA, and
 * PrivateUse1 keys are registered. This file adds the MPS key so that
 * collectives on MPS tensors route to the MCCL backend.
 *
 * Each implementation is a thin shim that extracts the Backend for MPS from
 * the ProcessGroup and forwards to the corresponding virtual method — the
 * same pattern PyTorch uses internally for CPU/CUDA in Ops.cpp.
 */

#include <torch/library.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

namespace mccl {
namespace {

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>>
allreduce_MPS(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<c10d::ReduceOp>& reduce_op,
    const std::optional<at::Tensor>& sparse_indices,
    bool async_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  c10d::AllreduceOptions opts;
  opts.reduceOp = *reduce_op;
  opts.sparseIndices = sparse_indices;
  opts.asyncOp = async_op;
  if (timeout >= 0)
    opts.timeout = std::chrono::milliseconds(timeout);
  auto work =
      process_group->getBackend(c10::DeviceType::MPS)->allreduce(tensor_vec, opts);
  return {std::move(tensor_vec), work};
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>>
broadcast_MPS(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    bool async_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  c10d::BroadcastOptions opts;
  opts.rootRank = root_rank;
  opts.rootTensor = root_tensor;
  opts.asyncOp = async_op;
  if (timeout >= 0)
    opts.timeout = std::chrono::milliseconds(timeout);
  auto work =
      process_group->getBackend(c10::DeviceType::MPS)->broadcast(tensor_vec, opts);
  return {std::move(tensor_vec), work};
}

c10::intrusive_ptr<c10d::Work> barrier_MPS(
    at::Tensor /* tensor */,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const std::vector<int64_t>& device_ids,
    bool async_op,
    int64_t timeout) {
  c10d::BarrierOptions opts;
  opts.device_ids = device_ids;
  opts.asyncOp = async_op;
  if (timeout >= 0)
    opts.timeout = std::chrono::milliseconds(timeout);
  return process_group->getBackend(c10::DeviceType::MPS)->barrier(opts);
}

std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<c10d::Work>>
allgather_MPS(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    bool async_op,
    int64_t timeout) {
  auto output_vec = output_tensors;
  auto input_vec = input_tensors.vec();
  c10d::AllgatherOptions opts;
  opts.asyncOp = async_op;
  if (timeout >= 0)
    opts.timeout = std::chrono::milliseconds(timeout);
  auto work = process_group->getBackend(c10::DeviceType::MPS)
                  ->allgather(output_vec, input_vec, opts);
  return {std::move(output_vec), work};
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>>
reduce_scatter_MPS(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<c10d::ReduceOp>& reduce_op,
    bool async_op,
    int64_t timeout) {
  auto output_vec = output_tensors.vec();
  auto input_vec = input_tensors;
  c10d::ReduceScatterOptions opts;
  opts.reduceOp = *reduce_op;
  opts.asyncOp = async_op;
  if (timeout >= 0)
    opts.timeout = std::chrono::milliseconds(timeout);
  auto work = process_group->getBackend(c10::DeviceType::MPS)
                  ->reduce_scatter(output_vec, input_vec, opts);
  return {std::move(output_vec), work};
}

c10::intrusive_ptr<c10d::Work> send_MPS(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t dst_rank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::MPS)
      ->send(tensor_vec, static_cast<int>(dst_rank), static_cast<int>(tag));
}

c10::intrusive_ptr<c10d::Work> recv_MPS(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    int64_t src_rank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::MPS)
      ->recv(tensor_vec, static_cast<int>(src_rank), static_cast<int>(tag));
}

TORCH_LIBRARY_IMPL(c10d, MPS, m) {
  m.impl("allreduce_", allreduce_MPS);
  m.impl("broadcast_", broadcast_MPS);
  m.impl("barrier", barrier_MPS);
  m.impl("allgather_", allgather_MPS);
  m.impl("reduce_scatter_", reduce_scatter_MPS);
  m.impl("send", send_MPS);
  m.impl("recv_", recv_MPS);
}

} // anonymous namespace
} // namespace mccl
