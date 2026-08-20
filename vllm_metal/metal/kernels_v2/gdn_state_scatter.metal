#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

// Scatter compact update rows into a slot-indexed GDN state pool, in place.
//
//   pool:    [num_slots, row_elems]   flattened; written in place
//   src:     [n, row_elems]           compact update rows
//   dst_ids: [n]                      destination slot for each update row
//
// One threadgroup per update row, threads striding the row.  Cost is
// O(n * row_elems).  MLX's own indexed assignment cannot be used on this path
// because it donates the destination buffer only when it holds the sole
// reference to it, and a state pool is aliased into every sibling layer that
// shares it -- so each write rewrites the whole pool, which under align-mode
// prefix caching grows with cache occupancy.
//
// Destination slots must be distinct: two update rows naming the same slot
// would race.  Callers enforce that (pool siblings own disjoint block ids).
template <typename T>
[[kernel]] void gdn_state_scatter_rows(
    device T *pool [[buffer(0)]], const device T *src [[buffer(1)]],
    const device int *dst_ids [[buffer(2)]],
    device const int &row_elems [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]) {
  const int64_t src_off = static_cast<int64_t>(gid) * row_elems;
  const int64_t dst_off = static_cast<int64_t>(dst_ids[gid]) * row_elems;
  for (int i = tid; i < row_elems; i += threads_per_threadgroup) {
    pool[dst_off + i] = src[src_off + i];
  }
}

#define instantiate_gdn_state_scatter_rows(type)                          \
  template [[host_name("gdn_state_scatter_rows_" #type)]] [[kernel]] void \
  gdn_state_scatter_rows<type>(                                           \
      device type *, const device type *, const device int *,             \
      device const int &, uint, uint, uint);

instantiate_gdn_state_scatter_rows(float);
instantiate_gdn_state_scatter_rows(bfloat16_t);
instantiate_gdn_state_scatter_rows(half);
