#include "utils.metal"
#include <metal_stdlib>

using namespace metal;

// Scatter compact update rows into a slot-indexed GDN state pool, in place.
//
//   pool:    [num_slots, row_elems]   flattened; written in place
//   src:     [n, row_elems]           compact update rows
//   dst_ids: [n]                      destination slot for each update row
//
// A 2D thread grid: x walks the row, y selects the update. One threadgroup
// per row would leave the GPU idle -- a recurrent row is 16x128x128 floats,
// so a single 256-wide threadgroup would run 1024 scalar iterations while
// most of the machine sits unused.
//
// MLX's own indexed assignment cannot be used on this path because it donates
// the destination buffer only when it holds the sole reference to it, and a
// state pool is aliased into every sibling layer that shares it -- so each
// write rewrites the whole pool, which under align-mode prefix caching grows
// with cache occupancy.
//
// Destination slots must be distinct: two update rows naming the same slot
// would race.  Callers enforce that (pool siblings own disjoint block ids).
template <typename T>
[[kernel]] void gdn_state_scatter_rows(
    device T *pool [[buffer(0)]], const device T *src [[buffer(1)]],
    const device int *dst_ids [[buffer(2)]],
    device const int &row_elems [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int i = int(gid.x);
  if (i >= row_elems) {
    return;
  }
  const int64_t row = int64_t(gid.y);
  const int64_t dst = int64_t(dst_ids[gid.y]) * row_elems;
  pool[dst + i] = src[row * row_elems + i];
}

// Same, four elements per thread. Selected by the host when row_elems is a
// multiple of 4 -- true for every GDN state layout in practice (a conv row is
// (kernel-1)*conv_dim, a recurrent row is heads*d_v*d_k).
template <typename T>
[[kernel]] void gdn_state_scatter_rows_vec4(
    device T *pool [[buffer(0)]], const device T *src [[buffer(1)]],
    const device int *dst_ids [[buffer(2)]],
    device const int &row_vec4s [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int i = int(gid.x);
  if (i >= row_vec4s) {
    return;
  }
  const int64_t row = int64_t(gid.y);
  const int64_t dst = int64_t(dst_ids[gid.y]) * row_vec4s;
  reinterpret_cast<device vec<T, 4> *>(pool)[dst + i] =
      reinterpret_cast<const device vec<T, 4> *>(src)[row * row_vec4s + i];
}

#define instantiate_gdn_state_scatter_rows(type)                          \
  template [[host_name("gdn_state_scatter_rows_" #type)]] [[kernel]] void \
  gdn_state_scatter_rows<type>(                                           \
      device type *, const device type *, const device int *,             \
      device const int &, uint2);                                         \
  template [[host_name("gdn_state_scatter_rows_vec4_" #type)]] [[kernel]] \
  void gdn_state_scatter_rows_vec4<type>(                                 \
      device type *, const device type *, const device int *,             \
      device const int &, uint2);

instantiate_gdn_state_scatter_rows(float);
instantiate_gdn_state_scatter_rows(bfloat16_t);
instantiate_gdn_state_scatter_rows(half);
