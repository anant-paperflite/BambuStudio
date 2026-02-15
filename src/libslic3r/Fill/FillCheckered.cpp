#include "FillCheckered.hpp"

#include "../AABBTreeIndirect.hpp"
#include "../Format/OBJ.hpp"
#include "../Point.hpp"
#include "../SVG.hpp"
#include "../TriangleMesh.hpp"
#include "../Utils.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

// Enable to write debug SVGs (XY segments and UV-space segments) to
// g_data_dir/SVG/
#define CHECKERED_INFILL_DEBUG_SVG

namespace Slic3r {

namespace {

// Default grid resolution for UV space [0,1]^2
constexpr int DEFAULT_GRID_COLS = 12;
constexpr int DEFAULT_GRID_ROWS = 12;

struct CachedUVMesh {
  TriangleMesh mesh;
  std::vector<std::array<Vec2f, 3>> uvs;
  AABBTreeIndirect::Tree<3, float> tree;
  bool valid{false};
  Vec3f bbox_min{0.f, 0.f, 0.f};
  Vec3f bbox_max{0.f, 0.f, 0.f};

  static std::optional<CachedUVMesh> load(const std::string &path) {
    if (path.empty()) {
      printf("FillCheckered: UV map path empty, skip load\n");
      return std::nullopt;
    }
    printf("FillCheckered: loading UV map from %s\n", path.c_str());
    TriangleMesh mesh;
    ObjInfo obj_info;
    std::string message;
    if (!load_obj(path.c_str(), &mesh, obj_info, message, false)) {
      printf("FillCheckered: load_obj failed: %s\n", message.c_str());
      return std::nullopt;
    }
    if (mesh.empty()) {
      printf("FillCheckered: UV map mesh is empty\n");
      return std::nullopt;
    }
    if (obj_info.uvs.size() != mesh.its.indices.size()) {
      printf("FillCheckered: UV count %zu != triangle count %zu, need one UV "
             "per face\n",
             obj_info.uvs.size(), mesh.its.indices.size());
      return std::nullopt;
    }
    AABBTreeIndirect::Tree<3, float> tree =
        AABBTreeIndirect::build_aabb_tree_over_indexed_triangle_set(
            mesh.its.vertices, mesh.its.indices);
    CachedUVMesh out;
    out.mesh = std::move(mesh);
    out.uvs = std::move(obj_info.uvs);
    out.tree = std::move(tree);
    out.valid = true;
    const auto &v = out.mesh.its.vertices;
    if (!v.empty()) {
      Vec3f mn = v.front(), mx = v.front();
      for (const Vec3f &p : v) {
        mn = mn.cwiseMin(p);
        mx = mx.cwiseMax(p);
      }
      out.bbox_min = mn;
      out.bbox_max = mx;
      printf("FillCheckered: UV mesh bbox mm: x[%.3f,%.3f] y[%.3f,%.3f] "
             "z[%.3f,%.3f]\n",
             mn.x(), mx.x(), mn.y(), mx.y(), mn.z(), mx.z());
    }
    printf("FillCheckered: UV map loaded, %zu triangles\n",
           out.mesh.its.indices.size());
    return out;
  }
};

static std::mutex s_cache_mutex;
static std::map<std::string, std::shared_ptr<CachedUVMesh>> s_uv_cache;

std::shared_ptr<CachedUVMesh> get_or_load_uv_mesh(const std::string &path) {
  if (path.empty())
    return nullptr;
  std::lock_guard<std::mutex> lock(s_cache_mutex);
  auto it = s_uv_cache.find(path);
  if (it != s_uv_cache.end()) {
    printf("FillCheckered: UV map cache hit for %s\n", path.c_str());
    return it->second;
  }
  auto opt = CachedUVMesh::load(path);
  if (!opt)
    return nullptr;
  auto ptr = std::make_shared<CachedUVMesh>(std::move(*opt));
  s_uv_cache[path] = ptr;
  return ptr;
}

// Barycentric coordinates of point Q in triangle (A, B, C): Q = w0*A + w1*B +
// w2*C. Uses 3D dot-product method (Real-time collision detection, Ericson).
// Returns nullopt if degenerate.
static std::optional<std::array<float, 3>>
barycentric_coords_3d(const Vec3d &Q, const Vec3d &A, const Vec3d &B,
                      const Vec3d &C) {
  Vec3d v1 = B - A;
  Vec3d v2 = C - A;
  Vec3d v3 = Q - A;
  double d00 = v1.dot(v1);
  double d01 = v1.dot(v2);
  double d11 = v2.dot(v2);
  double d20 = v3.dot(v1);
  double d21 = v3.dot(v2);
  double denom = d00 * d11 - d01 * d01;
  const double eps = 1e-12;
  if (std::abs(denom) < eps)
    return std::nullopt;
  double s = (d11 * d20 - d01 * d21) / denom;
  double t = (d00 * d21 - d01 * d20) / denom;
  double w0 = 1.0 - s - t;
  return std::array<float, 3>{float(w0), float(s), float(t)};
}

static Point point_to_model_surface_mm(const CachedUVMesh &cache, Point p, double outward_offset_mm) {
  double x_mm = unscale_(p.x());
  double y_mm = unscale_(p.y());

  double bbox_center_x = (cache.bbox_min.x() + cache.bbox_max.x()) / 2;
  double bbox_center_y = (cache.bbox_min.y() + cache.bbox_max.y()) / 2;

  if (x_mm < 0) {
    x_mm = x_mm - outward_offset_mm;
  } else {
    x_mm = x_mm + outward_offset_mm;
  }

  if (y_mm < 0) {
    y_mm = y_mm - outward_offset_mm;
  } else {
    y_mm = y_mm + outward_offset_mm;
  }

  x_mm = bbox_center_x + x_mm;
  y_mm = bbox_center_y + y_mm;

  return Point(x_mm, y_mm);
}

static Point point_mm_to_model_surface(const CachedUVMesh &cache, double x_mm, double y_mm, double outward_offset_mm) {
  double bbox_center_x = (cache.bbox_min.x() + cache.bbox_max.x()) / 2;
  double bbox_center_y = (cache.bbox_min.y() + cache.bbox_max.y()) / 2;

  x_mm = x_mm - bbox_center_x;
  y_mm = y_mm - bbox_center_y;

  if (x_mm < 0) {
    x_mm = x_mm + outward_offset_mm;
  } else {
    x_mm = x_mm - outward_offset_mm;
  }

  if (y_mm < 0) {
    y_mm = y_mm + outward_offset_mm;
  } else {
    y_mm = y_mm - outward_offset_mm;
  }

  return Point::new_scale(x_mm, y_mm);
}

// Map 3D point (x,y,z) in mm to (u,v) in [0,1]^2 by finding which face the
// point lies on, then interpolating that face's UV map. Returns nullopt if
// point is not on mesh surface. If the UV mesh has Z in [-H, 0] (e.g. top=0,
// bottom negative), we use -z_mm for query.
// When next_xyz_mm is provided and the hit lies on a seam (u or v at 0 or 1),
// we raycast the next point and canonicalize UV so the returned value is on the
// same "side" of the seam as the direction toward the next point.
static std::optional<Vec2f> point_to_uv(const CachedUVMesh &cache, double x_mm,
                                        double y_mm, double z_mm,
                                        std::optional<Vec3d> next_xyz_mm = std::nullopt) {
  const indexed_triangle_set &its = cache.mesh.its;
  if (its.vertices.empty() || its.indices.empty() ||
      cache.uvs.size() != its.indices.size()) {
    printf("FillCheckered: point_to_uv: vertices empty or indices empty or uvs "
           "size != indices size: %zu %zu %zu\n",
           its.vertices.size(), its.indices.size(), cache.uvs.size());
    return std::nullopt;
  }
  double pz = (cache.bbox_max.z() <= 0.f) ? -z_mm : z_mm;
  Vec3d P(x_mm, y_mm, pz);

  size_t hit_idx = 0;
  Vec3d hit_point;
  double sqr_dist = AABBTreeIndirect::squared_distance_to_indexed_triangle_set(
      its.vertices, its.indices, cache.tree, P, hit_idx, hit_point);

  const double epsilon_sq = 1e-6; // mm^2
  if (sqr_dist < 0 || sqr_dist > epsilon_sq) {
    printf("No hit\n");
    return std::nullopt;
  }

  if (hit_idx >= cache.uvs.size()) {
    printf("Hit index out of bounds\n");
    return std::nullopt;
  }

  const Vec3i &face = its.indices[hit_idx];
  Vec3d A = its.vertices[face(0)].cast<double>();
  Vec3d B = its.vertices[face(1)].cast<double>();
  Vec3d C = its.vertices[face(2)].cast<double>();
  auto bary = barycentric_coords_3d(hit_point, A, B, C);
  if (!bary)
    return std::nullopt;

  float w0 = (*bary)[0], w1 = (*bary)[1], w2 = (*bary)[2];
  const std::array<Vec2f, 3> &uv_arr = cache.uvs[hit_idx];

  float u = w0 * uv_arr[0].x() + w1 * uv_arr[1].x() + w2 * uv_arr[2].x();
  float v = w0 * uv_arr[0].y() + w1 * uv_arr[1].y() + w2 * uv_arr[2].y();
  u = std::clamp(u, 0.f, 1.f);
  v = std::clamp(v, 0.f, 1.f);

  constexpr float seam_eps = 1e-6f;
  const bool on_u_seam = (u <= seam_eps || u >= 1.f - seam_eps);
  const bool on_v_seam = (v <= seam_eps || v >= 1.f - seam_eps);
  if ((on_u_seam || on_v_seam) && next_xyz_mm) {
    std::optional<Vec2f> next_uv = point_to_uv(cache, next_xyz_mm->x(), next_xyz_mm->y(), next_xyz_mm->z());
    if (next_uv) {
      if (on_u_seam) {
        if (u >= 1.f - seam_eps && next_uv->x() < 0.5f)
          u = 0.f;
        else if (u <= seam_eps && next_uv->x() > 0.5f)
          u = 1.f;
      }
      if (on_v_seam) {
        if (v >= 1.f - seam_eps && next_uv->y() < 0.5f)
          v = 0.f;
        else if (v <= seam_eps && next_uv->y() > 0.5f)
          v = 1.f;
      }
    }
  }

  return Vec2f(u, v);
}

// Barycentric coordinates of point P in 2D triangle (A, B, C): P = w0*A + w1*B + w2*C.
// Returns nullopt if degenerate or P is outside the triangle.
static std::optional<std::array<float, 3>>
barycentric_coords_2d(const Vec2f &P, const Vec2f &A, const Vec2f &B, const Vec2f &C) {
  float v0x = B.x() - A.x();
  float v0y = B.y() - A.y();
  float v1x = C.x() - A.x();
  float v1y = C.y() - A.y();
  float v2x = P.x() - A.x();
  float v2y = P.y() - A.y();
  float d00 = v0x * v0x + v0y * v0y;
  float d01 = v0x * v1x + v0y * v1y;
  float d11 = v1x * v1x + v1y * v1y;
  float d20 = v2x * v0x + v2y * v0y;
  float d21 = v2x * v1x + v2y * v1y;
  float denom = d00 * d11 - d01 * d01;
  const float eps_denom = 1e-12f;
  if (std::abs(denom) < eps_denom){
    printf("Barycentric coords 2d: degenerate triangle\n");
    return std::nullopt;
  }
  float s = (d11 * d20 - d01 * d21) / denom;
  float t = (d00 * d21 - d01 * d20) / denom;
  float w0 = 1.f - s - t;

  // Inside triangle iff all barycentrics in [0, 1]. Use tolerance for float
  // rounding: point_to_uv -> uv_to_point round-trip can yield s/w0/t just
  // outside [0,1] (e.g. s=-1e-7, w0=-3e-8); accept within 1e-5.
  const float eps_inside = 1e-5f;
  if (w0 < -eps_inside || w0 > 1.f + eps_inside || s < -eps_inside || s > 1.f + eps_inside || t < -eps_inside || t > 1.f + eps_inside)
    return std::nullopt;

  return std::array<float, 3>{w0, s , t};
}

// Map (u, v) in [0,1]^2 to a 3D point on the mesh surface (mm, same frame as mesh).
// Finds the first triangle whose UV triangle contains (u,v) and interpolates
// the 3D position. Returns nullopt if no triangle contains (u,v).
static std::optional<Vec3d> uv_to_point(const CachedUVMesh &cache, float u, float v) {
  const indexed_triangle_set &its = cache.mesh.its;
  if (its.vertices.empty() || its.indices.empty() ||
      cache.uvs.size() != its.indices.size()){
    printf("UV to point: vertices empty or indices empty or uvs size != indices size: %zu %zu %zu\n",
           its.vertices.size(), its.indices.size(), cache.uvs.size());
    return std::nullopt;
  }

  u = std::clamp(u, 0.f, 1.f);
  v = std::clamp(v, 0.f, 1.f);
  Vec2f P(u, v);
  for (size_t i = 0; i < cache.uvs.size(); ++i) {
    const std::array<Vec2f, 3> &uv_arr = cache.uvs[i];
    auto bary = barycentric_coords_2d(P, uv_arr[0], uv_arr[1], uv_arr[2]);
    if (!bary)
      continue;
    const Vec3i &face = its.indices[i];
    Vec3f V0 = its.vertices[face(0)];
    Vec3f V1 = its.vertices[face(1)];
    Vec3f V2 = its.vertices[face(2)];
    float w0 = (*bary)[0], w1 = (*bary)[1], w2 = (*bary)[2];

    double x = w0 * double(V0.x()) + w1 * double(V1.x()) + w2 * double(V2.x());
    double y = w0 * double(V0.y()) + w1 * double(V1.y()) + w2 * double(V2.y());
    double z = w0 * double(V0.z()) + w1 * double(V1.z()) + w2 * double(V2.z());

    return Vec3d(x, y, z);
  }
  printf("UV to point: no triangle contains (u,v)\n");
  return std::nullopt;
}

// Map 3D point (x_mm, y_mm, z_mm) to UV grid cell (i, j) using the cached UV
// mesh.
std::optional<std::pair<int, int>>
point_to_grid_cell(const CachedUVMesh &cache, double x_mm, double y_mm,
                   double z_mm, int grid_cols, int grid_rows) {
  auto uv = point_to_uv(cache, x_mm, y_mm, z_mm);
  if (!uv)
    return std::nullopt;
  float u = uv->x(), v = uv->y();
  int i = static_cast<int>(std::floor(u * grid_cols));
  int j = static_cast<int>(std::floor((v) * grid_rows));
  
  i = std::clamp(i, 0, grid_cols - 1);
  j = std::clamp(j, 0, grid_rows - 1);

  return std::make_pair(i, j);
}

// Epsilon for "point on grid edge" detection.
constexpr float UV_GRID_EDGE_EPS = 1e-6f;

// Returns true when (u,v) lies on a vertical grid line u=k/grid_cols, horizontal
// grid line v=m/grid_rows, or on domain boundary u<=eps, u>=1-eps, v<=eps, v>=1-eps.
static bool is_uv_on_grid_edge(float u, float v, int grid_cols, int grid_rows) {
  if (u <= UV_GRID_EDGE_EPS || u >= 1.f - UV_GRID_EDGE_EPS)
    return true;
  if (v <= UV_GRID_EDGE_EPS || v >= 1.f - UV_GRID_EDGE_EPS)
    return true;
  float fu = u * float(grid_cols);
  float fv = v * float(grid_rows);
  if (std::abs(fu - std::round(fu)) <= UV_GRID_EDGE_EPS)
    return true;
  if (std::abs(fv - std::round(fv)) <= UV_GRID_EDGE_EPS)
    return true;
  return false;
}

// Map UV point (u, v) in [0,1]^2 to grid cell (i, j). u=1/v=1 map to last cell.
// When (u,v) is on a grid edge and (dir_u, dir_v) is non-zero, step by a minimal
// amount along the direction and use that point's cell to disambiguate.
// Seam (0 vs 1) is canonicalized in point_to_uv when next point is provided;
// here we only step to pick the cell the segment actually enters.
static std::pair<int, int> uv_to_grid_cell(float u, float v, int grid_cols,
                                            int grid_rows, float dir_u = 0.f,
                                            float dir_v = 0.f) {
  constexpr float step_eps = 1e-6f;
  constexpr float dir_eps = 1e-12f;
  bool has_direction = (std::abs(dir_u) > dir_eps || std::abs(dir_v) > dir_eps);
  if (has_direction && is_uv_on_grid_edge(u, v, grid_cols, grid_rows)) {
    u = u + step_eps * dir_u;
    v = v + step_eps * dir_v;
    u = std::clamp(u, 0.f, 1.f);
    v = std::clamp(v, 0.f, 1.f);
  }
  int i = static_cast<int>(std::floor(u * grid_cols));
  int j = static_cast<int>(std::floor(v * grid_rows));
  if (i >= grid_cols)
    i = grid_cols - 1;
  if (j >= grid_rows)
    j = grid_rows - 1;
  i = std::clamp(i, 0, grid_cols - 1);
  j = std::clamp(j, 0, grid_rows - 1);
  return {i, j};
}

// Black grid cell is the opposite of fill (checkered pattern)
inline bool is_black_cell(int i, int j) { return (i + j) % 2 == 0; }

// Outward unit normal at contour vertex i (CCW contour: interior to the left,
// so outward = right of edge). Returns (0,0) if contour too small or
// degenerate.
static Vec2d outward_unit_normal(const Polygon &contour, size_t i) {
  const size_t n = contour.points.size();
  if (n < 2)
    return Vec2d(0., 0.);
  const size_t prev = (i + n - 1) % n;
  const size_t next = (i + 1) % n;
  Vec2d e1 = (contour.points[i] - contour.points[prev]).cast<double>();
  Vec2d e2 = (contour.points[next] - contour.points[i]).cast<double>();
  // Right of edge = outward for CCW: (edge.y(), -edge.x())
  Vec2d n1(e1.y(), -e1.x());
  Vec2d n2(e2.y(), -e2.x());
  double l1 = n1.norm(), l2 = n2.norm();
  if (l1 < 1e-10 && l2 < 1e-10)
    return Vec2d(0., 0.);
  Vec2d out = (l1 >= 1e-10 ? n1 / l1 : Vec2d(0., 0.)) +
              (l2 >= 1e-10 ? n2 / l2 : Vec2d(0., 0.));
  double L = out.norm();
  return L >= 1e-10 ? out / L : Vec2d(0., 0.);
}

// Per-point result: where the contour point lies in UV space and whether that
// cell is black.
struct ContourPointUVInfo {
  Point point;                                  // contour point (XY)
  std::optional<Vec2f> uv;                      // UV in [0,1]^2 if ray hit
  std::optional<std::pair<int, int>> grid_cell; // (i, j) if uv is valid
  bool is_black_cell{false}; // true iff grid_cell is set and that cell is black
};

// Map every point on a contour to UV and detect if it lies on a black grid
// cell. origin_x_mm, origin_y_mm: added to contour mm so ray is in UV mesh (raw
// model) coords; contour is object-centered. outward_offset_mm: move ray origin
// outward so it lies on the mesh face (contour is inset by fill offset).
std::vector<ContourPointUVInfo>
get_contour_points_uv_info(const Polygon &contour, double z_mm,
                           const CachedUVMesh &cache, int grid_cols,
                           int grid_rows, double origin_x_mm,
                           double origin_y_mm, double outward_offset_mm) {
  std::vector<ContourPointUVInfo> out;
  out.reserve(contour.points.size());
  const size_t n_pts = contour.points.size();
  for (size_t i = 0; i < n_pts; ++i) {
    const Point &pt = contour.points[i];
    double x_mm = unscale_(pt.x()) + origin_x_mm;
    double y_mm = unscale_(pt.y()) + origin_y_mm;
    if (outward_offset_mm > 0.) {
      Vec2d n = outward_unit_normal(contour, i);
      x_mm += outward_offset_mm * n.x();
      y_mm += outward_offset_mm * n.y();
    }
    std::optional<Vec3d> next_mm;
    const Point &next_pt = contour.points[(i + 1) % n_pts];
    double nx_mm = unscale_(next_pt.x()) + origin_x_mm;
    double ny_mm = unscale_(next_pt.y()) + origin_y_mm;
    if (outward_offset_mm > 0.) {
      Vec2d n_next = outward_unit_normal(contour, (i + 1) % n_pts);
      nx_mm += outward_offset_mm * n_next.x();
      ny_mm += outward_offset_mm * n_next.y();
    }
    next_mm = Vec3d(nx_mm, ny_mm, z_mm);
    std::optional<Vec2f> uv = point_to_uv(cache, x_mm, y_mm, z_mm, next_mm);
    ContourPointUVInfo info;
    info.point = pt;
    if (uv) {
      info.uv = *uv;
      auto cell = uv_to_grid_cell(uv->x(), uv->y(), grid_cols, grid_rows);
      info.grid_cell = cell;
      info.is_black_cell = is_black_cell(cell.first, cell.second);
    }
    out.push_back(std::move(info));
  }
  return out;
}

// Cell bounds in UV [0,1]^2 for cell (i, j).
static void cell_bounds(int i, int j, int grid_cols, int grid_rows,
                        float &u_min, float &u_max, float &v_min,
                        float &v_max) {
  u_min = float(i) / float(grid_cols);
  u_max = float(i + 1) / float(grid_cols);
  v_min = float(j) / float(grid_rows);
  v_max = float(j + 1) / float(grid_rows);
  u_min = std::clamp(u_min, 0.f, 1.f);
  u_max = std::clamp(u_max, 0.f, 1.f);
  v_min = std::clamp(v_min, 0.f, 1.f);
  v_max = std::clamp(v_max, 0.f, 1.f);
}

// Intersect segment a->b (UV) with a vertical line u = u_edge, segment from
// (u_edge, v_lo) to (u_edge, v_hi). Returns t in [0,1] if hit, else nullopt. t
// is parameter for a + t*(b-a).
static std::optional<float> segment_intersect_vertical(
  float u_edge, float v_lo,
  float v_hi,
  const Vec2f &a,
  const Vec2f &b) {

  float du = b.x() - a.x();
  if (std::abs(du) < 1e-9f){
    return std::nullopt;
  }

  float t = (u_edge - a.x()) / du;
  if (t < 0.f || t > 1.f){
    return std::nullopt;
  }

  float v = a.y() + t * (b.y() - a.y());
  if (v < v_lo || v > v_hi){
    return std::nullopt;
  }

  return t;
}

// Intersect segment a->b (UV) with a horizontal line v = v_edge, segment from
// (u_lo, v_edge) to (u_hi, v_edge).
static std::optional<float> segment_intersect_horizontal(
  float u_lo, float u_hi,
  float v_edge,
  const Vec2f &a,
  const Vec2f &b) {

  float dv = b.y() - a.y();
  if (std::abs(dv) < 1e-9f)
    return std::nullopt;
  float t = (v_edge - a.y()) / dv;
  if (t < 0.f || t > 1.f)
    return std::nullopt;
  float u = a.x() + t * (b.x() - a.x());
  if (u < u_lo || u > u_hi)
    return std::nullopt;
  return t;
}

// Epsilon for "point on cell edge" so we consistently treat boundary points.
constexpr float UV_EDGE_EPS = 1e-6f;

// Exit parameter: smallest t in (0, 1] where segment a->b (UV) exits cell (ci,
// cj), or 1 if b is inside. Returns (t, exit_edge): exit_edge 0=left, 1=right,
// 2=bottom, 3=top (for adjacent cell).
// When a lies on a cell edge (or corner), we may return (0, edge) so the walk
// immediately steps to the adjacent cell; the exit edge is chosen using the
// segment direction (b - a) so we step into the cell the segment is heading toward.
static std::pair<float, int> segment_exit_cell(const Vec2f &a, const Vec2f &b,
                                               int ci, int cj, int grid_cols,
                                               int grid_rows) {
  float u_min, u_max, v_min, v_max;
  cell_bounds(ci, cj, grid_cols, grid_rows, u_min, u_max, v_min, v_max);
  const float du = b.x() - a.x();
  const float dv = b.y() - a.y();

  // If start is on a cell boundary, decide if we should "exit" immediately (t=0)
  // so the walk doesn't get stuck. Use segment direction: outward normal dot
  // (du,dv) > 0 means we leave through that edge. Pick the edge we're on that
  // the segment exits through (at corners, pick the one most aligned with direction).
  {
    const bool on_left   = (a.x() <= u_min + UV_EDGE_EPS);
    const bool on_right  = (a.x() >= u_max - UV_EDGE_EPS);
    const bool on_bottom = (a.y() <= v_min + UV_EDGE_EPS);
    const bool on_top    = (a.y() >= v_max - UV_EDGE_EPS);
    if (on_left || on_right || on_bottom || on_top) {
      // Outward normals: left (-1,0), right (1,0), bottom (0,-1), top (0,1).
      int best_edge = -1;
      float best_dot = 0.f;
      if (on_left   && -du > best_dot) { best_dot = -du;   best_edge = 0; }
      if (on_right  &&  du > best_dot) { best_dot =  du;   best_edge = 1; }
      if (on_bottom && -dv > best_dot) { best_dot = -dv;   best_edge = 2; }
      if (on_top    &&  dv > best_dot) { best_dot =  dv;   best_edge = 3; }
      if (best_edge >= 0)
        return {0.f, best_edge};
    }
  }

  // Ignore intersections with t very close to 0: when the segment start is
  // just outside a cell edge (float noise), we'd otherwise "exit" through that
  // edge immediately and the walk can break. Use a small minimum t so we take
  // the real exit (e.g. right edge) instead.
  constexpr float t_exit_min = 1e-5f;
  float t_best = 1.f;
  int edge_best = -1;
  auto consider = [&](std::optional<float> t, int edge) {
    if (t && *t >= t_exit_min && *t < t_best) {
      t_best = *t;
      edge_best = edge;
    }
  };

  consider(segment_intersect_vertical(u_min, v_min, v_max, a, b), 0);   // left
  consider(segment_intersect_vertical(u_max, v_min, v_max, a, b), 1);   // right
  consider(segment_intersect_horizontal(u_min, u_max, v_min, a, b), 2); // bottom
  consider(segment_intersect_horizontal(u_min, u_max, v_max, a, b), 3); // top

  if (edge_best < 0)
    return {1.f, -1};

  return {t_best, edge_best};
}

// New cell when exiting through edge: 0=left, 1=right, 2=bottom, 3=top.
// When next cell exceeds max row/col, wrap to 0; when below 0, wrap to max.
static std::pair<int, int> adjacent_cell(int ci, int cj, int edge,
                                         int grid_cols, int grid_rows) {
  int ni = ci, nj = cj;
  if (edge == 0)
    ni = ci - 1;
  else if (edge == 1)
    ni = ci + 1;
  else if (edge == 2)
    nj = cj - 1;
  else if (edge == 3)
    nj = cj + 1;
  
  ni = std::clamp(ni, 0, grid_cols - 1);
  nj = std::clamp(nj, 0, grid_rows - 1);

  return {ni, nj};
}

// Adjacent cell clamped to a segment's min/max cell range (so the walk stays
// within the bounding box of the two endpoint cells).
static std::pair<int, int> adjacent_cell_bounded(int ci, int cj, int edge,
                                                  int min_ci, int max_ci,
                                                  int min_cj, int max_cj) {
  int ni = ci, nj = cj;
  if (edge == 0)
    ni = ci - 1;
  else if (edge == 1)
    ni = ci + 1;
  else if (edge == 2)
    nj = cj - 1;
  else if (edge == 3)
    nj = cj + 1;
  ni = std::clamp(ni, min_ci, max_ci);
  nj = std::clamp(nj, min_cj, max_cj);
  return {ni, nj};
}

// One segment of a UV line that lies entirely inside a single grid cell.
struct UVSegmentInCell {
  Vec2f start_uv;
  Vec2f end_uv;
  int ci{0};
  int cj{0};
};

// Subdivide UV segment a->b into segments per grid cell. Each returned segment
// is the portion of the line inside one cell, in order along the line.
static std::vector<UVSegmentInCell> subdivide_uv_segment_by_grid(
    const Vec2f &a, const Vec2f &b, int grid_cols, int grid_rows) {
  std::vector<UVSegmentInCell> out;
  float du = b.x() - a.x(), dv = b.y() - a.y();
  auto [ci, cj] = uv_to_grid_cell(a.x(), a.y(), grid_cols, grid_rows, du, dv);
  auto [bi, bj] = uv_to_grid_cell(b.x(), b.y(), grid_cols, grid_rows, -du, -dv);
  const int min_ci = std::min(ci, bi);
  const int max_ci = std::max(ci, bi);
  const int min_cj = std::min(cj, bj);
  const int max_cj = std::max(cj, bj);
  Vec2f current_uv = a;
  constexpr float t_done_eps = 1e-6f;

  for (;;) {
    auto [t, exit_edge] =
        segment_exit_cell(current_uv, b, ci, cj, grid_cols, grid_rows);
    const float end_u = current_uv.x() + t * (b.x() - current_uv.x());
    const float end_v = current_uv.y() + t * (b.y() - current_uv.y());
    Vec2f end_uv(end_u, end_v);
    out.push_back({current_uv, end_uv, ci, cj});

    if (t >= 1.f - t_done_eps)
      break;

    current_uv = end_uv;
    auto next =
        adjacent_cell_bounded(ci, cj, exit_edge, min_ci, max_ci, min_cj, max_cj);
    if (next.first == ci && next.second == cj)
      break;
    ci = next.first;
    cj = next.second;
  }
  return out;
}

// Extract contour segments that lie in black grid cells by clipping each edge
// to UV cells. origin_x_mm, origin_y_mm: contour is object-centered; add to get
// ray in UV mesh (raw model) coords. outward_offset_mm: move ray origin outward
// so it lies on the mesh face.
static Polylines
extract_black_contour_segments(const Polygon &contour, double z_mm,
                               const CachedUVMesh &cache, int grid_cols,
                               int grid_rows, double origin_x_mm,
                               double origin_y_mm, double outward_offset_mm) {
  Polylines result;
  const size_t n = contour.points.size();
  if (n == 0)
    return result;

  std::vector<std::pair<Point, Point>> segments;

  for (size_t k = 0; k < n; ++k) {
    const Point A_xy = point_to_model_surface_mm(cache, contour.points[k], outward_offset_mm);
    const Point B_xy = point_to_model_surface_mm(cache, contour.points[(k + 1) % n], outward_offset_mm);
    double ax_mm = A_xy.x();
    double ay_mm = A_xy.y();
    double bx_mm = B_xy.x();
    double by_mm = B_xy.y();
    
    auto A_uv = point_to_uv(cache, ax_mm, ay_mm, z_mm, Vec3d(bx_mm, by_mm, z_mm));
    auto B_uv = point_to_uv(cache, bx_mm, by_mm, z_mm, Vec3d(ax_mm, ay_mm, z_mm));
    if (!A_uv || !B_uv){
      printf("No UV hit\n");
      continue;
    }

    float du = B_uv->x() - A_uv->x(), dv = B_uv->y() - A_uv->y();
    auto cell_A = uv_to_grid_cell(A_uv->x(), A_uv->y(), grid_cols, grid_rows, du, dv);
    auto cell_B = uv_to_grid_cell(B_uv->x(), B_uv->y(), grid_cols, grid_rows, -du, -dv);
    const int min_ci = std::min(cell_A.first, cell_B.first);
    const int max_ci = std::max(cell_A.first, cell_B.first);
    const int min_cj = std::min(cell_A.second, cell_B.second);
    const int max_cj = std::max(cell_A.second, cell_B.second);

    printf("A xy: %d, %d, B xy: %d, %d\n", A_xy.x(), A_xy.y(), B_xy.x(), B_xy.y());
    printf("A cell: %d, %d, B cell: %d, %d\n", cell_A.first, cell_A.second, cell_B.first, cell_B.second);

    int ci = cell_A.first, cj = cell_A.second;

    Vec2f current_uv = *A_uv;
    Point current_xy = A_xy;
    
    for (;;) {
      auto [t, exit_edge] = segment_exit_cell(current_uv, *B_uv, ci, cj, grid_cols, grid_rows);

      //printf("Exit edge: %d, t: %f\n", exit_edge, t);

      const double end_uv_x = double(current_uv.x()) +
                           t * (double(B_uv->x()) - double(current_uv.x()));
      const double end_uv_y = double(current_uv.y()) +
                           t * (double(B_uv->y()) - double(current_uv.y()));

      //printf("End uv: %f, %f\n", end_uv_x, end_uv_y);

      auto uv_to_point_opt = uv_to_point(cache, end_uv_x, end_uv_y);
      if (!uv_to_point_opt) {
        printf("No UV to point hit\n");
        continue;
      }

      //printf("UV to point: x: %f, y: %f, z: %f\n", uv_to_point_opt->x(), uv_to_point_opt->y(), uv_to_point_opt->z());
      Point end_xy = Point(uv_to_point_opt->x(), uv_to_point_opt->y());

      if (t > 1e-9f && is_black_cell(ci, cj)){
        printf("Adding segment: %d, %d -> %d, %d\n", current_xy.x(), current_xy.y(), end_xy.x(), end_xy.y());
        Point start_point = point_mm_to_model_surface(cache, current_xy.x(), current_xy.y(), outward_offset_mm);
        Point end_point = point_mm_to_model_surface(cache, end_xy.x(), end_xy.y(), outward_offset_mm);
        segments.push_back({start_point, end_point});
      }

      if (t >= 1.f - 1e-6f)
        break;

      current_xy = end_xy;
      current_uv = Vec2f(current_uv.x() + t * (B_uv->x() - current_uv.x()),
                         current_uv.y() + t * (B_uv->y() - current_uv.y()));

      if (exit_edge >= 0) {
        auto next = adjacent_cell_bounded(ci, cj, exit_edge, min_ci, max_ci, min_cj, max_cj);
        if (next.first == ci && next.second == cj)
          break;
        ci = next.first;
        cj = next.second;
      }
    }
  }

  printf("Segments: %zu\n", segments.size());
  for (const auto &segment : segments) {
    std::cout << "Segment: " << segment.first.x() << ", " << segment.first.y() << " -> " << segment.second.x() << ", " << segment.second.y() << std::endl;
    Polyline pl;
    pl.points.push_back(segment.first);
    pl.points.push_back(segment.second);
    result.push_back(pl);
  }

  // Merge consecutive segments that share an endpoint into polylines.
  // const coord_t eps2 = scale_(0.001) * scale_(0.001);
  // auto same_point = [eps2](const Point &a, const Point &b) {
  //   Vec2d d = (a - b).cast<double>();
  //   return d.squaredNorm() <= eps2;
  // };
  // std::vector<bool> used(segments.size(), false);
  // for (size_t i = 0; i < segments.size(); ++i) {
  //   if (used[i])
  //     continue;
  //   Polyline pl;
  //   pl.points.push_back(segments[i].first);
  //   pl.points.push_back(segments[i].second);
  //   used[i] = true;
  //   bool changed;
  //   do {
  //     changed = false;
  //     for (size_t j = 0; j < segments.size(); ++j) {
  //       if (used[j])
  //         continue;
  //       const Point &s0 = segments[j].first, &s1 = segments[j].second;
  //       if (same_point(pl.points.back(), s0)) {
  //         pl.points.push_back(s1);
  //         used[j] = true;
  //         changed = true;
  //       } else if (same_point(pl.points.back(), s1)) {
  //         pl.points.push_back(s0);
  //         used[j] = true;
  //         changed = true;
  //       } else if (same_point(pl.points.front(), s0)) {
  //         pl.points.insert(pl.points.begin(), s1);
  //         used[j] = true;
  //         changed = true;
  //       } else if (same_point(pl.points.front(), s1)) {
  //         pl.points.insert(pl.points.begin(), s0);
  //         used[j] = true;
  //         changed = true;
  //       }
  //     }
  //   } while (changed);
  //   if (pl.points.size() >= 2)
  //     result.push_back(std::move(pl));
  // }
  return result;
}

} // namespace

#ifdef CHECKERED_INFILL_DEBUG_SVG
void FillCheckered::write_debug_svgs(
    const Surface *surface, const Polylines &polylines_before_filter,
    const Polylines &polylines_after_filter,
    const std::vector<std::pair<Vec2f, Vec2f>> &uv_segments) const {
  static int s_svg_run = 0;
  const int run = s_svg_run++;
  std::string path_xy_before =
      debug_out_path("fill_checkered_xy_before_layer%d_z%.2f_run%d.svg",
                     int(this->layer_id), this->z, run);
  std::string path_xy_after =
      debug_out_path("fill_checkered_xy_after_layer%d_z%.2f_run%d.svg",
                     int(this->layer_id), this->z, run);
  std::string path_uv =
      debug_out_path("fill_checkered_uv_layer%d_z%.2f_run%d.svg",
                     int(this->layer_id), this->z, run);

  BoundingBox bbox = get_extents(surface->expolygon);
  bbox.offset(scale_(2.));
  {
    SVG svg(path_xy_before, bbox);
    if (svg.is_opened()) {
      svg.draw_outline(surface->expolygon, "blue", "cyan", scale_(0.05));
      svg.draw(polylines_before_filter, "green", scale_(0.08));
      svg.add_comment("Checkered infill XY before filter (full grid)");
    }
  }
  {
    SVG svg(path_xy_after, bbox);
    if (svg.is_opened()) {
      svg.draw_outline(surface->expolygon, "blue", "cyan", scale_(0.05));
      svg.draw(polylines_after_filter, "red", scale_(0.08));
      svg.add_comment("Checkered infill XY after UV filter");
    }
  }
  {
    BoundingBox uv_bbox(Point(0, 0),
                        Point(coord_t(scale_(100.)), coord_t(scale_(100.))));
    SVG svg(path_uv, uv_bbox, scale_(1.), true);
    if (svg.is_opened()) {
      for (int g = 0; g <= 10; ++g) {
        coord_t c = scale_(g * 10.);
        svg.draw(Line(Point(c, 0), Point(c, coord_t(scale_(100.)))),
                 "lightgray", scale_(0.2));
        svg.draw(Line(Point(0, c), Point(coord_t(scale_(100.)), c)),
                 "lightgray", scale_(0.2));
      }
      for (const auto &seg : uv_segments) {
        float u1 = seg.first.x() * 100.f, v1 = (1.f - seg.first.y()) * 100.f;
        float u2 = seg.second.x() * 100.f, v2 = (1.f - seg.second.y()) * 100.f;
        Point p1(coord_t(scale_(u1)), coord_t(scale_(v1))),
            p2(coord_t(scale_(u2)), coord_t(scale_(v2)));
        svg.draw(Line(p1, p2), "blue", scale_(0.5));
      }
      svg.add_comment("Checkered infill in UV space; overlay on texture image "
                      "to verify segments line on black lines");
    }
  }
  printf("FillCheckered: wrote debug SVGs %s %s %s\n", path_xy_before.c_str(),
         path_xy_after.c_str(), path_uv.c_str());
}
#else
void FillCheckered::write_debug_svgs(
    const Surface *, const Polylines &, const Polylines &,
    const std::vector<std::pair<Vec2f, Vec2f>> &) const {}
#endif

void FillCheckered::_fill_surface_single(
    const FillParams &params, unsigned int thickness_layers,
    const std::pair<float, Point> &direction, ExPolygon expolygon,
    Polylines &polylines_out) {

  Polygon outer_contour = expolygon.contour;
  double z_mm = this->z;

  if(size_t(this->layer_id) == 158) {
    std::shared_ptr<CachedUVMesh> cache =
        get_or_load_uv_mesh(m_uv_map_file_path);

    if (cache && cache->valid) {
      printf("Thickness %i", thickness_layers);

      const double outward_offset_mm =
          std::max(0., 0.5 * this->spacing - this->overlap);
      double ox = m_contour_to_mesh_origin_mm.x();
      double oy = m_contour_to_mesh_origin_mm.y();

      printf("Origin %f, %f \n", ox, oy);
      printf("Outward Offset %f \n", outward_offset_mm);

      printf("Bbox min: %f, %f\n", cache->bbox_min.x(), cache->bbox_min.y());
      printf("Bbox max: %f, %f\n", cache->bbox_max.x(), cache->bbox_max.y());

      // calcule the center of the bbox
      double bbox_center_x = (cache->bbox_min.x() + cache->bbox_max.x()) / 2;
      double bbox_center_y = (cache->bbox_min.y() + cache->bbox_max.y()) / 2;
      printf("Bbox center: %f, %f\n", bbox_center_x, bbox_center_y);

      // polylines_out = extract_black_contour_segments(
      //     outer_contour, z_mm, *cache, DEFAULT_GRID_COLS, DEFAULT_GRID_ROWS, ox,
      //     oy, outward_offset_mm);

      // printf("Polylines out: %zu\n", polylines_out.size());
      // for (const Polyline &pl : polylines_out) {
      //   for (const Point &p : pl.points) {
      //     //std::cout << "Point: " << p.x() << ", " << p.y() << std::endl;
      //   }
      // }

      size_t n = outer_contour.points.size();
      if (n == 0)
        return;

      for (size_t k = 0; k < n; ++k) {
        const Point A_xy = point_to_model_surface_mm(*cache, outer_contour.points[k], outward_offset_mm);
        const Point B_xy = point_to_model_surface_mm(*cache, outer_contour.points[(k + 1) % n], outward_offset_mm);
        double ax_mm = A_xy.x();
        double ay_mm = A_xy.y();
        double bx_mm = B_xy.x();
        double by_mm = B_xy.y();
        
        auto A_uv = point_to_uv(*cache, ax_mm, ay_mm, z_mm, Vec3d(bx_mm, by_mm, z_mm));
        auto B_uv = point_to_uv(*cache, bx_mm, by_mm, z_mm, Vec3d(ax_mm, ay_mm, z_mm));
        if (!A_uv || !B_uv){
          printf("No UV hit\n");
          continue;
        }

        printf("A uv: %f, %f, B uv: %f, %f\n", A_uv->x(), A_uv->y(), B_uv->x(), B_uv->y());

        auto segments = subdivide_uv_segment_by_grid(*A_uv, *B_uv, DEFAULT_GRID_COLS, DEFAULT_GRID_ROWS);
        for (const auto &segment : segments) {
          printf("Segment: %f, %f -> %f, %f in cell %d, %d\n", segment.start_uv.x(), segment.start_uv.y(), segment.end_uv.x(), segment.end_uv.y(), segment.ci, segment.cj);
        }

        printf("Segments: %zu\n", segments.size());
      }

    }
  }else{
    polylines_out.push_back(Polyline({Point(0, 0), Point(10, 10)}));
  }
}

} // namespace Slic3r
