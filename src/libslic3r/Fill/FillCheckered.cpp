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
#include <fstream>
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

// Map 3D point (x,y,z) in mm to (u,v) in [0,1]^2 by finding which face the
// point lies on, then interpolating that face's UV map. Returns nullopt if
// point is not on mesh surface. If the UV mesh has Z in [-H, 0] (e.g. top=0,
// bottom negative), we use -z_mm for query.
static std::optional<Vec2f> point_to_uv(const CachedUVMesh &cache, double x_mm,
                                        double y_mm, double z_mm) {
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

  // printf("Hit point: %f, %f, %f\n", hit_point.x(), hit_point.y(),
  //        hit_point.z());
  // printf("Hit index: %zu\n", hit_idx);

  const Vec3i &face = its.indices[hit_idx];
  Vec3d A = its.vertices[face(0)].cast<double>();
  Vec3d B = its.vertices[face(1)].cast<double>();
  Vec3d C = its.vertices[face(2)].cast<double>();
  auto bary = barycentric_coords_3d(hit_point, A, B, C);
  if (!bary)
    return std::nullopt;

  // printf("Barycentric: %f, %f, %f\n", (*bary)[0], (*bary)[1], (*bary)[2]);

  float w0 = (*bary)[0], w1 = (*bary)[1], w2 = (*bary)[2];
  const std::array<Vec2f, 3> &uv_arr = cache.uvs[hit_idx];
  // printf("UV array x: %f, %f, %f\n", uv_arr[0].x(), uv_arr[1].x(),
  //        uv_arr[2].x());
  // printf("UV array y: %f, %f, %f\n", uv_arr[0].y(), uv_arr[1].y(),
  //        uv_arr[2].y());
  float u = w0 * uv_arr[0].x() + w1 * uv_arr[1].x() + w2 * uv_arr[2].x();
  float v = w0 * uv_arr[0].y() + w1 * uv_arr[1].y() + w2 * uv_arr[2].y();
  u = std::clamp(u, 0.f, 1.f);
  v = std::clamp(v, 0.f, 1.f);
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

  return std::array<float, 3>{w0, s, t};
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
  if (u >= 1.f)
    i = grid_cols - 1;
  if (v >= 1.f)
    j = grid_rows - 1;
  i = std::clamp(i, 0, grid_cols - 1);
  j = std::clamp(j, 0, grid_rows - 1);
  return std::make_pair(i, j);
}

// Checkered pattern: fill cell iff (i + j) % 2 == 0
inline bool is_fill_cell(int i, int j) { return (i + j) % 2 == 0; }

// Black grid cell is the opposite of fill (checkered pattern)
inline bool is_black_cell(int i, int j) { return (i + j) % 2 == 1; }

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
  for (size_t i = 0; i < contour.points.size(); ++i) {
    const Point &pt = contour.points[i];
    double x_mm = unscale_(pt.x()) + origin_x_mm;
    double y_mm = unscale_(pt.y()) + origin_y_mm;
    if (outward_offset_mm > 0.) {
      Vec2d n = outward_unit_normal(contour, i);
      x_mm += outward_offset_mm * n.x();
      y_mm += outward_offset_mm * n.y();
    }
    std::optional<Vec2f> uv = point_to_uv(cache, x_mm, y_mm, z_mm);
    ContourPointUVInfo info;
    info.point = pt;
    if (uv) {
      info.uv = *uv;
      auto cell =
          point_to_grid_cell(cache, x_mm, y_mm, z_mm, grid_cols, grid_rows);
      if (cell) {
        info.grid_cell = *cell;
        info.is_black_cell = is_black_cell(cell->first, cell->second);
      }
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

// Exit parameter: smallest t in (0, 1] where segment a->b (UV) exits cell (ci,
// cj), or 1 if b is inside. Returns (t, exit_edge): exit_edge 0=left, 1=right,
// 2=bottom, 3=top (for adjacent cell).
static std::pair<float, int> segment_exit_cell(const Vec2f &a, const Vec2f &b,
                                               int ci, int cj, int grid_cols,
                                               int grid_rows) {
  float u_min, u_max, v_min, v_max;
  cell_bounds(ci, cj, grid_cols, grid_rows, u_min, u_max, v_min, v_max);
  float t_best = 1.f;
  int edge_best = -1;
  auto consider = [&](std::optional<float> t, int edge) {
    if (t && *t > 1e-9f && *t < t_best) {
      t_best = *t;
      edge_best = edge;
    }
  };

  consider(segment_intersect_vertical(u_min, v_min, v_max, a, b), 0); // left
  consider(segment_intersect_vertical(u_max, v_min, v_max, a, b), 1); // right
  consider(segment_intersect_horizontal(u_min, u_max, v_min, a, b),2); // bottom
  consider(segment_intersect_horizontal(u_min, u_max, v_max, a, b), 3); // top

  if (edge_best < 0)
    return {1.f, -1};
  return {t_best, edge_best};
}

// New cell when exiting through edge: 0=left, 1=right, 2=bottom, 3=top.
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
    
    auto A_uv = point_to_uv(cache, ax_mm, ay_mm, z_mm);
    auto B_uv = point_to_uv(cache, bx_mm, by_mm, z_mm);
    if (!A_uv || !B_uv){
      printf("No UV hit\n");
      continue;
    }

    auto cell_opt =
        point_to_grid_cell(cache, ax_mm, ay_mm, z_mm, grid_cols, grid_rows);
    if (!cell_opt) {
      printf("No grid cell hit\n");
      continue;
    }
    
    printf("Grid cell hit: %d, %d\n", cell_opt->first, cell_opt->second);

    int ci = cell_opt->first, cj = cell_opt->second;

    Vec2f current_uv = *A_uv;
    Point current_xy = A_xy;

    for (;;) {
      auto [t, exit_edge] =
          segment_exit_cell(current_uv, *B_uv, ci, cj, grid_cols, grid_rows);

      printf("Exit edge: %d, t: %f\n", exit_edge, t);

      printf("End uv: %f, %f\n", double(current_uv.x()) +
      t * (double(B_uv->x()) - double(current_uv.x())), double(current_uv.y()) +
      t * (double(B_uv->y()) - double(current_uv.y())));

      Point end_uv(
        coord_t(std::round(double(current_uv.x()) +
                           t * (double(B_uv->x()) - double(current_uv.x())))),
        coord_t(std::round(double(current_uv.y()) +
                           t * (double(B_uv->y()) - double(current_uv.y())))));

      printf("End uv: %f, %f\n", end_uv.x(), end_uv.y());

      auto uv_to_point_opt = uv_to_point(cache, end_uv.x(), end_uv.y());
      if (!uv_to_point_opt) {
        printf("No UV to point hit\n");
        continue;
      }

      Point end_xy = Point::new_scale(uv_to_point_opt->x(), uv_to_point_opt->y());

      printf("End xy: %f, %f\n", end_xy.x(), end_xy.y());

      if (is_black_cell(ci, cj)){
        printf("Adding segment: %f, %f -> %f, %f\n", current_xy.x(), current_xy.y(), end_xy.x(), end_xy.y());
        segments.push_back({current_xy, end_xy});
      }

      if (t >= 1.f - 1e-6f)
        break;

      current_xy = end_xy;
      current_uv = Vec2f(current_uv.x() + t * (B_uv->x() - current_uv.x()),
                         current_uv.y() + t * (B_uv->y() - current_uv.y()));
      if (exit_edge >= 0) {
        auto next = adjacent_cell(ci, cj, exit_edge, grid_cols, grid_rows);
        ci = next.first;
        cj = next.second;
      }
    }
  }

  // Merge consecutive segments that share an endpoint into polylines.
  const coord_t eps2 = scale_(0.001) * scale_(0.001);
  auto same_point = [eps2](const Point &a, const Point &b) {
    Vec2d d = (a - b).cast<double>();
    return d.squaredNorm() <= eps2;
  };
  std::vector<bool> used(segments.size(), false);
  for (size_t i = 0; i < segments.size(); ++i) {
    if (used[i])
      continue;
    Polyline pl;
    pl.points.push_back(segments[i].first);
    pl.points.push_back(segments[i].second);
    used[i] = true;
    bool changed;
    do {
      changed = false;
      for (size_t j = 0; j < segments.size(); ++j) {
        if (used[j])
          continue;
        const Point &s0 = segments[j].first, &s1 = segments[j].second;
        if (same_point(pl.points.back(), s0)) {
          pl.points.push_back(s1);
          used[j] = true;
          changed = true;
        } else if (same_point(pl.points.back(), s1)) {
          pl.points.push_back(s0);
          used[j] = true;
          changed = true;
        } else if (same_point(pl.points.front(), s0)) {
          pl.points.insert(pl.points.begin(), s1);
          used[j] = true;
          changed = true;
        } else if (same_point(pl.points.front(), s1)) {
          pl.points.insert(pl.points.begin(), s0);
          used[j] = true;
          changed = true;
        }
      }
    } while (changed);
    if (pl.points.size() >= 2)
      result.push_back(std::move(pl));
  }
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

  // Detect where every contour point lies in UV space and whether it's on a
  // black grid cell.
  // std::shared_ptr<CachedUVMesh> cache =
  // get_or_load_uv_mesh(m_uv_map_file_path);
  // if (cache && cache->valid) {
  //     const double outward_offset_mm = std::max(0., 0.5 * this->spacing -
  //     this->overlap); double ox = m_contour_to_mesh_origin_mm.x(); double oy
  //     = m_contour_to_mesh_origin_mm.y(); std::vector<ContourPointUVInfo>
  //     uv_info = get_contour_points_uv_info(
  //         outer_contour, z_mm, *cache, DEFAULT_GRID_COLS, DEFAULT_GRID_ROWS,
  //         ox, oy, outward_offset_mm);
  //     size_t on_black = 0, on_white = 0, no_uv = 0;
  //     for (const ContourPointUVInfo &info : uv_info) {
  //         if (!info.grid_cell) {
  //             ++no_uv;
  //             continue;
  //         }
  //         if (info.is_black_cell)
  //             ++on_black;
  //         else
  //             ++on_white;
  //     }
  //     // If all rays missed, OBJ may be in object-centered coords: retry with
  //     origin 0,0 if (no_uv == uv_info.size() &&
  //     !outer_contour.points.empty()) {
  //         std::vector<ContourPointUVInfo> uv_info0 =
  //         get_contour_points_uv_info(
  //             outer_contour, z_mm, *cache, DEFAULT_GRID_COLS,
  //             DEFAULT_GRID_ROWS, 0., 0., outward_offset_mm);
  //         size_t no_uv0 = 0;
  //         for (const ContourPointUVInfo &info : uv_info0)
  //             if (!info.grid_cell) ++no_uv0;
  //         if (no_uv0 < uv_info0.size()) {
  //             ox = 0.; oy = 0.;
  //             uv_info = std::move(uv_info0);
  //             on_black = on_white = no_uv = 0;
  //             for (const ContourPointUVInfo &info : uv_info) {
  //                 if (!info.grid_cell) { ++no_uv; continue; }
  //                 if (info.is_black_cell) ++on_black; else ++on_white;
  //             }
  //             printf("FillCheckered: all rays missed with center_offset; used
  //             origin=0 and got %zu hits\n", uv_info.size() - no_uv);
  //         }
  //     }
  //     printf("FillCheckered: contour points uv_info: %zu on_black=%zu
  //     on_white=%zu no_uv=%zu\n",
  //            uv_info.size(), on_black, on_white, no_uv);
  //     // Print ContourPointUVInfo to JSON
  //     {
  //         std::string path_json =
  //         "/Users/anant/Documents/Personal/BambuStudio/.cursor/fill_checkered_contour_uv_info.json";
  //         std::ofstream fj(path_json);
  //         if (fj) {
  //             fj << "{\"layer_id\":" << int(this->layer_id) << ",\"z_mm\":"
  //             << z_mm
  //                << ",\"contour_points\":[";
  //             for (size_t i = 0; i < uv_info.size(); ++i) {
  //                 const ContourPointUVInfo &info = uv_info[i];
  //                 if (i > 0) fj << ",";
  //                 fj << "{\"point_mm\":[" << unscale_(info.point.x()) << ","
  //                 << unscale_(info.point.y()) << "]"; if (info.uv) fj <<
  //                 ",\"uv\":[" << info.uv->x() << "," << info.uv->y() << "]";
  //                 else fj << ",\"uv\":null";
  //                 if (info.grid_cell) fj << ",\"grid_cell\":[" <<
  //                 info.grid_cell->first << "," << info.grid_cell->second <<
  //                 "]"; else fj << ",\"grid_cell\":null"; fj <<
  //                 ",\"is_black_cell\":" << (info.is_black_cell ? "true" :
  //                 "false") << "}";
  //             }
  //             fj << "]}\n";
  //             printf("FillCheckered: wrote ContourPointUVInfo to %s\n",
  //             path_json.c_str());
  //         }
  //     }
  //     // #region agent log — verification: mesh bbox, origin, ray,
  //     inside_bbox if (!outer_contour.points.empty()) {
  //         const Vec3f &bmn = cache->bbox_min, &bmx = cache->bbox_max;
  //         double c0x = unscale_(outer_contour.points.front().x());
  //         double c0y = unscale_(outer_contour.points.front().y());
  //         double rx = c0x + ox, ry = c0y + oy;
  //         const double tol = 0.01;
  //         bool inside = (rx >= double(bmn.x()) - tol && rx <= double(bmx.x())
  //         + tol &&
  //                        ry >= double(bmn.y()) - tol && ry <= double(bmx.y())
  //                        + tol && z_mm >= double(bmn.z()) - tol && z_mm <=
  //                        double(bmx.z()) + tol);
  //         std::ofstream
  //         f("/Users/anant/Documents/Personal/BambuStudio/.cursor/debug.log",
  //         std::ios::app); if (f)
  //             f << "FillCheckered mesh_bbox x[" << bmn.x() << "," << bmx.x()
  //             << "] y[" << bmn.y() << "," << bmx.y() << "] z[" << bmn.z() <<
  //             "," << bmx.z() << "] origin=(" << ox << "," << oy << ") ray0=("
  //             << rx << "," << ry << "," << z_mm << ") inside_bbox=" <<
  //             (inside ? 1 : 0) << " no_uv=" << no_uv << "\n";
  //     }
  // #endregion
  //}

  if (size_t(this->layer_id) == 10) {
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
      //     printf("Point: %f, %f\n", p.x(), p.y());
      //   }
      // }

      for (const Point &p : outer_contour.points) {

        double x_mm = unscale_(p.x());
        double y_mm = unscale_(p.y());

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
        // x_mm = 40;
        // y_mm = 10;
        z_mm = 10;

        printf("X mm: %f, Y mm: %f, Z mm: %f\n", x_mm, y_mm, z_mm);

        auto uv = point_to_uv(*cache, x_mm, y_mm, z_mm);
        if (!uv) {
          printf("No UV hit\n");
          continue;
        }

        printf("UV: %f, %f\n", uv->x(), uv->y());

        auto uv_to_point_opt = uv_to_point(*cache, uv->x(), uv->y());
        if (!uv_to_point_opt) {
          printf("No UV to point hit\n");
          continue;
        }

        Point end_xy = Point::new_scale(uv_to_point_opt->x(), uv_to_point_opt->y());

        printf("End xy: %f, %f\n", end_xy.x(), end_xy.y());
      }
    }
  }

  polylines_out.push_back(Polyline({Point(0, 0), Point(10, 10)}));
}

} // namespace Slic3r
