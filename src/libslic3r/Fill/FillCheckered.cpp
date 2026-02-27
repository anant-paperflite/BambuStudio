#include "FillCheckered.hpp"

#include "../AABBTreeIndirect.hpp"
#include "../Format/OBJ.hpp"
#include "../Line.hpp"
#include "../Point.hpp"
#include "../SVG.hpp"
#include "../ShortestPath.hpp"
#include "../TriangleMesh.hpp"
#include "../Utils.hpp"
#include "libslic3r/Polygon.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>

// Enable to write debug SVGs (XY segments and UV-space segments) to
// g_data_dir/SVG/
#define CHECKERED_INFILL_DEBUG_SVG

// Enable to print faces, vertices (mm), UV islands, and per-island grid spans
#define CHECKERED_INFILL_DEBUG_PRINT

namespace Slic3r {

namespace {

// Default grid resolution for UV space [0,1]^2 (12 cells per row/column,
// indices 0..11).
constexpr int DEFAULT_GRID_COLS = 12;
constexpr int DEFAULT_GRID_ROWS = 12;

struct CachedUVMesh {
  TriangleMesh mesh;
  std::vector<std::array<Vec2f, 3>> uvs;
  AABBTreeIndirect::Tree<3, float> tree;
  bool valid{false};
  Vec3f bbox_min{0.f, 0.f, 0.f};
  Vec3f bbox_max{0.f, 0.f, 0.f};

  // UV island data: face_idx -> island_id; island_id -> face indices;
  // per-island UV bounds.
  std::vector<int> face_to_island;
  std::vector<std::vector<size_t>> island_faces;
  std::vector<std::array<float, 4>> island_uv_bounds;

  static void compute_uv_islands(CachedUVMesh &out) {
    const indexed_triangle_set &its = out.mesh.its;
    const size_t n_faces = its.indices.size();
    if (n_faces == 0 || out.uvs.size() != n_faces)
      return;

    std::vector<Vec3i> face_neighbors = its_face_neighbors(its);

    // Fold UV to [0,1] for island continuity check (seams at 0/1 disconnect).
    auto fold_uv = [](float t) {
      if (t > 1.f && t <= 2.f)
        return 2.f - t;
      if (t > 2.f)
        return t - std::floor(t);
      if (t < 0.f)
        return t - std::floor(t);
      return t;
    };

    // Union-Find: parent[i] = representative of face i's island.
    std::vector<size_t> parent(n_faces);
    std::iota(parent.begin(), parent.end(), 0);

    auto find = [&parent](size_t i) {
      while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
      }
      return i;
    };

    auto unite = [&find, &parent](size_t a, size_t b) {
      a = find(a);
      b = find(b);
      if (a != b)
        parent[a] = b;
    };

    constexpr float uv_eps = 1e-5f;
    for (size_t fi = 0; fi < n_faces; ++fi) {
      const Vec3i &neighbors = face_neighbors[fi];
      const std::array<Vec2f, 3> &uv_a = out.uvs[fi];
      const Vec3i &face_a = its.indices[fi];

      for (int e = 0; e < 3; ++e) {
        int nbr = neighbors(e);
        if (nbr < 0)
          continue;

        size_t fj = size_t(nbr);
        const std::array<Vec2f, 3> &uv_b = out.uvs[fj];
        const Vec3i &face_b = its.indices[fj];

        // Edge e of face fi: vertices face_a(e) and face_a((e+1)%3).
        Vec2i edge_a(face_a(e), face_a((e + 1) % 3));
        if (edge_a(0) > edge_a(1))
          std::swap(edge_a(0), edge_a(1));

        // Find matching edge in face fj.
        int eb = its_triangle_edge_index(face_b, edge_a);
        if (eb < 0)
          continue;

        float ua0 = fold_uv(uv_a[e].x()), va0 = fold_uv(uv_a[e].y());
        float ua1 = fold_uv(uv_a[(e + 1) % 3].x()),
              va1 = fold_uv(uv_a[(e + 1) % 3].y());
        float ub0 = fold_uv(uv_b[eb].x()), vb0 = fold_uv(uv_b[eb].y());
        float ub1 = fold_uv(uv_b[(eb + 1) % 3].x()),
              vb1 = fold_uv(uv_b[(eb + 1) % 3].y());

        bool match0 =
            (std::abs(ua0 - ub0) < uv_eps && std::abs(va0 - vb0) < uv_eps) ||
            (std::abs(ua0 - ub1) < uv_eps && std::abs(va0 - vb1) < uv_eps);
        bool match1 =
            (std::abs(ua1 - ub0) < uv_eps && std::abs(va1 - vb0) < uv_eps) ||
            (std::abs(ua1 - ub1) < uv_eps && std::abs(va1 - vb1) < uv_eps);
        if (match0 && match1)
          unite(fi, fj);
      }
    }

    // Build island_id from union-find roots.
    std::map<size_t, int> root_to_island;
    out.face_to_island.assign(n_faces, -1);
    int num_islands = 0;
    for (size_t fi = 0; fi < n_faces; ++fi) {
      size_t r = find(fi);
      auto it = root_to_island.find(r);
      if (it == root_to_island.end()) {
        root_to_island[r] = num_islands++;
      }
      out.face_to_island[fi] = root_to_island[r];
    }

    out.island_faces.resize(num_islands);
    for (size_t fi = 0; fi < n_faces; ++fi)
      out.island_faces[out.face_to_island[fi]].push_back(fi);

    out.island_uv_bounds.resize(num_islands);
    for (int isl = 0; isl < num_islands; ++isl) {
      float u_min = 1.f, u_max = 0.f, v_min = 1.f, v_max = 0.f;
      for (size_t fi : out.island_faces[isl]) {
        for (const Vec2f &uv : out.uvs[fi]) {
          float u = fold_uv(uv.x()), v = fold_uv(uv.y());
          u_min = std::min(u_min, u);
          u_max = std::max(u_max, u);
          v_min = std::min(v_min, v);
          v_max = std::max(v_max, v);
        }
      }
      out.island_uv_bounds[isl] = {u_min, u_max, v_min, v_max};
    }
  }

  static std::optional<CachedUVMesh> load(const std::string &path) {
    if (path.empty())
      return std::nullopt;

    TriangleMesh mesh;
    ObjInfo obj_info;
    std::string message;
    if (!load_obj(path.c_str(), &mesh, obj_info, message, false))
      return std::nullopt;
    if (mesh.empty())
      return std::nullopt;
    if (obj_info.uvs.size() != mesh.its.indices.size())
      return std::nullopt;

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
    }

    compute_uv_islands(out);
    return out;
  }
};

static std::mutex s_cache_mutex;
static std::map<std::string, std::shared_ptr<CachedUVMesh>> s_uv_cache;

#ifdef CHECKERED_INFILL_DEBUG_PRINT
static void debug_print_uv_mesh_info(const CachedUVMesh &cache, int grid_cols,
                                     int grid_rows) {
  const indexed_triangle_set &its = cache.mesh.its;
  printf("=== Checkered UV Mesh Debug ===\n");
  printf("Faces and vertices (mm):\n");
  for (size_t fi = 0; fi < its.indices.size(); ++fi) {
    const Vec3i &face = its.indices[fi];
    const Vec3f &v0 = its.vertices[face(0)];
    const Vec3f &v1 = its.vertices[face(1)];
    const Vec3f &v2 = its.vertices[face(2)];
    printf("  face %zu: v0=(%.6f,%.6f,%.6f) v1=(%.6f,%.6f,%.6f) "
           "v2=(%.6f,%.6f,%.6f)\n",
           fi, double(v0.x()), double(v0.y()), double(v0.z()), double(v1.x()),
           double(v1.y()), double(v1.z()), double(v2.x()), double(v2.y()),
           double(v2.z()));
  }
  printf("UV Islands:\n");
  for (size_t isl = 0; isl < cache.island_faces.size(); ++isl) {
    const std::vector<size_t> &faces = cache.island_faces[isl];
    const std::array<float, 4> &bounds = cache.island_uv_bounds[isl];
    float u_min = bounds[0], u_max = bounds[1], v_min = bounds[2],
          v_max = bounds[3];
    int min_ci = std::max(0, static_cast<int>(std::floor(u_min * grid_cols)));
    int max_ci =
        std::min(grid_cols - 1,
                 static_cast<int>(std::floor((u_max - 1e-9f) * grid_cols)));
    int min_cj = std::max(0, static_cast<int>(std::floor(v_min * grid_rows)));
    int max_cj =
        std::min(grid_rows - 1,
                 static_cast<int>(std::floor((v_max - 1e-9f) * grid_rows)));
    min_ci = std::clamp(min_ci, 0, grid_cols - 1);
    max_ci = std::clamp(max_ci, 0, grid_cols - 1);
    min_cj = std::clamp(min_cj, 0, grid_rows - 1);
    max_cj = std::clamp(max_cj, 0, grid_rows - 1);
    printf("  island %zu: %zu faces [", isl, faces.size());
    for (size_t i = 0; i < faces.size(); ++i) {
      printf("%zu%s", faces[i], (i + 1 < faces.size()) ? "," : "");
    }
    printf("], bounds u=[%.4f,%.4f] v=[%.4f,%.4f], grid_numbers: [",
           double(u_min), double(u_max), double(v_min), double(v_max));
    bool first = true;
    for (int cj = min_cj; cj <= max_cj; ++cj) {
      for (int ci = min_ci; ci <= max_ci; ++ci) {
        int grid_number = (grid_cols - cj - 1) * grid_rows + ci;
        printf("%s%d", first ? "" : ",", grid_number);
        first = false;
      }
    }
    printf("]\n");
  }
}
#endif

std::shared_ptr<CachedUVMesh> get_or_load_uv_mesh(const std::string &path) {
  if (path.empty())
    return nullptr;
  std::lock_guard<std::mutex> lock(s_cache_mutex);
  auto it = s_uv_cache.find(path);
  if (it != s_uv_cache.end())
    return it->second;
  auto opt = CachedUVMesh::load(path);
  if (!opt)
    return nullptr;
  auto ptr = std::make_shared<CachedUVMesh>(std::move(*opt));
  s_uv_cache[path] = ptr;
#ifdef CHECKERED_INFILL_DEBUG_PRINT
  debug_print_uv_mesh_info(*ptr, DEFAULT_GRID_COLS, DEFAULT_GRID_ROWS);
#endif
  return ptr;
}

// Find mesh faces (triangle indices) whose intersection with plane z=z_mm
// contains or overlaps the 2D segment from (ax_mm,ay_mm) to (bx_mm,by_mm).
static std::vector<size_t> find_faces_for_segment(const CachedUVMesh &cache,
                                                  double ax_mm, double ay_mm,
                                                  double bx_mm, double by_mm,
                                                  double z_mm) {
  const indexed_triangle_set &its = cache.mesh.its;
  if (its.vertices.empty() || its.indices.empty())
    return {};

  double pz = (cache.bbox_max.z() <= 0.f) ? -z_mm : z_mm;
  double mid_x = 0.5 * (ax_mm + bx_mm);
  double mid_y = 0.5 * (ay_mm + by_mm);
  double seg_len_sq =
      (bx_mm - ax_mm) * (bx_mm - ax_mm) + (by_mm - ay_mm) * (by_mm - ay_mm);
  double radius_sq = seg_len_sq * 0.26 + 1e-6; // (0.5*seg_len)^2 + margin

  Vec3d mid(mid_x, mid_y, pz);
  std::vector<size_t> candidates = AABBTreeIndirect::all_triangles_in_radius(
      its.vertices, its.indices, cache.tree, mid, radius_sq);

  std::vector<size_t> out;
  constexpr double eps = 1e-9;

  for (size_t fi : candidates) {
    const Vec3i &face = its.indices[fi];
    Vec3d v0 = its.vertices[face(0)].cast<double>();
    Vec3d v1 = its.vertices[face(1)].cast<double>();
    Vec3d v2 = its.vertices[face(2)].cast<double>();

    double z0 = v0.z(), z1 = v1.z(), z2 = v2.z();
    if ((z0 > pz + eps && z1 > pz + eps && z2 > pz + eps) ||
        (z0 < pz - eps && z1 < pz - eps && z2 < pz - eps))
      continue;

    // Collect intersection points of triangle edges with plane z=pz.
    std::vector<std::pair<double, double>> pts;
    auto add_intersection = [&](const Vec3d &a, const Vec3d &b) {
      double za = a.z(), zb = b.z();
      if (std::abs(zb - za) < eps)
        return;
      double t = (pz - za) / (zb - za);
      if (t >= -eps && t <= 1.0 + eps) {
        double x = a.x() + t * (b.x() - a.x());
        double y = a.y() + t * (b.y() - a.y());
        pts.push_back({x, y});
      }
    };
    add_intersection(v0, v1);
    add_intersection(v1, v2);
    add_intersection(v2, v0);

    if (pts.size() < 2)
      continue;

    double tx0 = pts[0].first, ty0 = pts[0].second;
    double tx1 = pts[1].first, ty1 = pts[1].second;

    // Require contour segment AB to overlap triangle intersection CD.
    constexpr double eps_par = 1e-12;
    constexpr double eps_t = 1e-9;
    constexpr double eps_on = 1e-6;
    double denom =
        (ax_mm - bx_mm) * (ty0 - ty1) - (ay_mm - by_mm) * (tx0 - tx1);
    bool overlap = false;
    double t_a = 0, t_b = 0;
    if (std::abs(denom) <= eps_par) {
      double dx_cd = tx1 - tx0, dy_cd = ty1 - ty0;
      double len_sq = dx_cd * dx_cd + dy_cd * dy_cd;
      if (len_sq >= eps_par) {
        double cross_a = (ax_mm - tx0) * dy_cd - (ay_mm - ty0) * dx_cd;
        double cross_b = (bx_mm - tx0) * dy_cd - (by_mm - ty0) * dx_cd;
        if (std::abs(cross_a) <= eps_par && std::abs(cross_b) <= eps_par) {
          t_a = ((ax_mm - tx0) * dx_cd + (ay_mm - ty0) * dy_cd) / len_sq;
          t_b = ((bx_mm - tx0) * dx_cd + (by_mm - ty0) * dy_cd) / len_sq;
          double lo = std::min(t_a, t_b), hi = std::max(t_a, t_b);
          if (hi >= -eps_t && lo <= 1.0 + eps_t) {
            bool a_on_cd = (t_a >= -eps_on && t_a <= 1.0 + eps_on);
            bool b_on_cd = (t_b >= -eps_on && t_b <= 1.0 + eps_on);
            overlap = a_on_cd && b_on_cd; // segment fully on face intersection
          }
        }
      } else {
        overlap =
            (ax_mm - tx0) * (ax_mm - tx0) + (ay_mm - ty0) * (ay_mm - ty0) <
            eps_par;
      }
    }

    if (overlap)
      out.push_back(fi);
  }
  return out;
}

// Sub-segment with its adjacent mesh face(s). Each sub-segment is a mesh edge.
struct SubSegmentWithFaces {
  double ax_mm, ay_mm, bx_mm, by_mm;
  std::vector<size_t> faces;
};

// Partition point on segment AB: (x,y,t) and the mesh edge it lies on (va,vb).
// For a vertex: va == vb == vertex_id. For edge-plane intersection: (va,vb) is
// the edge.
struct PartitionPoint {
  double x, y, t;
  int edge_va, edge_vb;
};

// Split contour segment AB by mesh edges on the segment. Partition points are
// mesh vertices on the segment OR mesh edge-plane intersections. Each resulting
// sub-segment lies within a single mesh face.
// Returns empty if segment is degenerate (A≈B) or no partition points found.
static std::vector<SubSegmentWithFaces>
split_contour_segment_by_faces(const CachedUVMesh &cache, double ax_mm,
                               double ay_mm, double bx_mm, double by_mm,
                               double z_mm) {
  const indexed_triangle_set &its = cache.mesh.its;
  if (its.vertices.empty() || its.indices.empty())
    return {};

  constexpr double eps = 1e-6;
  constexpr double eps_cross = 1e-9;

  double dx = bx_mm - ax_mm, dy = by_mm - ay_mm;
  double len_sq = dx * dx + dy * dy;
  if (len_sq < eps * eps)
    return {};

  double pz = (cache.bbox_max.z() <= 0.f) ? -z_mm : z_mm;

  std::vector<PartitionPoint> pts;

  // 1. Mesh vertices on segment: on plane z=pz, on line AB, t in [0,1].
  for (size_t vi = 0; vi < its.vertices.size(); ++vi) {
    const Vec3f &v = its.vertices[vi];
    double vx = double(v.x()), vy = double(v.y()), vz = double(v.z());
    if (std::abs(vz - pz) > eps)
      continue;

    double px = vx - ax_mm, py = vy - ay_mm;
    double cross = px * dy - py * dx;
    if (std::abs(cross) > eps_cross * std::sqrt(len_sq))
      continue;

    double t = (px * dx + py * dy) / len_sq;
    if (t >= -eps && t <= 1.0 + eps)
      pts.push_back({vx, vy, t, int(vi), int(vi)});
  }

  // 2. Mesh edge-plane intersections on segment. Contour vertices are typically
  // these, not mesh vertices.
  for (size_t fi = 0; fi < its.indices.size(); ++fi) {
    const Vec3i &face = its.indices[fi];
    for (int e = 0; e < 3; ++e) {
      int va = face(e), vb = face((e + 1) % 3);
      const Vec3f &v0 = its.vertices[va];
      const Vec3f &v1 = its.vertices[vb];
      double z0 = double(v0.z()), z1 = double(v1.z());
      if (std::abs(z1 - z0) < eps)
        continue;
      double te = (pz - z0) / (z1 - z0);
      if (te < -eps || te > 1.0 + eps)
        continue;
      double px = double(v0.x()) + te * (double(v1.x()) - double(v0.x()));
      double py = double(v0.y()) + te * (double(v1.y()) - double(v0.y()));

      double seg_px = px - ax_mm, seg_py = py - ay_mm;
      double cross = seg_px * dy - seg_py * dx;
      if (std::abs(cross) > eps_cross * std::sqrt(len_sq))
        continue;
      double t = (seg_px * dx + seg_py * dy) / len_sq;
      if (t >= -eps && t <= 1.0 + eps) {
        int ea = va, eb = vb;
        if (ea > eb)
          std::swap(ea, eb);
        pts.push_back({px, py, t, ea, eb});
      }
    }
  }

  if (pts.empty())
    return {};

  // Deduplicate by (x,y) and sort by t.
  std::sort(pts.begin(), pts.end(),
            [](const PartitionPoint &a, const PartitionPoint &b) {
              return a.t < b.t;
            });
  auto last =
      std::unique(pts.begin(), pts.end(),
                  [](const PartitionPoint &a, const PartitionPoint &b) {
                    constexpr double e = 1e-6;
                    return std::abs(a.x - b.x) < e && std::abs(a.y - b.y) < e;
                  });
  pts.erase(last, pts.end());

  // 3. For mesh edge (va, vb), find face(s) containing it.
  auto faces_for_edge = [&](int va, int vb) -> std::vector<size_t> {
    std::vector<size_t> out;
    Vec2i edge(va, vb);
    if (edge(0) > edge(1))
      std::swap(edge(0), edge(1));
    for (size_t fi = 0; fi < its.indices.size(); ++fi) {
      int ei = its_triangle_edge_index(its.indices[fi], edge);
      if (ei >= 0) {
        out.push_back(fi);
        if (out.size() >= 2)
          break;
      }
    }
    return out;
  };

  // Face contains edge (va,vb)?
  auto face_has_edge = [&](size_t fi, int va, int vb) -> bool {
    Vec2i edge(va, vb);
    if (edge(0) > edge(1))
      std::swap(edge(0), edge(1));
    return its_triangle_edge_index(its.indices[fi], edge) >= 0;
  };

  // Face contains vertex vid?
  auto face_has_vertex = [&](size_t fi, int vid) -> bool {
    const Vec3i &tri = its.indices[fi];
    return tri(0) == vid || tri(1) == vid || tri(2) == vid;
  };

  // 4. Build sub-segments from consecutive partition points.
  std::vector<SubSegmentWithFaces> result;
  for (size_t i = 0; i + 1 < pts.size(); ++i) {
    const PartitionPoint &a = pts[i];
    const PartitionPoint &b = pts[i + 1];
    if (std::abs(a.t - b.t) < eps)
      continue;

    std::vector<size_t> faces;
    const bool a_is_vertex = (a.edge_va == a.edge_vb);
    const bool b_is_vertex = (b.edge_va == b.edge_vb);
    if (a_is_vertex && b_is_vertex) {
      faces = faces_for_edge(a.edge_va, b.edge_va);
    } else if (a_is_vertex) {
      for (size_t f : faces_for_edge(b.edge_va, b.edge_vb))
        if (face_has_vertex(f, a.edge_va))
          faces.push_back(f);
    } else if (b_is_vertex) {
      for (size_t f : faces_for_edge(a.edge_va, a.edge_vb))
        if (face_has_vertex(f, b.edge_va))
          faces.push_back(f);
    } else {
      for (size_t f : faces_for_edge(a.edge_va, a.edge_vb))
        if (face_has_edge(f, b.edge_va, b.edge_vb))
          faces.push_back(f);
    }
    if (!faces.empty()) {
      SubSegmentWithFaces sub;
      sub.ax_mm = a.x;
      sub.ay_mm = a.y;
      sub.bx_mm = b.x;
      sub.by_mm = b.y;
      sub.faces = std::move(faces);
      result.push_back(std::move(sub));
    } else {
      // Fallback: topology failed (consecutive edges may not share vertex when
      // segment crosses multiple mesh regions). Use overlap-based face lookup,
      // then filter to only faces that have partition-point edges (exclude
      // interior faces that overlap but lack those edges).
      faces = find_faces_for_segment(cache, a.x, a.y, b.x, b.y, z_mm);
      std::vector<size_t> filtered;
      for (size_t f : faces) {
        bool has_a = a_is_vertex ? face_has_vertex(f, a.edge_va)
                                 : face_has_edge(f, a.edge_va, a.edge_vb);
        bool has_b = b_is_vertex ? face_has_vertex(f, b.edge_va)
                                 : face_has_edge(f, b.edge_va, b.edge_vb);
        if (has_a || has_b)
          filtered.push_back(f);
      }
      faces = std::move(filtered);
      // Second fallback: when overlap-based lookup returns empty (e.g. strict
      // a_on_cd&&b_on_cd rejects), use union of faces from both partition
      // point edges. Sub-segment lies on a face that has at least one of the
      // two edges.
      if (faces.empty() && !a_is_vertex && !b_is_vertex) {
        for (size_t f : faces_for_edge(a.edge_va, a.edge_vb))
          faces.push_back(f);
        for (size_t f : faces_for_edge(b.edge_va, b.edge_vb)) {
          if (std::find(faces.begin(), faces.end(), f) == faces.end())
            faces.push_back(f);
        }
      } else if (faces.empty() && a_is_vertex && !b_is_vertex) {
        for (size_t f : faces_for_edge(b.edge_va, b.edge_vb))
          if (face_has_vertex(f, a.edge_va))
            faces.push_back(f);
      } else if (faces.empty() && !a_is_vertex && b_is_vertex) {
        for (size_t f : faces_for_edge(a.edge_va, a.edge_vb))
          if (face_has_vertex(f, b.edge_va))
            faces.push_back(f);
      }
      if (!faces.empty()) {
        SubSegmentWithFaces sub;
        sub.ax_mm = a.x;
        sub.ay_mm = a.y;
        sub.bx_mm = b.x;
        sub.by_mm = b.y;
        sub.faces = std::move(faces);
        result.push_back(std::move(sub));
      }
    }
  }

  return result;
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

// Convert point in model surface coordinates to point in mm coordinates
static Point point_to_model_surface_mm(const CachedUVMesh &cache, Point p,
                                       double outward_offset_mm) {
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

static Point point_mm_to_model_surface(const CachedUVMesh &cache, double x_mm,
                                       double y_mm, double outward_offset_mm) {
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
// When restrict_to_faces is non-null, only consider those face indices.
static std::optional<Vec2f>
point_to_uv(const CachedUVMesh &cache, double x_mm, double y_mm, double z_mm,
            const std::vector<size_t> *restrict_to_faces = nullptr) {
  const indexed_triangle_set &its = cache.mesh.its;
  if (its.vertices.empty() || its.indices.empty() ||
      cache.uvs.size() != its.indices.size())
    return std::nullopt;

  // double pz = (cache.bbox_max.z() <= 0.f) ? -z_mm : z_mm;
  double pz = z_mm;
  Vec3d P(x_mm, y_mm, pz);

  size_t hit_idx = 0;
  Vec3d hit_point;
  double best_sqr_dist = std::numeric_limits<double>::max();

  if (restrict_to_faces && !restrict_to_faces->empty()) {
    for (size_t fi : *restrict_to_faces) {
      if (fi >= its.indices.size())
        continue;
      const Vec3i &face = its.indices[fi];
      Vec3d a = its.vertices[face(0)].cast<double>();
      Vec3d b = its.vertices[face(1)].cast<double>();
      Vec3d c = its.vertices[face(2)].cast<double>();
      Vec3d ab = b - a, ac = c - a, ap = P - a;
      double d1 = ab.dot(ap), d2 = ac.dot(ap);

      Vec3d closest;
      if (d1 <= 0 && d2 <= 0) {
        closest = a;
      } else {
        Vec3d bp = P - b;
        double d3 = ab.dot(bp), d4 = ac.dot(bp);
        if (d3 >= 0 && d4 <= d3) {
          closest = b;
        } else {
          Vec3d cp = P - c;
          double d5 = ab.dot(cp), d6 = ac.dot(cp);
          if (d6 >= 0 && d5 <= d6) {
            closest = c;
          } else {
            double vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6,
                   va = d3 * d6 - d5 * d4;
            double denom = 1.0 / (va + vb + vc);
            double v = vb * denom, w = vc * denom;
            closest = a + ab * v + ac * w;
          }
        }
      }
      double sqr_dist = (P - closest).squaredNorm();
      if (sqr_dist < best_sqr_dist) {
        best_sqr_dist = sqr_dist;
        hit_idx = fi;
        hit_point = closest;
      }
    }
  } else {
    best_sqr_dist = AABBTreeIndirect::squared_distance_to_indexed_triangle_set(
        its.vertices, its.indices, cache.tree, P, hit_idx, hit_point);
  }

  const double epsilon_sq = 1e-6;

  if (best_sqr_dist < 0 || best_sqr_dist > epsilon_sq) {
    best_sqr_dist = AABBTreeIndirect::squared_distance_to_indexed_triangle_set(
        its.vertices, its.indices, cache.tree, P, hit_idx, hit_point);
  }

  if (best_sqr_dist < 0 || best_sqr_dist > epsilon_sq)
    return std::nullopt;
  if (hit_idx >= cache.uvs.size())
    return std::nullopt;

  const Vec3i &face = its.indices[hit_idx];
  Vec3d A = its.vertices[face(0)].cast<double>();
  Vec3d B = its.vertices[face(1)].cast<double>();
  Vec3d C = its.vertices[face(2)].cast<double>();
  auto bary = barycentric_coords_3d(hit_point, A, B, C);
  if (!bary)
    return std::nullopt;

  float w0 = (*bary)[0], w1 = (*bary)[1], w2 = (*bary)[2];
  const std::array<Vec2f, 3> &uv_arr = cache.uvs[hit_idx];
  auto fold_uv = [](float t) {
    if (t > 1.f && t <= 2.f)
      return 2.f - t;
    if (t > 2.f)
      return t - std::floor(t);
    if (t < 0.f)
      return t - std::floor(t);
    return t;
  };
  float u0 = fold_uv(uv_arr[0].x()), u1 = fold_uv(uv_arr[1].x()),
        u2 = fold_uv(uv_arr[2].x());
  float v0 = fold_uv(uv_arr[0].y()), v1 = fold_uv(uv_arr[1].y()),
        v2 = fold_uv(uv_arr[2].y());
  float u = w0 * u0 + w1 * u1 + w2 * u2;
  float v = w0 * v0 + w1 * v1 + w2 * v2;
  u = std::clamp(u, 0.f, 1.f);
  v = std::clamp(v, 0.f, 1.f);

  return Vec2f(u, v);
}

// Barycentric coordinates of point P in 2D triangle (A, B, C): P = w0*A + w1*B
// + w2*C. Returns nullopt if degenerate or P is outside the triangle.
static std::optional<std::array<float, 3>>
barycentric_coords_2d(const Vec2f &P, const Vec2f &A, const Vec2f &B,
                      const Vec2f &C) {
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
  if (std::abs(denom) < eps_denom)
    return std::nullopt;
  float s = (d11 * d20 - d01 * d21) / denom;
  float t = (d00 * d21 - d01 * d20) / denom;
  float w0 = 1.f - s - t;

  // Inside triangle iff all barycentrics in [0, 1]. Use tolerance for float
  // rounding: point_to_uv -> uv_to_point round-trip can yield s/w0/t just
  // outside [0,1] (e.g. s=-1e-7, w0=-3e-8); accept within 1e-5.
  const float eps_inside = 1e-5f;
  if (w0 < -eps_inside || w0 > 1.f + eps_inside || s < -eps_inside ||
      s > 1.f + eps_inside || t < -eps_inside || t > 1.f + eps_inside)
    return std::nullopt;

  return std::array<float, 3>{w0, s, t};
}

// Map (u, v) in [0,1]^2 to a 3D point on the mesh surface (mm, same frame as
// mesh). Finds the first triangle whose UV triangle contains (u,v) and
// interpolates the 3D position. Returns nullopt if no triangle contains (u,v).
// When restrict_to_faces is non-null, only search those face indices.
static std::optional<Vec3d>
uv_to_point(const CachedUVMesh &cache, float u, float v,
            const std::vector<size_t> *restrict_to_faces = nullptr) {
  const indexed_triangle_set &its = cache.mesh.its;
  if (its.vertices.empty() || its.indices.empty() ||
      cache.uvs.size() != its.indices.size())
    return std::nullopt;

  u = std::clamp(u, 0.f, 1.f);
  v = std::clamp(v, 0.f, 1.f);
  Vec2f P(u, v);

  const std::vector<size_t> *indices =
      restrict_to_faces && !restrict_to_faces->empty() ? restrict_to_faces
                                                       : nullptr;

  auto search = [&](size_t i) {
    const std::array<Vec2f, 3> &uv_arr = cache.uvs[i];
    auto bary = barycentric_coords_2d(P, uv_arr[0], uv_arr[1], uv_arr[2]);
    if (!bary)
      return std::optional<Vec3d>(std::nullopt);
    const Vec3i &face = its.indices[i];
    Vec3f V0 = its.vertices[face(0)];
    Vec3f V1 = its.vertices[face(1)];
    Vec3f V2 = its.vertices[face(2)];
    float w0 = (*bary)[0], w1 = (*bary)[1], w2 = (*bary)[2];
    double x = w0 * double(V0.x()) + w1 * double(V1.x()) + w2 * double(V2.x());
    double y = w0 * double(V0.y()) + w1 * double(V1.y()) + w2 * double(V2.y());
    double z = w0 * double(V0.z()) + w1 * double(V1.z()) + w2 * double(V2.z());
    return std::optional<Vec3d>(Vec3d(x, y, z));
  };

  if (indices) {
    for (size_t i : *indices) {
      if (i >= cache.uvs.size())
        continue;
      auto r = search(i);
      if (r)
        return r;
    }
  } else {
    for (size_t i = 0; i < cache.uvs.size(); ++i) {
      auto r = search(i);
      if (r)
        return r;
    }
  }
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
  int j = static_cast<int>(std::floor((v)*grid_rows));

  i = std::clamp(i, 0, grid_cols - 1);
  j = std::clamp(j, 0, grid_rows - 1);

  return std::make_pair(i, j);
}

// Epsilon for "point on grid edge" detection.
constexpr float UV_GRID_EDGE_EPS = 1e-6f;

// Returns true when (u,v) lies on a vertical grid line u=k/grid_cols,
// horizontal grid line v=m/grid_rows, or on domain boundary u<=eps, u>=1-eps,
// v<=eps, v>=1-eps.
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
// When (u,v) is on a grid edge and (dir_u, dir_v) is non-zero, step by a
// minimal amount along the direction and use that point's cell to disambiguate.
// Here we step to pick the cell the segment actually enters when on a grid
// edge.
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
    std::optional<Vec2f> uv = point_to_uv(cache, x_mm, y_mm, z_mm);
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
static std::optional<float> segment_intersect_vertical(float u_edge, float v_lo,
                                                       float v_hi,
                                                       const Vec2f &a,
                                                       const Vec2f &b) {

  float du = b.x() - a.x();
  if (std::abs(du) < 1e-9f) {
    return std::nullopt;
  }

  float t = (u_edge - a.x()) / du;
  if (t < 0.f || t > 1.f) {
    return std::nullopt;
  }

  float v = a.y() + t * (b.y() - a.y());
  if (v < v_lo || v > v_hi) {
    return std::nullopt;
  }

  return t;
}

// Intersect segment a->b (UV) with a horizontal line v = v_edge, segment from
// (u_lo, v_edge) to (u_hi, v_edge).
static std::optional<float> segment_intersect_horizontal(float u_lo, float u_hi,
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
// segment direction (b - a) so we step into the cell the segment is heading
// toward.
static std::pair<float, int> segment_exit_cell(const Vec2f &a, const Vec2f &b,
                                               int ci, int cj, int grid_cols,
                                               int grid_rows) {
  float u_min, u_max, v_min, v_max;
  cell_bounds(ci, cj, grid_cols, grid_rows, u_min, u_max, v_min, v_max);
  const float du = b.x() - a.x();
  const float dv = b.y() - a.y();

  // If start is on a cell boundary, decide if we should "exit" immediately
  // (t=0) so the walk doesn't get stuck. Use segment direction: outward normal
  // dot (du,dv) > 0 means we leave through that edge. Pick the edge we're on
  // that the segment exits through (at corners, pick the one most aligned with
  // direction).
  {
    const bool on_left = (a.x() <= u_min + UV_EDGE_EPS);
    const bool on_right = (a.x() >= u_max - UV_EDGE_EPS);
    const bool on_bottom = (a.y() <= v_min + UV_EDGE_EPS);
    const bool on_top = (a.y() >= v_max - UV_EDGE_EPS);
    if (on_left || on_right || on_bottom || on_top) {
      // Outward normals: left (-1,0), right (1,0), bottom (0,-1), top (0,1).
      int best_edge = -1;
      float best_dot = 0.f;
      if (on_left && -du > best_dot) {
        best_dot = -du;
        best_edge = 0;
      }
      if (on_right && du > best_dot) {
        best_dot = du;
        best_edge = 1;
      }
      if (on_bottom && -dv > best_dot) {
        best_dot = -dv;
        best_edge = 2;
      }
      if (on_top && dv > best_dot) {
        best_dot = dv;
        best_edge = 3;
      }
      if (best_edge >= 0)
        return {0.f, best_edge};
    }
  }

  // Ignore intersections with t very close to 0: when the segment start is
  // just outside a cell edge (float noise), we'd otherwise "exit" through that
  // edge immediately and the walk can break. Use a small minimum t so we take
  // the real exit (e.g. right edge) instead.
  // Only consider an edge if the segment is moving toward it (direction of
  // travel); otherwise we get spurious exits into the wrong column/row (e.g.
  // segment in column 10 with du<0 exiting "right" into column 11).
  constexpr float t_exit_min = 1e-5f;
  constexpr float dir_eps = 1e-9f;
  float t_best = 1.f;
  int edge_best = -1;
  auto consider = [&](std::optional<float> t, int edge) {
    if (!t || *t < t_exit_min || *t >= t_best)
      return;
    if (edge == 0 && du >= -dir_eps)
      return; // left: only if moving left (du < 0)
    if (edge == 1 && du <= dir_eps)
      return; // right: only if moving right (du > 0)
    if (edge == 2 && dv >= -dir_eps)
      return; // bottom: only if moving down (dv < 0)
    if (edge == 3 && dv <= dir_eps)
      return; // top: only if moving up (dv > 0)
    t_best = *t;
    edge_best = edge;
  };

  consider(segment_intersect_vertical(u_min, v_min, v_max, a, b), 0); // left
  consider(segment_intersect_vertical(u_max, v_min, v_max, a, b), 1); // right
  consider(segment_intersect_horizontal(u_min, u_max, v_min, a, b),
           2); // bottom
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
static std::vector<UVSegmentInCell>
subdivide_uv_segment_by_grid(const Vec2f &a, const Vec2f &b, int grid_cols,
                             int grid_rows) {
  std::vector<UVSegmentInCell> out;
  float du = b.x() - a.x(), dv = b.y() - a.y();
  auto [ci, cj] = uv_to_grid_cell(a.x(), a.y(), grid_cols, grid_rows, du, dv);
  auto [bi, bj] = uv_to_grid_cell(b.x(), b.y(), grid_cols, grid_rows, -du, -dv);
  const int min_ci = std::min(ci, bi);
  const int max_ci = std::max(ci, bi);
  const int min_cj = std::min(cj, bj);
  const int max_cj = std::max(cj, bj);

  // Clamp endpoints so the walk stays in the intended cell range. When the two
  // endpoints disagree on column (or row) due to float on the boundary (e.g.
  // u=11/12 → cell 11 vs 10), restrict to the midpoint's column (row) so we
  // don't step into the wrong cell.
  constexpr float uv_inset = 1e-6f;
  int clamp_ci_min = min_ci, clamp_ci_max = max_ci;
  int clamp_cj_min = min_cj, clamp_cj_max = max_cj;
  // When endpoints disagree on column due to float boundary ambiguity (e.g.
  // vertical UV segment with u=11/12, one endpoint rounds to cell 10, one to
  // 11), restrict to the midpoint's column. Only do this when the segment is
  // nearly vertical in UV (|du| tiny) so we don't collapse segments that
  // genuinely span multiple columns.
  constexpr float du_ambiguity_eps = 1e-6f;
  if (min_ci != max_ci && std::abs(b.x() - a.x()) < du_ambiguity_eps) {
    const float mid_u = 0.5f * (a.x() + b.x());
    const float mid_v = 0.5f * (a.y() + b.y());
    const float mid_du = b.x() - a.x();
    const float mid_dv = b.y() - a.y();
    auto [mid_ci, mid_cj] =
        uv_to_grid_cell(mid_u, mid_v, grid_cols, grid_rows, mid_du, mid_dv);
    clamp_ci_min = clamp_ci_max = mid_ci;
  }
  const float u_lo = float(clamp_ci_min) / float(grid_cols) + uv_inset;
  const float u_hi = float(clamp_ci_max + 1) / float(grid_cols) - uv_inset;
  const float v_lo = float(clamp_cj_min) / float(grid_rows) + uv_inset;
  const float v_hi = float(clamp_cj_max + 1) / float(grid_rows) - uv_inset;
  const Vec2f a_clamped(std::clamp(a.x(), u_lo, u_hi),
                        std::clamp(a.y(), v_lo, v_hi));
  const Vec2f b_clamped(std::clamp(b.x(), u_lo, u_hi),
                        std::clamp(b.y(), v_lo, v_hi));

  du = b_clamped.x() - a_clamped.x();
  dv = b_clamped.y() - a_clamped.y();
  Vec2f current_uv = a_clamped;
  constexpr float t_done_eps = 1e-6f;

  // Walk uses the clamp range so we never step into a column/row that was
  // excluded by midpoint disambiguation.
  const int walk_min_ci = clamp_ci_min;
  const int walk_max_ci = clamp_ci_max;
  const int walk_min_cj = clamp_cj_min;
  const int walk_max_cj = clamp_cj_max;

  // Start cell from clamped point so we're in the correct cell after clamping.
  auto start_cell = uv_to_grid_cell(a_clamped.x(), a_clamped.y(), grid_cols,
                                    grid_rows, du, dv);
  ci = std::clamp(start_cell.first, walk_min_ci, walk_max_ci);
  cj = std::clamp(start_cell.second, walk_min_cj, walk_max_cj);

  for (;;) {
    auto [t, exit_edge] =
        segment_exit_cell(current_uv, b_clamped, ci, cj, grid_cols, grid_rows);
    const float end_u = current_uv.x() + t * (b_clamped.x() - current_uv.x());
    const float end_v = current_uv.y() + t * (b_clamped.y() - current_uv.y());
    Vec2f end_uv(end_u, end_v);
    out.push_back({current_uv, end_uv, ci, cj});

    if (t >= 1.f - t_done_eps)
      break;

    current_uv = end_uv;
    auto next = adjacent_cell_bounded(ci, cj, exit_edge, walk_min_ci,
                                      walk_max_ci, walk_min_cj, walk_max_cj);
    if (next.first == ci && next.second == cj)
      break;
    ci = next.first;
    cj = next.second;
  }
  return out;
}

// Subdivide UV segment a->b by grid, restricted to island UV bounds
// [u_min,u_max] x [v_min,v_max]. Only returns segments whose grid cells fall
// within the island. Uses subdivide_uv_segment_by_grid with clamped cell range.
static std::vector<UVSegmentInCell>
subdivide_uv_segment_by_grid_clamped(const Vec2f &a, const Vec2f &b,
                                     int grid_cols, int grid_rows, float u_min,
                                     float u_max, float v_min, float v_max) {
  float du = b.x() - a.x(), dv = b.y() - a.y();
  auto [ci_a, cj_a] =
      uv_to_grid_cell(a.x(), a.y(), grid_cols, grid_rows, du, dv);
  auto [ci_b, cj_b] =
      uv_to_grid_cell(b.x(), b.y(), grid_cols, grid_rows, -du, -dv);

  int min_ci = std::max(0, static_cast<int>(std::floor(u_min * grid_cols)));
  int max_ci = std::min(
      grid_cols - 1, static_cast<int>(std::floor((u_max - 1e-9f) * grid_cols)));
  int min_cj = std::max(0, static_cast<int>(std::floor(v_min * grid_rows)));
  int max_cj = std::min(
      grid_rows - 1, static_cast<int>(std::floor((v_max - 1e-9f) * grid_rows)));

  min_ci = std::clamp(min_ci, 0, grid_cols - 1);
  max_ci = std::clamp(max_ci, 0, grid_cols - 1);
  min_cj = std::clamp(min_cj, 0, grid_rows - 1);
  max_cj = std::clamp(max_cj, 0, grid_rows - 1);

  const int walk_min_ci = std::max(min_ci, std::min(ci_a, ci_b));
  const int walk_max_ci = std::min(max_ci, std::max(ci_a, ci_b));
  const int walk_min_cj = std::max(min_cj, std::min(cj_a, cj_b));
  const int walk_max_cj = std::min(max_cj, std::max(cj_a, cj_b));

  std::vector<UVSegmentInCell> out;
  float u_lo = float(walk_min_ci) / float(grid_cols) + 1e-6f;
  float u_hi = float(walk_max_ci + 1) / float(grid_cols) - 1e-6f;
  float v_lo = float(walk_min_cj) / float(grid_rows) + 1e-6f;
  float v_hi = float(walk_max_cj + 1) / float(grid_rows) - 1e-6f;
  Vec2f a_clamped(std::clamp(a.x(), u_lo, u_hi), std::clamp(a.y(), v_lo, v_hi));
  Vec2f b_clamped(std::clamp(b.x(), u_lo, u_hi), std::clamp(b.y(), v_lo, v_hi));

  du = b_clamped.x() - a_clamped.x();
  dv = b_clamped.y() - a_clamped.y();
  Vec2f current_uv = a_clamped;
  auto [ci_start, cj_start] = uv_to_grid_cell(a_clamped.x(), a_clamped.y(),
                                              grid_cols, grid_rows, du, dv);
  int ci = std::clamp(ci_start, walk_min_ci, walk_max_ci);
  int cj = std::clamp(cj_start, walk_min_cj, walk_max_cj);

  constexpr float t_done_eps = 1e-6f;
  for (;;) {
    auto [t, exit_edge] =
        segment_exit_cell(current_uv, b_clamped, ci, cj, grid_cols, grid_rows);
    float end_u = current_uv.x() + t * (b_clamped.x() - current_uv.x());
    float end_v = current_uv.y() + t * (b_clamped.y() - current_uv.y());
    Vec2f end_uv(end_u, end_v);
    out.push_back({current_uv, end_uv, ci, cj});

    if (t >= 1.f - t_done_eps)
      break;

    current_uv = end_uv;
    auto next = adjacent_cell_bounded(ci, cj, exit_edge, walk_min_ci,
                                      walk_max_ci, walk_min_cj, walk_max_cj);
    if (next.first == ci && next.second == cj)
      break;
    ci = next.first;
    cj = next.second;
  }
  return out;
}

// Extract contour segments that lie in black grid cells by clipping each edge
// to UV cells. Uses face-based algorithm: find faces for each segment, get UV
// island, restrict projection and grid subdivision to that island.
static Polylines
extract_black_contour_segments(const Polygon &contour, double z_mm,
                               const CachedUVMesh &cache, int grid_cols,
                               int grid_rows, double origin_x_mm,
                               double origin_y_mm, double outward_offset_mm) {
  Polylines result;
  const size_t n = contour.points.size();
  if (n == 0)
    return result;

  if (cache.face_to_island.empty() || cache.island_faces.empty())
    return result;

  std::vector<std::pair<Point, Point>> segments;

  for (size_t k = 0; k < n; ++k) {
    Point p_a = contour.points[k];
    Point p_b = contour.points[(k + 1) % n];
    const Point A_xy = point_to_model_surface_mm(cache, p_a, outward_offset_mm);
    const Point B_xy = point_to_model_surface_mm(cache, p_b, outward_offset_mm);

    double ax_mm = double(A_xy.x());
    double ay_mm = double(A_xy.y());
    double bx_mm = double(B_xy.x());
    double by_mm = double(B_xy.y());

    std::vector<SubSegmentWithFaces> sub_segments =
        split_contour_segment_by_faces(cache, ax_mm, ay_mm, bx_mm, by_mm, z_mm);

    for (const SubSegmentWithFaces &sub : sub_segments) {
      if (sub.faces.empty())
        continue;

      int island_id = cache.face_to_island[sub.faces[0]];
      if (island_id < 0 || size_t(island_id) >= cache.island_uv_bounds.size())
        continue;

      const std::vector<size_t> &island_face_list =
          cache.island_faces[island_id];

      auto A_uv =
          point_to_uv(cache, sub.ax_mm, sub.ay_mm, z_mm, &island_face_list);
      auto B_uv =
          point_to_uv(cache, sub.bx_mm, sub.by_mm, z_mm, &island_face_list);

      if (!A_uv || !B_uv)
        continue;

      std::vector<UVSegmentInCell> cell_segments =
          subdivide_uv_segment_by_grid(*A_uv, *B_uv, grid_cols, grid_rows);

      for (const UVSegmentInCell &seg : cell_segments) {
        if (!is_black_cell(seg.ci, seg.cj))
          continue;

        auto start_opt = uv_to_point(cache, seg.start_uv.x(), seg.start_uv.y(),
                                     &island_face_list);
        auto end_opt = uv_to_point(cache, seg.end_uv.x(), seg.end_uv.y(),
                                   &island_face_list);

        if (!start_opt || !end_opt)
          continue;

        Point start_pt = point_mm_to_model_surface(
            cache, start_opt->x(), start_opt->y(), outward_offset_mm);
        Point end_pt = point_mm_to_model_surface(
            cache, end_opt->x(), end_opt->y(), outward_offset_mm);
        segments.push_back({start_pt, end_pt});
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
  result = chain_polylines(std::move(result));
  return result;
}

// Cast a ray from centroid through outer_pt and find where it intersects the
// inner contour. Returns the intersection point if found.
static bool ray_intersect_inner_contour(const Point &centroid,
                                        const Point &outer_pt,
                                        const Polygon &inner_contour,
                                        Point *intersection_out) {
  if (inner_contour.points.size() < 2)
    return false;
  // Ray from centroid through outer_pt. first_intersection returns the point
  // closest to line.a (centroid), i.e. the inner boundary.
  Line ray(centroid, outer_pt);
  return inner_contour.first_intersection(ray, intersection_out);
}

// Find which polygon edge (0..n-1) the point lies on. Returns -1 if not on contour.
static int find_edge_for_point(const Polygon &contour, const Point &pt) {
  const size_t n = contour.points.size();
  if (n < 2)
    return -1;
  const double eps2 = double(scale_(0.001)) * scale_(0.001);
  int best = -1;
  double best_d2 = std::numeric_limits<double>::max();
  for (size_t i = 0; i < n; ++i) {
    const Point &a = contour.points[i];
    const Point &b = contour.points[(i + 1) % n];
    Line seg(a, b);
    Point nearest;
    double d2 = line_alg::distance_to_squared(seg, pt, &nearest);
    if (d2 < best_d2 && d2 <= eps2) {
      best_d2 = d2;
      best = int(i);
    }
  }
  return best;
}

// Path from a to b along the contour (shorter arc). All points lie on the contour.
static Polyline path_along_contour(const Polygon &contour, const Point &a,
                                   const Point &b) {
  const size_t n = contour.points.size();
  if (n < 2)
    return Polyline();

  const coord_t eps2 = scale_(0.001) * scale_(0.001);
  auto same_pt = [eps2](const Point &p, const Point &q) {
    return (p - q).cast<double>().squaredNorm() <= eps2;
  };
  if (same_pt(a, b))
    return Polyline{a};

  int ei = find_edge_for_point(contour, a);
  int ej = find_edge_for_point(contour, b);
  if (ei < 0 || ej < 0)
    return Polyline();

  if (ei == ej) {
    return Polyline{a, b};
  }

  // Compute arc lengths: forward from vertex (ei+1)%n to ej, backward from ei to (ej+1)%n.
  auto arc_length_forward = [&]() -> double {
    double len = 0;
    for (size_t i = (ei + 1) % n;; i = (i + 1) % n) {
      len += (contour.points[(i + 1) % n] - contour.points[i])
                 .cast<double>()
                 .norm();
      if ((i + 1) % n == size_t(ej))
        break;
    }
    return len;
  };
  auto arc_length_backward = [&]() -> double {
    double len = 0;
    for (size_t i = ei;; i = (i + n - 1) % n) {
      len += (contour.points[i] - contour.points[(i + n - 1) % n])
                 .cast<double>()
                 .norm();
      if (i == (ej + 1) % n)
        break;
    }
    return len;
  };
  double len_fwd = arc_length_forward();
  double len_bwd = arc_length_backward();

  Polyline path;
  path.points.push_back(a);
  if (len_fwd <= len_bwd) {
    for (size_t k = (ei + 1) % n;; k = (k + 1) % n) {
      path.points.push_back(contour.points[k]);
      if (k == size_t(ej))
        break;
    }
  } else {
    for (size_t k = ei;; k = (k + n - 1) % n) {
      path.points.push_back(contour.points[k]);
      if (k == (ej + 1) % n)
        break;
    }
  }
  path.points.push_back(b);
  return path;
}

// Project each segment endpoint of outer polylines onto the inner contour via
// radial projection from centroid. All points and segments lie on the inner
// contour. Preserves filled flag. Filters by min path length.
static Polylines project_segments_radially(const Polylines &outer_polylines,
                                           const Polygon &inner_contour,
                                           const Point &centroid,
                                           double min_segment_length_mm) {
  Polylines result;
  const double min_len = scale_(min_segment_length_mm);
  std::vector<Polyline> paths;

  for (const Polyline &pl : outer_polylines) {
    if (pl.points.size() < 2)
      continue;
    for (size_t i = 0; i + 1 < pl.points.size(); ++i) {
      Point proj_a, proj_b;
      if (!ray_intersect_inner_contour(centroid, pl.points[i], inner_contour,
                                       &proj_a))
        continue;
      if (!ray_intersect_inner_contour(centroid, pl.points[i + 1], inner_contour,
                                       &proj_b))
        continue;
      proj_a = inner_contour.point_projection(proj_a);
      proj_b = inner_contour.point_projection(proj_b);

      Polyline path = path_along_contour(inner_contour, proj_a, proj_b);
      if (path.length() >= min_len && path.points.size() >= 2)
        paths.push_back(std::move(path));
    }
  }

  const coord_t eps2 = scale_(0.001) * scale_(0.001);
  auto same_point = [eps2](const Point &a, const Point &b) {
    Vec2d d = (a - b).cast<double>();
    return d.squaredNorm() <= eps2;
  };

  std::vector<bool> used(paths.size(), false);
  for (size_t i = 0; i < paths.size(); ++i) {
    if (used[i])
      continue;
    Polyline pl = std::move(paths[i]);
    used[i] = true;
    bool changed;
    do {
      changed = false;
      for (size_t j = 0; j < paths.size(); ++j) {
        if (used[j])
          continue;
        const Polyline &p = paths[j];
        if (p.points.size() < 2)
          continue;
        if (same_point(pl.points.back(), p.points.front())) {
          for (size_t k = 1; k < p.points.size(); ++k)
            pl.points.push_back(p.points[k]);
          used[j] = true;
          changed = true;
        } else if (same_point(pl.points.back(), p.points.back())) {
          for (size_t k = p.points.size() - 1; k-- > 0;)
            pl.points.push_back(p.points[k]);
          used[j] = true;
          changed = true;
        } else if (same_point(pl.points.front(), p.points.back())) {
          for (size_t k = p.points.size() - 1; k-- > 0;)
            pl.points.insert(pl.points.begin(), p.points[k]);
          used[j] = true;
          changed = true;
        } else if (same_point(pl.points.front(), p.points.front())) {
          for (size_t k = 1; k < p.points.size(); ++k)
            pl.points.insert(pl.points.begin(), p.points[k]);
          used[j] = true;
          changed = true;
        }
      }
    } while (changed);
    if (pl.points.size() >= 2)
      result.push_back(std::move(pl));
  }
  return chain_polylines(std::move(result));
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

  std::shared_ptr<CachedUVMesh> cache = get_or_load_uv_mesh(m_uv_map_file_path);

  if (cache && cache->valid) {
    const double outward_offset_mm =
        std::max(0., 0.5 * this->spacing - this->overlap);
    const double ox = m_contour_to_mesh_origin_mm.x();
    const double oy = m_contour_to_mesh_origin_mm.y();

    polylines_out = extract_black_contour_segments(
        outer_contour, z_mm, *cache, DEFAULT_GRID_COLS, DEFAULT_GRID_ROWS, ox,
        oy, outward_offset_mm);

    if (!expolygon.holes.empty()) {
      const Polygon &inner_contour = expolygon.holes[0];
      Point centroid = outer_contour.centroid();
      const double min_segment_length_mm =
          std::max(0.2, 0.5 * unscale_(this->spacing));
      Polylines inner_polylines = project_segments_radially(
          polylines_out, inner_contour, centroid, min_segment_length_mm);
      polylines_out.insert(polylines_out.end(), inner_polylines.begin(),
                           inner_polylines.end());
    }
  }
}

} // namespace Slic3r
