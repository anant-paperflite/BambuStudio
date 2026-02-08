#include "FillCheckered.hpp"

#include "../AABBTreeIndirect.hpp"
#include "../Format/OBJ.hpp"
#include "../Point.hpp"
#include "../TriangleMesh.hpp"
#include "../SVG.hpp"
#include "../Utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

// Enable to write debug SVGs (XY segments and UV-space segments) to g_data_dir/SVG/
#define CHECKERED_INFILL_DEBUG_SVG

namespace Slic3r {

namespace {

// Default grid resolution for UV space [0,1]^2
constexpr int DEFAULT_GRID_COLS = 10;
constexpr int DEFAULT_GRID_ROWS = 10;

struct CachedUVMesh {
    TriangleMesh                                    mesh;
    std::vector<std::array<Vec2f, 3>>               uvs;
    AABBTreeIndirect::Tree<3, float>               tree;
    bool                                            valid{false};
    Vec3f                                           bbox_min{0.f, 0.f, 0.f};
    Vec3f                                           bbox_max{0.f, 0.f, 0.f};

    static std::optional<CachedUVMesh> load(const std::string &path) {
        if (path.empty()) {
            printf("FillCheckered: UV map path empty, skip load\n");
            return std::nullopt;
        }
        printf("FillCheckered: loading UV map from %s\n", path.c_str());
        TriangleMesh mesh;
        ObjInfo      obj_info;
        std::string  message;
        if (!load_obj(path.c_str(), &mesh, obj_info, message, false)) {
            printf("FillCheckered: load_obj failed: %s\n", message.c_str());
            return std::nullopt;
        }
        if (mesh.empty()) {
            printf("FillCheckered: UV map mesh is empty\n");
            return std::nullopt;
        }
        if (obj_info.uvs.size() != mesh.its.indices.size()) {
            printf("FillCheckered: UV count %zu != triangle count %zu, need one UV per face\n",
                   obj_info.uvs.size(), mesh.its.indices.size());
            return std::nullopt;
        }
        AABBTreeIndirect::Tree<3, float> tree =
            AABBTreeIndirect::build_aabb_tree_over_indexed_triangle_set(
                mesh.its.vertices, mesh.its.indices);
        CachedUVMesh out;
        out.mesh  = std::move(mesh);
        out.uvs   = std::move(obj_info.uvs);
        out.tree  = std::move(tree);
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
            printf("FillCheckered: UV mesh bbox mm: x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]\n",
                   mn.x(), mx.x(), mn.y(), mx.y(), mn.z(), mx.z());
        }
        printf("FillCheckered: UV map loaded, %zu triangles\n", out.mesh.its.indices.size());
        return out;
    }
};

static std::mutex                                    s_cache_mutex;
static std::map<std::string, std::shared_ptr<CachedUVMesh>> s_uv_cache;

std::shared_ptr<CachedUVMesh> get_or_load_uv_mesh(const std::string &path) {
    if (path.empty()) return nullptr;
    std::lock_guard<std::mutex> lock(s_cache_mutex);
    auto it = s_uv_cache.find(path);
    if (it != s_uv_cache.end()) {
        printf("FillCheckered: UV map cache hit for %s\n", path.c_str());
        return it->second;
    }
    auto opt = CachedUVMesh::load(path);
    if (!opt) return nullptr;
    auto ptr = std::make_shared<CachedUVMesh>(std::move(*opt));
    s_uv_cache[path] = ptr;
    return ptr;
}

// Map 3D point to (u,v) in [0,1]^2; returns nullopt if ray misses mesh.
// If the UV mesh has Z in [-H, 0] (e.g. top=0, bottom negative), we use ray_z = -z_mm to match slice height.
static std::optional<Vec2f> point_to_uv(
    const CachedUVMesh &cache, double x_mm, double y_mm, double z_mm)
{
    const indexed_triangle_set &its = cache.mesh.its;
    if (its.vertices.empty() || its.indices.empty() || cache.uvs.size() != its.indices.size()){
      printf("FillCheckered: point_to_uv: vertices empty or indices empty or uvs size != indices size: %zu != %zu\n", its.vertices.size(), its.indices.size() , cache.uvs.size());
      return std::nullopt;
    }
    // Mesh Z non-positive (e.g. [−40,0]) vs slicer print_z positive → use -z_mm for ray
    double ray_z = (cache.bbox_max.z() <= 0.f) ? -z_mm : z_mm;
    Vec3d origin(x_mm, y_mm, ray_z);
    igl::Hit hit;
    const double eps = 1e-6;
    if (!AABBTreeIndirect::intersect_ray_first_hit(
            its.vertices, its.indices, cache.tree, origin, Vec3d(0., 0., 1.), hit, eps)
        && !AABBTreeIndirect::intersect_ray_first_hit(
            its.vertices, its.indices, cache.tree, origin, Vec3d(0., 0., -1.), hit, eps))
        return std::nullopt;
    if (hit.id < 0 || size_t(hit.id) >= cache.uvs.size())
        return std::nullopt;
    const float w0 = 1.f - hit.u - hit.v;
    const float w1 = hit.u;
    const float w2 = hit.v;
    const std::array<Vec2f, 3> &uv_arr = cache.uvs[hit.id];
    float u = w0 * uv_arr[0].x() + w1 * uv_arr[1].x() + w2 * uv_arr[2].x();
    float v = w0 * uv_arr[0].y() + w1 * uv_arr[1].y() + w2 * uv_arr[2].y();
    u = std::clamp(u, 0.f, 1.f);
    v = std::clamp(v, 0.f, 1.f);
    return Vec2f(u, v);
}

// Map 3D point (x_mm, y_mm, z_mm) to UV grid cell (i, j) using the cached UV mesh.
std::optional<std::pair<int, int>> point_to_grid_cell(
    const CachedUVMesh                              &cache,
    double                                          x_mm,
    double                                          y_mm,
    double                                          z_mm,
    int                                             grid_cols,
    int                                             grid_rows)
{
    auto uv = point_to_uv(cache, x_mm, y_mm, z_mm);
    if (!uv) return std::nullopt;
    float u = uv->x(), v = uv->y();
    int i = static_cast<int>(std::floor(u * grid_cols));
    int j = static_cast<int>(std::floor(v * grid_rows));
    if (u >= 1.f) i = grid_cols - 1;
    if (v >= 1.f) j = grid_rows - 1;
    i = std::clamp(i, 0, grid_cols - 1);
    j = std::clamp(j, 0, grid_rows - 1);
    return std::make_pair(i, j);
}

// Checkered pattern: fill cell iff (i + j) % 2 == 0
inline bool is_fill_cell(int i, int j) { return (i + j) % 2 == 0; }

// Black grid cell is the opposite of fill (checkered pattern)
inline bool is_black_cell(int i, int j) { return (i + j) % 2 == 1; }

// Per-point result: where the contour point lies in UV space and whether that cell is black.
struct ContourPointUVInfo {
    Point                    point;         // contour point (XY)
    std::optional<Vec2f>     uv;            // UV in [0,1]^2 if ray hit
    std::optional<std::pair<int, int>> grid_cell;  // (i, j) if uv is valid
    bool                     is_black_cell{false}; // true iff grid_cell is set and that cell is black
};

// Map every point on a contour to UV and detect if it lies on a black grid cell.
// origin_x_mm, origin_y_mm: added to contour mm so ray is in UV mesh (raw model) coords; contour is object-centered.
std::vector<ContourPointUVInfo> get_contour_points_uv_info(
    const Polygon          &contour,
    double                  z_mm,
    const CachedUVMesh     &cache,
    int                     grid_cols,
    int                     grid_rows,
    double                  origin_x_mm,
    double                  origin_y_mm)
{
    std::vector<ContourPointUVInfo> out;
    out.reserve(contour.points.size());
    for (const Point &pt : contour.points) {
        double x_mm = unscale_(pt.x()) + origin_x_mm;
        double y_mm = unscale_(pt.y()) + origin_y_mm;
        std::optional<Vec2f> uv = point_to_uv(cache, x_mm, y_mm, z_mm);
        ContourPointUVInfo info;
        info.point = pt;
        if (uv) {
            info.uv = *uv;
            auto cell = point_to_grid_cell(cache, x_mm, y_mm, z_mm, grid_cols, grid_rows);
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
    float &u_min, float &u_max, float &v_min, float &v_max)
{
    u_min = float(i) / float(grid_cols);
    u_max = float(i + 1) / float(grid_cols);
    v_min = float(j) / float(grid_rows);
    v_max = float(j + 1) / float(grid_rows);
    u_min = std::clamp(u_min, 0.f, 1.f);
    u_max = std::clamp(u_max, 0.f, 1.f);
    v_min = std::clamp(v_min, 0.f, 1.f);
    v_max = std::clamp(v_max, 0.f, 1.f);
}

// Intersect segment a->b (UV) with a vertical line u = u_edge, segment from (u_edge, v_lo) to (u_edge, v_hi).
// Returns t in [0,1] if hit, else nullopt. t is parameter for a + t*(b-a).
static std::optional<float> segment_intersect_vertical(float u_edge, float v_lo, float v_hi,
    const Vec2f &a, const Vec2f &b)
{
    float du = b.x() - a.x();
    if (std::abs(du) < 1e-9f) return std::nullopt;
    float t = (u_edge - a.x()) / du;
    if (t < 0.f || t > 1.f) return std::nullopt;
    float v = a.y() + t * (b.y() - a.y());
    if (v < v_lo || v > v_hi) return std::nullopt;
    return t;
}

// Intersect segment a->b (UV) with a horizontal line v = v_edge, segment from (u_lo, v_edge) to (u_hi, v_edge).
static std::optional<float> segment_intersect_horizontal(float u_lo, float u_hi, float v_edge,
    const Vec2f &a, const Vec2f &b)
{
    float dv = b.y() - a.y();
    if (std::abs(dv) < 1e-9f) return std::nullopt;
    float t = (v_edge - a.y()) / dv;
    if (t < 0.f || t > 1.f) return std::nullopt;
    float u = a.x() + t * (b.x() - a.x());
    if (u < u_lo || u > u_hi) return std::nullopt;
    return t;
}

// Exit parameter: smallest t in (0, 1] where segment a->b (UV) exits cell (ci, cj), or 1 if b is inside.
// Returns (t, exit_edge): exit_edge 0=left, 1=right, 2=bottom, 3=top (for adjacent cell).
static std::pair<float, int> segment_exit_cell(const Vec2f &a, const Vec2f &b, int ci, int cj,
    int grid_cols, int grid_rows)
{
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
    consider(segment_intersect_vertical(u_min, v_min, v_max, a, b), 0);  // left
    consider(segment_intersect_vertical(u_max, v_min, v_max, a, b), 1);  // right
    consider(segment_intersect_horizontal(u_min, u_max, v_min, a, b), 2);  // bottom
    consider(segment_intersect_horizontal(u_min, u_max, v_max, a, b), 3);  // top
    if (edge_best < 0)
        return {1.f, -1};
    return {t_best, edge_best};
}

// New cell when exiting through edge: 0=left, 1=right, 2=bottom, 3=top.
static std::pair<int, int> adjacent_cell(int ci, int cj, int edge, int grid_cols, int grid_rows)
{
    int ni = ci, nj = cj;
    if (edge == 0) ni = ci - 1;
    else if (edge == 1) ni = ci + 1;
    else if (edge == 2) nj = cj - 1;
    else if (edge == 3) nj = cj + 1;
    ni = std::clamp(ni, 0, grid_cols - 1);
    nj = std::clamp(nj, 0, grid_rows - 1);
    return {ni, nj};
}

// Extract contour segments that lie in black grid cells by clipping each edge to UV cells.
// origin_x_mm, origin_y_mm: contour is object-centered; add to get ray in UV mesh (raw model) coords.
static Polylines extract_black_contour_segments(
    const Polygon       &contour,
    double               z_mm,
    const CachedUVMesh  &cache,
    int                  grid_cols,
    int                  grid_rows,
    double               origin_x_mm,
    double               origin_y_mm)
{
    Polylines result;
    const size_t n = contour.points.size();
    if (n == 0) return result;

    std::vector<std::pair<Point, Point>> segments;

    for (size_t k = 0; k < n; ++k) {
        const Point A_xy = contour.points[k];
        const Point B_xy = contour.points[(k + 1) % n];
        double ax_mm = unscale_(A_xy.x()) + origin_x_mm;
        double ay_mm = unscale_(A_xy.y()) + origin_y_mm;
        double bx_mm = unscale_(B_xy.x()) + origin_x_mm;
        double by_mm = unscale_(B_xy.y()) + origin_y_mm;

        auto A_uv = point_to_uv(cache, ax_mm, ay_mm, z_mm);
        auto B_uv = point_to_uv(cache, bx_mm, by_mm, z_mm);
        if (!A_uv || !B_uv) continue;

        printf("A_uv: %f, %f\n", A_uv->x(), A_uv->y());
        printf("B_uv: %f, %f\n", B_uv->x(), B_uv->y());

        auto cell_opt = point_to_grid_cell(cache, ax_mm, ay_mm, z_mm, grid_cols, grid_rows);
        if (!cell_opt) continue;
        int ci = cell_opt->first, cj = cell_opt->second;

        Vec2f current_uv = *A_uv;
        Point current_xy = A_xy;

        for (;;) {
            auto [t, exit_edge] = segment_exit_cell(current_uv, *B_uv, ci, cj, grid_cols, grid_rows);

            Point end_xy(
                coord_t(std::round(double(current_xy.x()) + t * (double(B_xy.x()) - double(current_xy.x())))),
                coord_t(std::round(double(current_xy.y()) + t * (double(B_xy.y()) - double(current_xy.y())))));
            if (is_black_cell(ci, cj))
                segments.push_back({current_xy, end_xy});

            if (t >= 1.f - 1e-6f) break;

            current_xy = end_xy;
            current_uv = Vec2f(
                current_uv.x() + t * (B_uv->x() - current_uv.x()),
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
        if (used[i]) continue;
        Polyline pl;
        pl.points.push_back(segments[i].first);
        pl.points.push_back(segments[i].second);
        used[i] = true;
        bool changed;
        do {
            changed = false;
            for (size_t j = 0; j < segments.size(); ++j) {
                if (used[j]) continue;
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
void FillCheckered::write_debug_svgs(const Surface *surface,
                                     const Polylines &polylines_before_filter,
                                     const Polylines &polylines_after_filter,
                                     const std::vector<std::pair<Vec2f, Vec2f>> &uv_segments) const
{
    static int s_svg_run = 0;
    const int run = s_svg_run++;
    std::string path_xy_before = debug_out_path("fill_checkered_xy_before_layer%d_z%.2f_run%d.svg",
        int(this->layer_id), this->z, run);
    std::string path_xy_after  = debug_out_path("fill_checkered_xy_after_layer%d_z%.2f_run%d.svg",
        int(this->layer_id), this->z, run);
    std::string path_uv       = debug_out_path("fill_checkered_uv_layer%d_z%.2f_run%d.svg",
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
        BoundingBox uv_bbox(Point(0, 0), Point(coord_t(scale_(100.)), coord_t(scale_(100.))));
        SVG svg(path_uv, uv_bbox, scale_(1.), true);
        if (svg.is_opened()) {
            for (int g = 0; g <= 10; ++g) {
                coord_t c = scale_(g * 10.);
                svg.draw(Line(Point(c, 0), Point(c, coord_t(scale_(100.)))), "lightgray", scale_(0.2));
                svg.draw(Line(Point(0, c), Point(coord_t(scale_(100.)), c)), "lightgray", scale_(0.2));
            }
            for (const auto &seg : uv_segments) {
                float u1 = seg.first.x() * 100.f, v1 = (1.f - seg.first.y()) * 100.f;
                float u2 = seg.second.x() * 100.f, v2 = (1.f - seg.second.y()) * 100.f;
                Point p1(coord_t(scale_(u1)), coord_t(scale_(v1))), p2(coord_t(scale_(u2)), coord_t(scale_(v2)));
                svg.draw(Line(p1, p2), "blue", scale_(0.5));
            }
            svg.add_comment("Checkered infill in UV space; overlay on texture image to verify segments line on black lines");
        }
    }
    printf("FillCheckered: wrote debug SVGs %s %s %s\n", path_xy_before.c_str(), path_xy_after.c_str(), path_uv.c_str());
}
#else
void FillCheckered::write_debug_svgs(const Surface *,
                                     const Polylines &,
                                     const Polylines &,
                                     const std::vector<std::pair<Vec2f, Vec2f>> &) const
{
}
#endif


void FillCheckered::_fill_surface_single(
    const FillParams &params,
    unsigned int thickness_layers,
    const std::pair<float, Point> &direction,
    ExPolygon expolygon,
    Polylines &polylines_out)
{
    Polygon outer_contour = expolygon.contour;
    const double z_mm = this->z;

    // Detect where every contour point lies in UV space and whether it's on a black grid cell.
    std::shared_ptr<CachedUVMesh> cache = get_or_load_uv_mesh(m_uv_map_file_path);
    if (cache && cache->valid) {
        double ox = m_contour_to_mesh_origin_mm.x();
        double oy = m_contour_to_mesh_origin_mm.y();
        std::vector<ContourPointUVInfo> uv_info = get_contour_points_uv_info(
            outer_contour, z_mm, *cache, DEFAULT_GRID_COLS, DEFAULT_GRID_ROWS, ox, oy);
        size_t on_black = 0, on_white = 0, no_uv = 0;
        for (const ContourPointUVInfo &info : uv_info) {
            if (!info.grid_cell) {
                ++no_uv;
                continue;
            }
            if (info.is_black_cell)
                ++on_black;
            else
                ++on_white;
        }
        // If all rays missed, OBJ may be in object-centered coords: retry with origin 0,0
        if (no_uv == uv_info.size() && !outer_contour.points.empty()) {
            std::vector<ContourPointUVInfo> uv_info0 = get_contour_points_uv_info(
                outer_contour, z_mm, *cache, DEFAULT_GRID_COLS, DEFAULT_GRID_ROWS, 0., 0.);
            size_t no_uv0 = 0;
            for (const ContourPointUVInfo &info : uv_info0)
                if (!info.grid_cell) ++no_uv0;
            if (no_uv0 < uv_info0.size()) {
                ox = 0.; oy = 0.;
                uv_info = std::move(uv_info0);
                on_black = on_white = no_uv = 0;
                for (const ContourPointUVInfo &info : uv_info) {
                    if (!info.grid_cell) { ++no_uv; continue; }
                    if (info.is_black_cell) ++on_black; else ++on_white;
                }
                printf("FillCheckered: all rays missed with center_offset; used origin=0 and got %zu hits\n", uv_info.size() - no_uv);
            }
        }
        printf("FillCheckered: contour points uv_info: %zu on_black=%zu on_white=%zu no_uv=%zu\n",
               uv_info.size(), on_black, on_white, no_uv);
        // #region agent log — verification: mesh bbox, origin, ray, inside_bbox
        if (!outer_contour.points.empty()) {
            const Vec3f &bmn = cache->bbox_min, &bmx = cache->bbox_max;
            double c0x = unscale_(outer_contour.points.front().x());
            double c0y = unscale_(outer_contour.points.front().y());
            double rx = c0x + ox, ry = c0y + oy;
            const double tol = 0.01;
            bool inside = (rx >= double(bmn.x()) - tol && rx <= double(bmx.x()) + tol &&
                           ry >= double(bmn.y()) - tol && ry <= double(bmx.y()) + tol &&
                           z_mm >= double(bmn.z()) - tol && z_mm <= double(bmx.z()) + tol);
            std::ofstream f("/Users/anant/Documents/Personal/BambuStudio/.cursor/debug.log", std::ios::app);
            if (f)
                f << "FillCheckered mesh_bbox x[" << bmn.x() << "," << bmx.x() << "] y[" << bmn.y() << "," << bmx.y() << "] z[" << bmn.z() << "," << bmx.z() << "] origin=(" << ox << "," << oy << ") ray0=(" << rx << "," << ry << "," << z_mm << ") inside_bbox=" << (inside ? 1 : 0) << " no_uv=" << no_uv << "\n";
        }
        // #endregion

        // Extract segments that lie on black grid cells and export to SVG.
        Polylines black_polylines = extract_black_contour_segments(
            outer_contour, z_mm, *cache, DEFAULT_GRID_COLS, DEFAULT_GRID_ROWS, ox, oy);

        printf("FillCheckered: black_polylines size: %zu\n", black_polylines.size());

        if (!black_polylines.empty()) {
#ifdef CHECKERED_INFILL_DEBUG_SVG
            std::string path = debug_out_path("fill_checkered_black_segments_layer%d_z%.2f.svg",
                int(this->layer_id), this->z);
            BoundingBox bbox = get_extents(outer_contour);
            bbox.offset(scale_(2.));
            SVG svg(path, bbox);
            if (svg.is_opened()) {
                svg.draw_outline(outer_contour, "blue", scale_(0.05));
                svg.draw(black_polylines, "red", scale_(0.08));
                svg.Close();
            }
            printf("FillCheckered: wrote black segments SVG %s\n", path.c_str());
#endif
        }
    }

    polylines_out.push_back(Polyline({Point(0, 0), Point(10, 10)}));
}

// Polylines FillCheckered::fill_surface22(const Surface *surface, const FillParams &params)
// {
//     BOOST_LOG_TRIVIAL(debug) << "FillCheckered::fill_surface() layer_id=" << this->layer_id << " z=" << this->z;

//     Polylines polylines_out;
//     if (!this->fill_surface_by_multilines(
//             surface, params,
//             {{0.f, 0.f}, {float(M_PI / 2.), 0.f}},
//             polylines_out))
//         BOOST_LOG_TRIVIAL(error) << "FillCheckered::fill_surface() failed to fill a region.";

//     printf("layer_id: %d\n", this->layer_id);
//     printf("polylines_out size: %zu\n", polylines_out.size());

//     if( this->layer_id == 69) {
//       SVG::export_polyline("/Users/anant/Desktop/debug/out.svg", polylines_out[0]);                              // black, default width
//     }

//     if (this->layer_id % 2 == 1)
//         for (size_t i = 0; i < polylines_out.size(); i++)
//             std::reverse(polylines_out[i].begin(), polylines_out[i].end());

//     size_t segments_before = 0;
//     for (const Polyline &pl : polylines_out)
//         segments_before += (pl.points.size() < 2) ? 0 : (pl.points.size() - 1);

//     // If UV map path is set, filter segments by UV grid cell (checkered pattern)
//     if (!m_uv_map_file_path.empty()) {
//         std::shared_ptr<CachedUVMesh> cache = get_or_load_uv_mesh(m_uv_map_file_path);
//         if (cache && cache->valid) {
//             const double z_mm   = this->z;
//             const int    gcols  = DEFAULT_GRID_COLS;
//             const int    grows  = DEFAULT_GRID_ROWS;
//             Polylines    filtered;
//             size_t       segments_kept = 0;
//             size_t       segments_dropped = 0;
//             size_t       ray_miss = 0;

// #ifdef CHECKERED_INFILL_DEBUG_SVG
//             std::vector<std::pair<Vec2f, Vec2f>> uv_segments; // for UV-space SVG
//             Polylines polylines_before_filter = polylines_out; // copy for before-SVG
// #endif
//             for (Polyline &pline : polylines_out) {
//                 if (pline.points.size() < 2) continue;
//                 Polyline out_pline;
//                 for (size_t k = 0; k + 1 < pline.points.size(); ++k) {
//                     const Point &a = pline.points[k];
//                     const Point &b = pline.points[k + 1];
//                     double mx = (unscale_(a.x()) + unscale_(b.x())) * 0.5;
//                     double my = (unscale_(a.y()) + unscale_(b.y())) * 0.5;
//                     auto cell = point_to_grid_cell(*cache, mx, my, z_mm, gcols, grows);
//                     if (!cell) ++ray_miss;
//                     if (cell && is_fill_cell(cell->first, cell->second)) {
//                         ++segments_kept;
//                         if (out_pline.points.empty())
//                             out_pline.points.push_back(a);
//                         out_pline.points.push_back(b);
// #ifdef CHECKERED_INFILL_DEBUG_SVG
//                         auto uva = point_to_uv(*cache, unscale_(a.x()), unscale_(a.y()), z_mm);
//                         auto uvb = point_to_uv(*cache, unscale_(b.x()), unscale_(b.y()), z_mm);
//                         if (uva && uvb) uv_segments.push_back({*uva, *uvb});
// #endif
//                     } else {
//                         ++segments_dropped;
//                         if (!out_pline.points.empty()) {
//                             filtered.push_back(std::move(out_pline));
//                             out_pline = Polyline();
//                         }
//                     }
//                 }
//                 if (!out_pline.points.empty())
//                     filtered.push_back(std::move(out_pline));
//             }
//             polylines_out = std::move(filtered);
//             BOOST_LOG_TRIVIAL(debug) << "FillCheckered: segments before=" << segments_before
//                 << " kept=" << segments_kept << " dropped=" << segments_dropped
//                 << " ray_miss=" << ray_miss << " polylines_out=" << polylines_out.size();

// #ifdef CHECKERED_INFILL_DEBUG_SVG
//             write_debug_svgs(surface, polylines_before_filter, polylines_out, uv_segments);
// #endif
//         } else {
//             BOOST_LOG_TRIVIAL(debug) << "FillCheckered: UV map not available, using full grid (no filter)";
//         }
//     } else {
//         BOOST_LOG_TRIVIAL(debug) << "FillCheckered: no UV map path, using full grid";
//     }

//     return polylines_out;
// }

} // namespace Slic3r
