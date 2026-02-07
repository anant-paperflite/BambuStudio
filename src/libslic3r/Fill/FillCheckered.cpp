#include "FillCheckered.hpp"

#include "../AABBTreeIndirect.hpp"
#include "../Format/OBJ.hpp"
#include "../Point.hpp"
#include "../TriangleMesh.hpp"
#include "../SVG.hpp"
#include "../Utils.hpp"

#include <boost/log/trivial.hpp>

#include <algorithm>
#include <cmath>
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

    static std::optional<CachedUVMesh> load(const std::string &path) {
        if (path.empty()) {
            BOOST_LOG_TRIVIAL(debug) << "FillCheckered: UV map path empty, skip load";
            return std::nullopt;
        }
        BOOST_LOG_TRIVIAL(debug) << "FillCheckered: loading UV map from " << path;
        TriangleMesh mesh;
        ObjInfo      obj_info;
        std::string  message;
        if (!load_obj(path.c_str(), &mesh, obj_info, message, false)) {
            BOOST_LOG_TRIVIAL(warning) << "FillCheckered: load_obj failed: " << message;
            return std::nullopt;
        }
        if (mesh.empty()) {
            BOOST_LOG_TRIVIAL(warning) << "FillCheckered: UV map mesh is empty";
            return std::nullopt;
        }
        if (obj_info.uvs.size() != mesh.its.indices.size()) {
            BOOST_LOG_TRIVIAL(warning) << "FillCheckered: UV count " << obj_info.uvs.size()
                << " != triangle count " << mesh.its.indices.size() << ", need one UV per face";
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
        BOOST_LOG_TRIVIAL(debug) << "FillCheckered: UV map loaded, " << out.mesh.its.indices.size() << " triangles";
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
        BOOST_LOG_TRIVIAL(debug) << "FillCheckered: UV map cache hit for " << path;
        return it->second;
    }
    auto opt = CachedUVMesh::load(path);
    if (!opt) return nullptr;
    auto ptr = std::make_shared<CachedUVMesh>(std::move(*opt));
    s_uv_cache[path] = ptr;
    return ptr;
}

// Map 3D point to (u,v) in [0,1]^2; returns nullopt if ray misses mesh.
static std::optional<Vec2f> point_to_uv(
    const CachedUVMesh &cache, double x_mm, double y_mm, double z_mm)
{
    const indexed_triangle_set &its = cache.mesh.its;
    if (its.vertices.empty() || its.indices.empty() || cache.uvs.size() != its.indices.size())
        return std::nullopt;
    Vec3d origin(x_mm, y_mm, z_mm);
    Vec3d dir(0., 0., -1.);
    igl::Hit hit;
    const double eps = 1e-6;
    if (!AABBTreeIndirect::intersect_ray_first_hit(
            its.vertices, its.indices, cache.tree, origin, dir, hit, eps))
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
    BOOST_LOG_TRIVIAL(debug) << "FillCheckered: wrote debug SVGs " << path_xy_before << " " << path_xy_after << " " << path_uv;
}
#else
void FillCheckered::write_debug_svgs(const Surface *,
                                     const Polylines &,
                                     const Polylines &,
                                     const std::vector<std::pair<Vec2f, Vec2f>> &) const
{
}
#endif

Polylines FillCheckered::fill_surface(const Surface *surface, const FillParams &params)
{
    BOOST_LOG_TRIVIAL(debug) << "FillCheckered::fill_surface() layer_id=" << this->layer_id << " z=" << this->z;

    Polylines polylines_out;
    if (!this->fill_surface_by_multilines(
            surface, params,
            {{0.f, 0.f}, {float(M_PI / 2.), 0.f}},
            polylines_out))
        BOOST_LOG_TRIVIAL(error) << "FillCheckered::fill_surface() failed to fill a region.";

    if (this->layer_id % 2 == 1)
        for (size_t i = 0; i < polylines_out.size(); i++)
            std::reverse(polylines_out[i].begin(), polylines_out[i].end());

    size_t segments_before = 0;
    for (const Polyline &pl : polylines_out)
        segments_before += (pl.points.size() < 2) ? 0 : (pl.points.size() - 1);

    // If UV map path is set, filter segments by UV grid cell (checkered pattern)
    if (!m_uv_map_file_path.empty()) {
        std::shared_ptr<CachedUVMesh> cache = get_or_load_uv_mesh(m_uv_map_file_path);
        if (cache && cache->valid) {
            const double z_mm   = this->z;
            const int    gcols  = DEFAULT_GRID_COLS;
            const int    grows  = DEFAULT_GRID_ROWS;
            Polylines    filtered;
            size_t       segments_kept = 0;
            size_t       segments_dropped = 0;
            size_t       ray_miss = 0;

#ifdef CHECKERED_INFILL_DEBUG_SVG
            std::vector<std::pair<Vec2f, Vec2f>> uv_segments; // for UV-space SVG
            Polylines polylines_before_filter = polylines_out; // copy for before-SVG
#endif
            for (Polyline &pline : polylines_out) {
                if (pline.points.size() < 2) continue;
                Polyline out_pline;
                for (size_t k = 0; k + 1 < pline.points.size(); ++k) {
                    const Point &a = pline.points[k];
                    const Point &b = pline.points[k + 1];
                    double mx = (unscale_(a.x()) + unscale_(b.x())) * 0.5;
                    double my = (unscale_(a.y()) + unscale_(b.y())) * 0.5;
                    auto cell = point_to_grid_cell(*cache, mx, my, z_mm, gcols, grows);
                    if (!cell) ++ray_miss;
                    if (cell && is_fill_cell(cell->first, cell->second)) {
                        ++segments_kept;
                        if (out_pline.points.empty())
                            out_pline.points.push_back(a);
                        out_pline.points.push_back(b);
#ifdef CHECKERED_INFILL_DEBUG_SVG
                        auto uva = point_to_uv(*cache, unscale_(a.x()), unscale_(a.y()), z_mm);
                        auto uvb = point_to_uv(*cache, unscale_(b.x()), unscale_(b.y()), z_mm);
                        if (uva && uvb) uv_segments.push_back({*uva, *uvb});
#endif
                    } else {
                        ++segments_dropped;
                        if (!out_pline.points.empty()) {
                            filtered.push_back(std::move(out_pline));
                            out_pline = Polyline();
                        }
                    }
                }
                if (!out_pline.points.empty())
                    filtered.push_back(std::move(out_pline));
            }
            polylines_out = std::move(filtered);
            BOOST_LOG_TRIVIAL(debug) << "FillCheckered: segments before=" << segments_before
                << " kept=" << segments_kept << " dropped=" << segments_dropped
                << " ray_miss=" << ray_miss << " polylines_out=" << polylines_out.size();

#ifdef CHECKERED_INFILL_DEBUG_SVG
            write_debug_svgs(surface, polylines_before_filter, polylines_out, uv_segments);
#endif
        } else {
            BOOST_LOG_TRIVIAL(debug) << "FillCheckered: UV map not available, using full grid (no filter)";
        }
    } else {
        BOOST_LOG_TRIVIAL(debug) << "FillCheckered: no UV map path, using full grid";
    }

    return polylines_out;
}

} // namespace Slic3r
