#ifndef slic3r_FillCheckered_hpp_
#define slic3r_FillCheckered_hpp_

#include <string>
#include <utility>
#include <vector>

#include "FillBase.hpp"
#include "FillRectilinear.hpp"
#include "../Point.hpp"
#include "../Polyline.hpp"

namespace Slic3r {

class Surface;

class FillCheckered : public Fill
{
public:
    Fill* clone() const override { return new FillCheckered(*this); }
    ~FillCheckered() override = default;
    void set_uv_map_file_path(const std::string &path) override { m_uv_map_file_path = path; }
    // Offset in mm from object-centered contour to UV mesh (raw model) coordinates. Set from PrintObject::center_offset().
    void set_contour_to_mesh_origin_mm(double x_mm, double y_mm) { m_contour_to_mesh_origin_mm = Vec2d(x_mm, y_mm); }
    bool is_self_crossing() override { return true; }

protected:
    float _layer_angle(size_t idx) const override { return 0.f; }
    void _fill_surface_single(
      const FillParams &params,
      unsigned int thickness_layers,
      const std::pair<float, Point> &direction,
      ExPolygon expolygon,
      Polylines &polylines_out) override;

private:
    void write_debug_svgs(const Surface *surface,
                          const Polylines &polylines_before_filter,
                          const Polylines &polylines_after_filter,
                          const std::vector<std::pair<Vec2f, Vec2f>> &uv_segments) const;

    std::string m_uv_map_file_path;
    Vec2d       m_contour_to_mesh_origin_mm{0., 0.};
};

} // namespace Slic3r

#endif // slic3r_FillCheckered_hpp_
