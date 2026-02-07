#ifndef slic3r_FillCheckered_hpp_
#define slic3r_FillCheckered_hpp_

#include <string>

#include "FillBase.hpp"
#include "FillRectilinear.hpp"

namespace Slic3r {

class Surface;

class FillCheckered : public FillRectilinear
{
public:
    Fill* clone() const override { return new FillCheckered(*this); }
    ~FillCheckered() override = default;
    void set_uv_map_file_path(const std::string &path) override { m_uv_map_file_path = path; }
    Polylines fill_surface(const Surface *surface, const FillParams &params) override;
    bool is_self_crossing() override { return true; }

protected:
    float _layer_angle(size_t idx) const override { return 0.f; }

private:
    std::string m_uv_map_file_path;
};

} // namespace Slic3r

#endif // slic3r_FillCheckered_hpp_
