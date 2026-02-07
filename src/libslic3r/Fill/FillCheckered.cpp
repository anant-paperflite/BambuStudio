#include "FillCheckered.hpp"

#include <boost/log/trivial.hpp>

namespace Slic3r {

Polylines FillCheckered::fill_surface(const Surface *surface, const FillParams &params)
{
    // Stub: generate a grid pattern (same as FillGrid). UV map file path (m_uv_map_file_path)
    // can be used later to decide fill vs no-fill per cell when the format is defined.
    Polylines polylines_out;
    if (!this->fill_surface_by_multilines(
            surface, params,
            {{0.f, 0.f}, {float(M_PI / 2.), 0.f}},
            polylines_out))
        BOOST_LOG_TRIVIAL(error) << "FillCheckered::fill_surface() failed to fill a region.";

    if (this->layer_id % 2 == 1)
        for (size_t i = 0; i < polylines_out.size(); i++)
            std::reverse(polylines_out[i].begin(), polylines_out[i].end());
    return polylines_out;
}

} // namespace Slic3r
