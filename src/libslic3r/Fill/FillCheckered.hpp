#ifndef slic3r_FillCheckered_hpp_
#define slic3r_FillCheckered_hpp_

#include <map>

#include "../libslic3r.h"

#include "FillBase.hpp"

namespace Slic3r
{

  class FillCheckered : public Fill
  {

  using Segments = std::vector<Polylines>;

  public:
    Fill *clone() const override { return new FillCheckered(*this); };
    ~FillCheckered() override {}
    bool is_self_crossing() override { return true; }

  protected:
    void _fill_surface_single(
        const FillParams &params,
        unsigned int thickness_layers,
        const std::pair<float, Point> &direction,
        ExPolygon expolygon,
        Polylines &polylines_out) override;
  };

} // namespace Slic3r

#endif // slic3r_FillCrossHatch_hpp_