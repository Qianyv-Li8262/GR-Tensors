#pragma once

// Cached counterpart of tensors_all.hpp.  Include exactly one aggregate
// header in a translation unit so the cached and baseline Field definitions
// never collide.
#include "Index_and_variance.hpp"
#include "contraction_and_arithmetic.hpp"
#include "diff_forms.hpp"
#include "field_wrapper_cached.hpp"
#include "num_deriv.hpp"
#include "symm_asymm.hpp"
#include "tensor_class.hpp"
#include "type_lists.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>
