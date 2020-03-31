// The libMesh Finite Element Library.
// Copyright (C) 2002-2020 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA



#ifndef LIBMESH_POINT_H
#define LIBMESH_POINT_H

// Local includes
#include "libmesh/type_vector.h"

namespace libMesh
{

/**
 * A \p Point defines a location in LIBMESH_DIM dimensional Real space.  Points
 * are always real-valued, even if the library is configured with
 * \p --enable-complex.
 *
 * \author Benjamin S. Kirk
 * \date 2003
 * \brief A geometric point in (x,y,z) space.
 */
class Point : public TypeVector<GeomReal>
{
public:

  /**
   * Constructor.  By default sets all entries to 0.  Gives the point
   * 0 in \p LIBMESH_DIM dimensions.
   */
  Point (const GeomReal x=0.,
         const GeomReal y=0.,
         const GeomReal z=0.) :
    TypeVector<GeomReal> (x,y,z)
  {}

  /**
   * Copy-constructor.
   */
  Point (const Point & p) :
    TypeVector<GeomReal> (p)
  {}

  /**
   * Copy-constructor.
   */
  Point (const TypeVector<GeomReal> & p) :
    TypeVector<GeomReal> (p)
  {}

  /**
   * Copy-assignment operator.
   */
  Point& operator=(const Point & p) = default;

  /**
   * Disambiguate constructing from non-Real scalars
   */
  template <typename T,
            typename = typename
              boostcopy::enable_if_c<ScalarTraits<T>::value,void>::type>
  Point (const T x) :
    TypeVector<GeomReal> (x,0,0)
  {}

  /**
   * Empty.
   */
  ~Point() {}

protected:

  /**
   * Make the derived class a friend.
   */
  friend class Node;
};

} // namespace libMesh

#endif // LIBMESH_POINT_H
