// The libMesh Finite Element Library.
// Copyright (C) 2002-2019 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

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

#ifndef LIBMESH_FACE_TRI6_IMPL_H
#define LIBMESH_FACE_TRI6_IMPL_H

// Local includes
#include "libmesh/side.h"
#include "libmesh/edge_edge3.h"
#include "libmesh/face_tri6.h"
#include "libmesh/enum_io_package.h"
#include "libmesh/enum_order.h"

namespace libMesh
{

// ------------------------------------------------------------
// Tri6 class member functions

template <typename RealType>
bool Tri6Templ<RealType>::is_vertex(const unsigned int i) const
{
  if (i < 3)
    return true;
  return false;
}

template <typename RealType>
bool Tri6Templ<RealType>::is_edge(const unsigned int i) const
{
  if (i < 3)
    return false;
  return true;
}

template <typename RealType>
bool Tri6Templ<RealType>::is_face(const unsigned int) const
{
  return false;
}

template <typename RealType>
bool Tri6Templ<RealType>::is_node_on_side(const unsigned int n,
                           const unsigned int s) const
{
  libmesh_assert_less (s, this->n_sides());
  return std::find(std::begin(side_nodes_map[s]),
                   std::end(side_nodes_map[s]),
                   n) != std::end(side_nodes_map[s]);
}

template <typename RealType>
std::vector<unsigned>
Tri6Templ<RealType>::nodes_on_side(const unsigned int s) const
{
  libmesh_assert_less(s, this->n_sides());
  return {std::begin(side_nodes_map[s]), std::end(side_nodes_map[s])};
}

template <typename RealType>
bool Tri6Templ<RealType>::has_affine_map() const
{
  // Make sure edges are straight
  if (!this->point(3).relative_fuzzy_equals
      ((this->point(0) + this->point(1))/2.))
    return false;
  if (!this->point(4).relative_fuzzy_equals
      ((this->point(1) + this->point(2))/2.))
    return false;
  if (!this->point(5).relative_fuzzy_equals
      ((this->point(2) + this->point(0))/2.))
    return false;

  return true;
}



template <typename RealType>
Order Tri6Templ<RealType>::default_order() const
{
  return SECOND;
}



template <typename RealType>
dof_id_type Tri6Templ<RealType>::key (const unsigned int s) const
{
  libmesh_assert_less (s, this->n_sides());

  switch (s)
    {
    case 0:

      return
        this->compute_key (this->node_id(3));

    case 1:

      return
        this->compute_key (this->node_id(4));

    case 2:

      return
        this->compute_key (this->node_id(5));

    default:
      libmesh_error_msg("Invalid side s = " << s);
    }
}



template <typename RealType>
unsigned int Tri6Templ<RealType>::which_node_am_i(unsigned int side,
                                   unsigned int side_node) const
{
  libmesh_assert_less (side, this->n_sides());
  libmesh_assert_less (side_node, Tri6::nodes_per_side);

  return Tri6::side_nodes_map[side][side_node];
}



template <typename RealType>
std::unique_ptr<ElemTempl<RealType>> Tri6Templ<RealType>::build_side_ptr (const unsigned int i,
                                            bool proxy)
{
  libmesh_assert_less (i, this->n_sides());

  if (proxy)
    return libmesh_make_unique<Side<Edge3,Tri6>>(this,i);

  else
    {
      std::unique_ptr<Elem> edge = libmesh_make_unique<Edge3>();
      edge->subdomain_id() = this->subdomain_id();

      // Set the nodes
      for (auto n : edge->node_index_range())
        edge->set_node(n) = this->node_ptr(Tri6::side_nodes_map[i][n]);

      return edge;
    }
}



template <typename RealType>
void Tri6Templ<RealType>::build_side_ptr (std::unique_ptr<Elem> & side,
                           const unsigned int i)
{
  this->template simple_build_side_ptr<Tri6>(side, i, EDGE3);
}



template <typename RealType>
void Tri6Templ<RealType>::connectivity(const unsigned int sf,
                        const IOPackage iop,
                        std::vector<dof_id_type> & conn) const
{
  libmesh_assert_less (sf, this->n_sub_elem());
  libmesh_assert_not_equal_to (iop, INVALID_IO_PACKAGE);

  switch (iop)
    {
    case TECPLOT:
      {
        conn.resize(4);
        switch(sf)
          {
          case 0:
            // linear sub-triangle 0
            conn[0] = this->node_id(0)+1;
            conn[1] = this->node_id(3)+1;
            conn[2] = this->node_id(5)+1;
            conn[3] = this->node_id(5)+1;

            return;

          case 1:
            // linear sub-triangle 1
            conn[0] = this->node_id(3)+1;
            conn[1] = this->node_id(1)+1;
            conn[2] = this->node_id(4)+1;
            conn[3] = this->node_id(4)+1;

            return;

          case 2:
            // linear sub-triangle 2
            conn[0] = this->node_id(5)+1;
            conn[1] = this->node_id(4)+1;
            conn[2] = this->node_id(2)+1;
            conn[3] = this->node_id(2)+1;

            return;

          case 3:
            // linear sub-triangle 3
            conn[0] = this->node_id(3)+1;
            conn[1] = this->node_id(4)+1;
            conn[2] = this->node_id(5)+1;
            conn[3] = this->node_id(5)+1;

            return;

          default:
            libmesh_error_msg("Invalid sf = " << sf);
          }
      }

    case VTK:
      {
        // VTK_QUADRATIC_TRIANGLE has same numbering as libmesh TRI6
        conn.resize(6);
        conn[0] = this->node_id(0);
        conn[1] = this->node_id(1);
        conn[2] = this->node_id(2);
        conn[3] = this->node_id(3);
        conn[4] = this->node_id(4);
        conn[5] = this->node_id(5);
        return;

        // Used to write out linear sub-triangles for VTK...
        /*
          conn.resize(3);
          switch(sf)
          {
          case 0:
          // linear sub-triangle 0
          conn[0] = this->node_id(0);
          conn[1] = this->node_id(3);
          conn[2] = this->node_id(5);

          return;

          case 1:
          // linear sub-triangle 1
          conn[0] = this->node_id(3);
          conn[1] = this->node_id(1);
          conn[2] = this->node_id(4);

          return;

          case 2:
          // linear sub-triangle 2
          conn[0] = this->node_id(5);
          conn[1] = this->node_id(4);
          conn[2] = this->node_id(2);

          return;

          case 3:
          // linear sub-triangle 3
          conn[0] = this->node_id(3);
          conn[1] = this->node_id(4);
          conn[2] = this->node_id(5);

          return;

          default:
          libmesh_error_msg("Invalid sf = " << sf);
          }
        */
      }

    default:
      libmesh_error_msg("Unsupported IO package " << iop);
    }
}



template <typename RealType>
BoundingBoxTempl<RealType> Tri6Templ<RealType>::loose_bounding_box () const
{
  // This might have curved edges, or might be a curved surface in
  // 3-space, in which case the full bounding box can be larger than
  // the bounding box of just the nodes.
  //
  //
  // FIXME - I haven't yet proven the formula below to be correct for
  // quadratics in 2D - RHS
  Point pmin, pmax;

  for (unsigned d=0; d<LIBMESH_DIM; ++d)
    {
      Real center = this->point(0)(d);
      for (unsigned int p=1; p != 6; ++p)
        center += this->point(p)(d);
      center /= 6;

      Real hd = std::abs(center - this->point(0)(d));
      for (unsigned int p=1; p != 6; ++p)
        hd = std::max(hd, std::abs(center - this->point(p)(d)));

      pmin(d) = center - hd;
      pmax(d) = center + hd;
    }

  return BoundingBox(pmin, pmax);
}



template <typename RealType>
RealType Tri6Templ<RealType>::volume () const
{
  // This specialization is good for Lagrange mappings only
  if (this->mapping_type() != LAGRANGE_MAP)
    return this->Elem::volume();

  RealType vol=0.;

#if LIBMESH_DIM > 1
  // Make copies of our points.  It makes the subsequent calculations a bit
  // shorter and avoids dereferencing the same pointer multiple times.
  Point
    x0 = this->point(0), x1 = this->point(1), x2 = this->point(2),
    x3 = this->point(3), x4 = this->point(4), x5 = this->point(5);

  // Construct constant data vectors.
  // \vec{x}_{\xi}  = \vec{a1}*xi + \vec{b1}*eta + \vec{c1}
  // \vec{x}_{\eta} = \vec{a2}*xi + \vec{b2}*eta + \vec{c2}
  Point
    a1 =  4*x0 + 4*x1 - 8*x3,
    b1 =  4*x0 - 4*x3 + 4*x4 - 4*x5, /*=a2*/
    c1 = -3*x0 - 1*x1 + 4*x3,
    b2 =  4*x0 + 4*x2 - 8*x5,
    c2 = -3*x0 - 1*x2 + 4*x5;

  // If a1 == b1 == a2 == b2 == 0, this is a TRI6 with straight sides,
  // and we can use the TRI3 formula to compute the volume.
  if (a1.relative_fuzzy_equals(Point(0,0,0)) &&
      b1.relative_fuzzy_equals(Point(0,0,0)) &&
      b2.relative_fuzzy_equals(Point(0,0,0)))
    return 0.5 * cross_norm(c1, c2);

  // 7-point rule, exact for quintics.
  const unsigned int N = 7;

  // Parameters of the quadrature rule
  const static Real
    w1 = Real(31)/480 + Real(std::sqrt(15.0L)/2400),
    w2 = Real(31)/480 - Real(std::sqrt(15.0L)/2400),
    q1 = Real(2)/7 + Real(std::sqrt(15.0L)/21),
    q2 = Real(2)/7 - Real(std::sqrt(15.0L)/21);

  const static Real xi[N]  = {Real(1)/3,  q1, q1,     1-2*q1, q2, q2,     1-2*q2};
  const static Real eta[N] = {Real(1)/3,  q1, 1-2*q1, q1,     q2, 1-2*q2, q2};
  const static Real wts[N] = {Real(9)/80, w1, w1,     w1,     w2, w2,     w2};

  // Approximate the area with quadrature
  for (unsigned int q=0; q<N; ++q)
    vol += wts[q] * cross_norm(xi[q]*a1 + eta[q]*b1 + c1,
                               xi[q]*b1 + eta[q]*b2 + c2);
#endif // LIBMESH_DIM > 1

  return vol;
}



template <typename RealType>
unsigned short int Tri6Templ<RealType>::second_order_adjacent_vertex (const unsigned int n,
                                                       const unsigned int v) const
{
  libmesh_assert_greater_equal (n, this->n_vertices());
  libmesh_assert_less (n, this->n_nodes());
  libmesh_assert_less (v, 2);
  return _second_order_adjacent_vertices[n-this->n_vertices()][v];
}



template <typename RealType>
std::pair<unsigned short int, unsigned short int>
Tri6Templ<RealType>::second_order_child_vertex (const unsigned int n) const
{
  libmesh_assert_greater_equal (n, this->n_vertices());
  libmesh_assert_less (n, this->n_nodes());
  return std::pair<unsigned short int, unsigned short int>
    (_second_order_vertex_child_number[n],
     _second_order_vertex_child_index[n]);
}



} // namespace libMesh

#endif // LIBMESH_FACE_TRI6_IMPL_H
