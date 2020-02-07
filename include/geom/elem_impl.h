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

#ifndef LIBMESH_ELEM_IMPL_H
#define LIBMESH_ELEM_IMPL_H

// C++ includes
#include <algorithm> // for std::sort
#include <array>
#include <iterator>  // for std::ostream_iterator
#include <sstream>
#include <limits>    // for std::numeric_limits<>
#include <cmath>     // for std::sqrt()

// Local includes
#include "libmesh/auto_ptr.h" // libmesh_make_unique
#include "libmesh/elem.h"
#include "libmesh/fe_type.h"
#include "libmesh/fe_interface.h"
#include "libmesh/node_elem.h"
#include "libmesh/edge_edge2.h"
#include "libmesh/edge_edge3.h"
#include "libmesh/edge_edge4.h"
#include "libmesh/edge_inf_edge2.h"
#include "libmesh/face_tri3.h"
#include "libmesh/face_tri3_subdivision.h"
#include "libmesh/face_tri3_shell.h"
#include "libmesh/face_tri6.h"
#include "libmesh/face_quad4.h"
#include "libmesh/face_quad4_shell.h"
#include "libmesh/face_quad8.h"
#include "libmesh/face_quad8_shell.h"
#include "libmesh/face_quad9.h"
#include "libmesh/face_inf_quad4.h"
#include "libmesh/face_inf_quad6.h"
#include "libmesh/cell_tet4.h"
#include "libmesh/cell_tet10.h"
#include "libmesh/cell_hex8.h"
#include "libmesh/cell_hex20.h"
#include "libmesh/cell_hex27.h"
#include "libmesh/cell_inf_hex8.h"
#include "libmesh/cell_inf_hex16.h"
#include "libmesh/cell_inf_hex18.h"
#include "libmesh/cell_prism6.h"
#include "libmesh/cell_prism15.h"
#include "libmesh/cell_prism18.h"
#include "libmesh/cell_inf_prism6.h"
#include "libmesh/cell_inf_prism12.h"
#include "libmesh/cell_pyramid5.h"
#include "libmesh/cell_pyramid13.h"
#include "libmesh/cell_pyramid14.h"
#include "libmesh/fe_base.h"
#include "libmesh/mesh_base.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/remote_elem.h"
#include "libmesh/reference_elem.h"
#include "libmesh/enum_to_string.h"
#include "libmesh/threads.h"
#include "libmesh/enum_elem_quality.h"
#include "libmesh/enum_io_package.h"
#include "libmesh/enum_order.h"
#include "libmesh/elem_internal.h"
#include "libmesh/mesh_refinement.h"

#ifdef LIBMESH_ENABLE_PERIODIC
#include "libmesh/mesh.h"
#include "libmesh/periodic_boundaries.h"
#include "libmesh/boundary_info.h"
#endif

namespace libMesh
{

extern Threads::spin_mutex parent_indices_mutex;
extern Threads::spin_mutex parent_bracketing_nodes_mutex;

// ------------------------------------------------------------
// Elem class member functions
template <typename RealType>
std::unique_ptr<ElemTempl<RealType>> ElemTempl<RealType>::build(const ElemType type,
                                  ElemTempl<RealType> * p)
{
  switch (type)
    {
      // 0D elements
    case NODEELEM:
      return libmesh_make_unique<NodeElem>(p);

      // 1D elements
    case EDGE2:
      return libmesh_make_unique<Edge2>(p);
    case EDGE3:
      return libmesh_make_unique<Edge3>(p);
    case EDGE4:
      return libmesh_make_unique<Edge4>(p);

      // 2D elements
    case TRI3:
      return libmesh_make_unique<Tri3>(p);
    case TRISHELL3:
      return libmesh_make_unique<TriShell3>(p);
    case TRI3SUBDIVISION:
      return libmesh_make_unique<Tri3Subdivision>(p);
    case TRI6:
      return libmesh_make_unique<Tri6>(p);
    case QUAD4:
      return libmesh_make_unique<Quad4>(p);
    case QUADSHELL4:
      return libmesh_make_unique<QuadShell4>(p);
    case QUAD8:
      return libmesh_make_unique<Quad8>(p);
    case QUADSHELL8:
      return libmesh_make_unique<QuadShell8>(p);
    case QUAD9:
      return libmesh_make_unique<Quad9>(p);

      // 3D elements
    case TET4:
      return libmesh_make_unique<Tet4>(p);
    case TET10:
      return libmesh_make_unique<Tet10>(p);
    case HEX8:
      return libmesh_make_unique<Hex8>(p);
    case HEX20:
      return libmesh_make_unique<Hex20>(p);
    case HEX27:
      return libmesh_make_unique<Hex27>(p);
    case PRISM6:
      return libmesh_make_unique<Prism6>(p);
    case PRISM15:
      return libmesh_make_unique<Prism15>(p);
    case PRISM18:
      return libmesh_make_unique<Prism18>(p);
    case PYRAMID5:
      return libmesh_make_unique<Pyramid5>(p);
    case PYRAMID13:
      return libmesh_make_unique<Pyramid13>(p);
    case PYRAMID14:
      return libmesh_make_unique<Pyramid14>(p);

#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
      // 1D infinite elements
    case INFEDGE2:
      return libmesh_make_unique<InfEdge2>(p);

      // 2D infinite elements
    case INFQUAD4:
      return libmesh_make_unique<InfQuad4>(p);
    case INFQUAD6:
      return libmesh_make_unique<InfQuad6>(p);

      // 3D infinite elements
    case INFHEX8:
      return libmesh_make_unique<InfHex8>(p);
    case INFHEX16:
      return libmesh_make_unique<InfHex16>(p);
    case INFHEX18:
      return libmesh_make_unique<InfHex18>(p);
    case INFPRISM6:
      return libmesh_make_unique<InfPrism6>(p);
    case INFPRISM12:
      return libmesh_make_unique<InfPrism12>(p);
#endif

    default:
      libmesh_error_msg("ERROR: Undefined element type!");
    }
}


template <typename RealType>
PointTempl<RealType> ElemTempl<RealType>::centroid() const
{
  PointTempl<RealType> cp;

  const auto n_vertices = this->n_vertices();

  for (unsigned int n=0; n<n_vertices; n++)
    cp.add (this->point(n));

  return (cp /= static_cast<Real>(n_vertices));
}



template <typename RealType>
RealType ElemTempl<RealType>::hmin() const
{
  RealType h_min=std::numeric_limits<Real>::max();

  // Avoid calling a virtual a lot of times
  const auto n_vertices = this->n_vertices();

  for (unsigned int n_outer=0; n_outer<n_vertices; n_outer++)
    for (unsigned int n_inner=n_outer+1; n_inner<n_vertices; n_inner++)
      {
        const auto diff = (this->point(n_outer) - this->point(n_inner));

        h_min = std::min(h_min, diff.norm_sq());
      }

  return std::sqrt(h_min);
}



template <typename RealType>
RealType ElemTempl<RealType>::hmax() const
{
  RealType h_max=0;

  // Avoid calling a virtual a lot of times
  const auto n_vertices = this->n_vertices();

  for (unsigned int n_outer=0; n_outer<n_vertices; n_outer++)
    for (unsigned int n_inner=n_outer+1; n_inner<n_vertices; n_inner++)
      {
        const auto diff = (this->point(n_outer) - this->point(n_inner));

        h_max = std::max(h_max, diff.norm_sq());
      }

  return std::sqrt(h_max);
}



template <typename RealType>
RealType ElemTempl<RealType>::length(const unsigned int n1,
                                     const unsigned int n2) const
{
  libmesh_assert_less ( n1, this->n_vertices() );
  libmesh_assert_less ( n2, this->n_vertices() );

  return (this->point(n1) - this->point(n2)).norm();
}



template <typename RealType>
dof_id_type ElemTempl<RealType>::key () const
{
  const unsigned short n_n = this->n_nodes();

  std::array<dof_id_type, ElemTempl<RealType>::max_n_nodes> node_ids;

  for (unsigned short n=0; n != n_n; ++n)
    node_ids[n] = this->node_id(n);

  // Always sort, so that different local node numberings hash to the
  // same value.
  std::sort (node_ids.begin(), node_ids.begin()+n_n);

  return Utility::hashword(node_ids.data(), n_n);
}



template <typename RealType>
bool ElemTempl<RealType>::operator == (const ElemTempl<RealType> & rhs) const
{
  // If the elements aren't the same type, they aren't equal
  if (this->type() != rhs.type())
    return false;

  const unsigned short n_n = this->n_nodes();
  libmesh_assert_equal_to(n_n, rhs.n_nodes());

  // Make two sorted arrays of global node ids and compare them for
  // equality.
  std::array<dof_id_type, ElemTempl<RealType>::max_n_nodes> this_ids, rhs_ids;

  for (unsigned short n = 0; n != n_n; n++)
    {
      this_ids[n] = this->node_id(n);
      rhs_ids[n] = rhs.node_id(n);
    }

  // Sort the vectors to rule out different local node numberings.
  std::sort(this_ids.begin(), this_ids.begin()+n_n);
  std::sort(rhs_ids.begin(), rhs_ids.begin()+n_n);

  // If the node ids match, the elements are equal!
  for (unsigned short n = 0; n != n_n; ++n)
    if (this_ids[n] != rhs_ids[n])
      return false;
  return true;
}



template <typename RealType>
bool ElemTempl<RealType>::is_semilocal(const processor_id_type my_pid) const
{
  std::set<const ElemTempl<RealType> *> point_neighbors;

  this->find_point_neighbors(point_neighbors);

  for (const auto & elem : point_neighbors)
    if (elem->processor_id() == my_pid)
      return true;

  return false;
}



template <typename RealType>
unsigned int ElemTempl<RealType>::which_side_am_i (const ElemTempl<RealType> * e) const
{
  libmesh_assert(e);

  const unsigned int ns = this->n_sides();
  const unsigned int nn = this->n_nodes();

  const unsigned int en = e->n_nodes();

  // e might be on any side until proven otherwise
  std::vector<bool> might_be_side(ns, true);

  for (unsigned int i=0; i != en; ++i)
    {
      PointTempl<RealType> side_point = e->point(i);
      unsigned int local_node_id = libMesh::invalid_uint;

      // Look for a node of this that's contiguous with node i of
      // e. Note that the exact floating point comparison of PointTempl<RealType>
      // positions is intentional, see the class documentation for
      // this function.
      for (unsigned int j=0; j != nn; ++j)
        if (this->point(j) == side_point)
          local_node_id = j;

      // If a node of e isn't contiguous with some node of this, then
      // e isn't a side of this.
      if (local_node_id == libMesh::invalid_uint)
        return libMesh::invalid_uint;

      // If a node of e isn't contiguous with some node on side s of
      // this, then e isn't on side s.
      for (unsigned int s=0; s != ns; ++s)
        if (!this->is_node_on_side(local_node_id, s))
          might_be_side[s] = false;
    }

  for (unsigned int s=0; s != ns; ++s)
    if (might_be_side[s])
      {
#ifdef DEBUG
        for (unsigned int s2=s+1; s2 < ns; ++s2)
          libmesh_assert (!might_be_side[s2]);
#endif
        return s;
      }

  // Didn't find any matching side
  return libMesh::invalid_uint;
}



template <typename RealType>
bool ElemTempl<RealType>::contains_vertex_of(const ElemTempl<RealType> * e) const
{
  // Our vertices are the first numbered nodes
  for (auto n : IntRange<unsigned int>(0, e->n_vertices()))
    if (this->contains_point(e->point(n)))
      return true;
  return false;
}



template <typename RealType>
bool ElemTempl<RealType>::contains_edge_of(const ElemTempl<RealType> * e) const
{
  unsigned int num_contained_edges = 0;

  // Our vertices are the first numbered nodes
  for (auto n : IntRange<unsigned int>(0, e->n_vertices()))
    {
      if (this->contains_point(e->point(n)))
        {
          num_contained_edges++;
          if (num_contained_edges>=2)
            {
              return true;
            }
        }
    }
  return false;
}



template <typename RealType>
void ElemTempl<RealType>::find_point_neighbors(const PointTempl<RealType> & p,
                                std::set<const ElemTempl<RealType> *> & neighbor_set) const
{
  libmesh_assert(this->contains_point(p));
  libmesh_assert(this->active());

  neighbor_set.clear();
  neighbor_set.insert(this);

  std::set<const ElemTempl<RealType> *> untested_set, next_untested_set;
  untested_set.insert(this);

  while (!untested_set.empty())
    {
      // Loop over all the elements in the patch that haven't already
      // been tested
      for (const auto & elem : untested_set)
          for (auto current_neighbor : elem->neighbor_ptr_range())
            {
              if (current_neighbor &&
                  current_neighbor != RemoteElem::get_instance())    // we have a real neighbor on this side
                {
                  if (current_neighbor->active())                // ... if it is active
                    {
                      if (current_neighbor->contains_point(p))   // ... and touches p
                        {
                          // Make sure we'll test it
                          if (!neighbor_set.count(current_neighbor))
                            next_untested_set.insert (current_neighbor);

                          // And add it
                          neighbor_set.insert (current_neighbor);
                        }
                    }
#ifdef LIBMESH_ENABLE_AMR
                  else                                 // ... the neighbor is *not* active,
                    {                                  // ... so add *all* neighboring
                                                       // active children that touch p
                      std::vector<const ElemTempl<RealType> *> active_neighbor_children;

                      current_neighbor->active_family_tree_by_neighbor
                        (active_neighbor_children, elem);

                      for (const auto & current_child : active_neighbor_children)
                          if (current_child->contains_point(p))
                            {
                              // Make sure we'll test it
                              if (!neighbor_set.count(current_child))
                                next_untested_set.insert (current_child);

                              neighbor_set.insert (current_child);
                            }
                    }
#endif // #ifdef LIBMESH_ENABLE_AMR
                }
            }
      untested_set.swap(next_untested_set);
      next_untested_set.clear();
    }
}



template <typename RealType>
void ElemTempl<RealType>::find_point_neighbors(std::set<const ElemTempl<RealType> *> & neighbor_set) const
{
  this->find_point_neighbors(neighbor_set, this);
}



template <typename RealType>
void ElemTempl<RealType>::find_point_neighbors(std::set<const ElemTempl<RealType> *> & neighbor_set,
                                const ElemTempl<RealType> * start_elem) const
{
  ElemInternal::find_point_neighbors(this, neighbor_set, start_elem);
}



template <typename RealType>
void ElemTempl<RealType>::find_point_neighbors(std::set<ElemTempl<RealType> *> & neighbor_set,
                                ElemTempl<RealType> * start_elem)
{
  ElemInternal::find_point_neighbors(this, neighbor_set, start_elem);
}



template <typename RealType>
void ElemTempl<RealType>::find_edge_neighbors(const PointTempl<RealType> & p1,
                               const PointTempl<RealType> & p2,
                               std::set<const ElemTempl<RealType> *> & neighbor_set) const
{
  // Simple but perhaps suboptimal code: find elements containing the
  // first point, then winnow this set down by removing elements which
  // don't also contain the second point

  libmesh_assert(this->contains_point(p2));
  this->find_point_neighbors(p1, neighbor_set);

  typename std::set<const ElemTempl<RealType> *>::iterator        it = neighbor_set.begin();
  const typename std::set<const ElemTempl<RealType> *>::iterator end = neighbor_set.end();

  while (it != end)
    {
      // As of C++11, set::erase returns an iterator to the element
      // following the erased element, or end.
      if (!(*it)->contains_point(p2))
        it = neighbor_set.erase(it);
      else
        ++it;
    }
}



template <typename RealType>
void ElemTempl<RealType>::find_edge_neighbors(std::set<const ElemTempl<RealType> *> & neighbor_set) const
{
  neighbor_set.clear();
  neighbor_set.insert(this);

  std::set<const ElemTempl<RealType> *> untested_set, next_untested_set;
  untested_set.insert(this);

  while (!untested_set.empty())
    {
      // Loop over all the elements in the patch that haven't already
      // been tested
      for (const auto & elem : untested_set)
        {
          for (auto current_neighbor : elem->neighbor_ptr_range())
            {
              if (current_neighbor &&
                  current_neighbor != RemoteElem::get_instance())    // we have a real neighbor on this side
                {
                  if (current_neighbor->active())                // ... if it is active
                    {
                      if (this->contains_edge_of(current_neighbor) // ... and touches us
                          || current_neighbor->contains_edge_of(this))
                        {
                          // Make sure we'll test it
                          if (!neighbor_set.count(current_neighbor))
                            next_untested_set.insert (current_neighbor);

                          // And add it
                          neighbor_set.insert (current_neighbor);
                        }
                    }
#ifdef LIBMESH_ENABLE_AMR
                  else                                 // ... the neighbor is *not* active,
                    {                                  // ... so add *all* neighboring
                                                       // active children
                      std::vector<const ElemTempl<RealType> *> active_neighbor_children;

                      current_neighbor->active_family_tree_by_neighbor
                        (active_neighbor_children, elem);

                      for (const auto & current_child : active_neighbor_children)
                        if (this->contains_edge_of(current_child) || current_child->contains_edge_of(this))
                          {
                            // Make sure we'll test it
                            if (!neighbor_set.count(current_child))
                              next_untested_set.insert (current_child);

                            neighbor_set.insert (current_child);
                          }
                    }
#endif // #ifdef LIBMESH_ENABLE_AMR
                }
            }
        }
      untested_set.swap(next_untested_set);
      next_untested_set.clear();
    }
}



template <typename RealType>
void ElemTempl<RealType>::find_interior_neighbors(std::set<const Elem *> & neighbor_set) const
{
  ElemInternal::find_interior_neighbors(this, neighbor_set);
}



template <typename RealType>
void ElemTempl<RealType>::find_interior_neighbors(std::set<ElemTempl<RealType> *> & neighbor_set)
{
  ElemInternal::find_interior_neighbors(this, neighbor_set);
}



template <typename RealType>
const ElemTempl<RealType> * ElemTempl<RealType>::interior_parent () const
{
  // interior parents make no sense for full-dimensional elements.
  libmesh_assert_less (this->dim(), LIBMESH_DIM);

  // they USED TO BE only good for level-0 elements, but we now
  // support keeping interior_parent() valid on refined boundary
  // elements.
  // if (this->level() != 0)
  // return this->parent()->interior_parent();

  // We store the interior_parent pointer after both the parent
  // neighbor and neighbor pointers
  ElemTempl<RealType> * interior_p = _elemlinks[1+this->n_sides()];

  // If we have an interior_parent, we USED TO assume it was a
  // one-higher-dimensional interior element, but we now allow e.g.
  // edge elements to have a 3D interior_parent with no
  // intermediate 2D element.
  // libmesh_assert (!interior_p ||
  //                interior_p->dim() == (this->dim()+1));
  libmesh_assert (!interior_p ||
                  (interior_p == RemoteElem::get_instance()) ||
                  (interior_p->dim() > this->dim()));

  // We require consistency between AMR of interior and of boundary
  // elements
  if (interior_p && (interior_p != RemoteElem::get_instance()))
    libmesh_assert_less_equal (interior_p->level(), this->level());

  return interior_p;
}



template <typename RealType>
ElemTempl<RealType> * ElemTempl<RealType>::interior_parent ()
{
  // See the const version for comments
  libmesh_assert_less (this->dim(), LIBMESH_DIM);
  ElemTempl<RealType> * interior_p = _elemlinks[1+this->n_sides()];

  libmesh_assert (!interior_p ||
                  (interior_p == RemoteElem::get_instance()) ||
                  (interior_p->dim() > this->dim()));
  if (interior_p && (interior_p != RemoteElem::get_instance()))
    libmesh_assert_less_equal (interior_p->level(), this->level());

  return interior_p;
}



template <typename RealType>
void ElemTempl<RealType>::set_interior_parent (ElemTempl<RealType> * p)
{
  // interior parents make no sense for full-dimensional elements.
  libmesh_assert_less (this->dim(), LIBMESH_DIM);

  // If we have an interior_parent, we USED TO assume it was a
  // one-higher-dimensional interior element, but we now allow e.g.
  // edge elements to have a 3D interior_parent with no
  // intermediate 2D element.
  // libmesh_assert (!p ||
  //                 p->dim() == (this->dim()+1));
  libmesh_assert (!p ||
                  (p == RemoteElem::get_instance()) ||
                  (p->dim() > this->dim()));

  _elemlinks[1+this->n_sides()] = p;
}



#ifdef LIBMESH_ENABLE_PERIODIC

template <typename RealType>
ElemTempl<RealType> * ElemTempl<RealType>::topological_neighbor (const unsigned int i,
                                   MeshBase & mesh,
                                   const PointLocatorBase & point_locator,
                                   const PeriodicBoundaries * pb)
{
  libmesh_assert_less (i, this->n_neighbors());

  ElemTempl<RealType> * neighbor_i = this->neighbor_ptr(i);
  if (neighbor_i != nullptr)
    return neighbor_i;

  if (pb)
    {
      // Since the neighbor is nullptr it must be on a boundary. We need
      // see if this is a periodic boundary in which case it will have a
      // topological neighbor
      std::vector<boundary_id_type> bc_ids;
      mesh.get_boundary_info().boundary_ids(this, cast_int<unsigned short>(i), bc_ids);
      for (const auto & id : bc_ids)
        if (pb->boundary(id))
          {
            // Since the point locator inside of periodic boundaries
            // returns a const pointer we will retrieve the proper
            // pointer directly from the mesh object.
            const ElemTempl<RealType> * const cn = pb->neighbor(id, point_locator, this, i);
            neighbor_i = const_cast<ElemTempl<RealType> *>(cn);

            // Since coarse elements do not have more refined
            // neighbors we need to make sure that we don't return one
            // of these types of neighbors.
            if (neighbor_i)
              while (level() < neighbor_i->level())
                neighbor_i = neighbor_i->parent();
            return neighbor_i;
          }
    }

  return nullptr;
}



template <typename RealType>
const ElemTempl<RealType> * ElemTempl<RealType>::topological_neighbor (const unsigned int i,
                                         const MeshBase & mesh,
                                         const PointLocatorBase & point_locator,
                                         const PeriodicBoundaries * pb) const
{
  libmesh_assert_less (i, this->n_neighbors());

  const ElemTempl<RealType> * neighbor_i = this->neighbor_ptr(i);
  if (neighbor_i != nullptr)
    return neighbor_i;

  if (pb)
    {
      // Since the neighbor is nullptr it must be on a boundary. We need
      // see if this is a periodic boundary in which case it will have a
      // topological neighbor
      std::vector<boundary_id_type> bc_ids;
      mesh.get_boundary_info().boundary_ids(this, cast_int<unsigned short>(i), bc_ids);
      for (const auto & id : bc_ids)
        if (pb->boundary(id))
          {
            neighbor_i = pb->neighbor(id, point_locator, this, i);

            // Since coarse elements do not have more refined
            // neighbors we need to make sure that we don't return one
            // of these types of neighbors.
            if (neighbor_i)
              while (level() < neighbor_i->level())
                neighbor_i = neighbor_i->parent();
            return neighbor_i;
          }
    }

  return nullptr;
}


template <typename RealType>
bool ElemTempl<RealType>::has_topological_neighbor (const ElemTempl<RealType> * elem,
                                     const MeshBase & mesh,
                                     const PointLocatorBase & point_locator,
                                     const PeriodicBoundaries * pb) const
{
  // First see if this is a normal "interior" neighbor
  if (has_neighbor(elem))
    return true;

  for (auto n : this->side_index_range())
    if (this->topological_neighbor(n, mesh, point_locator, pb))
      return true;

  return false;
}


#endif

#ifdef DEBUG

template <typename RealType>
void ElemTempl<RealType>::libmesh_assert_valid_node_pointers() const
{
  libmesh_assert(this->valid_id());
  for (auto n : this->node_index_range())
    {
      libmesh_assert(this->node_ptr(n));
      libmesh_assert(this->node_ptr(n)->valid_id());
    }
}



template <typename RealType>
void ElemTempl<RealType>::libmesh_assert_valid_neighbors() const
{
  for (auto n : this->side_index_range())
    {
      const ElemTempl<RealType> * neigh = this->neighbor_ptr(n);

      // Any element might have a remote neighbor; checking
      // to make sure that's not inaccurate is tough.
      if (neigh == RemoteElem::get_instance())
        continue;

      if (neigh)
        {
          // Only subactive elements have subactive neighbors
          libmesh_assert (this->subactive() || !neigh->subactive());

          const ElemTempl<RealType> * elem = this;

          // If we're subactive but our neighbor isn't, its
          // return neighbor link will be to our first active
          // ancestor OR to our inactive ancestor of the same
          // level as neigh,
          if (this->subactive() && !neigh->subactive())
            {
              for (elem = this; !elem->active();
                   elem = elem->parent())
                libmesh_assert(elem);
            }
          else
            {
              unsigned int rev = neigh->which_neighbor_am_i(elem);
              libmesh_assert_less (rev, neigh->n_neighbors());

              if (this->subactive() && !neigh->subactive())
                {
                  while (neigh->neighbor_ptr(rev) != elem)
                    {
                      libmesh_assert(elem->parent());
                      elem = elem->parent();
                    }
                }
              else
                {
                  const ElemTempl<RealType> * nn = neigh->neighbor_ptr(rev);
                  libmesh_assert(nn);

                  for (; elem != nn; elem = elem->parent())
                    libmesh_assert(elem);
                }
            }
        }
      // If we don't have a neighbor and we're not subactive, our
      // ancestors shouldn't have any neighbors in this same
      // direction.
      else if (!this->subactive())
        {
          const ElemTempl<RealType> * my_parent = this->parent();
          if (my_parent &&
              // A parent with a different dimension isn't really one of
              // our ancestors, it means we're on a boundary mesh and this
              // is an interior mesh element for which we're on a side.
              // Nothing to test for in that case.
              (my_parent->dim() == this->dim()))
            libmesh_assert (!my_parent->neighbor_ptr(n));
        }
    }
}

#endif // DEBUG



template <typename RealType>
void ElemTempl<RealType>::make_links_to_me_local(unsigned int n, unsigned int nn)
{
  ElemTempl<RealType> * neigh = this->neighbor_ptr(n);

  // Don't bother calling this function unless it's necessary
  libmesh_assert(neigh);
  libmesh_assert(!neigh->is_remote());

  // We never have neighbors more refined than us
  libmesh_assert_less_equal (neigh->level(), this->level());

  // We never have subactive neighbors of non subactive elements
  libmesh_assert(!neigh->subactive() || this->subactive());

  // If we have a neighbor less refined than us then it must not
  // have any more refined descendants we could have pointed to
  // instead.
  libmesh_assert((neigh->level() == this->level()) ||
                 (neigh->active() && !this->subactive()) ||
                 (!neigh->has_children() && this->subactive()));

  // If neigh is at our level, then its family might have
  // remote_elem neighbor links which need to point to us
  // instead, but if not, then we're done.
  if (neigh->level() != this->level())
    return;

  // What side of neigh are we on?  nn.
  //
  // We can't use the usual Elem method because we're in the middle of
  // restoring topology.  We can't compare side_ptr nodes because
  // users want to abuse neighbor_ptr to point to
  // not-technically-neighbors across mesh slits.  We can't compare
  // node locations because users want to move those
  // not-technically-neighbors until they're
  // not-even-geometrically-neighbors.

  // Find any elements that ought to point to elem
  std::vector<ElemTempl<RealType> *> neigh_family;
#ifdef LIBMESH_ENABLE_AMR
  if (this->active())
    neigh->family_tree_by_side(neigh_family, nn);
  else
#endif
    neigh_family.push_back(neigh);

  // And point them to elem
  for (auto & neigh_family_member : neigh_family)
    {
      // Only subactive elements point to other subactive elements
      if (this->subactive() && !neigh_family_member->subactive())
        continue;

      // Ideally, the neighbor link ought to either be correct
      // already or ought to be to remote_elem.
      //
      // However, if we're redistributing a newly created elem,
      // after an AMR step but before find_neighbors has fixed up
      // neighbor links, we might have an out of date neighbor
      // link to elem's parent instead.
#ifdef LIBMESH_ENABLE_AMR
      libmesh_assert((neigh_family_member->neighbor_ptr(nn) &&
                      (neigh_family_member->neighbor_ptr(nn)->active() ||
                       neigh_family_member->neighbor_ptr(nn)->is_ancestor_of(this))) ||
                     (neigh_family_member->neighbor_ptr(nn) == RemoteElem::get_instance()) ||
                     ((this->refinement_flag() == JUST_REFINED) &&
                      (this->parent() != nullptr) &&
                      (neigh_family_member->neighbor_ptr(nn) == this->parent())));
#else
      libmesh_assert((neigh_family_member->neighbor_ptr(nn) == this) ||
                     (neigh_family_member->neighbor_ptr(nn) == RemoteElem::get_instance()));
#endif

      neigh_family_member->set_neighbor(nn, this);
    }
}


template <typename RealType>
void ElemTempl<RealType>::make_links_to_me_remote()
{
  libmesh_assert_not_equal_to (this, RemoteElem::get_instance());

  // We need to have handled any children first
#if defined(LIBMESH_ENABLE_AMR) && defined(DEBUG)
  if (this->has_children())
    for (auto & child : this->child_ref_range())
      libmesh_assert_equal_to (&child, RemoteElem::get_instance());
#endif

  // Remotify any neighbor links
  for (auto neigh : this->neighbor_ptr_range())
    {
      if (neigh && neigh != RemoteElem::get_instance())
        {
          // My neighbor should never be more refined than me; my real
          // neighbor would have been its parent in that case.
          libmesh_assert_greater_equal (this->level(), neigh->level());

          if (this->level() == neigh->level() &&
              neigh->has_neighbor(this))
            {
#ifdef LIBMESH_ENABLE_AMR
              // My neighbor may have descendants which also consider me a
              // neighbor
              std::vector<ElemTempl<RealType> *> family;
              neigh->total_family_tree_by_neighbor (family, this);

              // FIXME - There's a lot of ugly const_casts here; we
              // may want to make remote_elem non-const
              for (auto & n : family)
                {
                  libmesh_assert (n);
                  if (n == RemoteElem::get_instance())
                    continue;
                  unsigned int my_s = n->which_neighbor_am_i(this);
                  libmesh_assert_less (my_s, n->n_neighbors());
                  libmesh_assert_equal_to (n->neighbor_ptr(my_s), this);
                  n->set_neighbor(my_s, const_cast<RemoteElem *>(RemoteElem::get_instance()));
                }
#else
              unsigned int my_s = neigh->which_neighbor_am_i(this);
              libmesh_assert_less (my_s, neigh->n_neighbors());
              libmesh_assert_equal_to (neigh->neighbor_ptr(my_s), this);
              neigh->set_neighbor(my_s, const_cast<RemoteElem *>(RemoteElem::get_instance()));
#endif
            }
#ifdef LIBMESH_ENABLE_AMR
          // Even if my neighbor doesn't link back to me, it might
          // have subactive descendants which do
          else if (neigh->has_children())
            {
              // If my neighbor at the same level doesn't have me as a
              // neighbor, I must be subactive
              libmesh_assert(this->level() > neigh->level() ||
                             this->subactive());

              // My neighbor must have some ancestor of mine as a
              // neighbor
              ElemTempl<RealType> * my_ancestor = this->parent();
              libmesh_assert(my_ancestor);
              while (!neigh->has_neighbor(my_ancestor))
                {
                  my_ancestor = my_ancestor->parent();
                  libmesh_assert(my_ancestor);
                }

              // My neighbor may have descendants which consider me a
              // neighbor
              std::vector<ElemTempl<RealType> *> family;
              neigh->total_family_tree_by_subneighbor (family, my_ancestor, this);

              for (auto & n : family)
                {
                  libmesh_assert (n);
                  if (n->is_remote())
                    continue;
                  unsigned int my_s = n->which_neighbor_am_i(this);
                  libmesh_assert_less (my_s, n->n_neighbors());
                  libmesh_assert_equal_to (n->neighbor_ptr(my_s), this);
                  // TODO: we may want to make RemoteElem::get_instance() non-const.
                  n->set_neighbor(my_s, const_cast<RemoteElem *>(RemoteElem::get_instance()));
                }
            }
#endif
        }
    }

#ifdef LIBMESH_ENABLE_AMR
  // Remotify parent's child link
  ElemTempl<RealType> * my_parent = this->parent();
  if (my_parent &&
      // As long as it's not already remote
      my_parent != RemoteElem::get_instance() &&
      // And it's a real parent, not an interior parent
      this->dim() == my_parent->dim())
    {
      unsigned int me = my_parent->which_child_am_i(this);
      libmesh_assert_equal_to (my_parent->child_ptr(me), this);
      my_parent->set_child(me, const_cast<RemoteElem *>(RemoteElem::get_instance()));
    }
#endif
}


template <typename RealType>
void ElemTempl<RealType>::remove_links_to_me()
{
  libmesh_assert_not_equal_to (this, RemoteElem::get_instance());

  // We need to have handled any children first
#ifdef LIBMESH_ENABLE_AMR
  libmesh_assert (!this->has_children());
#endif

  // Nullify any neighbor links
  for (auto neigh : this->neighbor_ptr_range())
    {
      if (neigh && neigh != RemoteElem::get_instance())
        {
          // My neighbor should never be more refined than me; my real
          // neighbor would have been its parent in that case.
          libmesh_assert_greater_equal (this->level(), neigh->level());

          if (this->level() == neigh->level() &&
              neigh->has_neighbor(this))
            {
#ifdef LIBMESH_ENABLE_AMR
              // My neighbor may have descendants which also consider me a
              // neighbor
              std::vector<ElemTempl<RealType> *> family;
              neigh->total_family_tree_by_neighbor (family, this);

              for (auto & n : family)
                {
                  libmesh_assert (n);
                  if (n->is_remote())
                    continue;
                  unsigned int my_s = n->which_neighbor_am_i(this);
                  libmesh_assert_less (my_s, n->n_neighbors());
                  libmesh_assert_equal_to (n->neighbor_ptr(my_s), this);
                  n->set_neighbor(my_s, nullptr);
                }
#else
              unsigned int my_s = neigh->which_neighbor_am_i(this);
              libmesh_assert_less (my_s, neigh->n_neighbors());
              libmesh_assert_equal_to (neigh->neighbor_ptr(my_s), this);
              neigh->set_neighbor(my_s, nullptr);
#endif
            }
#ifdef LIBMESH_ENABLE_AMR
          // Even if my neighbor doesn't link back to me, it might
          // have subactive descendants which do
          else if (neigh->has_children())
            {
              // If my neighbor at the same level doesn't have me as a
              // neighbor, I must be subactive
              libmesh_assert(this->level() > neigh->level() ||
                             this->subactive());

              // My neighbor must have some ancestor of mine as a
              // neighbor
              ElemTempl<RealType> * my_ancestor = this->parent();
              libmesh_assert(my_ancestor);
              while (!neigh->has_neighbor(my_ancestor))
                {
                  my_ancestor = my_ancestor->parent();
                  libmesh_assert(my_ancestor);
                }

              // My neighbor may have descendants which consider me a
              // neighbor
              std::vector<ElemTempl<RealType> *> family;
              neigh->total_family_tree_by_subneighbor (family, my_ancestor, this);

              for (auto & n : family)
                {
                  libmesh_assert (n);
                  if (n->is_remote())
                    continue;
                  unsigned int my_s = n->which_neighbor_am_i(this);
                  libmesh_assert_less (my_s, n->n_neighbors());
                  libmesh_assert_equal_to (n->neighbor_ptr(my_s), this);
                  n->set_neighbor(my_s, nullptr);
                }
            }
#endif
        }
    }

#ifdef LIBMESH_ENABLE_AMR
  // We can't currently delete a child with a parent!
  libmesh_assert (!this->parent());
#endif
}



template <typename RealType>
void ElemTempl<RealType>::write_connectivity (std::ostream & out_stream,
                               const IOPackage iop) const
{
  libmesh_assert (out_stream.good());
  libmesh_assert(_nodes);
  libmesh_assert_not_equal_to (iop, INVALID_IO_PACKAGE);

  switch (iop)
    {
    case TECPLOT:
      {
        // This connectivity vector will be used repeatedly instead
        // of being reconstructed inside the loop.
        std::vector<dof_id_type> conn;
        for (auto sc : IntRange<unsigned int>(0, this->n_sub_elem()))
          {
            this->connectivity(sc, TECPLOT, conn);

            std::copy(conn.begin(),
                      conn.end(),
                      std::ostream_iterator<dof_id_type>(out_stream, " "));

            out_stream << '\n';
          }
        return;
      }

    case UCD:
      {
        for (auto i : this->node_index_range())
          out_stream << this->node_id(i)+1 << "\t";

        out_stream << '\n';
        return;
      }

    default:
      libmesh_error_msg("Unsupported IO package " << iop);
    }
}



template <typename RealType>
Real ElemTempl<RealType>::quality (const ElemQuality q) const
{
  switch (q)
    {
      /**
       * I don't know what to do for this metric.
       */
    default:
      {
        libmesh_do_once( libmesh_here();

                         libMesh::err << "ERROR:  unknown quality metric: "
                         << Utility::enum_to_string(q)
                         << std::endl
                         << "Cowardly returning 1."
                         << std::endl; );

        return 1.;
      }
    }
}



template <typename RealType>
bool ElemTempl<RealType>::ancestor() const
{
#ifdef LIBMESH_ENABLE_AMR

  // Use a fast, DistributedMesh-safe definition
  const bool is_ancestor =
    !this->active() && !this->subactive();

  // But check for inconsistencies if we have time
#ifdef DEBUG
  if (!is_ancestor && this->has_children())
    {
      for (auto & c : this->child_ref_range())
        {
          if (&c != RemoteElem::get_instance())
            {
              libmesh_assert(!c.active());
              libmesh_assert(!c.ancestor());
            }
        }
    }
#endif // DEBUG

  return is_ancestor;

#else
  return false;
#endif
}



#ifdef LIBMESH_ENABLE_AMR

template <typename RealType>
void ElemTempl<RealType>::add_child (ElemTempl<RealType> * elem)
{
  const unsigned int nc = this->n_children();

  if (_children == nullptr)
    {
      _children = new ElemTempl<RealType> *[nc];

      for (unsigned int c = 0; c != nc; c++)
        this->set_child(c, nullptr);
    }

  for (unsigned int c = 0; c != nc; c++)
    {
      if (this->_children[c] == nullptr || this->_children[c] == RemoteElem::get_instance())
        {
          libmesh_assert_equal_to (this, elem->parent());
          this->set_child(c, elem);
          return;
        }
    }

  libmesh_error_msg("Error: Tried to add a child to an element with full children array");
}



template <typename RealType>
void ElemTempl<RealType>::add_child (ElemTempl<RealType> * elem, unsigned int c)
{
  if (!this->has_children())
    {
      const unsigned int nc = this->n_children();
      _children = new ElemTempl<RealType> *[nc];

      for (unsigned int i = 0; i != nc; i++)
        this->set_child(i, nullptr);
    }

  libmesh_assert (this->_children[c] == nullptr || this->child_ptr(c) == RemoteElem::get_instance());
  libmesh_assert (elem == RemoteElem::get_instance() || this == elem->parent());

  this->set_child(c, elem);
}



template <typename RealType>
void ElemTempl<RealType>::replace_child (ElemTempl<RealType> * elem, unsigned int c)
{
  libmesh_assert(this->has_children());

  libmesh_assert(this->child_ptr(c));

  this->set_child(c, elem);
}



template <typename RealType>
void ElemTempl<RealType>::family_tree (std::vector<const ElemTempl<RealType> *> & family,
                          bool reset) const
{
  ElemInternal::family_tree(this, family, reset);
}



template <typename RealType>
void ElemTempl<RealType>::family_tree (std::vector<ElemTempl<RealType> *> & family,
                        bool reset)
{
  ElemInternal::family_tree(this, family, reset);
}



template <typename RealType>
void ElemTempl<RealType>::total_family_tree (std::vector<const ElemTempl<RealType> *> & family,
                              bool reset) const
{
  ElemInternal::total_family_tree(this, family, reset);
}



template <typename RealType>
void ElemTempl<RealType>::total_family_tree (std::vector<ElemTempl<RealType> *> & family,
                              bool reset)
{
  ElemInternal::total_family_tree(this, family, reset);
}



template <typename RealType>
void ElemTempl<RealType>::active_family_tree (std::vector<const ElemTempl<RealType> *> & active_family,
                               bool reset) const
{
  ElemInternal::active_family_tree(this, active_family, reset);
}



template <typename RealType>
void ElemTempl<RealType>::active_family_tree (std::vector<ElemTempl<RealType> *> & active_family,
                               bool reset)
{
  ElemInternal::active_family_tree(this, active_family, reset);
}



template <typename RealType>
void ElemTempl<RealType>::family_tree_by_side (std::vector<const ElemTempl<RealType> *> & family,
                                unsigned int side,
                                bool reset) const
{
  ElemInternal::family_tree_by_side(this, family, side, reset);
}



template <typename RealType>
void ElemTempl<RealType>:: family_tree_by_side (std::vector<ElemTempl<RealType> *> & family,
                                 unsigned int side,
                                 bool reset)
{
  ElemInternal::family_tree_by_side(this, family, side, reset);
}



template <typename RealType>
void ElemTempl<RealType>::active_family_tree_by_side (std::vector<const ElemTempl<RealType> *> & family,
                                       unsigned int side,
                                       bool reset) const
{
  ElemInternal::active_family_tree_by_side(this, family, side, reset);
}



template <typename RealType>
void ElemTempl<RealType>::active_family_tree_by_side (std::vector<ElemTempl<RealType> *> & family,
                                       unsigned int side,
                                       bool reset)
{
  ElemInternal::active_family_tree_by_side(this, family, side, reset);
}



template <typename RealType>
void ElemTempl<RealType>::family_tree_by_neighbor (std::vector<const ElemTempl<RealType> *> & family,
                                    const ElemTempl<RealType> * neighbor,
                                    bool reset) const
{
  ElemInternal::family_tree_by_neighbor(this, family, neighbor, reset);
}



template <typename RealType>
void ElemTempl<RealType>::family_tree_by_neighbor (std::vector<ElemTempl<RealType> *> & family,
                                    ElemTempl<RealType> * neighbor,
                                    bool reset)
{
  ElemInternal::family_tree_by_neighbor(this, family, neighbor, reset);
}



template <typename RealType>
void ElemTempl<RealType>::total_family_tree_by_neighbor (std::vector<const ElemTempl<RealType> *> & family,
                                          const ElemTempl<RealType> * neighbor,
                                          bool reset) const
{
  ElemInternal::total_family_tree_by_neighbor(this, family, neighbor, reset);
}



template <typename RealType>
void ElemTempl<RealType>::total_family_tree_by_neighbor (std::vector<ElemTempl<RealType> *> & family,
                                          ElemTempl<RealType> * neighbor,
                                          bool reset)
{
  ElemInternal::total_family_tree_by_neighbor(this, family, neighbor, reset);
}



template <typename RealType>
void ElemTempl<RealType>::family_tree_by_subneighbor (std::vector<const ElemTempl<RealType> *> & family,
                                       const ElemTempl<RealType> * neighbor,
                                       const ElemTempl<RealType> * subneighbor,
                                       bool reset) const
{
  ElemInternal::family_tree_by_subneighbor(this, family, neighbor, subneighbor, reset);
}



template <typename RealType>
void ElemTempl<RealType>::family_tree_by_subneighbor (std::vector<ElemTempl<RealType> *> & family,
                                       ElemTempl<RealType> * neighbor,
                                       ElemTempl<RealType> * subneighbor,
                                       bool reset)
{
  ElemInternal::family_tree_by_subneighbor(this, family, neighbor, subneighbor, reset);
}



template <typename RealType>
void ElemTempl<RealType>::total_family_tree_by_subneighbor (std::vector<const ElemTempl<RealType> *> & family,
                                             const ElemTempl<RealType> * neighbor,
                                             const ElemTempl<RealType> * subneighbor,
                                             bool reset) const
{
  ElemInternal::total_family_tree_by_subneighbor(this, family, neighbor, subneighbor, reset);
}



template <typename RealType>
void ElemTempl<RealType>::total_family_tree_by_subneighbor (std::vector<ElemTempl<RealType> *> & family,
                                             ElemTempl<RealType> * neighbor,
                                             ElemTempl<RealType> * subneighbor,
                                             bool reset)
{
  ElemInternal::total_family_tree_by_subneighbor(this, family, neighbor, subneighbor, reset);
}



template <typename RealType>
void ElemTempl<RealType>::active_family_tree_by_neighbor (std::vector<const ElemTempl<RealType> *> & family,
                                           const ElemTempl<RealType> * neighbor,
                                           bool reset) const
{
  ElemInternal::active_family_tree_by_neighbor(this, family, neighbor, reset);
}



template <typename RealType>
void ElemTempl<RealType>::active_family_tree_by_neighbor (std::vector<ElemTempl<RealType> *> & family,
                                           ElemTempl<RealType> * neighbor,
                                           bool reset)
{
  ElemInternal::active_family_tree_by_neighbor(this, family, neighbor, reset);
}



template <typename RealType>
void ElemTempl<RealType>::active_family_tree_by_topological_neighbor (std::vector<const ElemTempl<RealType> *> & family,
                                                       const ElemTempl<RealType> * neighbor,
                                                       const MeshBase & mesh,
                                                       const PointLocatorBase & point_locator,
                                                       const PeriodicBoundaries * pb,
                                                       bool reset) const
{
  ElemInternal::active_family_tree_by_topological_neighbor(this, family, neighbor,
                                                           mesh, point_locator, pb,
                                                           reset);
}



template <typename RealType>
void ElemTempl<RealType>::active_family_tree_by_topological_neighbor (std::vector<ElemTempl<RealType> *> & family,
                                                       ElemTempl<RealType> * neighbor,
                                                       const MeshBase & mesh,
                                                       const PointLocatorBase & point_locator,
                                                       const PeriodicBoundaries * pb,
                                                       bool reset)
{
  ElemInternal::active_family_tree_by_topological_neighbor(this, family, neighbor,
                                                           mesh, point_locator, pb,
                                                           reset);
}


template <typename RealType>
bool ElemTempl<RealType>::is_child_on_edge(const unsigned int libmesh_dbg_var(c),
                            const unsigned int e) const
{
  libmesh_assert_less (c, this->n_children());
  libmesh_assert_less (e, this->n_edges());

  std::unique_ptr<const ElemTempl<RealType>> my_edge = this->build_edge_ptr(e);
  std::unique_ptr<const ElemTempl<RealType>> child_edge = this->build_edge_ptr(e);

  // We're assuming that an overlapping child edge has the same
  // number and orientation as its parent
  return (child_edge->node_id(0) == my_edge->node_id(0) ||
          child_edge->node_id(1) == my_edge->node_id(1));
}



template <typename RealType>
unsigned int ElemTempl<RealType>::min_p_level_by_neighbor(const ElemTempl<RealType> * neighbor_in,
                                           unsigned int current_min) const
{
  libmesh_assert(!this->subactive());
  libmesh_assert(neighbor_in->active());

  // If we're an active element this is simple
  if (this->active())
    return std::min(current_min, this->p_level());

  libmesh_assert(has_neighbor(neighbor_in));

  // The p_level() of an ancestor element is already the minimum
  // p_level() of its children - so if that's high enough, we don't
  // need to examine any children.
  if (current_min <= this->p_level())
    return current_min;

  unsigned int min_p_level = current_min;

  for (auto & c : this->child_ref_range())
    if (&c != RemoteElem::get_instance() && c.has_neighbor(neighbor_in))
      min_p_level =
        c.min_p_level_by_neighbor(neighbor_in, min_p_level);

  return min_p_level;
}


template <typename RealType>
unsigned int ElemTempl<RealType>::min_new_p_level_by_neighbor(const ElemTempl<RealType> * neighbor_in,
                                               unsigned int current_min) const
{
  libmesh_assert(!this->subactive());
  libmesh_assert(neighbor_in->active());

  // If we're an active element this is simple
  if (this->active())
    {
      unsigned int new_p_level = this->p_level();
      if (this->p_refinement_flag() == ElemTempl<RealType>::REFINE)
        new_p_level += 1;
      if (this->p_refinement_flag() == ElemTempl<RealType>::COARSEN)
        {
          libmesh_assert_greater (new_p_level, 0);
          new_p_level -= 1;
        }
      return std::min(current_min, new_p_level);
    }

  libmesh_assert(has_neighbor(neighbor_in));

  unsigned int min_p_level = current_min;

  for (auto & c : this->child_ref_range())
    if (&c != RemoteElem::get_instance() && c.has_neighbor(neighbor_in))
      min_p_level =
        c.min_new_p_level_by_neighbor(neighbor_in, min_p_level);

  return min_p_level;
}



template <typename RealType>
unsigned int ElemTempl<RealType>::as_parent_node (unsigned int child,
                                   unsigned int child_node) const
{
  const unsigned int nc = this->n_children();
  libmesh_assert_less(child, nc);

  // Cached return values, indexed first by embedding_matrix version,
  // then by child number, then by child node number.
  std::vector<std::vector<std::vector<signed char>>> &
    cached_parent_indices = this->_get_parent_indices_cache();

  unsigned int em_vers = this->embedding_matrix_version();

  // We may be updating the cache on one thread, and while that
  // happens we can't safely access the cache from other threads.
  Threads::spin_mutex::scoped_lock lock(parent_indices_mutex);

  if (em_vers >= cached_parent_indices.size())
    cached_parent_indices.resize(em_vers+1);

  if (child >= cached_parent_indices[em_vers].size())
    {
      const signed char nn = cast_int<signed char>(this->n_nodes());

      cached_parent_indices[em_vers].resize(nc);

      for (unsigned int c = 0; c != nc; ++c)
        {
          const unsigned int ncn = this->n_nodes_in_child(c);
          cached_parent_indices[em_vers][c].resize(ncn);
          for (unsigned int cn = 0; cn != ncn; ++cn)
            {
              for (signed char n = 0; n != nn; ++n)
                {
                  const float em_val = this->embedding_matrix
                    (c, cn, n);
                  if (em_val == 1)
                    {
                      cached_parent_indices[em_vers][c][cn] = n;
                      break;
                    }

                  if (em_val != 0)
                    {
                      cached_parent_indices[em_vers][c][cn] =
                        -1;
                      break;
                    }

                  // We should never see an all-zero embedding matrix
                  // row
                  libmesh_assert_not_equal_to (n+1, nn);
                }
            }
        }
    }

  const signed char cache_val =
    cached_parent_indices[em_vers][child][child_node];
  if (cache_val == -1)
    return libMesh::invalid_uint;

  return cached_parent_indices[em_vers][child][child_node];
}



template <typename RealType>
const std::vector<std::pair<unsigned char, unsigned char>> &
ElemTempl<RealType>::parent_bracketing_nodes(unsigned int child,
                              unsigned int child_node) const
{
  // Indexed first by embedding matrix type, then by child id, then by
  // child node, then by bracketing pair
  std::vector<std::vector<std::vector<std::vector<std::pair<unsigned char, unsigned char>>>>> &
    cached_bracketing_nodes = this->_get_bracketing_node_cache();

  const unsigned int em_vers = this->embedding_matrix_version();

  // We may be updating the cache on one thread, and while that
  // happens we can't safely access the cache from other threads.
  Threads::spin_mutex::scoped_lock lock(parent_bracketing_nodes_mutex);

  if (cached_bracketing_nodes.size() <= em_vers)
    cached_bracketing_nodes.resize(em_vers+1);

  const unsigned int nc = this->n_children();

  // If we haven't cached the bracketing nodes corresponding to this
  // embedding matrix yet, let's do so now.
  if (cached_bracketing_nodes[em_vers].size() < nc)
    {
      // If we're a second-order element but we're not a full-order
      // element, then some of our bracketing nodes may not exist
      // except on the equivalent full-order element.  Let's build an
      // equivalent full-order element and make a copy of its cache to
      // use.
      if (this->default_order() != FIRST &&
          second_order_equivalent_type(this->type(), /*full_ordered=*/ true) != this->type())
        {
          // Check that we really are the non-full-order type
          libmesh_assert_equal_to
            (second_order_equivalent_type (this->type(), false),
             this->type());

          // Build the full-order type
          ElemType full_type =
            second_order_equivalent_type(this->type(), /*full_ordered=*/ true);
          std::unique_ptr<ElemTempl<RealType>> full_elem = ElemTempl<RealType>::build(full_type);

          // This won't work for elements with multiple
          // embedding_matrix versions, but every such element is full
          // order anyways.
          libmesh_assert_equal_to(em_vers, 0);

          // Make sure its cache has been built.  We temporarily
          // release our mutex lock so that the inner call can
          // re-acquire it.
          lock.release();
          full_elem->parent_bracketing_nodes(0,0);

          // And then we need to lock again, so that if someone *else*
          // grabbed our lock before we did we don't risk accessing
          // cached_bracketing_nodes while they're working on it.
          // Threading is hard.
          lock.acquire(parent_bracketing_nodes_mutex);

          // Copy its cache
          cached_bracketing_nodes =
            full_elem->_get_bracketing_node_cache();

          // Now we don't need to build the cache ourselves.
          return cached_bracketing_nodes[em_vers][child][child_node];
        }

      cached_bracketing_nodes[em_vers].resize(nc);

      const unsigned int nn = this->n_nodes();

      // We have to examine each child
      for (unsigned int c = 0; c != nc; ++c)
        {
          const unsigned int ncn = this->n_nodes_in_child(c);

          cached_bracketing_nodes[em_vers][c].resize(ncn);

          // We have to examine each node in that child
          for (unsigned int n = 0; n != ncn; ++n)
            {
              // If this child node isn't a vertex or an infinite
              // child element's mid-infinite-edge node, then we need
              // to find bracketing nodes on the child.
              if (!this->is_vertex_on_child(c, n)
#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
                  && !this->is_mid_infinite_edge_node(n)
#endif
                  )
                {
                  // Use the embedding matrix to find the child node
                  // location in parent master element space
                  PointTempl<RealType> bracketed_pt;

                  for (unsigned int pn = 0; pn != nn; ++pn)
                    {
                      const float em_val =
                        this->embedding_matrix(c,n,pn);

                      libmesh_assert_not_equal_to (em_val, 1);
                      if (em_val != 0.)
                        bracketed_pt.add_scaled(this->master_point(pn), em_val);
                    }

                  // Check each pair of nodes on the child which are
                  // also both parent nodes
                  for (unsigned int n1 = 0; n1 != ncn; ++n1)
                    {
                      if (n1 == n)
                        continue;

                      unsigned int parent_n1 =
                        this->as_parent_node(c,n1);

                      if (parent_n1 == libMesh::invalid_uint)
                        continue;

                      PointTempl<RealType> p1 = this->master_point(parent_n1);

                      for (unsigned int n2 = n1+1; n2 < nn; ++n2)
                        {
                          if (n2 == n)
                            continue;

                          unsigned int parent_n2 =
                            this->as_parent_node(c,n2);

                          if (parent_n2 == libMesh::invalid_uint)
                            continue;

                          PointTempl<RealType> p2 = this->master_point(parent_n2);

                          PointTempl<RealType> pmid = (p1 + p2)/2;

                          if (pmid == bracketed_pt)
                            {
                              cached_bracketing_nodes[em_vers][c][n].push_back
                                (std::make_pair(parent_n1,parent_n2));
                              break;
                            }
                          else
                            libmesh_assert(!pmid.absolute_fuzzy_equals(bracketed_pt));
                        }
                    }
                }
              // If this child node is a parent node, we need to
              // find bracketing nodes on the parent.
              else
                {
                  unsigned int parent_node = this->as_parent_node(c,n);

                  PointTempl<RealType> bracketed_pt;

                  // If we're not a parent node, use the embedding
                  // matrix to find the child node location in parent
                  // master element space
                  if (parent_node == libMesh::invalid_uint)
                    {
                      for (unsigned int pn = 0; pn != nn; ++pn)
                        {
                          const float em_val =
                            this->embedding_matrix(c,n,pn);

                          libmesh_assert_not_equal_to (em_val, 1);
                          if (em_val != 0.)
                            bracketed_pt.add_scaled(this->master_point(pn), em_val);
                        }
                    }
                  // If we're a parent node then we need no arithmetic
                  else
                    bracketed_pt = this->master_point(parent_node);

                  for (unsigned int n1 = 0; n1 != nn; ++n1)
                    {
                      if (n1 == parent_node)
                        continue;

                      PointTempl<RealType> p1 = this->master_point(n1);

                      for (unsigned int n2 = n1+1; n2 < nn; ++n2)
                        {
                          if (n2 == parent_node)
                            continue;

                          PointTempl<RealType> pmid = (p1 + this->master_point(n2))/2;

                          if (pmid == bracketed_pt)
                            {
                              cached_bracketing_nodes[em_vers][c][n].push_back
                                (std::make_pair(n1,n2));
                              break;
                            }
                          else
                            libmesh_assert(!pmid.absolute_fuzzy_equals(bracketed_pt));
                        }
                    }
                }
            }
        }
    }

  return cached_bracketing_nodes[em_vers][child][child_node];
}


template <typename RealType>
const std::vector<std::pair<dof_id_type, dof_id_type>>
ElemTempl<RealType>::bracketing_nodes(unsigned int child,
                       unsigned int child_node) const
{
  std::vector<std::pair<dof_id_type, dof_id_type>> returnval;

  const std::vector<std::pair<unsigned char, unsigned char>> & pbc =
    this->parent_bracketing_nodes(child,child_node);

  for (const auto & pb : pbc)
    {
      const unsigned short n_n = this->n_nodes();
      if (pb.first < n_n && pb.second < n_n)
        returnval.push_back(std::make_pair(this->node_id(pb.first),
                                           this->node_id(pb.second)));
      else
        {
          // We must be on a non-full-order higher order element...
          libmesh_assert_not_equal_to(this->default_order(), FIRST);
          libmesh_assert_not_equal_to
            (second_order_equivalent_type (this->type(), true),
             this->type());
          libmesh_assert_equal_to
            (second_order_equivalent_type (this->type(), false),
             this->type());

          // And that's a shame, because this is a nasty search:

          // Build the full-order type
          ElemType full_type =
            second_order_equivalent_type(this->type(), /*full_ordered=*/ true);
          std::unique_ptr<ElemTempl<RealType>> full_elem = ElemTempl<RealType>::build(full_type);

          dof_id_type pt1 = DofObject::invalid_id;
          dof_id_type pt2 = DofObject::invalid_id;

          // Find the bracketing nodes by figuring out what
          // already-created children will have them.

          // This only doesn't break horribly because we add children
          // and nodes in straightforward + hierarchical orders...
          for (unsigned int c=0; c <= child; ++c)
            for (auto n : IntRange<unsigned int>(0, this->n_nodes_in_child(c)))
              {
                if (c == child && n == child_node)
                  break;

                if (pb.first == full_elem->as_parent_node(c,n))
                  {
                    // We should be consistent
                    if (pt1 != DofObject::invalid_id)
                      libmesh_assert_equal_to(pt1, this->child_ptr(c)->node_id(n));

                    pt1 = this->child_ptr(c)->node_id(n);
                  }

                if (pb.second == full_elem->as_parent_node(c,n))
                  {
                    // We should be consistent
                    if (pt2 != DofObject::invalid_id)
                      libmesh_assert_equal_to(pt2, this->child_ptr(c)->node_id(n));

                    pt2 = this->child_ptr(c)->node_id(n);
                  }
              }

          // We should *usually* find all bracketing nodes by the time
          // we query them (again, because of the child & node add
          // order)
          //
          // The exception is if we're a HEX20, in which case we will
          // find pairs of vertex nodes and edge nodes bracketing the
          // new central node but we *won't* find the pairs of face
          // nodes which we would have had on a HEX27.  In that case
          // we'll still have enough bracketing nodes for a
          // topological lookup, but we won't be able to make the
          // following assertions.
          if (this->type() != HEX20)
            {
              libmesh_assert_not_equal_to (pt1, DofObject::invalid_id);
              libmesh_assert_not_equal_to (pt2, DofObject::invalid_id);
            }

          if (pt1 != DofObject::invalid_id &&
              pt2 != DofObject::invalid_id)
            returnval.push_back(std::make_pair(pt1, pt2));
        }
    }

  return returnval;
}
#endif // #ifdef LIBMESH_ENABLE_AMR




template <typename RealType>
bool ElemTempl<RealType>::contains_point (const PointTempl<RealType> & p, Real tol) const
{
  // We currently allow the user to enlarge the bounding box by
  // providing a tol > TOLERANCE (so this routine is identical to
  // ElemTempl<RealType>::close_to_point()), but print a warning so that the
  // user can eventually switch his code over to calling close_to_point()
  // instead, which is intended to be used for this purpose.
  if (tol > TOLERANCE)
    {
      libmesh_do_once(libMesh::err
                      << "WARNING: Resizing bounding box to match user-specified tolerance!\n"
                      << "In the future, calls to ElemTempl<RealType>::contains_point() with tol > TOLERANCE\n"
                      << "will be more optimized, but should not be used\n"
                      << "to search for points 'close to' elements!\n"
                      << "Instead, use ElemTempl<RealType>::close_to_point() for this purpose.\n"
                      << std::endl;);
      return this->point_test(p, tol, tol);
    }
  else
    return this->point_test(p, TOLERANCE, tol);
}




template <typename RealType>
bool ElemTempl<RealType>::close_to_point (const PointTempl<RealType> & p, Real tol) const
{
  // This test uses the user's passed-in tolerance for the
  // bounding box test as well, thereby allowing the routine to
  // find points which are not only "in" the element, but also
  // "nearby" to within some tolerance.
  return this->point_test(p, tol, tol);
}




template <typename RealType>
bool ElemTempl<RealType>::point_test(const PointTempl<RealType> & p, Real box_tol, Real map_tol) const
{
  libmesh_assert_greater (box_tol, 0.);
  libmesh_assert_greater (map_tol, 0.);

  // This is a great optimization on first order elements, but it
  // could return false negatives on higher orders
  if (this->default_order() == FIRST)
    {
      // Check to make sure the element *could* contain this point, so we
      // can avoid an expensive inverse_map call if it doesn't.
      bool
#if LIBMESH_DIM > 2
        point_above_min_z = false,
        point_below_max_z = false,
#endif
#if LIBMESH_DIM > 1
        point_above_min_y = false,
        point_below_max_y = false,
#endif
        point_above_min_x = false,
        point_below_max_x = false;

      // For relative bounding box checks in physical space
      const RealType my_hmax = this->hmax();

      for (auto & n : this->node_ref_range())
        {
          point_above_min_x = point_above_min_x || (n(0) - my_hmax*box_tol <= p(0));
          point_below_max_x = point_below_max_x || (n(0) + my_hmax*box_tol >= p(0));
#if LIBMESH_DIM > 1
          point_above_min_y = point_above_min_y || (n(1) - my_hmax*box_tol <= p(1));
          point_below_max_y = point_below_max_y || (n(1) + my_hmax*box_tol >= p(1));
#endif
#if LIBMESH_DIM > 2
          point_above_min_z = point_above_min_z || (n(2) - my_hmax*box_tol <= p(2));
          point_below_max_z = point_below_max_z || (n(2) + my_hmax*box_tol >= p(2));
#endif
        }

      if (
#if LIBMESH_DIM > 2
          !point_above_min_z ||
          !point_below_max_z ||
#endif
#if LIBMESH_DIM > 1
          !point_above_min_y ||
          !point_below_max_y ||
#endif
          !point_above_min_x ||
          !point_below_max_x)
        return false;
    }

  // To be on the safe side, we converge the inverse_map() iteration
  // to a slightly tighter tolerance than that requested by the
  // user...
  const PointTempl<RealType> mapped_point = FEMap::inverse_map(this->dim(),
                                                this,
                                                p,
                                                0.1*map_tol, // <- this is |dx| tolerance, the Newton residual should be ~ |dx|^2
                                                /*secure=*/ false);

  // Check that the refspace point maps back to p!  This is only necessary
  // for 1D and 2D elements, 3D elements always live in 3D.
  //
  // TODO: The contains_point() function could most likely be implemented
  // more efficiently in the element sub-classes themselves, at least for
  // the linear element types.
  if (this->dim() < 3)
    {
      PointTempl<RealType> xyz = FEMap::map(this->dim(), this, mapped_point);

      // Compute the distance between the original point and the re-mapped point.
      // They should be in the same place.
      RealType dist = (xyz - p).norm();


      // If dist is larger than some fraction of the tolerance, then return false.
      // This can happen when e.g. a 2D element is living in 3D, and
      // FEMap::inverse_map() maps p onto the projection of the element,
      // effectively "tricking" FEInterface::on_reference_element().
      if (dist > this->hmax() * map_tol)
        return false;
    }



  return FEInterface::on_reference_element(mapped_point, this->type(), map_tol);
}




template <typename RealType>
void ElemTempl<RealType>::print_info (std::ostream & os) const
{
  os << this->get_info()
     << std::endl;
}



template <typename RealType>
std::string ElemTempl<RealType>::get_info () const
{
  std::ostringstream oss;

  oss << "  Elem Information"                                      << '\n'
      << "   id()=";

  if (this->valid_id())
    oss << this->id();
  else
    oss << "invalid";

#ifdef LIBMESH_ENABLE_UNIQUE_ID
  oss << ", unique_id()=";
  if (this->valid_unique_id())
    oss << this->unique_id();
  else
    oss << "invalid";
#endif

  oss << ", processor_id()=" << this->processor_id()               << '\n';

  oss << "   type()="    << Utility::enum_to_string(this->type())  << '\n'
      << "   dim()="     << this->dim()                            << '\n'
      << "   n_nodes()=" << this->n_nodes()                        << '\n';

  for (auto n : this->node_index_range())
    oss << "    " << n << this->node_ref(n);

  oss << "   n_sides()=" << this->n_sides()                        << '\n';

  for (auto s : this->side_index_range())
    {
      oss << "    neighbor(" << s << ")=";
      if (this->neighbor_ptr(s))
        oss << this->neighbor_ptr(s)->id() << '\n';
      else
        oss << "nullptr\n";
    }

#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
  if (!this->infinite())
    {
#endif
    oss << "   hmin()=" << this->hmin()
        << ", hmax()=" << this->hmax()                             << '\n'
        << "   volume()=" << this->volume()                        << '\n';
#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
    }
#endif
    oss << "   active()=" << this->active()
      << ", ancestor()=" << this->ancestor()
      << ", subactive()=" << this->subactive()
      << ", has_children()=" << this->has_children()               << '\n'
      << "   parent()=";
  if (this->parent())
    oss << this->parent()->id() << '\n';
  else
    oss << "nullptr\n";
  oss << "   level()=" << this->level()
      << ", p_level()=" << this->p_level()                         << '\n'
#ifdef LIBMESH_ENABLE_AMR
      << "   refinement_flag()=" << Utility::enum_to_string(this->refinement_flag())        << '\n'
      << "   p_refinement_flag()=" << Utility::enum_to_string(this->p_refinement_flag())    << '\n'
#endif
#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
      << "   infinite()=" << this->infinite()    << '\n';
  if (this->infinite())
    oss << "   origin()=" << this->origin()    << '\n'
#endif
      ;

  oss << "   DoFs=";
  for (auto s : IntRange<unsigned int>(0, this->n_systems()))
    for (auto v : IntRange<unsigned int>(0, this->n_vars(s)))
      for (auto c : IntRange<unsigned int>(0, this->n_comp(s,v)))
        oss << '(' << s << '/' << v << '/' << this->dof_number(s,v,c) << ") ";


  return oss.str();
}



template <typename RealType>
void ElemTempl<RealType>::nullify_neighbors ()
{
  // Tell any of my neighbors about my death...
  // Looks strange, huh?
  for (auto n : this->side_index_range())
    {
      ElemTempl<RealType> * current_neighbor = this->neighbor_ptr(n);
      if (current_neighbor && current_neighbor != RemoteElem::get_instance())
        {
          // Note:  it is possible that I see the neighbor
          // (which is coarser than me)
          // but they don't see me, so avoid that case.
          if (current_neighbor->level() == this->level())
            {
              const unsigned int w_n_a_i = current_neighbor->which_neighbor_am_i(this);
              libmesh_assert_less (w_n_a_i, current_neighbor->n_neighbors());
              current_neighbor->set_neighbor(w_n_a_i, nullptr);
              this->set_neighbor(n, nullptr);
            }
        }
    }
}




template <typename RealType>
unsigned int ElemTempl<RealType>::n_second_order_adjacent_vertices (const unsigned int) const
{
  // for linear elements, always return 0
  return 0;
}



template <typename RealType>
unsigned short int ElemTempl<RealType>::second_order_adjacent_vertex (const unsigned int,
                                                       const unsigned int) const
{
  // for linear elements, always return 0
  return 0;
}



template <typename RealType>
std::pair<unsigned short int, unsigned short int>
ElemTempl<RealType>::second_order_child_vertex (const unsigned int) const
{
  // for linear elements, always return 0
  return std::pair<unsigned short int, unsigned short int>(0,0);
}



template <typename RealType>
ElemType ElemTempl<RealType>::first_order_equivalent_type (const ElemType et)
{
  switch (et)
    {
    case NODEELEM:
      return NODEELEM;
    case EDGE2:
    case EDGE3:
    case EDGE4:
      return EDGE2;
    case TRI3:
    case TRI6:
      return TRI3;
    case TRISHELL3:
      return TRISHELL3;
    case QUAD4:
    case QUAD8:
    case QUAD9:
      return QUAD4;
    case QUADSHELL4:
    case QUADSHELL8:
      return QUADSHELL4;
    case TET4:
    case TET10:
      return TET4;
    case HEX8:
    case HEX27:
    case HEX20:
      return HEX8;
    case PRISM6:
    case PRISM15:
    case PRISM18:
      return PRISM6;
    case PYRAMID5:
    case PYRAMID13:
    case PYRAMID14:
      return PYRAMID5;

#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS

    case INFEDGE2:
      return INFEDGE2;
    case INFQUAD4:
    case INFQUAD6:
      return INFQUAD4;
    case INFHEX8:
    case INFHEX16:
    case INFHEX18:
      return INFHEX8;
    case INFPRISM6:
    case INFPRISM12:
      return INFPRISM6;

#endif

    default:
      // unknown element
      return INVALID_ELEM;
    }
}



template <typename RealType>
ElemType ElemTempl<RealType>::second_order_equivalent_type (const ElemType et,
                                             const bool full_ordered)
{
  switch (et)
    {
    case NODEELEM:
      return NODEELEM;
    case EDGE2:
    case EDGE3:
      {
        // full_ordered not relevant
        return EDGE3;
      }

    case EDGE4:
      {
        // full_ordered not relevant
        return EDGE4;
      }

    case TRI3:
    case TRI6:
      {
        // full_ordered not relevant
        return TRI6;
      }

      // Currently there is no TRISHELL6, so similarly to other types
      // where this is the case, we just return the input.
    case TRISHELL3:
      return TRISHELL3;

    case QUAD4:
    case QUAD8:
      {
        if (full_ordered)
          return QUAD9;
        else
          return QUAD8;
      }

    case QUADSHELL4:
    case QUADSHELL8:
      {
        // There is no QUADSHELL9, so in that sense QUADSHELL8 is the
        // "full ordered" element.
        return QUADSHELL8;
      }

    case QUAD9:
      {
        // full_ordered not relevant
        return QUAD9;
      }

    case TET4:
    case TET10:
      {
        // full_ordered not relevant
        return TET10;
      }

    case HEX8:
    case HEX20:
      {
        // see below how this correlates with INFHEX8
        if (full_ordered)
          return HEX27;
        else
          return HEX20;
      }

    case HEX27:
      {
        // full_ordered not relevant
        return HEX27;
      }

    case PRISM6:
    case PRISM15:
      {
        if (full_ordered)
          return PRISM18;
        else
          return PRISM15;
      }

    case PRISM18:
      {
        // full_ordered not relevant
        return PRISM18;
      }

    case PYRAMID5:
    case PYRAMID13:
      {
        if (full_ordered)
          return PYRAMID14;
        else
          return PYRAMID13;
      }

    case PYRAMID14:
      {
        // full_ordered not relevant
        return PYRAMID14;
      }



#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS

      // infinite elements
    case INFEDGE2:
      {
        return INFEDGE2;
      }

    case INFQUAD4:
    case INFQUAD6:
      {
        // full_ordered not relevant
        return INFQUAD6;
      }

    case INFHEX8:
    case INFHEX16:
      {
        /*
         * Note that this matches with \p Hex8:
         * For full-ordered, \p InfHex18 and \p Hex27
         * belong together, and for not full-ordered,
         * \p InfHex16 and \p Hex20 belong together.
         */
        if (full_ordered)
          return INFHEX18;
        else
          return INFHEX16;
      }

    case INFHEX18:
      {
        // full_ordered not relevant
        return INFHEX18;
      }

    case INFPRISM6:
    case INFPRISM12:
      {
        // full_ordered not relevant
        return INFPRISM12;
      }

#endif


    default:
      {
        // what did we miss?
        libmesh_error_msg("No second order equivalent element type for et =  "
                          << Utility::enum_to_string(et));
      }
    }
}



template <typename RealType>
typename ElemTempl<RealType>::side_iterator ElemTempl<RealType>::boundary_sides_begin()
{
  Predicates::BoundarySide<SideIter> bsp;
  return side_iterator(this->_first_side(), this->_last_side(), bsp);
}




template <typename RealType>
typename ElemTempl<RealType>::side_iterator ElemTempl<RealType>::boundary_sides_end()
{
  Predicates::BoundarySide<SideIter> bsp;
  return side_iterator(this->_last_side(), this->_last_side(), bsp);
}




template <typename RealType>
RealType ElemTempl<RealType>::volume () const
{
  // The default implementation builds a finite element of the correct
  // order and sums up the JxW contributions.  This can be expensive,
  // so the various element types can overload this method and compute
  // the volume more efficiently.
  const FEFamily mapping_family = FEMap::map_fe_type(*this);
  const FEType fe_type(this->default_order(), mapping_family);

  std::unique_ptr<FEBase> fe (FEBase::build(this->dim(),
                                            fe_type));

  const auto & JxW = fe->get_JxW();

  // The default quadrature rule should integrate the mass matrix,
  // thus it should be plenty to compute the area
  QGauss qrule (this->dim(), fe_type.default_quadrature_order());

  fe->attach_quadrature_rule(&qrule);

  fe->reinit(this);

  RealType vol=0.;
  for (auto jxw : JxW)
    vol += jxw;

  return vol;

}



template <typename RealType>
BoundingBoxTempl<RealType> ElemTempl<RealType>::loose_bounding_box () const
{
  PointTempl<RealType> pmin = this->point(0);
  PointTempl<RealType> pmax = pmin;

  unsigned int n_points = this->n_nodes();
  for (unsigned int p=0; p != n_points; ++p)
    for (unsigned d=0; d<LIBMESH_DIM; ++d)
      {
        const PointTempl<RealType> & pt = this->point(p);
        if (pmin(d) > pt(d))
          pmin(d) = pt(d);

        if (pmax(d) < pt(d))
          pmax(d) = pt(d);
      }

  return BoundingBox(pmin, pmax);
}



template <typename RealType>
bool ElemTempl<RealType>::is_vertex_on_parent(unsigned int c,
                               unsigned int n) const
{
#ifdef LIBMESH_ENABLE_AMR

  unsigned int my_n_vertices = this->n_vertices();
  for (unsigned int n_parent = 0; n_parent != my_n_vertices;
       ++n_parent)
    if (this->node_ptr(n_parent) == this->child_ptr(c)->node_ptr(n))
      return true;
  return false;

#else

  // No AMR?
  libmesh_ignore(c,n);
  libmesh_error_msg("ERROR: AMR disabled, how did we get here?");
  return true;

#endif
}



template <typename RealType>
unsigned int ElemTempl<RealType>::opposite_side(const unsigned int /*s*/) const
{
  // If the subclass didn't rederive this, using it is an error
  libmesh_not_implemented();
}



template <typename RealType>
unsigned int ElemTempl<RealType>::opposite_node(const unsigned int /*n*/,
                                 const unsigned int /*s*/) const
{
  // If the subclass didn't rederive this, using it is an error
  libmesh_not_implemented();
}

/**
 * The following functions only apply when
 * AMR is enabled and thus are not present
 * otherwise.
 */
#ifdef LIBMESH_ENABLE_AMR

template <typename RealType>
void ElemTempl<RealType>::set_p_level(unsigned int p)
{
  // Maintain the parent's p level as the minimum of it's children
  if (this->parent() != nullptr)
    {
      unsigned int parent_p_level = this->parent()->p_level();

      // If our new p level is less than our parents, our parents drops
      if (parent_p_level > p)
        {
          this->parent()->set_p_level(p);

          // And we should keep track of the drop, in case we need to
          // do a projection later.
          this->parent()->set_p_refinement_flag(Elem::JUST_COARSENED);
        }
      // If we are the lowest p level and it increases, so might
      // our parent's, but we have to check every other child to see
      else if (parent_p_level == _p_level && _p_level < p)
        {
          _p_level = cast_int<unsigned char>(p);
          parent_p_level = cast_int<unsigned char>(p);
          for (auto & c : this->parent()->child_ref_range())
            parent_p_level = std::min(parent_p_level,
                                      c.p_level());

          // When its children all have a higher p level, the parent's
          // should rise
          if (parent_p_level > this->parent()->p_level())
            {
              this->parent()->set_p_level(parent_p_level);

              // And we should keep track of the rise, in case we need to
              // do a projection later.
              this->parent()->set_p_refinement_flag(Elem::JUST_REFINED);
            }

          return;
        }
    }

  _p_level = cast_int<unsigned char>(p);
}



template <typename RealType>
void ElemTempl<RealType>::refine (MeshRefinement & mesh_refinement)
{
  libmesh_assert_equal_to (this->refinement_flag(), Elem::REFINE);
  libmesh_assert (this->active());

  const unsigned int nc = this->n_children();

  // Create my children if necessary
  if (!_children)
    {
      _children = new Elem *[nc];

      unsigned int parent_p_level = this->p_level();
      const unsigned int nei = this->n_extra_integers();
      for (unsigned int c = 0; c != nc; c++)
        {
          _children[c] = Elem::build(this->type(), this).release();
          Elem * current_child = this->child_ptr(c);

          current_child->set_refinement_flag(Elem::JUST_REFINED);
          current_child->set_p_level(parent_p_level);
          current_child->set_p_refinement_flag(this->p_refinement_flag());

          for (auto cnode : current_child->node_index_range())
            {
              Node * node =
                mesh_refinement.add_node(*this, c, cnode,
                                         current_child->processor_id());
              node->set_n_systems (this->n_systems());
              current_child->set_node(cnode) = node;
            }

          mesh_refinement.add_elem (current_child);
          current_child->set_n_systems(this->n_systems());
          libmesh_assert_equal_to (current_child->n_extra_integers(),
                                   this->n_extra_integers());
          for (unsigned int i=0; i != nei; ++i)
            current_child->set_extra_integer(i, this->get_extra_integer(i));
        }
    }
  else
    {
      unsigned int parent_p_level = this->p_level();
      for (unsigned int c = 0; c != nc; c++)
        {
          Elem * current_child = this->child_ptr(c);
          if (current_child != RemoteElem::get_instance())
            {
              libmesh_assert(current_child->subactive());
              current_child->set_refinement_flag(Elem::JUST_REFINED);
              current_child->set_p_level(parent_p_level);
              current_child->set_p_refinement_flag(this->p_refinement_flag());
            }
        }
    }

  // Un-set my refinement flag now
  this->set_refinement_flag(Elem::INACTIVE);

  // Leave the p refinement flag set - we will need that later to get
  // projection operations correct
  // this->set_p_refinement_flag(Elem::INACTIVE);

#ifndef NDEBUG
  for (unsigned int c = 0; c != nc; c++)
    if (this->child_ptr(c) != RemoteElem::get_instance())
      {
        libmesh_assert_equal_to (this->child_ptr(c)->parent(), this);
        libmesh_assert(this->child_ptr(c)->active());
      }
#endif
  libmesh_assert (this->ancestor());
}



template <typename RealType>
void ElemTempl<RealType>::coarsen()
{
  libmesh_assert_equal_to (this->refinement_flag(), Elem::COARSEN_INACTIVE);
  libmesh_assert (!this->active());

  // We no longer delete children until MeshRefinement::contract()
  // delete [] _children;
  // _children = nullptr;

  unsigned int parent_p_level = 0;

  const unsigned int n_n = this->n_nodes();

  // re-compute hanging node nodal locations
  for (unsigned int c = 0, nc = this->n_children(); c != nc; ++c)
    {
      Elem * mychild = this->child_ptr(c);
      if (mychild == RemoteElem::get_instance())
        continue;
      for (auto cnode : mychild->node_index_range())
        {
          Point new_pos;
          bool calculated_new_pos = false;

          for (unsigned int n=0; n<n_n; n++)
            {
              // The value from the embedding matrix
              const float em_val = this->embedding_matrix(c,cnode,n);

              // The node location is somewhere between existing vertices
              if ((em_val != 0.) && (em_val != 1.))
                {
                  new_pos.add_scaled (this->point(n), em_val);
                  calculated_new_pos = true;
                }
            }

          if (calculated_new_pos)
            {
              //Move the existing node back into it's original location
              for (unsigned int i=0; i<LIBMESH_DIM; i++)
                {
                  Point & child_node = mychild->point(cnode);
                  child_node(i)=new_pos(i);
                }
            }
        }
    }

  for (auto & mychild : this->child_ref_range())
    {
      if (&mychild == RemoteElem::get_instance())
        continue;
      libmesh_assert_equal_to (mychild.refinement_flag(), Elem::COARSEN);
      mychild.set_refinement_flag(Elem::INACTIVE);
      if (mychild.p_level() > parent_p_level)
        parent_p_level = mychild.p_level();
    }

  this->set_refinement_flag(Elem::JUST_COARSENED);
  this->set_p_level(parent_p_level);

  libmesh_assert (this->active());
}



template <typename RealType>
void ElemTempl<RealType>::contract()
{
  // Subactive elements get deleted entirely, not contracted
  libmesh_assert (this->active());

  // Active contracted elements no longer can have children
  delete [] _children;
  _children = nullptr;

  if (this->refinement_flag() == Elem::JUST_COARSENED)
    this->set_refinement_flag(Elem::DO_NOTHING);
}

#endif // #ifdef LIBMESH_ENABLE_AMR

} // namespace libMesh

#endif // LIBMESH_ELEM_IMPL_H
