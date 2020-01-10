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

#ifndef LIBMESH_DOF_MAP_IMPL_H
#define LIBMESH_DOF_MAP_IMPL_H

#include "libmesh/dof_map.h"
#include "libmesh/default_coupling.h"
#include "libmesh/periodic_boundaries.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/auto_ptr.h"
#include "libmesh/elem.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fem_function_base.h"
#include "libmesh/function_base.h"
#include "libmesh/system.h"
#include "libmesh/quadrature.h"
#include "libmesh/raw_accessor.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/coupling_matrix.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/enum_to_string.h"

#include "timpi/parallel_implementation.h"
#include "timpi/parallel_sync.h"

#include <algorithm>

namespace libMesh
{

template <typename RealType>
DofMap::DofMap(const unsigned int number,
               MeshBaseTempl<RealType> & mesh) :
  ParallelObject (mesh.comm()),
  _dof_coupling(nullptr),
  _error_on_constraint_loop(false),
  _variables(),
  _variable_groups(),
  _variable_group_numbers(),
  _sys_number(number),
  _mesh(mesh),
  _matrices(),
  _first_df(),
  _end_df(),
  _first_scalar_df(),
  _send_list(),
  _augment_sparsity_pattern(nullptr),
  _extra_sparsity_function(nullptr),
  _extra_sparsity_context(nullptr),
  _augment_send_list(nullptr),
  _extra_send_list_function(nullptr),
  _extra_send_list_context(nullptr),
  need_full_sparsity_pattern(false),
  _n_nz(nullptr),
  _n_oz(nullptr),
  _n_dfs(0),
  _n_SCALAR_dofs(0)
#ifdef LIBMESH_ENABLE_AMR
  , _n_old_dfs(0),
  _first_old_df(),
  _end_old_df(),
  _first_old_scalar_df()
#endif
#ifdef LIBMESH_ENABLE_CONSTRAINTS
  , _dof_constraints()
  , _stashed_dof_constraints()
  , _primal_constraint_values()
  , _adjoint_constraint_values()
#endif
#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
  , _node_constraints()
#endif
#ifdef LIBMESH_ENABLE_PERIODIC
  , _periodic_boundaries(libmesh_make_unique<PeriodicBoundaries>())
#endif
#ifdef LIBMESH_ENABLE_DIRICHLET
  , _dirichlet_boundaries(libmesh_make_unique<DirichletBoundaries>())
  , _adjoint_dirichlet_boundaries()
#endif
  , _implicit_neighbor_dofs_initialized(false),
  _implicit_neighbor_dofs(false)
{
  _matrices.clear();

  auto default_coupling = libmesh_make_unique<DefaultCouplingTempl<RealType>>();
  auto default_evaluating = libmesh_make_unique<DefaultCouplingTempl<RealType>>();

  default_coupling->set_mesh(&mesh);
  default_evaluating->set_mesh(&mesh);
  default_evaluating->set_n_levels(1);

#ifdef LIBMESH_ENABLE_PERIODIC
  default_coupling->set_periodic_boundaries(_periodic_boundaries.get());
  default_evaluating->set_periodic_boundaries(_periodic_boundaries.get());
#endif

  _default_coupling = std::move(default_coupling);
  _default_evaluating = std::move(default_evaluating);

  this->add_coupling_functor(*_default_coupling);
  this->add_algebraic_ghosting_functor(*_default_evaluating);
}

// Anonymous namespace to hold helper classes
namespace {

using namespace libMesh;

class ComputeConstraints
{
public:
  ComputeConstraints (DofConstraints & constraints,
                      DofMap & dof_map,
#ifdef LIBMESH_ENABLE_PERIODIC
                      PeriodicBoundaries & periodic_boundaries,
#endif
                      const MeshBase & mesh,
                      const unsigned int variable_number) :
    _constraints(constraints),
    _dof_map(dof_map),
#ifdef LIBMESH_ENABLE_PERIODIC
    _periodic_boundaries(periodic_boundaries),
#endif
    _mesh(mesh),
    _variable_number(variable_number)
  {}

  void operator()(const ConstElemRange & range) const
  {
    const Variable & var_description = _dof_map.variable(_variable_number);

#ifdef LIBMESH_ENABLE_PERIODIC
    std::unique_ptr<PointLocatorBase> point_locator;
    const bool have_periodic_boundaries =
      !_periodic_boundaries.empty();
    if (have_periodic_boundaries && !range.empty())
      point_locator = _mesh.sub_point_locator();
#endif

    for (const auto & elem : range)
      if (var_description.active_on_subdomain(elem->subdomain_id()))
        {
#ifdef LIBMESH_ENABLE_AMR
          FEInterface::compute_constraints (_constraints,
                                            _dof_map,
                                            _variable_number,
                                            elem);
#endif
#ifdef LIBMESH_ENABLE_PERIODIC
          // FIXME: periodic constraints won't work on a non-serial
          // mesh unless it's kept ghost elements from opposing
          // boundaries!
          if (have_periodic_boundaries)
            FEInterface::compute_periodic_constraints (_constraints,
                                                       _dof_map,
                                                       _periodic_boundaries,
                                                       _mesh,
                                                       point_locator.get(),
                                                       _variable_number,
                                                       elem);
#endif
        }
  }

private:
  DofConstraints & _constraints;
  DofMap & _dof_map;
#ifdef LIBMESH_ENABLE_PERIODIC
  PeriodicBoundaries & _periodic_boundaries;
#endif
  const MeshBase & _mesh;
  const unsigned int _variable_number;
};



#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
class ComputeNodeConstraints
{
public:
  ComputeNodeConstraints (NodeConstraints & node_constraints,
#ifdef LIBMESH_ENABLE_PERIODIC
                          PeriodicBoundaries & periodic_boundaries,
#endif
                          const MeshBase & mesh) :
    _node_constraints(node_constraints),
#ifdef LIBMESH_ENABLE_PERIODIC
    _periodic_boundaries(periodic_boundaries),
#endif
    _mesh(mesh)
  {}

  void operator()(const ConstElemRange & range) const
  {
#ifdef LIBMESH_ENABLE_PERIODIC
    std::unique_ptr<PointLocatorBase> point_locator;
    bool have_periodic_boundaries = !_periodic_boundaries.empty();
    if (have_periodic_boundaries && !range.empty())
      point_locator = _mesh.sub_point_locator();
#endif

    for (const auto & elem : range)
      {
#ifdef LIBMESH_ENABLE_AMR
        FEBase::compute_node_constraints (_node_constraints, elem);
#endif
#ifdef LIBMESH_ENABLE_PERIODIC
        // FIXME: periodic constraints won't work on a non-serial
        // mesh unless it's kept ghost elements from opposing
        // boundaries!
        if (have_periodic_boundaries)
          FEBase::compute_periodic_node_constraints (_node_constraints,
                                                     _periodic_boundaries,
                                                     _mesh,
                                                     point_locator.get(),
                                                     elem);
#endif
      }
  }

private:
  NodeConstraints & _node_constraints;
#ifdef LIBMESH_ENABLE_PERIODIC
  PeriodicBoundaries & _periodic_boundaries;
#endif
  const MeshBase & _mesh;
};
#endif // LIBMESH_ENABLE_NODE_CONSTRAINTS


#ifdef LIBMESH_ENABLE_DIRICHLET

/**
 * This functor class hierarchy adds a constraint row to a DofMap
 */
class AddConstraint
{
protected:
  DofMap                  & dof_map;

public:
  AddConstraint(DofMap & dof_map_in) : dof_map(dof_map_in) {}

  virtual void operator()(dof_id_type dof_number,
                          const DofConstraintRow & constraint_row,
                          const Number constraint_rhs) const = 0;
};

class AddPrimalConstraint : public AddConstraint
{
public:
  AddPrimalConstraint(DofMap & dof_map_in) : AddConstraint(dof_map_in) {}

  virtual void operator()(dof_id_type dof_number,
                          const DofConstraintRow & constraint_row,
                          const Number constraint_rhs) const
  {
    if (!dof_map.is_constrained_dof(dof_number))
      dof_map.add_constraint_row (dof_number, constraint_row,
                                  constraint_rhs, true);
  }
};

class AddAdjointConstraint : public AddConstraint
{
private:
  const unsigned int qoi_index;

public:
  AddAdjointConstraint(DofMap & dof_map_in, unsigned int qoi_index_in)
    : AddConstraint(dof_map_in), qoi_index(qoi_index_in) {}

  virtual void operator()(dof_id_type dof_number,
                          const DofConstraintRow & constraint_row,
                          const Number constraint_rhs) const
  {
    dof_map.add_adjoint_constraint_row
      (qoi_index, dof_number, constraint_row, constraint_rhs,
       true);
  }
};



/**
 * This class implements turning an arbitrary
 * boundary function into Dirichlet constraints.  It
 * may be executed in parallel on multiple threads.
 */
class ConstrainDirichlet
{
private:
  DofMap                  & dof_map;
  const MeshBase          & mesh;
  const Real               time;
  const DirichletBoundary  dirichlet;

  const AddConstraint     & add_fn;

  static Number f_component (FunctionBase<Number> * f,
                             FEMFunctionBase<Number> * f_fem,
                             const FEMContext * c,
                             unsigned int i,
                             const Point & p,
                             Real time)
  {
    if (f_fem)
      {
        if (c)
          return f_fem->component(*c, i, p, time);
        else
          return std::numeric_limits<Real>::quiet_NaN();
      }
    return f->component(i, p, time);
  }

  static Gradient g_component (FunctionBase<Gradient> * g,
                               FEMFunctionBase<Gradient> * g_fem,
                               const FEMContext * c,
                               unsigned int i,
                               const Point & p,
                               Real time)
  {
    if (g_fem)
      {
        if (c)
          return g_fem->component(*c, i, p, time);
        else
          return std::numeric_limits<Number>::quiet_NaN();
      }
    return g->component(i, p, time);
  }

  template<typename OutputType>
  void apply_dirichlet_impl(const ConstElemRange & range,
                            const unsigned int var,
                            const Variable & variable,
                            const FEType & fe_type) const
  {
    typedef OutputType                                                      OutputShape;
    typedef typename TensorTools::IncrementRank<OutputShape>::type          OutputGradient;
    //typedef typename TensorTools::IncrementRank<OutputGradient>::type       OutputTensor;
    typedef typename TensorTools::MakeNumber<OutputShape>::type             OutputNumber;
    typedef typename TensorTools::IncrementRank<OutputNumber>::type         OutputNumberGradient;
    //typedef typename TensorTools::IncrementRank<OutputNumberGradient>::type OutputNumberTensor;

    FunctionBase<Number> * f = dirichlet.f.get();
    FunctionBase<Gradient> * g = dirichlet.g.get();

    FEMFunctionBase<Number> * f_fem = dirichlet.f_fem.get();
    FEMFunctionBase<Gradient> * g_fem = dirichlet.g_fem.get();

    const System * f_system = dirichlet.f_system;

    const std::set<boundary_id_type> & b = dirichlet.b;

    // We need data to project
    libmesh_assert(f || f_fem);
    libmesh_assert(!(f && f_fem));

    // Iff our data depends on a system, we should have it.
    libmesh_assert(!(f && f_system));
    libmesh_assert(!(f_fem && !f_system));

    // The element matrix and RHS for projections.
    // Note that Ke is always real-valued, whereas
    // Fe may be complex valued if complex number
    // support is enabled
    DenseMatrix<Real> Ke;
    DenseVector<Number> Fe;
    // The new element coefficients
    DenseVector<Number> Ue;

    // The dimensionality of the current mesh
    const unsigned int dim = mesh.mesh_dimension();

    // Boundary info for the current mesh
    const BoundaryInfo & boundary_info = mesh.get_boundary_info();

    unsigned int n_vec_dim = FEInterface::n_vec_dim(mesh, fe_type);

    const unsigned int var_component =
      variable.first_scalar_number();

    // Get FE objects of the appropriate type
    std::unique_ptr<FEGenericBase<OutputType>> fe = FEGenericBase<OutputType>::build(dim, fe_type);

    // Set tolerance on underlying FEMap object. This will allow us to
    // avoid spurious negative Jacobian errors while imposing BCs by
    // simply ignoring them. This should only be required in certain
    // special cases, see the DirichletBoundaries comments on this
    // parameter for more information.
    fe->get_fe_map().set_jacobian_tolerance(dirichlet.jacobian_tolerance);

    // Prepare variables for projection
    std::unique_ptr<QBase> qedgerule (fe_type.default_quadrature_rule(1));
    std::unique_ptr<QBase> qsiderule (fe_type.default_quadrature_rule(dim-1));
    std::unique_ptr<QBase> qrule (fe_type.default_quadrature_rule(dim));

    // The values of the shape functions at the quadrature
    // points
    const std::vector<std::vector<OutputShape>> & phi = fe->get_phi();

    // The gradients of the shape functions at the quadrature
    // points on the child element.
    const std::vector<std::vector<OutputGradient>> * dphi = nullptr;

    const FEContinuity cont = fe->get_continuity();

    if ((cont == C_ONE) && (fe_type.family != SUBDIVISION))
      {
        // We'll need gradient data for a C1 projection
        libmesh_assert(g || g_fem);

        // We currently demand that either neither nor both function
        // object depend on current FEM data.
        libmesh_assert(!(g && g_fem));
        libmesh_assert(!(f && g_fem));
        libmesh_assert(!(f_fem && g));

        const std::vector<std::vector<OutputGradient>> & ref_dphi = fe->get_dphi();
        dphi = &ref_dphi;
      }

    // The Jacobian * quadrature weight at the quadrature points
    const std::vector<Real> & JxW = fe->get_JxW();

    // The XYZ locations of the quadrature points
    const std::vector<Point> & xyz_values = fe->get_xyz();

    // The global DOF indices
    std::vector<dof_id_type> dof_indices;
    // Side/edge local DOF indices
    std::vector<unsigned int> side_dofs;

    // If our supplied functions require a FEMContext, and if we have
    // an initialized solution to use with that FEMContext, then
    // create one
    std::unique_ptr<FEMContext> context;
    if (f_fem)
      {
        libmesh_assert(f_system);
        if (f_system->current_local_solution->initialized())
          {
            context = libmesh_make_unique<FEMContext>(*f_system);
            f_fem->init_context(*context);
            if (g_fem)
              g_fem->init_context(*context);
          }
      }

    // Iterate over all the elements in the range
    for (const auto & elem : range)
      {
        // We only calculate Dirichlet constraints on active
        // elements
        if (!elem->active())
          continue;

        // Per-subdomain variables don't need to be projected on
        // elements where they're not active
        if (!variable.active_on_subdomain(elem->subdomain_id()))
          continue;

        const unsigned short n_sides = elem->n_sides();
        const unsigned short n_edges = elem->n_edges();
        const unsigned short n_nodes = elem->n_nodes();

        // Find out which nodes, edges, sides and shellfaces are on a requested
        // boundary:
        std::vector<bool> is_boundary_node(n_nodes, false),
          is_boundary_edge(n_edges, false),
          is_boundary_side(n_sides, false),
          is_boundary_shellface(2, false);

        // We also maintain a separate list of nodeset-based boundary nodes
        std::vector<bool> is_boundary_nodeset(n_nodes, false);

        // Update has_dirichlet_constraint below, and if it remains false then
        // we can skip this element since there are not constraints to impose.
        bool has_dirichlet_constraint = false;

        // Container to catch boundary ids handed back for sides,
        // nodes, and edges in the loops below.
        std::vector<boundary_id_type> ids_vec;

        for (unsigned char s = 0; s != n_sides; ++s)
          {
            // First see if this side has been requested
            boundary_info.boundary_ids (elem, s, ids_vec);

            bool do_this_side = false;
            for (const auto & bc_id : ids_vec)
              if (b.count(bc_id))
                {
                  do_this_side = true;
                  break;
                }
            if (!do_this_side)
              continue;

            is_boundary_side[s] = true;
            has_dirichlet_constraint = true;

            // Then see what nodes and what edges are on it
            for (unsigned int n = 0; n != n_nodes; ++n)
              if (elem->is_node_on_side(n,s))
                is_boundary_node[n] = true;
            for (unsigned int e = 0; e != n_edges; ++e)
              if (elem->is_edge_on_side(e,s))
                is_boundary_edge[e] = true;
          }

        // We can also impose Dirichlet boundary conditions on nodes, so we should
        // also independently check whether the nodes have been requested
        for (unsigned int n=0; n != n_nodes; ++n)
          {
            boundary_info.boundary_ids (elem->node_ptr(n), ids_vec);

            for (const auto & bc_id : ids_vec)
              if (b.count(bc_id))
                {
                  is_boundary_node[n] = true;
                  is_boundary_nodeset[n] = true;
                  has_dirichlet_constraint = true;
                }
          }

        // We can also impose Dirichlet boundary conditions on edges, so we should
        // also independently check whether the edges have been requested
        for (unsigned short e=0; e != n_edges; ++e)
          {
            boundary_info.edge_boundary_ids (elem, e, ids_vec);

            for (const auto & bc_id : ids_vec)
              if (b.count(bc_id))
                {
                  is_boundary_edge[e] = true;
                  has_dirichlet_constraint = true;
                }
          }

        // We can also impose Dirichlet boundary conditions on shellfaces, so we should
        // also independently check whether the shellfaces have been requested
        for (unsigned short shellface=0; shellface != 2; ++shellface)
          {
            boundary_info.shellface_boundary_ids (elem, shellface, ids_vec);

            for (const auto & bc_id : ids_vec)
              if (b.count(bc_id))
                {
                  is_boundary_shellface[shellface] = true;
                  has_dirichlet_constraint = true;
                }
          }

        if(!has_dirichlet_constraint)
          {
            continue;
          }

        // There's a chicken-and-egg problem with FEMFunction-based
        // Dirichlet constraints: we can't evaluate the FEMFunction
        // until we have an initialized local solution vector, we
        // can't initialize local solution vectors until we have a
        // send list, and we can't generate a send list until we know
        // all our constraints
        //
        // We don't generate constraints on uninitialized systems;
        // currently user code will have to reinit() before any
        // FEMFunction-based constraints will be correct.  This should
        // be fine, since user code would want to reinit() after
        // setting initial conditions anyway.
        if (f_system && context.get())
          context->pre_fe_reinit(*f_system, elem);

        // Update the DOF indices for this element based on
        // the current mesh
        dof_map.dof_indices (elem, dof_indices, var);

        // The number of DOFs on the element
        const unsigned int n_dofs =
          cast_int<unsigned int>(dof_indices.size());

        // Fixed vs. free DoFs on edge/face projections
        std::vector<char> dof_is_fixed(n_dofs, false); // bools
        std::vector<int> free_dof(n_dofs, 0);

        // The element type
        const ElemType elem_type = elem->type();

        // Zero the interpolated values
        Ue.resize (n_dofs); Ue.zero();

        // In general, we need a series of
        // projections to ensure a unique and continuous
        // solution.  We start by interpolating boundary nodes, then
        // hold those fixed and project boundary edges, then hold
        // those fixed and project boundary faces,

        // Interpolate node values first. Note that we have a special
        // case for nodes that have a boundary nodeset, since we do
        // need to interpolate them directly, even if they're non-vertex
        // nodes.
        unsigned int current_dof = 0;
        for (unsigned int n=0; n!= n_nodes; ++n)
          {
            // FIXME: this should go through the DofMap,
            // not duplicate dof_indices code badly!
            const unsigned int nc =
              FEInterface::n_dofs_at_node (dim, fe_type, elem_type,
                                           n);
            if ((!elem->is_vertex(n) || !is_boundary_node[n]) &&
                !is_boundary_nodeset[n])
              {
                current_dof += nc;
                continue;
              }
            if (cont == DISCONTINUOUS)
              {
                libmesh_assert_equal_to (nc, 0);
              }
            // Assume that C_ZERO elements have a single nodal
            // value shape function
            else if ((cont == C_ZERO) || (fe_type.family == SUBDIVISION))
              {
                libmesh_assert_equal_to (nc, n_vec_dim);
                for (unsigned int c = 0; c < n_vec_dim; c++)
                  {
                    Ue(current_dof+c) =
                      f_component(f, f_fem, context.get(), var_component+c,
                                  elem->point(n), time);
                    dof_is_fixed[current_dof+c] = true;
                  }
                current_dof += n_vec_dim;
              }
            // The hermite element vertex shape functions are weird
            else if (fe_type.family == HERMITE)
              {
                Ue(current_dof) =
                  f_component(f, f_fem, context.get(), var_component,
                              elem->point(n), time);
                dof_is_fixed[current_dof] = true;
                current_dof++;
                Gradient grad =
                  g_component(g, g_fem, context.get(), var_component,
                              elem->point(n), time);
                // x derivative
                Ue(current_dof) = grad(0);
                dof_is_fixed[current_dof] = true;
                current_dof++;
                if (dim > 1)
                  {
                    // We'll finite difference mixed derivatives
                    Point nxminus = elem->point(n),
                      nxplus = elem->point(n);
                    nxminus(0) -= TOLERANCE;
                    nxplus(0) += TOLERANCE;
                    Gradient gxminus =
                      g_component(g, g_fem, context.get(), var_component,
                                  nxminus, time);
                    Gradient gxplus =
                      g_component(g, g_fem, context.get(), var_component,
                                  nxplus, time);
                    // y derivative
                    Ue(current_dof) = grad(1);
                    dof_is_fixed[current_dof] = true;
                    current_dof++;
                    // xy derivative
                    Ue(current_dof) = (gxplus(1) - gxminus(1))
                      / 2. / TOLERANCE;
                    dof_is_fixed[current_dof] = true;
                    current_dof++;

                    if (dim > 2)
                      {
                        // z derivative
                        Ue(current_dof) = grad(2);
                        dof_is_fixed[current_dof] = true;
                        current_dof++;
                        // xz derivative
                        Ue(current_dof) = (gxplus(2) - gxminus(2))
                          / 2. / TOLERANCE;
                        dof_is_fixed[current_dof] = true;
                        current_dof++;
                        // We need new points for yz
                        Point nyminus = elem->point(n),
                          nyplus = elem->point(n);
                        nyminus(1) -= TOLERANCE;
                        nyplus(1) += TOLERANCE;
                        Gradient gyminus =
                          g_component(g, g_fem, context.get(), var_component,
                                      nyminus, time);
                        Gradient gyplus =
                          g_component(g, g_fem, context.get(), var_component,
                                      nyplus, time);
                        // xz derivative
                        Ue(current_dof) = (gyplus(2) - gyminus(2))
                          / 2. / TOLERANCE;
                        dof_is_fixed[current_dof] = true;
                        current_dof++;
                        // Getting a 2nd order xyz is more tedious
                        Point nxmym = elem->point(n),
                          nxmyp = elem->point(n),
                          nxpym = elem->point(n),
                          nxpyp = elem->point(n);
                        nxmym(0) -= TOLERANCE;
                        nxmym(1) -= TOLERANCE;
                        nxmyp(0) -= TOLERANCE;
                        nxmyp(1) += TOLERANCE;
                        nxpym(0) += TOLERANCE;
                        nxpym(1) -= TOLERANCE;
                        nxpyp(0) += TOLERANCE;
                        nxpyp(1) += TOLERANCE;
                        Gradient gxmym =
                          g_component(g, g_fem, context.get(), var_component,
                                      nxmym, time);
                        Gradient gxmyp =
                          g_component(g, g_fem, context.get(), var_component,
                                      nxmyp, time);
                        Gradient gxpym =
                          g_component(g, g_fem, context.get(), var_component,
                                      nxpym, time);
                        Gradient gxpyp =
                          g_component(g, g_fem, context.get(), var_component,
                                      nxpyp, time);
                        Number gxzplus = (gxpyp(2) - gxmyp(2))
                          / 2. / TOLERANCE;
                        Number gxzminus = (gxpym(2) - gxmym(2))
                          / 2. / TOLERANCE;
                        // xyz derivative
                        Ue(current_dof) = (gxzplus - gxzminus)
                          / 2. / TOLERANCE;
                        dof_is_fixed[current_dof] = true;
                        current_dof++;
                      }
                  }
              }
            // Assume that other C_ONE elements have a single nodal
            // value shape function and nodal gradient component
            // shape functions
            else if (cont == C_ONE)
              {
                libmesh_assert_equal_to (nc, 1 + dim);
                Ue(current_dof) =
                  f_component(f, f_fem, context.get(), var_component,
                              elem->point(n), time);
                dof_is_fixed[current_dof] = true;
                current_dof++;
                Gradient grad =
                  g_component(g, g_fem, context.get(), var_component,
                              elem->point(n), time);
                for (unsigned int i=0; i!= dim; ++i)
                  {
                    Ue(current_dof) = grad(i);
                    dof_is_fixed[current_dof] = true;
                    current_dof++;
                  }
              }
            else
              libmesh_error_msg("Unknown continuity cont = " << cont);
          }

        // In 3D, project any edge values next
        if (dim > 2 && cont != DISCONTINUOUS)
          for (unsigned int e=0; e != n_edges; ++e)
            {
              if (!is_boundary_edge[e])
                continue;

              FEInterface::dofs_on_edge(elem, dim, fe_type, e,
                                        side_dofs);

              const unsigned int n_side_dofs =
                cast_int<unsigned int>(side_dofs.size());

              // Some edge dofs are on nodes and already
              // fixed, others are free to calculate
              unsigned int free_dofs = 0;
              for (unsigned int i=0; i != n_side_dofs; ++i)
                if (!dof_is_fixed[side_dofs[i]])
                  free_dof[free_dofs++] = i;

              // There may be nothing to project
              if (!free_dofs)
                continue;

              Ke.resize (free_dofs, free_dofs); Ke.zero();
              Fe.resize (free_dofs); Fe.zero();
              // The new edge coefficients
              DenseVector<Number> Uedge(free_dofs);

              // Initialize FE data on the edge
              fe->attach_quadrature_rule (qedgerule.get());
              fe->edge_reinit (elem, e);
              const unsigned int n_qp = qedgerule->n_points();

              // Loop over the quadrature points
              for (unsigned int qp=0; qp<n_qp; qp++)
                {
                  // solution at the quadrature point
                  OutputNumber fineval(0);
                  libMesh::RawAccessor<OutputNumber> f_accessor( fineval, dim );

                  for (unsigned int c = 0; c < n_vec_dim; c++)
                    f_accessor(c) =
                      f_component(f, f_fem, context.get(), var_component+c,
                                  xyz_values[qp], time);

                  // solution grad at the quadrature point
                  OutputNumberGradient finegrad;
                  libMesh::RawAccessor<OutputNumberGradient> g_accessor( finegrad, dim );

                  unsigned int g_rank;
                  switch( FEInterface::field_type( fe_type ) )
                    {
                    case TYPE_SCALAR:
                      {
                        g_rank = 1;
                        break;
                      }
                    case TYPE_VECTOR:
                      {
                        g_rank = 2;
                        break;
                      }
                    default:
                      libmesh_error_msg("Unknown field type!");
                    }

                  if (cont == C_ONE)
                    for (unsigned int c = 0; c < n_vec_dim; c++)
                      for (unsigned int d = 0; d < g_rank; d++)
                        g_accessor(c + d*dim ) =
                          g_component(g, g_fem, context.get(), var_component,
                                      xyz_values[qp], time)(c);

                  // Form edge projection matrix
                  for (unsigned int sidei=0, freei=0; sidei != n_side_dofs; ++sidei)
                    {
                      unsigned int i = side_dofs[sidei];
                      // fixed DoFs aren't test functions
                      if (dof_is_fixed[i])
                        continue;
                      for (unsigned int sidej=0, freej=0; sidej != n_side_dofs; ++sidej)
                        {
                          unsigned int j = side_dofs[sidej];
                          if (dof_is_fixed[j])
                            Fe(freei) -= phi[i][qp] * phi[j][qp] *
                              JxW[qp] * Ue(j);
                          else
                            Ke(freei,freej) += phi[i][qp] *
                              phi[j][qp] * JxW[qp];
                          if (cont == C_ONE)
                            {
                              if (dof_is_fixed[j])
                                Fe(freei) -= ((*dphi)[i][qp].contract((*dphi)[j][qp]) ) *
                                  JxW[qp] * Ue(j);
                              else
                                Ke(freei,freej) += ((*dphi)[i][qp].contract((*dphi)[j][qp]))
                                  * JxW[qp];
                            }
                          if (!dof_is_fixed[j])
                            freej++;
                        }
                      Fe(freei) += phi[i][qp] * fineval * JxW[qp];
                      if (cont == C_ONE)
                        Fe(freei) += (finegrad.contract( (*dphi)[i][qp]) ) *
                          JxW[qp];
                      freei++;
                    }
                }

              Ke.cholesky_solve(Fe, Uedge);

              // Transfer new edge solutions to element
              for (unsigned int i=0; i != free_dofs; ++i)
                {
                  Number & ui = Ue(side_dofs[free_dof[i]]);
                  libmesh_assert(std::abs(ui) < TOLERANCE ||
                                 std::abs(ui - Uedge(i)) < TOLERANCE);
                  ui = Uedge(i);
                  dof_is_fixed[side_dofs[free_dof[i]]] = true;
                }
            }

        // Project any side values (edges in 2D, faces in 3D)
        if (dim > 1 && cont != DISCONTINUOUS)
          for (unsigned int s=0; s != n_sides; ++s)
            {
              if (!is_boundary_side[s])
                continue;

              FEInterface::dofs_on_side(elem, dim, fe_type, s,
                                        side_dofs);

              const unsigned int n_side_dofs =
                cast_int<unsigned int>(side_dofs.size());

              // Some side dofs are on nodes/edges and already
              // fixed, others are free to calculate
              unsigned int free_dofs = 0;
              for (unsigned int i=0; i != n_side_dofs; ++i)
                if (!dof_is_fixed[side_dofs[i]])
                  free_dof[free_dofs++] = i;

              // There may be nothing to project
              if (!free_dofs)
                continue;

              Ke.resize (free_dofs, free_dofs); Ke.zero();
              Fe.resize (free_dofs); Fe.zero();
              // The new side coefficients
              DenseVector<Number> Uside(free_dofs);

              // Initialize FE data on the side
              fe->attach_quadrature_rule (qsiderule.get());
              fe->reinit (elem, s);
              const unsigned int n_qp = qsiderule->n_points();

              // Loop over the quadrature points
              for (unsigned int qp=0; qp<n_qp; qp++)
                {
                  // solution at the quadrature point
                  OutputNumber fineval(0);
                  libMesh::RawAccessor<OutputNumber> f_accessor( fineval, dim );

                  for (unsigned int c = 0; c < n_vec_dim; c++)
                    f_accessor(c) =
                      f_component(f, f_fem, context.get(), var_component+c,
                                  xyz_values[qp], time);

                  // solution grad at the quadrature point
                  OutputNumberGradient finegrad;
                  libMesh::RawAccessor<OutputNumberGradient> g_accessor( finegrad, dim );

                  unsigned int g_rank;
                  switch( FEInterface::field_type( fe_type ) )
                    {
                    case TYPE_SCALAR:
                      {
                        g_rank = 1;
                        break;
                      }
                    case TYPE_VECTOR:
                      {
                        g_rank = 2;
                        break;
                      }
                    default:
                      libmesh_error_msg("Unknown field type!");
                    }

                  if (cont == C_ONE)
                    for (unsigned int c = 0; c < n_vec_dim; c++)
                      for (unsigned int d = 0; d < g_rank; d++)
                        g_accessor(c + d*dim ) =
                          g_component(g, g_fem, context.get(), var_component,
                                      xyz_values[qp], time)(c);

                  // Form side projection matrix
                  for (unsigned int sidei=0, freei=0; sidei != n_side_dofs; ++sidei)
                    {
                      unsigned int i = side_dofs[sidei];
                      // fixed DoFs aren't test functions
                      if (dof_is_fixed[i])
                        continue;
                      for (unsigned int sidej=0, freej=0; sidej != n_side_dofs; ++sidej)
                        {
                          unsigned int j = side_dofs[sidej];
                          if (dof_is_fixed[j])
                            Fe(freei) -= phi[i][qp] * phi[j][qp] *
                              JxW[qp] * Ue(j);
                          else
                            Ke(freei,freej) += phi[i][qp] *
                              phi[j][qp] * JxW[qp];
                          if (cont == C_ONE)
                            {
                              if (dof_is_fixed[j])
                                Fe(freei) -= ((*dphi)[i][qp].contract((*dphi)[j][qp])) *
                                  JxW[qp] * Ue(j);
                              else
                                Ke(freei,freej) += ((*dphi)[i][qp].contract((*dphi)[j][qp]))
                                  * JxW[qp];
                            }
                          if (!dof_is_fixed[j])
                            freej++;
                        }
                      Fe(freei) += (fineval * phi[i][qp]) * JxW[qp];
                      if (cont == C_ONE)
                        Fe(freei) += (finegrad.contract((*dphi)[i][qp])) *
                          JxW[qp];
                      freei++;
                    }
                }

              Ke.cholesky_solve(Fe, Uside);

              // Transfer new side solutions to element
              for (unsigned int i=0; i != free_dofs; ++i)
                {
                  Number & ui = Ue(side_dofs[free_dof[i]]);
                  libmesh_assert(std::abs(ui) < TOLERANCE ||
                                 std::abs(ui - Uside(i)) < TOLERANCE);
                  ui = Uside(i);
                  dof_is_fixed[side_dofs[free_dof[i]]] = true;
                }
            }

        // Project any shellface values
        if (dim == 2 && cont != DISCONTINUOUS)
          for (unsigned int shellface=0; shellface != 2; ++shellface)
            {
              if (!is_boundary_shellface[shellface])
                continue;

              // A shellface has the same dof indices as the element itself
              std::vector<unsigned int> shellface_dofs(n_dofs);
              std::iota(shellface_dofs.begin(), shellface_dofs.end(), 0);

              // Some shellface dofs are on nodes/edges and already
              // fixed, others are free to calculate
              unsigned int free_dofs = 0;
              for (unsigned int i=0; i != n_dofs; ++i)
                if (!dof_is_fixed[shellface_dofs[i]])
                  free_dof[free_dofs++] = i;

              // There may be nothing to project
              if (!free_dofs)
                continue;

              Ke.resize (free_dofs, free_dofs); Ke.zero();
              Fe.resize (free_dofs); Fe.zero();
              // The new shellface coefficients
              DenseVector<Number> Ushellface(free_dofs);

              // Initialize FE data on the element
              fe->attach_quadrature_rule (qrule.get());
              fe->reinit (elem);
              const unsigned int n_qp = qrule->n_points();

              // Loop over the quadrature points
              for (unsigned int qp=0; qp<n_qp; qp++)
                {
                  // solution at the quadrature point
                  OutputNumber fineval(0);
                  libMesh::RawAccessor<OutputNumber> f_accessor( fineval, dim );

                  for (unsigned int c = 0; c < n_vec_dim; c++)
                    f_accessor(c) =
                      f_component(f, f_fem, context.get(), var_component+c,
                                  xyz_values[qp], time);

                  // solution grad at the quadrature point
                  OutputNumberGradient finegrad;
                  libMesh::RawAccessor<OutputNumberGradient> g_accessor( finegrad, dim );

                  unsigned int g_rank;
                  switch( FEInterface::field_type( fe_type ) )
                    {
                    case TYPE_SCALAR:
                      {
                        g_rank = 1;
                        break;
                      }
                    case TYPE_VECTOR:
                      {
                        g_rank = 2;
                        break;
                      }
                    default:
                      libmesh_error_msg("Unknown field type!");
                    }

                  if (cont == C_ONE)
                    for (unsigned int c = 0; c < n_vec_dim; c++)
                      for (unsigned int d = 0; d < g_rank; d++)
                        g_accessor(c + d*dim ) =
                          g_component(g, g_fem, context.get(), var_component,
                                      xyz_values[qp], time)(c);

                  // Form shellface projection matrix
                  for (unsigned int shellfacei=0, freei=0;
                       shellfacei != n_dofs; ++shellfacei)
                    {
                      unsigned int i = shellface_dofs[shellfacei];
                      // fixed DoFs aren't test functions
                      if (dof_is_fixed[i])
                        continue;
                      for (unsigned int shellfacej=0, freej=0;
                           shellfacej != n_dofs; ++shellfacej)
                        {
                          unsigned int j = shellface_dofs[shellfacej];
                          if (dof_is_fixed[j])
                            Fe(freei) -= phi[i][qp] * phi[j][qp] *
                              JxW[qp] * Ue(j);
                          else
                            Ke(freei,freej) += phi[i][qp] *
                              phi[j][qp] * JxW[qp];
                          if (cont == C_ONE)
                            {
                              if (dof_is_fixed[j])
                                Fe(freei) -= ((*dphi)[i][qp].contract((*dphi)[j][qp])) *
                                  JxW[qp] * Ue(j);
                              else
                                Ke(freei,freej) += ((*dphi)[i][qp].contract((*dphi)[j][qp]))
                                  * JxW[qp];
                            }
                          if (!dof_is_fixed[j])
                            freej++;
                        }
                      Fe(freei) += (fineval * phi[i][qp]) * JxW[qp];
                      if (cont == C_ONE)
                        Fe(freei) += (finegrad.contract((*dphi)[i][qp])) *
                          JxW[qp];
                      freei++;
                    }
                }

              Ke.cholesky_solve(Fe, Ushellface);

              // Transfer new shellface solutions to element
              for (unsigned int i=0; i != free_dofs; ++i)
                {
                  Number & ui = Ue(shellface_dofs[free_dof[i]]);
                  libmesh_assert(std::abs(ui) < TOLERANCE ||
                                 std::abs(ui - Ushellface(i)) < TOLERANCE);
                  ui = Ushellface(i);
                  dof_is_fixed[shellface_dofs[free_dof[i]]] = true;
                }
            }

        // Lock the DofConstraints since it is shared among threads.
        {
          Threads::spin_mutex::scoped_lock lock(Threads::spin_mtx);

          for (unsigned int i = 0; i < n_dofs; i++)
            {
              DofConstraintRow empty_row;
              if (dof_is_fixed[i] && !libmesh_isnan(Ue(i)))
                add_fn (dof_indices[i], empty_row, Ue(i));
            }
        }
      }

  } // apply_dirichlet_impl

public:
  ConstrainDirichlet (DofMap & dof_map_in,
                      const MeshBase & mesh_in,
                      const Real time_in,
                      const DirichletBoundary & dirichlet_in,
                      const AddConstraint & add_in) :
    dof_map(dof_map_in),
    mesh(mesh_in),
    time(time_in),
    dirichlet(dirichlet_in),
    add_fn(add_in) { }

  ConstrainDirichlet (const ConstrainDirichlet & in) :
    dof_map(in.dof_map),
    mesh(in.mesh),
    time(in.time),
    dirichlet(in.dirichlet),
    add_fn(in.add_fn) { }

  void operator()(const ConstElemRange & range) const
  {
    /**
     * This method examines an arbitrary boundary solution to calculate
     * corresponding Dirichlet constraints on the current mesh.  The
     * input function \p f gives the arbitrary solution.
     */

    // Loop over all the variables we've been requested to project
    for (const auto & var : dirichlet.variables)
      {
        const Variable & variable = dof_map.variable(var);

        const FEType & fe_type = variable.type();

        if (fe_type.family == SCALAR)
          continue;

        switch( FEInterface::field_type( fe_type ) )
          {
          case TYPE_SCALAR:
            {
              this->apply_dirichlet_impl<Real>( range, var, variable, fe_type );
              break;
            }
          case TYPE_VECTOR:
            {
              this->apply_dirichlet_impl<RealGradient>( range, var, variable, fe_type );
              break;
            }
          default:
            libmesh_error_msg("Unknown field type!");
          }

      }
  }

}; // class ConstrainDirichlet


#endif // LIBMESH_ENABLE_DIRICHLET


} // anonymous namespace


template <typename RealType>
void DofMap::create_dof_constraints(const MeshBaseTempl<RealType> & mesh, Real time)
{
  parallel_object_only();

  LOG_SCOPE("create_dof_constraints()", "DofMap");

  libmesh_assert (mesh.is_prepared());

  // The user might have set boundary conditions after the mesh was
  // prepared; we should double-check that those boundary conditions
  // are still consistent.
#ifdef DEBUG
  MeshTools::libmesh_assert_valid_boundary_ids(mesh);
#endif

  // We might get constraint equations from AMR hanging nodes in 2D/3D
  // or from boundary conditions in any dimension
  const bool possible_local_constraints = false
    || !mesh.n_elem()
#ifdef LIBMESH_ENABLE_AMR
    || mesh.mesh_dimension() > 1
#endif
#ifdef LIBMESH_ENABLE_PERIODIC
    || !_periodic_boundaries->empty()
#endif
#ifdef LIBMESH_ENABLE_DIRICHLET
    || !_dirichlet_boundaries->empty()
#endif
    ;

  // Even if we don't have constraints, another processor might.
  bool possible_global_constraints = possible_local_constraints;
#if defined(LIBMESH_ENABLE_PERIODIC) || defined(LIBMESH_ENABLE_DIRICHLET) || defined(LIBMESH_ENABLE_AMR)
  libmesh_assert(this->comm().verify(mesh.is_serial()));

  this->comm().max(possible_global_constraints);
#endif

  if (!possible_global_constraints)
    {
      // Clear out any old constraints; maybe the user just deleted
      // their last remaining dirichlet/periodic/user constraint?
      // Note: any _stashed_dof_constraints are not cleared as it
      // may be the user's intention to restore them later.
#ifdef LIBMESH_ENABLE_CONSTRAINTS
      _dof_constraints.clear();
      _primal_constraint_values.clear();
      _adjoint_constraint_values.clear();
#endif
#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
      _node_constraints.clear();
#endif

      return;
    }

  // Here we build the hanging node constraints.  This is done
  // by enforcing the condition u_a = u_b along hanging sides.
  // u_a = u_b is collocated at the nodes of side a, which gives
  // one row of the constraint matrix.

  // Processors only compute their local constraints
  ConstElemRange range (mesh.local_elements_begin(),
                        mesh.local_elements_end());

  // Global computation fails if we're using a FEMFunctionBase BC on a
  // ReplicatedMesh in parallel
  // ConstElemRange range (mesh.elements_begin(),
  //                       mesh.elements_end());

  // compute_periodic_constraints requires a point_locator() from our
  // Mesh, but point_locator() construction is parallel and threaded.
  // Rather than nest threads within threads we'll make sure it's
  // preconstructed.
#ifdef LIBMESH_ENABLE_PERIODIC
  bool need_point_locator = !_periodic_boundaries->empty() && !range.empty();

  this->comm().max(need_point_locator);

  if (need_point_locator)
    mesh.sub_point_locator();
#endif

#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
  // recalculate node constraints from scratch
  _node_constraints.clear();

  Threads::parallel_for (range,
                         ComputeNodeConstraints (_node_constraints,
#ifdef LIBMESH_ENABLE_PERIODIC
                                                 *_periodic_boundaries,
#endif
                                                 mesh));
#endif // LIBMESH_ENABLE_NODE_CONSTRAINTS


  // recalculate dof constraints from scratch
  // Note: any _stashed_dof_constraints are not cleared as it
  // may be the user's intention to restore them later.
  _dof_constraints.clear();
  _primal_constraint_values.clear();
  _adjoint_constraint_values.clear();

  // Look at all the variables in the system.  Reset the element
  // range at each iteration -- there is no need to reconstruct it.
  for (unsigned int variable_number=0; variable_number<this->n_variables();
       ++variable_number, range.reset())
    Threads::parallel_for (range,
                           ComputeConstraints (_dof_constraints,
                                               *this,
#ifdef LIBMESH_ENABLE_PERIODIC
                                               *_periodic_boundaries,
#endif
                                               mesh,
                                               variable_number));

#ifdef LIBMESH_ENABLE_DIRICHLET
  for (DirichletBoundaries::iterator
         i = _dirichlet_boundaries->begin();
       i != _dirichlet_boundaries->end(); ++i, range.reset())
    {
      // Sanity check that the boundary ids associated with the DirichletBoundary
      // objects are actually present in the mesh
      this->check_dirichlet_bcid_consistency(mesh,**i);

      Threads::parallel_for
        (range,
         ConstrainDirichlet(*this, mesh, time, **i,
                            AddPrimalConstraint(*this))
         );
    }

  for (unsigned int qoi_index = 0,
       n_qois = cast_int<unsigned int>(_adjoint_dirichlet_boundaries.size());
       qoi_index != n_qois; ++qoi_index)
    {
      for (DirichletBoundaries::iterator
             i = _adjoint_dirichlet_boundaries[qoi_index]->begin();
           i != _adjoint_dirichlet_boundaries[qoi_index]->end();
           ++i, range.reset())
        {
          // Sanity check that the boundary ids associated with the DirichletBoundary
          // objects are actually present in the mesh
          this->check_dirichlet_bcid_consistency(mesh,**i);

          Threads::parallel_for
            (range,
             ConstrainDirichlet(*this, mesh, time, **i,
                                AddAdjointConstraint(*this, qoi_index))
             );
        }
    }

#endif // LIBMESH_ENABLE_DIRICHLET
}

template <typename RealType>
void DofMap::allgather_recursive_constraints(MeshBaseTempl<RealType> & mesh)
{
  // This function must be run on all processors at once
  parallel_object_only();

  // Return immediately if there's nothing to gather
  if (this->n_processors() == 1)
    return;

  // We might get to return immediately if none of the processors
  // found any constraints
  unsigned int has_constraints = !_dof_constraints.empty()
#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
    || !_node_constraints.empty()
#endif // LIBMESH_ENABLE_NODE_CONSTRAINTS
    ;
  this->comm().max(has_constraints);
  if (!has_constraints)
    return;

  // If we have heterogenous adjoint constraints we need to
  // communicate those too.
  const unsigned int max_qoi_num =
    _adjoint_constraint_values.empty() ?
    0 : _adjoint_constraint_values.rbegin()->first;

  // We might have calculated constraints for constrained dofs
  // which have support on other processors.
  // Push these out first.
  {
    std::map<processor_id_type, std::set<dof_id_type>> pushed_ids;

#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
    std::map<processor_id_type, std::set<dof_id_type>> pushed_node_ids;
#endif

    const unsigned int sys_num = this->sys_number();

    // Collect the constraints to push to each processor
    for (auto & elem : as_range(mesh.active_not_local_elements_begin(),
                                mesh.active_not_local_elements_end()))
      {
        const unsigned short n_nodes = elem->n_nodes();

        // Just checking dof_indices on the foreign element isn't
        // enough.  Consider a central hanging node between a coarse
        // Q2/Q1 element and its finer neighbors on a higher-ranked
        // processor.  The coarse element's processor will own the node,
        // and will thereby own the pressure dof on that node, despite
        // the fact that that pressure dof doesn't directly exist on the
        // coarse element!
        //
        // So, we loop through dofs manually.

        {
          const unsigned int n_vars = elem->n_vars(sys_num);
          for (unsigned int v=0; v != n_vars; ++v)
            {
              const unsigned int n_comp = elem->n_comp(sys_num,v);
              for (unsigned int c=0; c != n_comp; ++c)
                {
                  const unsigned int id =
                    elem->dof_number(sys_num,v,c);
                  if (this->is_constrained_dof(id))
                    pushed_ids[elem->processor_id()].insert(id);
                }
            }
        }

        for (unsigned short n = 0; n != n_nodes; ++n)
          {
            const Node & node = elem->node_ref(n);
            const unsigned int n_vars = node.n_vars(sys_num);
            for (unsigned int v=0; v != n_vars; ++v)
              {
                const unsigned int n_comp = node.n_comp(sys_num,v);
                for (unsigned int c=0; c != n_comp; ++c)
                  {
                    const unsigned int id =
                      node.dof_number(sys_num,v,c);
                    if (this->is_constrained_dof(id))
                      pushed_ids[elem->processor_id()].insert(id);
                  }
              }
          }

#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
        for (unsigned short n = 0; n != n_nodes; ++n)
          if (this->is_constrained_node(elem->node_ptr(n)))
            pushed_node_ids[elem->processor_id()].insert(elem->node_id(n));
#endif
      }

    // Rewrite those id sets as vectors for sending and receiving,
    // then find the corresponding data for each id, then push it all.
    std::map<processor_id_type, std::vector<dof_id_type>>
      pushed_id_vecs, received_id_vecs;
    for (auto & p : pushed_ids)
      pushed_id_vecs[p.first].assign(p.second.begin(), p.second.end());

    std::map<processor_id_type, std::vector<std::vector<std::pair<dof_id_type,Real>>>>
      pushed_keys_vals, received_keys_vals;
    std::map<processor_id_type, std::vector<std::vector<Number>>> pushed_rhss, received_rhss;
    for (auto & p : pushed_id_vecs)
      {
        auto & keys_vals = pushed_keys_vals[p.first];
        keys_vals.reserve(p.second.size());

        auto & rhss = pushed_rhss[p.first];
        rhss.reserve(p.second.size());
        for (auto & pushed_id : p.second)
          {
            const DofConstraintRow & row = _dof_constraints[pushed_id];
            keys_vals.emplace_back(row.begin(), row.end());

            rhss.push_back(std::vector<Number>(max_qoi_num+1));
            std::vector<Number> & rhs = rhss.back();
            DofConstraintValueMap::const_iterator rhsit =
              _primal_constraint_values.find(pushed_id);
            rhs[max_qoi_num] =
              (rhsit == _primal_constraint_values.end()) ?
              0 : rhsit->second;
            for (unsigned int q = 0; q != max_qoi_num; ++q)
              {
                AdjointDofConstraintValues::const_iterator adjoint_map_it =
                  _adjoint_constraint_values.find(q);

                if (adjoint_map_it == _adjoint_constraint_values.end())
                  continue;

                const DofConstraintValueMap & constraint_map =
                  adjoint_map_it->second;

                DofConstraintValueMap::const_iterator adj_rhsit =
                  constraint_map.find(pushed_id);

                rhs[q] =
                  (adj_rhsit == constraint_map.end()) ?
                  0 : adj_rhsit->second;
              }
          }
      }

    auto ids_action_functor =
      [& received_id_vecs]
      (processor_id_type pid,
       const std::vector<dof_id_type> & data)
      {
        received_id_vecs[pid] = data;
      };

    Parallel::push_parallel_vector_data
      (this->comm(), pushed_id_vecs, ids_action_functor);

    auto keys_vals_action_functor =
      [& received_keys_vals]
      (processor_id_type pid,
       const std::vector<std::vector<std::pair<dof_id_type,Real>>> & data)
      {
        received_keys_vals[pid] = data;
      };

    Parallel::push_parallel_vector_data
      (this->comm(), pushed_keys_vals, keys_vals_action_functor);

    auto rhss_action_functor =
      [& received_rhss]
      (processor_id_type pid,
       const std::vector<std::vector<Number>> & data)
      {
        received_rhss[pid] = data;
      };

    Parallel::push_parallel_vector_data
      (this->comm(), pushed_rhss, rhss_action_functor);

    // Now we have all the DofConstraint rows and rhs values received
    // from others, so add the DoF constraints that we've been sent

#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
    std::map<processor_id_type, std::vector<dof_id_type>>
      pushed_node_id_vecs, received_node_id_vecs;
    for (auto & p : pushed_node_ids)
      pushed_node_id_vecs[p.first].assign(p.second.begin(), p.second.end());

    std::map<processor_id_type, std::vector<std::vector<std::pair<dof_id_type,Real>>>>
      pushed_node_keys_vals, received_node_keys_vals;
    std::map<processor_id_type, std::vector<Point>> pushed_offsets, received_offsets;

    for (auto & p : pushed_node_id_vecs)
      {
        auto & node_keys_vals = pushed_node_keys_vals[p.first];
        node_keys_vals.reserve(p.second.size());

        auto & offsets = pushed_offsets[p.first];
        offsets.reserve(p.second.size());

        for (auto & pushed_node_id : p.second)
          {
            const Node * node = mesh.node_ptr(pushed_node_id);
            NodeConstraintRow & row = _node_constraints[node].first;
            const std::size_t row_size = row.size();
            node_keys_vals.push_back
              (std::vector<std::pair<dof_id_type,Real>>());
            std::vector<std::pair<dof_id_type,Real>> & this_node_kv =
              node_keys_vals.back();
            this_node_kv.reserve(row_size);
            for (const auto & j : row)
              this_node_kv.push_back
                (std::make_pair(j.first->id(), j.second));

            offsets.push_back(_node_constraints[node].second);
          }
      }

    auto node_ids_action_functor =
      [& received_node_id_vecs]
      (processor_id_type pid,
       const std::vector<dof_id_type> & data)
      {
        received_node_id_vecs[pid] = data;
      };

    Parallel::push_parallel_vector_data
      (this->comm(), pushed_node_id_vecs, node_ids_action_functor);

    auto node_keys_vals_action_functor =
      [& received_node_keys_vals]
      (processor_id_type pid,
       const std::vector<std::vector<std::pair<dof_id_type,Real>>> & data)
      {
        received_node_keys_vals[pid] = data;
      };

    Parallel::push_parallel_vector_data
      (this->comm(), pushed_node_keys_vals,
       node_keys_vals_action_functor);

    auto node_offsets_action_functor =
      [& received_offsets]
      (processor_id_type pid,
       const std::vector<Point> & data)
      {
        received_offsets[pid] = data;
      };

    Parallel::push_parallel_vector_data
      (this->comm(), pushed_offsets, node_offsets_action_functor);

#endif

    // Add all the dof constraints that I've been sent
    for (auto & p : received_id_vecs)
      {
        const processor_id_type pid = p.first;
        const auto & pushed_ids_to_me = p.second;
        libmesh_assert(received_keys_vals.count(pid));
        libmesh_assert(received_rhss.count(pid));
        const auto & pushed_keys_vals_to_me = received_keys_vals.at(pid);
        const auto & pushed_rhss_to_me = received_rhss.at(pid);

        libmesh_assert_equal_to (pushed_ids_to_me.size(),
                                 pushed_keys_vals_to_me.size());
        libmesh_assert_equal_to (pushed_ids_to_me.size(),
                                 pushed_rhss_to_me.size());

        for (auto i : index_range(pushed_ids_to_me))
          {
            dof_id_type constrained = pushed_ids_to_me[i];

            // If we don't already have a constraint for this dof,
            // add the one we were sent
            if (!this->is_constrained_dof(constrained))
              {
                DofConstraintRow & row = _dof_constraints[constrained];
                for (auto & kv : pushed_keys_vals_to_me[i])
                  {
                    libmesh_assert_less(kv.first, this->n_dofs());
                    row[kv.first] = kv.second;
                  }

                const Number primal_rhs = pushed_rhss_to_me[i][max_qoi_num];

                if (libmesh_isnan(primal_rhs))
                  libmesh_assert(pushed_keys_vals_to_me[i].empty());

                if (primal_rhs != Number(0))
                  _primal_constraint_values[constrained] = primal_rhs;
                else
                  _primal_constraint_values.erase(constrained);

                for (unsigned int q = 0; q != max_qoi_num; ++q)
                  {
                    AdjointDofConstraintValues::iterator adjoint_map_it =
                      _adjoint_constraint_values.find(q);

                    const Number adj_rhs = pushed_rhss_to_me[i][q];

                    if ((adjoint_map_it == _adjoint_constraint_values.end()) &&
                        adj_rhs == Number(0))
                      continue;

                    if (adjoint_map_it == _adjoint_constraint_values.end())
                      adjoint_map_it = _adjoint_constraint_values.insert
                        (std::make_pair(q,DofConstraintValueMap())).first;

                    DofConstraintValueMap & constraint_map =
                      adjoint_map_it->second;

                    if (adj_rhs != Number(0))
                      constraint_map[constrained] = adj_rhs;
                    else
                      constraint_map.erase(constrained);
                  }
              }
          }
      }

#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
    // Add all the node constraints that I've been sent
    for (auto & p : received_node_id_vecs)
      {
        const processor_id_type pid = p.first;
        const auto & pushed_node_ids_to_me = p.second;
        libmesh_assert(received_node_keys_vals.count(pid));
        libmesh_assert(received_offsets.count(pid));
        const auto & pushed_node_keys_vals_to_me = received_node_keys_vals.at(pid);
        const auto & pushed_offsets_to_me = received_offsets.at(pid);

        libmesh_assert_equal_to (pushed_node_ids_to_me.size(),
                                 pushed_node_keys_vals_to_me.size());
        libmesh_assert_equal_to (pushed_node_ids_to_me.size(),
                                 pushed_offsets_to_me.size());

        for (auto i : index_range(pushed_node_ids_to_me))
          {
            dof_id_type constrained_id = pushed_node_ids_to_me[i];

            // If we don't already have a constraint for this node,
            // add the one we were sent
            const Node * constrained = mesh.node_ptr(constrained_id);
            if (!this->is_constrained_node(constrained))
              {
                NodeConstraintRow & row = _node_constraints[constrained].first;
                for (auto & kv : pushed_node_keys_vals_to_me[i])
                  {
                    const Node * key_node = mesh.node_ptr(kv.first);
                    libmesh_assert(key_node);
                    row[key_node] = kv.second;
                  }
                _node_constraints[constrained].second = pushed_offsets_to_me[i];
              }
          }
      }
#endif // LIBMESH_ENABLE_NODE_CONSTRAINTS
  }

  // Now start checking for any other constraints we need
  // to know about, requesting them recursively.

  // Create sets containing the DOFs and nodes we already depend on
  typedef std::set<dof_id_type> DoF_RCSet;
  DoF_RCSet unexpanded_dofs;

  for (const auto & i : _dof_constraints)
    unexpanded_dofs.insert(i.first);

  // Gather all the dof constraints we need
  this->gather_constraints(mesh, unexpanded_dofs, false);

  // Gather all the node constraints we need
#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
  typedef std::set<const Node *> Node_RCSet;
  Node_RCSet unexpanded_nodes;

  for (const auto & i : _node_constraints)
    unexpanded_nodes.insert(i.first);

  // We have to keep recursing while the unexpanded set is
  // nonempty on *any* processor
  bool unexpanded_set_nonempty = !unexpanded_nodes.empty();
  this->comm().max(unexpanded_set_nonempty);

  // We may be receiving packed_range sends out of order with
  // parallel_sync tags, so make sure they're received correctly.
  Parallel::MessageTag range_tag = this->comm().get_unique_tag();

  while (unexpanded_set_nonempty)
    {
      // Let's make sure we don't lose sync in this loop.
      parallel_object_only();

      // Request sets
      Node_RCSet node_request_set;

      // Request sets to send to each processor
      std::map<processor_id_type, std::vector<dof_id_type>>
        requested_node_ids;

      // And the sizes of each
      std::map<processor_id_type, dof_id_type> node_ids_on_proc;

      // Fill (and thereby sort and uniq!) the main request sets
      for (const auto & i : unexpanded_nodes)
        {
          NodeConstraintRow & row = _node_constraints[i].first;
          for (const auto & j : row)
            {
              const Node * const node = j.first;
              libmesh_assert(node);

              // If it's non-local and we haven't already got a
              // constraint for it, we might need to ask for one
              if ((node->processor_id() != this->processor_id()) &&
                  !_node_constraints.count(node))
                node_request_set.insert(node);
            }
        }

      // Clear the unexpanded constraint sets; we're about to expand
      // them
      unexpanded_nodes.clear();

      // Count requests by processor
      for (const auto & node : node_request_set)
        {
          libmesh_assert(node);
          libmesh_assert_less (node->processor_id(), this->n_processors());
          node_ids_on_proc[node->processor_id()]++;
        }

      for (auto pair : node_ids_on_proc)
        requested_node_ids[pair.first].reserve(pair.second);

      // Prepare each processor's request set
      for (const auto & node : node_request_set)
        requested_node_ids[node->processor_id()].push_back(node->id());

      typedef std::vector<std::pair<dof_id_type, Real>> row_datum;

      // We may need to send nodes ahead of data about them
      std::vector<Parallel::Request> packed_range_sends;

      auto node_row_gather_functor =
        [this,
         & mesh,
         & packed_range_sends,
         & range_tag]
        (processor_id_type pid,
         const std::vector<dof_id_type> & ids,
         std::vector<row_datum> & data)
        {
          // Do we need to keep track of requested nodes to send
          // later?
          const bool dist_mesh = !mesh.is_serial();

          // FIXME - this could be an unordered set, given a
          // hash<pointers> specialization
          std::set<const Node *> nodes_requested;

          // Fill those requests
          const std::size_t query_size = ids.size();

          data.resize(query_size);
          for (std::size_t i=0; i != query_size; ++i)
            {
              dof_id_type constrained_id = ids[i];
              const Node * constrained_node = mesh.node_ptr(constrained_id);
              if (_node_constraints.count(constrained_node))
                {
                  const NodeConstraintRow & row = _node_constraints[constrained_node].first;
                  std::size_t row_size = row.size();
                  data[i].reserve(row_size);
                  for (const auto & j : row)
                    {
                      const Node * node = j.first;
                      data[i].push_back(std::make_pair(node->id(), j.second));

                      // If we're not sure whether our send
                      // destination already has this node, let's give
                      // it a copy.
                      if (node->processor_id() != pid && dist_mesh)
                        nodes_requested.insert(node);

                      // We can have 0 nodal constraint
                      // coefficients, where no Lagrange constraint
                      // exists but non-Lagrange basis constraints
                      // might.
                      // libmesh_assert(j.second);
                    }
                }
              else
                {
                  // We have to distinguish "constraint with no
                  // constraining nodes" (e.g. due to user node
                  // constraint equations) from "no constraint".
                  // We'll use invalid_id for the latter.
                  data[i].push_back
                    (std::make_pair(DofObject::invalid_id, Real(0)));
                }
            }

          // Constraining nodes might not even exist on our
          // correspondant's subset of a distributed mesh, so let's
          // make them exist.
          if (dist_mesh)
            {
              packed_range_sends.push_back(Parallel::Request());
              this->comm().send_packed_range
                (pid, &mesh, nodes_requested.begin(), nodes_requested.end(),
                 packed_range_sends.back(), range_tag);
            }
        };

      typedef Point node_rhs_datum;

      auto node_rhs_gather_functor =
        [this,
         & mesh]
        (processor_id_type,
         const std::vector<dof_id_type> & ids,
         std::vector<node_rhs_datum> & data)
        {
          // Fill those requests
          const std::size_t query_size = ids.size();

          data.resize(query_size);
          for (std::size_t i=0; i != query_size; ++i)
            {
              dof_id_type constrained_id = ids[i];
              const Node * constrained_node = mesh.node_ptr(constrained_id);
              if (_node_constraints.count(constrained_node))
                data[i] = _node_constraints[constrained_node].second;
              else
                data[i](0) = std::numeric_limits<Real>::quiet_NaN();
            }
        };

      auto node_row_action_functor =
        [this,
         & mesh,
         & range_tag,
         & unexpanded_nodes]
        (processor_id_type pid,
         const std::vector<dof_id_type> & ids,
         const std::vector<row_datum> & data)
        {
          // Before we act on any new constraint rows, we may need to
          // make sure we have all the nodes involved!
          if (!mesh.is_serial())
            this->comm().receive_packed_range
              (pid, &mesh, mesh_inserter_iterator<Node>(mesh),
               (Node**)nullptr, range_tag);

          // Add any new constraint rows we've found
          const std::size_t query_size = ids.size();

          for (std::size_t i=0; i != query_size; ++i)
            {
              const dof_id_type constrained_id = ids[i];

              // An empty row is an constraint with an empty row; for
              // no constraint we use a "no row" placeholder
              if (data[i].empty())
                {
                  const Node * constrained_node = mesh.node_ptr(constrained_id);
                  NodeConstraintRow & row = _node_constraints[constrained_node].first;
                  row.clear();
                }
              else if (data[i][0].first != DofObject::invalid_id)
                {
                  const Node * constrained_node = mesh.node_ptr(constrained_id);
                  NodeConstraintRow & row = _node_constraints[constrained_node].first;
                  row.clear();
                  for (auto & pair : data[i])
                    {
                      const Node * key_node =
                        mesh.node_ptr(pair.first);
                      libmesh_assert(key_node);
                      row[key_node] = pair.second;
                    }

                  // And prepare to check for more recursive constraints
                  unexpanded_nodes.insert(constrained_node);
                }
            }
        };

      auto node_rhs_action_functor =
        [this,
         & mesh]
        (processor_id_type,
         const std::vector<dof_id_type> & ids,
         const std::vector<node_rhs_datum> & data)
        {
          // Add rhs data for any new node constraint rows we've found
          const std::size_t query_size = ids.size();

          for (std::size_t i=0; i != query_size; ++i)
            {
              dof_id_type constrained_id = ids[i];
              const Node * constrained_node = mesh.node_ptr(constrained_id);

              if (!libmesh_isnan(data[i](0)))
                _node_constraints[constrained_node].second = data[i];
              else
                _node_constraints.erase(constrained_node);
            }
        };

      // Now request node constraint rows from other processors
      row_datum * node_row_ex = nullptr;
      Parallel::pull_parallel_vector_data
        (this->comm(), requested_node_ids, node_row_gather_functor,
         node_row_action_functor, node_row_ex);

      // And request node constraint right hand sides from other procesors
      node_rhs_datum * node_rhs_ex = nullptr;
      Parallel::pull_parallel_vector_data
        (this->comm(), requested_node_ids, node_rhs_gather_functor,
         node_rhs_action_functor, node_rhs_ex);

      Parallel::wait(packed_range_sends);

      // We have to keep recursing while the unexpanded set is
      // nonempty on *any* processor
      unexpanded_set_nonempty = !unexpanded_nodes.empty();
      this->comm().max(unexpanded_set_nonempty);
    }
#endif // LIBMESH_ENABLE_NODE_CONSTRAINTS
}

template <typename RealType>
void DofMap::local_variable_indices(std::vector<dof_id_type> & idx,
                                    const MeshBaseTempl<RealType> & mesh,
                                    unsigned int var_num) const
{
  // Count dofs in the *exact* order that distribute_dofs numbered
  // them, so that we can assume ascending indices and use push_back
  // instead of find+insert.

  const unsigned int sys_num       = this->sys_number();

  // If this isn't a SCALAR variable, we need to find all its field
  // dofs on the mesh
  if (this->variable_type(var_num).family != SCALAR)
    {
      const Variable & var(this->variable(var_num));

      for (auto & elem : mesh.active_local_element_ptr_range())
        {
          if (!var.active_on_subdomain(elem->subdomain_id()))
            continue;

          // Only count dofs connected to active
          // elements on this processor.
          const unsigned int n_nodes = elem->n_nodes();

          // First get any new nodal DOFS
          for (unsigned int n=0; n<n_nodes; n++)
            {
              Node & node = elem->node_ref(n);

              if (node.processor_id() != this->processor_id())
                continue;

              const unsigned int n_comp = node.n_comp(sys_num, var_num);
              for(unsigned int i=0; i<n_comp; i++)
                {
                  const dof_id_type index = node.dof_number(sys_num,var_num,i);
                  libmesh_assert (this->local_index(index));

                  if (idx.empty() || index > idx.back())
                    idx.push_back(index);
                }
            }

          // Next get any new element DOFS
          const unsigned int n_comp = elem->n_comp(sys_num, var_num);
          for (unsigned int i=0; i<n_comp; i++)
            {
              const dof_id_type index = elem->dof_number(sys_num,var_num,i);
              if (idx.empty() || index > idx.back())
                idx.push_back(index);
            }
        } // done looping over elements


      // we may have missed assigning DOFs to nodes that we own
      // but to which we have no connected elements matching our
      // variable restriction criterion.  this will happen, for example,
      // if variable V is restricted to subdomain S.  We may not own
      // any elements which live in S, but we may own nodes which are
      // *connected* to elements which do.  in this scenario these nodes
      // will presently have unnumbered DOFs. we need to take care of
      // them here since we own them and no other processor will touch them.
      for (const auto & node : mesh.local_node_ptr_range())
        {
          libmesh_assert(node);

          const unsigned int n_comp = node->n_comp(sys_num, var_num);
          for (unsigned int i=0; i<n_comp; i++)
            {
              const dof_id_type index = node->dof_number(sys_num,var_num,i);
              if (idx.empty() || index > idx.back())
                idx.push_back(index);
            }
        }
    }
  // Otherwise, count up the SCALAR dofs, if we're on the processor
  // that holds this SCALAR variable
  else if (this->processor_id() == (this->n_processors()-1))
    {
      std::vector<dof_id_type> di_scalar;
      this->SCALAR_dof_indices(di_scalar,var_num);
      idx.insert( idx.end(), di_scalar.begin(), di_scalar.end());
    }
}

template <typename RealType>
void DofMap::reinit_send_list (MeshBaseTempl<RealType> & mesh)
{
  this->clear_send_list();
  this->add_neighbors_to_send_list(mesh);

#ifdef LIBMESH_ENABLE_CONSTRAINTS
  // This is assuming that we only need to recommunicate
  // the constraints and no new ones have been added since
  // a previous call to reinit_constraints.
  this->process_constraints(mesh);
#endif
  this->prepare_send_list();
}

template <typename RealType>
void DofMap::compute_sparsity(const MeshBaseTempl<RealType> & mesh)
{
  _sp = this->build_sparsity(mesh);

  // It is possible that some \p SparseMatrix implementations want to
  // see the sparsity pattern before we throw it away.  If so, we
  // share a view of its arrays, and we pass it in to the matrices.
  if (need_full_sparsity_pattern)
    {
      _n_nz = &_sp->n_nz;
      _n_oz = &_sp->n_oz;

      for (const auto & mat : _matrices)
        mat->update_sparsity_pattern (_sp->sparsity_pattern);
    }
  // If we don't need the full sparsity pattern anymore, steal the
  // arrays we do need and free the rest of the memory
  else
    {
      if (!_n_nz)
        _n_nz = new std::vector<dof_id_type>();
      _n_nz->swap(_sp->n_nz);
      if (!_n_oz)
        _n_oz = new std::vector<dof_id_type>();
      _n_oz->swap(_sp->n_oz);

      _sp.reset();
    }
}

template <typename RealType>
void DofMap::distribute_dofs (MeshBaseTempl<RealType> & mesh)
{
  // This function must be run on all processors at once
  parallel_object_only();

  // Log how long it takes to distribute the degrees of freedom
  LOG_SCOPE("distribute_dofs()", "DofMap");

  libmesh_assert (mesh.is_prepared());

  const processor_id_type proc_id = this->processor_id();
  const processor_id_type n_proc  = this->n_processors();

  //  libmesh_assert_greater (this->n_variables(), 0);
  libmesh_assert_less (proc_id, n_proc);

  // re-init in case the mesh has changed
  this->reinit(mesh);

  // By default distribute variables in a
  // var-major fashion, but allow run-time
  // specification
  bool node_major_dofs = libMesh::on_command_line ("--node-major-dofs");

  // The DOF counter, will be incremented as we encounter
  // new degrees of freedom
  dof_id_type next_free_dof = 0;

  // Clear the send list before we rebuild it
  this->clear_send_list();

  // Set temporary DOF indices on this processor
  if (node_major_dofs)
    this->distribute_local_dofs_node_major (next_free_dof, mesh);
  else
    this->distribute_local_dofs_var_major (next_free_dof, mesh);

  // Get DOF counts on all processors
  std::vector<dof_id_type> dofs_on_proc(n_proc, 0);
  this->comm().allgather(next_free_dof, dofs_on_proc);

  // Resize and fill the _first_df and _end_df arrays
#ifdef LIBMESH_ENABLE_AMR
  _first_old_df = _first_df;
  _end_old_df = _end_df;
#endif

  _first_df.resize(n_proc);
  _end_df.resize (n_proc);

  // Get DOF offsets
  _first_df[0] = 0;
  for (processor_id_type i=1; i < n_proc; ++i)
    _first_df[i] = _end_df[i-1] = _first_df[i-1] + dofs_on_proc[i-1];
  _end_df[n_proc-1] = _first_df[n_proc-1] + dofs_on_proc[n_proc-1];

  // Clear all the current DOF indices
  // (distribute_dofs expects them cleared!)
  this->invalidate_dofs(mesh);

  next_free_dof = _first_df[proc_id];

  // Set permanent DOF indices on this processor
  if (node_major_dofs)
    this->distribute_local_dofs_node_major (next_free_dof, mesh);
  else
    this->distribute_local_dofs_var_major (next_free_dof, mesh);

  libmesh_assert_equal_to (next_free_dof, _end_df[proc_id]);

  //------------------------------------------------------------
  // At this point, all n_comp and dof_number values on local
  // DofObjects should be correct, but a DistributedMesh might have
  // incorrect values on non-local DofObjects.  Let's request the
  // correct values from each other processor.

  if (this->n_processors() > 1)
    {
      this->set_nonlocal_dof_objects(mesh.nodes_begin(),
                                     mesh.nodes_end(),
                                     mesh, &DofMap::node_ptr);

      this->set_nonlocal_dof_objects(mesh.elements_begin(),
                                     mesh.elements_end(),
                                     mesh, &DofMap::elem_ptr);
    }

#ifdef DEBUG
  {
    const unsigned int
      sys_num = this->sys_number();

    // Processors should all agree on DoF ids for the newly numbered
    // system.
    MeshTools::libmesh_assert_valid_dof_ids(mesh, sys_num);

    // DoF processor ids should match DofObject processor ids
    for (auto & node : mesh.node_ptr_range())
      {
        DofObject const * const dofobj = node;
        const processor_id_type obj_proc_id = dofobj->processor_id();

        for (auto v : IntRange<unsigned int>(0, dofobj->n_vars(sys_num)))
          for (auto c : IntRange<unsigned int>(0, dofobj->n_comp(sys_num,v)))
            {
              const dof_id_type dofid = dofobj->dof_number(sys_num,v,c);
              libmesh_assert_greater_equal (dofid, this->first_dof(obj_proc_id));
              libmesh_assert_less (dofid, this->end_dof(obj_proc_id));
            }
      }

    for (auto & elem : mesh.element_ptr_range())
      {
        DofObject const * const dofobj = elem;
        const processor_id_type obj_proc_id = dofobj->processor_id();

        for (auto v : IntRange<unsigned int>(0, dofobj->n_vars(sys_num)))
          for (auto c : IntRange<unsigned int>(0, dofobj->n_comp(sys_num,v)))
            {
              const dof_id_type dofid = dofobj->dof_number(sys_num,v,c);
              libmesh_assert_greater_equal (dofid, this->first_dof(obj_proc_id));
              libmesh_assert_less (dofid, this->end_dof(obj_proc_id));
            }
      }
  }
#endif

  // Set the total number of degrees of freedom, then start finding
  // SCALAR degrees of freedom
#ifdef LIBMESH_ENABLE_AMR
  _n_old_dfs = _n_dfs;
  _first_old_scalar_df = _first_scalar_df;
#endif
  _n_dfs = _end_df[n_proc-1];
  _first_scalar_df.clear();
  _first_scalar_df.resize(this->n_variables(), DofObject::invalid_id);
  dof_id_type current_SCALAR_dof_index = n_dofs() - n_SCALAR_dofs();

  // Calculate and cache the initial DoF indices for SCALAR variables.
  // This is an O(N_vars) calculation so we want to do it once per
  // renumbering rather than once per SCALAR_dof_indices() call

  for (auto v : IntRange<unsigned int>(0, this->n_variables()))
    if (this->variable(v).type().family == SCALAR)
      {
        _first_scalar_df[v] = current_SCALAR_dof_index;
        current_SCALAR_dof_index += this->variable(v).type().order.get_order();
      }

  // Allow our GhostingFunctor objects to reinit if necessary
  for (const auto & gf : _algebraic_ghosting_functors)
    {
      libmesh_assert(gf);
      gf->dofmap_reinit();
    }

  for (const auto & gf : _coupling_functors)
    {
      libmesh_assert(gf);
      gf->dofmap_reinit();
    }

  // Note that in the add_neighbors_to_send_list nodes on processor
  // boundaries that are shared by multiple elements are added for
  // each element.
  this->add_neighbors_to_send_list(mesh);

  // Here we used to clean up that data structure; now System and
  // EquationSystems call that for us, after we've added constraint
  // dependencies to the send_list too.
  // this->sort_send_list ();
}

template <typename RealType>
void DofMap::scatter_constraints(MeshBaseTempl<RealType> & mesh)
{
  // At this point each processor with a constrained node knows
  // the corresponding constraint row, but we also need each processor
  // with a constrainer node to know the corresponding row(s).

  // This function must be run on all processors at once
  parallel_object_only();

  // Return immediately if there's nothing to gather
  if (this->n_processors() == 1)
    return;

  // We might get to return immediately if none of the processors
  // found any constraints
  unsigned int has_constraints = !_dof_constraints.empty()
#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
    || !_node_constraints.empty()
#endif // LIBMESH_ENABLE_NODE_CONSTRAINTS
    ;
  this->comm().max(has_constraints);
  if (!has_constraints)
    return;

  // We may be receiving packed_range sends out of order with
  // parallel_sync tags, so make sure they're received correctly.
  Parallel::MessageTag range_tag = this->comm().get_unique_tag();

#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
  std::map<processor_id_type, std::set<dof_id_type>> pushed_node_ids;
#endif // LIBMESH_ENABLE_NODE_CONSTRAINTS

  std::map<processor_id_type, std::set<dof_id_type>> pushed_ids;

  // Collect the dof constraints I need to push to each processor
  dof_id_type constrained_proc_id = 0;
  for (auto & i : _dof_constraints)
    {
      const dof_id_type constrained = i.first;
      while (constrained >= _end_df[constrained_proc_id])
        constrained_proc_id++;

      if (constrained_proc_id != this->processor_id())
        continue;

      DofConstraintRow & row = i.second;
      for (auto & j : row)
        {
          const dof_id_type constraining = j.first;

          processor_id_type constraining_proc_id = 0;
          while (constraining >= _end_df[constraining_proc_id])
            constraining_proc_id++;

          if (constraining_proc_id != this->processor_id() &&
              constraining_proc_id != constrained_proc_id)
            pushed_ids[constraining_proc_id].insert(constrained);
        }
    }

  // Pack the dof constraint rows and rhs's to push

  std::map<processor_id_type,
          std::vector<std::vector<std::pair<dof_id_type, Real>>>>
    pushed_keys_vals, pushed_keys_vals_to_me;

  std::map<processor_id_type, std::vector<std::pair<dof_id_type, Number>>>
    pushed_ids_rhss, pushed_ids_rhss_to_me;

  auto gather_ids =
    [this,
     & pushed_ids,
     & pushed_keys_vals,
     & pushed_ids_rhss]
    ()
    {
      for (auto & pid_id_pair : pushed_ids)
        {
          const processor_id_type pid = pid_id_pair.first;
          const std::set<dof_id_type>
            & pid_ids = pid_id_pair.second;

          const std::size_t ids_size = pid_ids.size();
          std::vector<std::vector<std::pair<dof_id_type, Real>>> &
            keys_vals = pushed_keys_vals[pid];
          std::vector<std::pair<dof_id_type,Number>> &
            ids_rhss = pushed_ids_rhss[pid];
          keys_vals.resize(ids_size);
          ids_rhss.resize(ids_size);

          std::size_t push_i;
          std::set<dof_id_type>::const_iterator it;
          for (push_i = 0, it = pid_ids.begin();
               it != pid_ids.end(); ++push_i, ++it)
            {
              const dof_id_type constrained = *it;
              DofConstraintRow & row = _dof_constraints[constrained];
              keys_vals[push_i].assign(row.begin(), row.end());

              DofConstraintValueMap::const_iterator rhsit =
                _primal_constraint_values.find(constrained);
              ids_rhss[push_i].first = constrained;
              ids_rhss[push_i].second =
                (rhsit == _primal_constraint_values.end()) ?
                0 : rhsit->second;
            }
        }
    };

  gather_ids();

  auto ids_rhss_action_functor =
    [& pushed_ids_rhss_to_me]
    (processor_id_type pid,
     const std::vector<std::pair<dof_id_type, Number>> & data)
    {
      pushed_ids_rhss_to_me[pid] = data;
    };

  auto keys_vals_action_functor =
    [& pushed_keys_vals_to_me]
    (processor_id_type pid,
     const std::vector<std::vector<std::pair<dof_id_type, Real>>> & data)
    {
      pushed_keys_vals_to_me[pid] = data;
    };

  Parallel::push_parallel_vector_data
    (this->comm(), pushed_ids_rhss, ids_rhss_action_functor);
  Parallel::push_parallel_vector_data
    (this->comm(), pushed_keys_vals, keys_vals_action_functor);

  // Now work on traded dof constraint rows
  auto receive_dof_constraints =
    [this,
     & pushed_ids_rhss_to_me,
     & pushed_keys_vals_to_me]
    ()
    {
      for (auto & pid_id_pair : pushed_ids_rhss_to_me)
        {
          const processor_id_type pid = pid_id_pair.first;
          const auto & ids_rhss = pid_id_pair.second;
          const auto & keys_vals = pushed_keys_vals_to_me[pid];

          libmesh_assert_equal_to
            (ids_rhss.size(), keys_vals.size());

          // Add the dof constraints that I've been sent
          for (auto i : index_range(ids_rhss))
            {
              dof_id_type constrained = ids_rhss[i].first;

              // If we don't already have a constraint for this dof,
              // add the one we were sent
              if (!this->is_constrained_dof(constrained))
                {
                  DofConstraintRow & row = _dof_constraints[constrained];
                  for (auto & key_val : keys_vals[i])
                    {
                      libmesh_assert_less(key_val.first, this->n_dofs());
                      row[key_val.first] = key_val.second;
                    }
                  if (ids_rhss[i].second != Number(0))
                    _primal_constraint_values[constrained] =
                      ids_rhss[i].second;
                  else
                    _primal_constraint_values.erase(constrained);
                }
            }
        }
    };

  receive_dof_constraints();

#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
  // Collect the node constraints to push to each processor
  for (auto & i : _node_constraints)
    {
      const Node * constrained = i.first;

      if (constrained->processor_id() != this->processor_id())
        continue;

      NodeConstraintRow & row = i.second.first;
      for (auto & j : row)
        {
          const Node * constraining = j.first;

          if (constraining->processor_id() != this->processor_id() &&
              constraining->processor_id() != constrained->processor_id())
            pushed_node_ids[constraining->processor_id()].insert(constrained->id());
        }
    }

  // Pack the node constraint rows and rhss to push
  std::map<processor_id_type,
          std::vector<std::vector<std::pair<dof_id_type,Real>>>>
    pushed_node_keys_vals, pushed_node_keys_vals_to_me;
  std::map<processor_id_type, std::vector<std::pair<dof_id_type, Point>>>
    pushed_node_ids_offsets, pushed_node_ids_offsets_to_me;
  std::map<processor_id_type, std::set<const Node *>> pushed_nodes;

  for (auto & pid_id_pair : pushed_node_ids)
    {
      const processor_id_type pid = pid_id_pair.first;
      const std::set<dof_id_type>
        & pid_ids = pid_id_pair.second;

      const std::size_t ids_size = pid_ids.size();
      std::vector<std::vector<std::pair<dof_id_type,Real>>> &
        keys_vals = pushed_node_keys_vals[pid];
      std::vector<std::pair<dof_id_type, Point>> &
        ids_offsets = pushed_node_ids_offsets[pid];
      keys_vals.resize(ids_size);
      ids_offsets.resize(ids_size);
      std::set<const Node *> & nodes = pushed_nodes[pid];

      std::size_t push_i;
      std::set<dof_id_type>::const_iterator it;
      for (push_i = 0, it = pid_ids.begin();
           it != pid_ids.end(); ++push_i, ++it)
        {
          const Node * constrained = mesh.node_ptr(*it);

          if (constrained->processor_id() != pid)
            nodes.insert(constrained);

          NodeConstraintRow & row = _node_constraints[constrained].first;
          std::size_t row_size = row.size();
          keys_vals[push_i].reserve(row_size);
          for (const auto & j : row)
            {
              const Node * constraining = j.first;

              keys_vals[push_i].push_back
                (std::make_pair(constraining->id(), j.second));

              if (constraining->processor_id() != pid)
                nodes.insert(constraining);
            }

          ids_offsets[push_i].first = *it;
          ids_offsets[push_i].second = _node_constraints[constrained].second;
        }
    }

  auto node_ids_offsets_action_functor =
    [& pushed_node_ids_offsets_to_me]
    (processor_id_type pid,
     const std::vector<std::pair<dof_id_type, Point>> & data)
    {
      pushed_node_ids_offsets_to_me[pid] = data;
    };

  auto node_keys_vals_action_functor =
    [& pushed_node_keys_vals_to_me]
    (processor_id_type pid,
     const std::vector<std::vector<std::pair<dof_id_type, Real>>> & data)
    {
      pushed_node_keys_vals_to_me[pid] = data;
    };

  // Trade pushed node constraint rows
  Parallel::push_parallel_vector_data
    (this->comm(), pushed_node_ids_offsets, node_ids_offsets_action_functor);
  Parallel::push_parallel_vector_data
    (this->comm(), pushed_node_keys_vals, node_keys_vals_action_functor);

  // Constraining nodes might not even exist on our subset of a
  // distributed mesh, so let's make them exist.
  std::vector<Parallel::Request> send_requests;
  if (!mesh.is_serial())
    {
      for (auto & pid_id_pair : pushed_node_ids_offsets)
        {
          const processor_id_type pid = pid_id_pair.first;
          send_requests.push_back(Parallel::Request());
          this->comm().send_packed_range
            (pid, &mesh,
             pushed_nodes[pid].begin(), pushed_nodes[pid].end(),
             send_requests.back(), range_tag);
        }
    }

  for (auto & pid_id_pair : pushed_node_ids_offsets_to_me)
    {
      const processor_id_type pid = pid_id_pair.first;
      const auto & ids_offsets = pid_id_pair.second;
      const auto & keys_vals = pushed_node_keys_vals_to_me[pid];

      libmesh_assert_equal_to
        (ids_offsets.size(), keys_vals.size());

      if (!mesh.is_serial())
        this->comm().receive_packed_range
          (pid, &mesh, mesh_inserter_iterator<Node>(mesh),
           (Node**)nullptr, range_tag);

      // Add the node constraints that I've been sent
      for (auto i : index_range(ids_offsets))
        {
          dof_id_type constrained_id = ids_offsets[i].first;

          // If we don't already have a constraint for this node,
          // add the one we were sent
          const Node * constrained = mesh.node_ptr(constrained_id);
          if (!this->is_constrained_node(constrained))
            {
              NodeConstraintRow & row = _node_constraints[constrained].first;
              for (auto & key_val : keys_vals[i])
                {
                  const Node * key_node = mesh.node_ptr(key_val.first);
                  row[key_node] = key_val.second;
                }
              _node_constraints[constrained].second =
                ids_offsets[i].second;
            }
        }
    }

  Parallel::wait(send_requests);

#endif // LIBMESH_ENABLE_NODE_CONSTRAINTS

  // Next we need to push constraints to processors which don't own
  // the constrained dof, don't own the constraining dof, but own an
  // element supporting the constraining dof.
  //
  // We need to be able to quickly look up constrained dof ids by what
  // constrains them, so that we can handle the case where we see a
  // foreign element containing one of our constraining DoF ids and we
  // need to push that constraint.
  //
  // Getting distributed adaptive sparsity patterns right is hard.

  typedef std::map<dof_id_type, std::set<dof_id_type>> DofConstrainsMap;
  DofConstrainsMap dof_id_constrains;

  for (auto & i : _dof_constraints)
    {
      const dof_id_type constrained = i.first;
      DofConstraintRow & row = i.second;
      for (const auto & j : row)
        {
          const dof_id_type constraining = j.first;

          dof_id_type constraining_proc_id = 0;
          while (constraining >= _end_df[constraining_proc_id])
            constraining_proc_id++;

          if (constraining_proc_id == this->processor_id())
            dof_id_constrains[constraining].insert(constrained);
        }
    }

  // Loop over all foreign elements, find any supporting our
  // constrained dof indices.
  pushed_ids.clear();

  for (const auto & elem : as_range(mesh.active_not_local_elements_begin(),
                                    mesh.active_not_local_elements_end()))
    {
      std::vector<dof_id_type> my_dof_indices;
      this->dof_indices (elem, my_dof_indices);

      for (const auto & dof : my_dof_indices)
        {
          DofConstrainsMap::const_iterator dcmi = dof_id_constrains.find(dof);
          if (dcmi != dof_id_constrains.end())
            {
              for (const auto & constrained : dcmi->second)
                {
                  dof_id_type the_constrained_proc_id = 0;
                  while (constrained >= _end_df[the_constrained_proc_id])
                    the_constrained_proc_id++;

                  const processor_id_type elemproc = elem->processor_id();
                  if (elemproc != the_constrained_proc_id)
                    pushed_ids[elemproc].insert(constrained);
                }
            }
        }
    }

  pushed_ids_rhss.clear();
  pushed_ids_rhss_to_me.clear();
  pushed_keys_vals.clear();
  pushed_keys_vals_to_me.clear();

  gather_ids();

  // Trade pushed dof constraint rows
  Parallel::push_parallel_vector_data
    (this->comm(), pushed_ids_rhss, ids_rhss_action_functor);
  Parallel::push_parallel_vector_data
    (this->comm(), pushed_keys_vals, keys_vals_action_functor);

  receive_dof_constraints();

  // Finally, we need to handle the case of remote dof coupling.  If a
  // processor's element is coupled to a ghost element, then the
  // processor needs to know about all constraints which affect the
  // dofs on that ghost element, so we'll have to query the ghost
  // element's owner.

  GhostingFunctor::map_type elements_to_couple;

  // Man, I wish we had guaranteed unique_ptr availability...
  std::set<CouplingMatrix*> temporary_coupling_matrices;

  this->merge_ghost_functor_outputs
    (elements_to_couple,
     temporary_coupling_matrices,
     this->coupling_functors_begin(),
     this->coupling_functors_end(),
     mesh.active_local_elements_begin(),
     mesh.active_local_elements_end(),
     this->processor_id());

  // Each ghost-coupled element's owner should get a request for its dofs
  std::set<dof_id_type> requested_dofs;

  for (const auto & pr : elements_to_couple)
    {
      const Elem * elem = pr.first;

      // FIXME - optimize for the non-fully-coupled case?
      std::vector<dof_id_type> element_dofs;
      this->dof_indices(elem, element_dofs);

      for (auto dof : element_dofs)
        requested_dofs.insert(dof);
    }

  this->gather_constraints(mesh, requested_dofs, false);
}

template <typename RealType>
void DofMap::process_constraints (MeshBaseTempl<RealType> & mesh)
{
  // We've computed our local constraints, but they may depend on
  // non-local constraints that we'll need to take into account.
  this->allgather_recursive_constraints(mesh);

  if (_error_on_constraint_loop)
  {
    // Optionally check for constraint loops and throw an error
    // if they're detected. We always do this check below in dbg/devel
    // mode but here we optionally do it in opt mode as well.
    check_for_constraint_loops();
  }

  // Create a set containing the DOFs we already depend on
  typedef std::set<dof_id_type> RCSet;
  RCSet unexpanded_set;

  for (const auto & i : _dof_constraints)
    unexpanded_set.insert(i.first);

  while (!unexpanded_set.empty())
    for (RCSet::iterator i = unexpanded_set.begin();
         i != unexpanded_set.end(); /* nothing */)
      {
        // If the DOF is constrained
        DofConstraints::iterator
          pos = _dof_constraints.find(*i);

        libmesh_assert (pos != _dof_constraints.end());

        DofConstraintRow & constraint_row = pos->second;

        DofConstraintValueMap::iterator rhsit =
          _primal_constraint_values.find(*i);
        Number constraint_rhs = (rhsit == _primal_constraint_values.end()) ?
          0 : rhsit->second;

        std::vector<dof_id_type> constraints_to_expand;

        for (const auto & item : constraint_row)
          if (item.first != *i && this->is_constrained_dof(item.first))
            {
              unexpanded_set.insert(item.first);
              constraints_to_expand.push_back(item.first);
            }

        for (const auto & expandable : constraints_to_expand)
          {
            const Real this_coef = constraint_row[expandable];

            DofConstraints::const_iterator
              subpos = _dof_constraints.find(expandable);

            libmesh_assert (subpos != _dof_constraints.end());

            const DofConstraintRow & subconstraint_row = subpos->second;

            for (const auto & item : subconstraint_row)
              {
                // Assert that the constraint does not form a cycle.
                libmesh_assert(item.first != expandable);
                constraint_row[item.first] += item.second * this_coef;
              }

            DofConstraintValueMap::const_iterator subrhsit =
              _primal_constraint_values.find(expandable);
            if (subrhsit != _primal_constraint_values.end())
              constraint_rhs += subrhsit->second * this_coef;

            constraint_row.erase(expandable);
          }

        if (rhsit == _primal_constraint_values.end())
          {
            if (constraint_rhs != Number(0))
              _primal_constraint_values[*i] = constraint_rhs;
            else
              _primal_constraint_values.erase(*i);
          }
        else
          {
            if (constraint_rhs != Number(0))
              rhsit->second = constraint_rhs;
            else
              _primal_constraint_values.erase(rhsit);
          }

        if (constraints_to_expand.empty())
          i = unexpanded_set.erase(i);
        else
          ++i;
      }

  // In parallel we can't guarantee that nodes/dofs which constrain
  // others are on processors which are aware of that constraint, yet
  // we need such awareness for sparsity pattern generation.  So send
  // other processors any constraints they might need to know about.
  this->scatter_constraints(mesh);

  // Now that we have our root constraint dependencies sorted out, add
  // them to the send_list
  this->add_constraints_to_send_list();
}

template <typename RealType>
void DofMap::reinit(MeshBaseTempl<RealType> & mesh)
{
  libmesh_assert (mesh.is_prepared());

  LOG_SCOPE("reinit()", "DofMap");

  // We ought to reconfigure our default coupling functor.
  //
  // The user might have removed it from our coupling functors set,
  // but if so, who cares, this reconfiguration is cheap.

  // Avoid calling set_dof_coupling() with an empty/non-nullptr
  // _dof_coupling matrix which may happen when there are actually no
  // variables on the system.
  if (this->_dof_coupling && this->_dof_coupling->empty() && !this->n_variables())
    this->_dof_coupling = nullptr;
  _default_coupling->set_dof_coupling(this->_dof_coupling);

  // By default we may want 0 or 1 levels of coupling
  unsigned int standard_n_levels =
    this->use_coupled_neighbor_dofs(mesh);
  _default_coupling->set_n_levels
    (std::max(_default_coupling->n_levels(), standard_n_levels));

  // But we *don't* want to restrict to a CouplingMatrix unless the
  // user does so manually; the original libMesh behavior was to put
  // ghost indices on the send_list regardless of variable.
  //_default_evaluating->set_dof_coupling(this->_dof_coupling);

  const unsigned int
    sys_num      = this->sys_number(),
    n_var_groups = this->n_variable_groups();

  // The DofObjects need to know how many variable groups we have, and
  // how many variables there are in each group.
  std::vector<unsigned int> n_vars_per_group; /**/ n_vars_per_group.reserve (n_var_groups);

  for (unsigned int vg=0; vg<n_var_groups; vg++)
    n_vars_per_group.push_back (this->variable_group(vg).n_variables());

#ifdef LIBMESH_ENABLE_AMR

  //------------------------------------------------------------
  // Clear the old_dof_objects for all the nodes
  // and elements so that we can overwrite them
  for (auto & node : mesh.node_ptr_range())
    {
      node->clear_old_dof_object();
      libmesh_assert (!node->old_dof_object);
    }

  for (auto & elem : mesh.element_ptr_range())
    {
      elem->clear_old_dof_object();
      libmesh_assert (!elem->old_dof_object);
    }


  //------------------------------------------------------------
  // Set the old_dof_objects for the elements that
  // weren't just created, if these old dof objects
  // had variables
  for (auto & elem : mesh.element_ptr_range())
    {
      // Skip the elements that were just refined
      if (elem->refinement_flag() == Elem::JUST_REFINED)
        continue;

      for (Node & node : elem->node_ref_range())
        if (node.old_dof_object == nullptr)
          if (node.has_dofs(sys_num))
            node.set_old_dof_object();

      libmesh_assert (!elem->old_dof_object);

      if (elem->has_dofs(sys_num))
        elem->set_old_dof_object();
    }

#endif // #ifdef LIBMESH_ENABLE_AMR


  //------------------------------------------------------------
  // Then set the number of variables for each \p DofObject
  // equal to n_variables() for this system.  This will
  // handle new \p DofObjects that may have just been created

  // All the nodes
  for (auto & node : mesh.node_ptr_range())
    node->set_n_vars_per_group(sys_num, n_vars_per_group);

  // All the elements
  for (auto & elem : mesh.element_ptr_range())
    elem->set_n_vars_per_group(sys_num, n_vars_per_group);

  // Zero _n_SCALAR_dofs, it will be updated below.
  this->_n_SCALAR_dofs = 0;

  //------------------------------------------------------------
  // Next allocate space for the DOF indices
  for (unsigned int vg=0; vg<n_var_groups; vg++)
    {
      const VariableGroup & vg_description = this->variable_group(vg);

      const unsigned int n_var_in_group = vg_description.n_variables();
      const FEType & base_fe_type        = vg_description.type();

      // Don't need to loop over elements for a SCALAR variable
      // Just increment _n_SCALAR_dofs
      if (base_fe_type.family == SCALAR)
        {
          this->_n_SCALAR_dofs += base_fe_type.order.get_order()*n_var_in_group;
          continue;
        }

      // This should be constant even on p-refined elements
      const bool extra_hanging_dofs =
        FEInterface::extra_hanging_dofs(base_fe_type);

      // For all the active elements, count vertex degrees of freedom.
      for (auto & elem : mesh.active_element_ptr_range())
        {
          libmesh_assert(elem);

          // Skip the numbering if this variable is
          // not active on this element's subdomain
          if (!vg_description.active_on_subdomain(elem->subdomain_id()))
            continue;

          const ElemType type = elem->type();
          const unsigned int dim = elem->dim();

          FEType fe_type = base_fe_type;

#ifdef LIBMESH_ENABLE_AMR
          // Make sure we haven't done more p refinement than we can
          // handle
          if (elem->p_level() + base_fe_type.order >
              FEInterface::max_order(base_fe_type, type))
            {
              libmesh_assert_less_msg(base_fe_type.order.get_order(),
                                      FEInterface::max_order(base_fe_type,type),
                                      "ERROR: Finite element "
                                      << Utility::enum_to_string(base_fe_type.family)
                                      << " on geometric element "
                                      << Utility::enum_to_string(type)
                                      << "\nonly supports FEInterface::max_order = "
                                      << FEInterface::max_order(base_fe_type,type)
                                      << ", not fe_type.order = "
                                      << base_fe_type.order);

#  ifdef DEBUG
              libMesh::err << "WARNING: Finite element "
                           << Utility::enum_to_string(base_fe_type.family)
                           << " on geometric element "
                           << Utility::enum_to_string(type) << std::endl
                           << "could not be p refined past FEInterface::max_order = "
                           << FEInterface::max_order(base_fe_type,type)
                           << std::endl;
#  endif
              elem->set_p_level(FEInterface::max_order(base_fe_type,type)
                                - base_fe_type.order);
            }
#endif

          fe_type.order = static_cast<Order>(fe_type.order +
                                             elem->p_level());

          // Allocate the vertex DOFs
          for (auto n : elem->node_index_range())
            {
              Node & node = elem->node_ref(n);

              if (elem->is_vertex(n))
                {
                  const unsigned int old_node_dofs =
                    node.n_comp_group(sys_num, vg);

                  const unsigned int vertex_dofs =
                    std::max(FEInterface::n_dofs_at_node(dim, fe_type,
                                                         type, n),
                             old_node_dofs);

                  // Some discontinuous FEs have no vertex dofs
                  if (vertex_dofs > old_node_dofs)
                    {
                      node.set_n_comp_group(sys_num, vg,
                                            vertex_dofs);

                      // Abusing dof_number to set a "this is a
                      // vertex" flag
                      node.set_vg_dof_base(sys_num, vg,
                                           vertex_dofs);

                      // libMesh::out << "sys_num,vg,old_node_dofs,vertex_dofs="
                      //       << sys_num << ","
                      //       << vg << ","
                      //       << old_node_dofs << ","
                      //       << vertex_dofs << '\n',
                      // node.debug_buffer();

                      // libmesh_assert_equal_to (vertex_dofs, node.n_comp(sys_num, vg));
                      // libmesh_assert_equal_to (vertex_dofs, node.vg_dof_base(sys_num, vg));
                    }
                }
            }
        } // done counting vertex dofs

      // count edge & face dofs next
      for (auto & elem : mesh.active_element_ptr_range())
        {
          libmesh_assert(elem);

          // Skip the numbering if this variable is
          // not active on this element's subdomain
          if (!vg_description.active_on_subdomain(elem->subdomain_id()))
            continue;

          const ElemType type = elem->type();
          const unsigned int dim = elem->dim();

          FEType fe_type = base_fe_type;
          fe_type.order = static_cast<Order>(fe_type.order +
                                             elem->p_level());

          // Allocate the edge and face DOFs
          for (auto n : elem->node_index_range())
            {
              Node & node = elem->node_ref(n);

              const unsigned int old_node_dofs =
                node.n_comp_group(sys_num, vg);

              const unsigned int vertex_dofs = old_node_dofs?
                cast_int<unsigned int>(node.vg_dof_base (sys_num,vg)):0;

              const unsigned int new_node_dofs =
                FEInterface::n_dofs_at_node(dim, fe_type, type, n);

              // We've already allocated vertex DOFs
              if (elem->is_vertex(n))
                {
                  libmesh_assert_greater_equal (old_node_dofs, vertex_dofs);
                  // //if (vertex_dofs < new_node_dofs)
                  //   libMesh::out << "sys_num,vg,old_node_dofs,vertex_dofs,new_node_dofs="
                  //                << sys_num << ","
                  //                << vg << ","
                  //                << old_node_dofs << ","
                  //                << vertex_dofs << ","
                  //                << new_node_dofs << '\n',
                  //     node.debug_buffer();

                  libmesh_assert_greater_equal (vertex_dofs,   new_node_dofs);
                }
              // We need to allocate the rest
              else
                {
                  // If this has no dofs yet, it needs no vertex
                  // dofs, so we just give it edge or face dofs
                  if (!old_node_dofs)
                    {
                      node.set_n_comp_group(sys_num, vg,
                                            new_node_dofs);
                      // Abusing dof_number to set a "this has no
                      // vertex dofs" flag
                      if (new_node_dofs)
                        node.set_vg_dof_base(sys_num, vg, 0);
                    }

                  // If this has dofs, but has no vertex dofs,
                  // it may still need more edge or face dofs if
                  // we're p-refined.
                  else if (vertex_dofs == 0)
                    {
                      if (new_node_dofs > old_node_dofs)
                        {
                          node.set_n_comp_group(sys_num, vg,
                                                new_node_dofs);

                          node.set_vg_dof_base(sys_num, vg,
                                               vertex_dofs);
                        }
                    }
                  // If this is another element's vertex,
                  // add more (non-overlapping) edge/face dofs if
                  // necessary
                  else if (extra_hanging_dofs)
                    {
                      if (new_node_dofs > old_node_dofs - vertex_dofs)
                        {
                          node.set_n_comp_group(sys_num, vg,
                                                vertex_dofs + new_node_dofs);

                          node.set_vg_dof_base(sys_num, vg,
                                               vertex_dofs);
                        }
                    }
                  // If this is another element's vertex, add any
                  // (overlapping) edge/face dofs if necessary
                  else
                    {
                      libmesh_assert_greater_equal (old_node_dofs, vertex_dofs);
                      if (new_node_dofs > old_node_dofs)
                        {
                          node.set_n_comp_group(sys_num, vg,
                                                new_node_dofs);

                          node.set_vg_dof_base (sys_num, vg,
                                                vertex_dofs);
                        }
                    }
                }
            }
          // Allocate the element DOFs
          const unsigned int dofs_per_elem =
            FEInterface::n_dofs_per_elem(dim, fe_type,
                                         type);

          elem->set_n_comp_group(sys_num, vg, dofs_per_elem);

        }
    } // end loop over variable groups

  // Calling DofMap::reinit() by itself makes little sense,
  // so we won't bother with nonlocal DofObjects.
  // Those will be fixed by distribute_dofs

  //------------------------------------------------------------
  // Finally, clear all the current DOF indices
  // (distribute_dofs expects them cleared!)
  this->invalidate_dofs(mesh);
}

template <typename RealType>
std::unique_ptr<SparsityPattern::Build>
DofMap::build_sparsity (const MeshBaseTempl<RealType> & mesh) const
{
  libmesh_assert (mesh.is_prepared());

  LOG_SCOPE("build_sparsity()", "DofMap");

  // Compute the sparsity structure of the global matrix.  This can be
  // fed into a PetscMatrix to allocate exactly the number of nonzeros
  // necessary to store the matrix.  This algorithm should be linear
  // in the (# of elements)*(# nodes per element)

  // We can be more efficient in the threaded sparsity pattern assembly
  // if we don't need the exact pattern.  For some sparse matrix formats
  // a good upper bound will suffice.

  // See if we need to include sparsity pattern entries for coupling
  // between neighbor dofs
  bool implicit_neighbor_dofs = this->use_coupled_neighbor_dofs(mesh);

  // We can compute the sparsity pattern in parallel on multiple
  // threads.  The goal is for each thread to compute the full sparsity
  // pattern for a subset of elements.  These sparsity patterns can
  // be efficiently merged in the SparsityPattern::Build::join()
  // method, especially if there is not too much overlap between them.
  // Even better, if the full sparsity pattern is not needed then
  // the number of nonzeros per row can be estimated from the
  // sparsity patterns created on each thread.
  auto sp = libmesh_make_unique<SparsityPattern::Build>
    (mesh,
     *this,
     this->_dof_coupling,
     this->_coupling_functors,
     implicit_neighbor_dofs,
     need_full_sparsity_pattern);

  Threads::parallel_reduce (ConstElemRange (mesh.active_local_elements_begin(),
                                            mesh.active_local_elements_end()), *sp);

  sp->parallel_sync();

#ifndef NDEBUG
  // Avoid declaring these variables unless asserts are enabled.
  const processor_id_type proc_id        = mesh.processor_id();
  const dof_id_type n_dofs_on_proc = this->n_dofs_on_processor(proc_id);
#endif
  libmesh_assert_equal_to (sp->sparsity_pattern.size(), n_dofs_on_proc);

  // Check to see if we have any extra stuff to add to the sparsity_pattern
  if (_extra_sparsity_function)
    {
      if (_augment_sparsity_pattern)
        {
          libmesh_here();
          libMesh::out << "WARNING:  You have specified both an extra sparsity function and object.\n"
                       << "          Are you sure this is what you meant to do??"
                       << std::endl;
        }

      _extra_sparsity_function
        (sp->sparsity_pattern, sp->n_nz,
         sp->n_oz, _extra_sparsity_context);
    }

  if (_augment_sparsity_pattern)
    _augment_sparsity_pattern->augment_sparsity_pattern
      (sp->sparsity_pattern, sp->n_nz, sp->n_oz);

  return std::unique_ptr<SparsityPattern::Build>(sp.release());
}

template <typename RealType>
void DofMap::invalidate_dofs(MeshBaseTempl<RealType> & mesh) const
{
  const unsigned int sys_num = this->sys_number();

  // All the nodes
  for (auto & node : mesh.node_ptr_range())
    node->invalidate_dofs(sys_num);

  // All the active elements.
  for (auto & elem : mesh.active_element_ptr_range())
    elem->invalidate_dofs(sys_num);
}

template <typename RealType>
DofObject * DofMap::node_ptr(MeshBaseTempl<RealType> & mesh, dof_id_type i) const
{
  return mesh.node_ptr(i);
}

template <typename RealType>
DofObject * DofMap::elem_ptr(MeshBaseTempl<RealType> & mesh, dof_id_type i) const
{
  return mesh.elem_ptr(i);
}

template <typename RealType, typename iterator_type>
void DofMap::set_nonlocal_dof_objects(iterator_type objects_begin,
                                      iterator_type objects_end,
                                      MeshBaseTempl<RealType> & mesh,
                                      dofobject_accessor<RealType> objects)
{
  // This function must be run on all processors at once
  parallel_object_only();

  // First, iterate over local objects to find out how many
  // are on each processor
  std::unordered_map<processor_id_type, dof_id_type> ghost_objects_from_proc;

  iterator_type it  = objects_begin;

  for (; it != objects_end; ++it)
    {
      DofObject * obj = *it;

      if (obj)
        {
          processor_id_type obj_procid = obj->processor_id();
          // We'd better be completely partitioned by now
          libmesh_assert_not_equal_to (obj_procid, DofObject::invalid_processor_id);
          ghost_objects_from_proc[obj_procid]++;
        }
    }

  // Request sets to send to each processor
  std::map<processor_id_type, std::vector<dof_id_type>>
    requested_ids;

  // We know how many of our objects live on each processor, so
  // reserve() space for requests from each.
  for (auto pair : ghost_objects_from_proc)
    {
      const processor_id_type p = pair.first;
      if (p != this->processor_id())
        requested_ids[p].reserve(pair.second);
    }

  for (it = objects_begin; it != objects_end; ++it)
    {
      DofObject * obj = *it;
      if (obj->processor_id() != DofObject::invalid_processor_id)
        requested_ids[obj->processor_id()].push_back(obj->id());
    }
#ifdef DEBUG
  for (auto p : IntRange<processor_id_type>(0, this->n_processors()))
    {
      if (ghost_objects_from_proc.count(p))
        libmesh_assert_equal_to (requested_ids[p].size(), ghost_objects_from_proc[p]);
      else
        libmesh_assert(!requested_ids.count(p));
    }
#endif

  typedef std::vector<dof_id_type> datum;

  auto gather_functor =
    [this, &mesh, &objects]
    (processor_id_type,
     const std::vector<dof_id_type> & ids,
     std::vector<datum> & data)
    {
      // Fill those requests
      const unsigned int
        sys_num      = this->sys_number(),
        n_var_groups = this->n_variable_groups();

      const std::size_t query_size = ids.size();

      data.resize(query_size);
      for (auto & d : data)
        d.resize(2 * n_var_groups);

      for (std::size_t i=0; i != query_size; ++i)
        {
          DofObject * requested = (this->*objects)(mesh, ids[i]);
          libmesh_assert(requested);
          libmesh_assert_equal_to (requested->processor_id(), this->processor_id());
          libmesh_assert_equal_to (requested->n_var_groups(sys_num), n_var_groups);
          for (unsigned int vg=0; vg != n_var_groups; ++vg)
            {
              unsigned int n_comp_g =
                requested->n_comp_group(sys_num, vg);
              data[i][vg] = n_comp_g;
              dof_id_type my_first_dof = n_comp_g ?
                requested->vg_dof_base(sys_num, vg) : 0;
              libmesh_assert_not_equal_to (my_first_dof, DofObject::invalid_id);
              data[i][n_var_groups+vg] = my_first_dof;
            }
        }
    };

  auto action_functor =
    [this, &mesh, &objects]
    (processor_id_type libmesh_dbg_var(pid),
     const std::vector<dof_id_type> & ids,
     const std::vector<datum> & data)
    {
      const unsigned int
        sys_num      = this->sys_number(),
        n_var_groups = this->n_variable_groups();

      // Copy the id changes we've now been informed of
      for (auto i : index_range(ids))
        {
          DofObject * requested = (this->*objects)(mesh, ids[i]);
          libmesh_assert(requested);
          libmesh_assert_equal_to (requested->processor_id(), pid);
          for (unsigned int vg=0; vg != n_var_groups; ++vg)
            {
              unsigned int n_comp_g =
                cast_int<unsigned int>(data[i][vg]);
              requested->set_n_comp_group(sys_num, vg, n_comp_g);
              if (n_comp_g)
                {
                  dof_id_type my_first_dof = data[i][n_var_groups+vg];
                  libmesh_assert_not_equal_to (my_first_dof, DofObject::invalid_id);
                  requested->set_vg_dof_base
                    (sys_num, vg, my_first_dof);
                }
            }
        }
    };

  datum * ex = nullptr;
  Parallel::pull_parallel_vector_data
    (this->comm(), requested_ids, gather_functor, action_functor, ex);

#ifdef DEBUG
  // Double check for invalid dofs
  for (it = objects_begin; it != objects_end; ++it)
    {
      DofObject * obj = *it;
      libmesh_assert (obj);
      unsigned int num_variables = obj->n_vars(this->sys_number());
      for (unsigned int v=0; v != num_variables; ++v)
        {
          unsigned int n_comp =
            obj->n_comp(this->sys_number(), v);
          dof_id_type my_first_dof = n_comp ?
            obj->dof_number(this->sys_number(), v, 0) : 0;
          libmesh_assert_not_equal_to (my_first_dof, DofObject::invalid_id);
        }
    }
#endif
}

template <typename RealType>
void DofMap::distribute_local_dofs_node_major(dof_id_type & next_free_dof,
                                              MeshBaseTempl<RealType> & mesh)
{
  const unsigned int sys_num       = this->sys_number();
  const unsigned int n_var_groups  = this->n_variable_groups();

  // Our numbering here must be kept consistent with the numbering
  // scheme assumed by DofMap::local_variable_indices!

  //-------------------------------------------------------------------------
  // First count and assign temporary numbers to local dofs
  for (auto & elem : mesh.active_local_element_ptr_range())
    {
      // Only number dofs connected to active
      // elements on this processor.
      const unsigned int n_nodes = elem->n_nodes();

      // First number the nodal DOFS
      for (unsigned int n=0; n<n_nodes; n++)
        {
          Node & node = elem->node_ref(n);

          for (unsigned vg=0; vg<n_var_groups; vg++)
            {
              const VariableGroup & vg_description(this->variable_group(vg));

              if ((vg_description.type().family != SCALAR) &&
                  (vg_description.active_on_subdomain(elem->subdomain_id())))
                {
                  // assign dof numbers (all at once) if this is
                  // our node and if they aren't already there
                  if ((node.n_comp_group(sys_num,vg) > 0) &&
                      (node.processor_id() == this->processor_id()) &&
                      (node.vg_dof_base(sys_num,vg) ==
                       DofObject::invalid_id))
                    {
                      node.set_vg_dof_base(sys_num, vg,
                                           next_free_dof);
                      next_free_dof += (vg_description.n_variables()*
                                        node.n_comp_group(sys_num,vg));
                      //node.debug_buffer();
                    }
                }
            }
        }

      // Now number the element DOFS
      for (unsigned vg=0; vg<n_var_groups; vg++)
        {
          const VariableGroup & vg_description(this->variable_group(vg));

          if ((vg_description.type().family != SCALAR) &&
              (vg_description.active_on_subdomain(elem->subdomain_id())))
            if (elem->n_comp_group(sys_num,vg) > 0)
              {
                libmesh_assert_equal_to (elem->vg_dof_base(sys_num,vg),
                                         DofObject::invalid_id);

                elem->set_vg_dof_base(sys_num,
                                      vg,
                                      next_free_dof);

                next_free_dof += (vg_description.n_variables()*
                                  elem->n_comp(sys_num,vg));
              }
        }
    } // done looping over elements


  // we may have missed assigning DOFs to nodes that we own
  // but to which we have no connected elements matching our
  // variable restriction criterion.  this will happen, for example,
  // if variable V is restricted to subdomain S.  We may not own
  // any elements which live in S, but we may own nodes which are
  // *connected* to elements which do.  in this scenario these nodes
  // will presently have unnumbered DOFs. we need to take care of
  // them here since we own them and no other processor will touch them.
  for (auto & node : mesh.local_node_ptr_range())
    for (unsigned vg=0; vg<n_var_groups; vg++)
      {
        const VariableGroup & vg_description(this->variable_group(vg));

        if (node->n_comp_group(sys_num,vg))
          if (node->vg_dof_base(sys_num,vg) == DofObject::invalid_id)
            {
              node->set_vg_dof_base (sys_num,
                                     vg,
                                     next_free_dof);

              next_free_dof += (vg_description.n_variables()*
                                node->n_comp(sys_num,vg));
            }
      }

  // Finally, count up the SCALAR dofs
  this->_n_SCALAR_dofs = 0;
  for (unsigned vg=0; vg<n_var_groups; vg++)
    {
      const VariableGroup & vg_description(this->variable_group(vg));

      if (vg_description.type().family == SCALAR)
        {
          this->_n_SCALAR_dofs += (vg_description.n_variables()*
                                   vg_description.type().order.get_order());
          continue;
        }
    }

  // Only increment next_free_dof if we're on the processor
  // that holds this SCALAR variable
  if (this->processor_id() == (this->n_processors()-1))
    next_free_dof += _n_SCALAR_dofs;

#ifdef DEBUG
  {
    // libMesh::out << "next_free_dof=" << next_free_dof << std::endl
    //       << "_n_SCALAR_dofs=" << _n_SCALAR_dofs << std::endl;

    // Make sure we didn't miss any nodes
    MeshTools::libmesh_assert_valid_procids<Node>(mesh);

    for (auto & node : mesh.local_node_ptr_range())
      {
        unsigned int n_var_g = node->n_var_groups(this->sys_number());
        for (unsigned int vg=0; vg != n_var_g; ++vg)
          {
            unsigned int n_comp_g =
              node->n_comp_group(this->sys_number(), vg);
            dof_id_type my_first_dof = n_comp_g ?
              node->vg_dof_base(this->sys_number(), vg) : 0;
            libmesh_assert_not_equal_to (my_first_dof, DofObject::invalid_id);
          }
      }
  }
#endif // DEBUG
}



template <typename RealType>
void DofMap::distribute_local_dofs_var_major(dof_id_type & next_free_dof,
                                             MeshBaseTempl<RealType> & mesh)
{
  const unsigned int sys_num      = this->sys_number();
  const unsigned int n_var_groups = this->n_variable_groups();

  // Our numbering here must be kept consistent with the numbering
  // scheme assumed by DofMap::local_variable_indices!

  //-------------------------------------------------------------------------
  // First count and assign temporary numbers to local dofs
  for (unsigned vg=0; vg<n_var_groups; vg++)
    {
      const VariableGroup & vg_description(this->variable_group(vg));

      const unsigned int n_vars_in_group = vg_description.n_variables();

      // Skip the SCALAR dofs
      if (vg_description.type().family == SCALAR)
        continue;

      for (auto & elem : mesh.active_local_element_ptr_range())
        {
          // Only number dofs connected to active elements on this
          // processor and only variables which are active on on this
          // element's subdomain.
          if (!vg_description.active_on_subdomain(elem->subdomain_id()))
            continue;

          const unsigned int n_nodes = elem->n_nodes();

          // First number the nodal DOFS
          for (unsigned int n=0; n<n_nodes; n++)
            {
              Node & node = elem->node_ref(n);

              // assign dof numbers (all at once) if this is
              // our node and if they aren't already there
              if ((node.n_comp_group(sys_num,vg) > 0) &&
                  (node.processor_id() == this->processor_id()) &&
                  (node.vg_dof_base(sys_num,vg) ==
                   DofObject::invalid_id))
                {
                  node.set_vg_dof_base(sys_num, vg, next_free_dof);

                  next_free_dof += (n_vars_in_group*
                                    node.n_comp_group(sys_num,vg));
                }
            }

          // Now number the element DOFS
          if (elem->n_comp_group(sys_num,vg) > 0)
            {
              libmesh_assert_equal_to (elem->vg_dof_base(sys_num,vg),
                                       DofObject::invalid_id);

              elem->set_vg_dof_base(sys_num,
                                    vg,
                                    next_free_dof);

              next_free_dof += (n_vars_in_group*
                                elem->n_comp_group(sys_num,vg));
            }
        } // end loop on elements

      // we may have missed assigning DOFs to nodes that we own
      // but to which we have no connected elements matching our
      // variable restriction criterion.  this will happen, for example,
      // if variable V is restricted to subdomain S.  We may not own
      // any elements which live in S, but we may own nodes which are
      // *connected* to elements which do.  in this scenario these nodes
      // will presently have unnumbered DOFs. we need to take care of
      // them here since we own them and no other processor will touch them.
      for (auto & node : mesh.local_node_ptr_range())
        if (node->n_comp_group(sys_num,vg))
          if (node->vg_dof_base(sys_num,vg) == DofObject::invalid_id)
            {
              node->set_vg_dof_base (sys_num,
                                     vg,
                                     next_free_dof);

              next_free_dof += (n_vars_in_group*
                                node->n_comp_group(sys_num,vg));
            }
    } // end loop on variable groups

  // Finally, count up the SCALAR dofs
  this->_n_SCALAR_dofs = 0;
  for (unsigned vg=0; vg<n_var_groups; vg++)
    {
      const VariableGroup & vg_description(this->variable_group(vg));

      if (vg_description.type().family == SCALAR)
        {
          this->_n_SCALAR_dofs += (vg_description.n_variables()*
                                   vg_description.type().order.get_order());
          continue;
        }
    }

  // Only increment next_free_dof if we're on the processor
  // that holds this SCALAR variable
  if (this->processor_id() == (this->n_processors()-1))
    next_free_dof += _n_SCALAR_dofs;

#ifdef DEBUG
  {
    // Make sure we didn't miss any nodes
    MeshTools::libmesh_assert_valid_procids<Node>(mesh);

    for (auto & node : mesh.local_node_ptr_range())
      {
        unsigned int n_var_g = node->n_var_groups(this->sys_number());
        for (unsigned int vg=0; vg != n_var_g; ++vg)
          {
            unsigned int n_comp_g =
              node->n_comp_group(this->sys_number(), vg);
            dof_id_type my_first_dof = n_comp_g ?
              node->vg_dof_base(this->sys_number(), vg) : 0;
            libmesh_assert_not_equal_to (my_first_dof, DofObject::invalid_id);
          }
      }
  }
#endif // DEBUG
}

template <typename RealType, template <typename> class ElemIterator>
void
DofMap::
merge_ghost_functor_outputs(gf_map_type<RealType> & elements_to_ghost,
                            std::set<CouplingMatrix *> & temporary_coupling_matrices,
                            const typename std::set<GhostingFunctorBase *>::iterator & gf_begin,
                            const typename std::set<GhostingFunctorBase *>::iterator & gf_end,
                            const ElemIterator<RealType> & elems_begin,
                            const ElemIterator<RealType> & elems_end,
                            processor_id_type p)
{
  typedef GhostingFunctorTempl<RealType> GhostingFunctor;

  for (const auto & gf_base : as_range(gf_begin, gf_end))
    {
      typename GhostingFunctor::map_type more_elements_to_ghost;

      auto gf = static_cast<GhostingFunctor *>(gf_base);

      libmesh_assert(gf);
      (*gf)(elems_begin, elems_end, p, more_elements_to_ghost);

      for (const auto & pr : more_elements_to_ghost)
        {
          typename GhostingFunctor::map_type::iterator existing_it =
            elements_to_ghost.find (pr.first);
          if (existing_it == elements_to_ghost.end())
            elements_to_ghost.insert(pr);
          else
            {
              if (existing_it->second)
                {
                  if (pr.second)
                    {
                      // If this isn't already a temporary
                      // then we need to make one so we'll
                      // have a non-const matrix to merge
                      if (temporary_coupling_matrices.empty() ||
                          temporary_coupling_matrices.find(const_cast<CouplingMatrix *>(existing_it->second)) == temporary_coupling_matrices.end())
                        {
                          CouplingMatrix * cm = new CouplingMatrix(*existing_it->second);
                          temporary_coupling_matrices.insert(cm);
                          existing_it->second = cm;
                        }
                      const_cast<CouplingMatrix &>(*existing_it->second) &= *pr.second;
                    }
                  else
                    {
                      // Any existing_it matrix merged with a full
                      // matrix (symbolized as nullptr) gives another
                      // full matrix (symbolizable as nullptr).

                      // So if existing_it->second is a temporary then
                      // we don't need it anymore; we might as well
                      // remove it to keep the set of temporaries
                      // small.
                      std::set<CouplingMatrix *>::iterator temp_it =
                        temporary_coupling_matrices.find(const_cast<CouplingMatrix *>(existing_it->second));
                      if (temp_it != temporary_coupling_matrices.end())
                        temporary_coupling_matrices.erase(temp_it);

                      existing_it->second = nullptr;
                    }
                }
              // else we have a nullptr already, then we have a full
              // coupling matrix, already, and merging with anything
              // else won't change that, so we're done.
            }
        }
    }
}



template <typename RealType>
void DofMap::add_neighbors_to_send_list(MeshBaseTempl<RealType> & mesh)
{
  LOG_SCOPE("add_neighbors_to_send_list()", "DofMap");

  // Return immediately if there's no ghost data
  if (this->n_processors() == 1)
    return;

  const unsigned int n_var  = this->n_variables();

  auto local_elem_it
    = mesh.active_local_elements_begin();
  auto local_elem_end
    = mesh.active_local_elements_end();

  GhostingFunctor::map_type elements_to_send;

  // Man, I wish we had guaranteed unique_ptr availability...
  std::set<CouplingMatrix *> temporary_coupling_matrices;

  // We need to add dofs to the send list if they've been directly
  // requested by an algebraic ghosting functor or they've been
  // indirectly requested by a coupling functor.
  this->merge_ghost_functor_outputs(elements_to_send,
                                    temporary_coupling_matrices,
                                    this->algebraic_ghosting_functors_begin(),
                                    this->algebraic_ghosting_functors_end(),
                                    local_elem_it, local_elem_end, mesh.processor_id());

  this->merge_ghost_functor_outputs(elements_to_send,
                                    temporary_coupling_matrices,
                                    this->coupling_functors_begin(),
                                    this->coupling_functors_end(),
                                    local_elem_it, local_elem_end, mesh.processor_id());

  // Making a list of non-zero coupling matrix columns is an
  // O(N_var^2) operation.  We cache it so we only have to do it once
  // per CouplingMatrix and not once per element.
  std::map<const CouplingMatrix *, std::vector<unsigned int>>
    column_variable_lists;

  for (auto & pr : elements_to_send)
    {
      const Elem * const partner = pr.first;

      // We asked ghosting functors not to give us local elements
      libmesh_assert_not_equal_to
        (partner->processor_id(), this->processor_id());

      const CouplingMatrix * ghost_coupling = pr.second;

      // Loop over any present coupling matrix column variables if we
      // have a coupling matrix, or just add all variables to
      // send_list if not.
      if (ghost_coupling)
        {
          libmesh_assert_equal_to (ghost_coupling->size(), n_var);

          // Try to find a cached list of column variables.
          std::map<const CouplingMatrix *, std::vector<unsigned int>>::const_iterator
            column_variable_list = column_variable_lists.find(ghost_coupling);

          // If we didn't find it, then we need to create it.
          if (column_variable_list == column_variable_lists.end())
            {
              std::pair<std::map<const CouplingMatrix *, std::vector<unsigned int>>::iterator, bool>
                inserted_variable_list_pair = column_variable_lists.insert(std::make_pair(ghost_coupling,
                                                                                          std::vector<unsigned int>()));
              column_variable_list = inserted_variable_list_pair.first;

              std::vector<unsigned int> & new_variable_list =
                inserted_variable_list_pair.first->second;

              std::vector<unsigned char> has_variable(n_var, false);

              for (unsigned int vi = 0; vi != n_var; ++vi)
                {
                  ConstCouplingRow ccr(vi, *ghost_coupling);

                  for (const auto & vj : ccr)
                    has_variable[vj] = true;
                }
              for (unsigned int vj = 0; vj != n_var; ++vj)
                {
                  if (has_variable[vj])
                    new_variable_list.push_back(vj);
                }
            }

          const std::vector<unsigned int> & variable_list =
            column_variable_list->second;

          for (const auto & vj : variable_list)
            {
              std::vector<dof_id_type> di;
              this->dof_indices (partner, di, vj);

              // Insert the remote DOF indices into the send list
              for (auto d : di)
                if (!this->local_index(d))
                  _send_list.push_back(d);
            }
        }
      else
        {
          std::vector<dof_id_type> di;
          this->dof_indices (partner, di);

          // Insert the remote DOF indices into the send list
          for (const auto & dof : di)
            if (!this->local_index(dof))
              _send_list.push_back(dof);
        }

    }

  // We're now done with any merged coupling matrices we had to create.
  for (auto & mat : temporary_coupling_matrices)
    delete mat;

  //-------------------------------------------------------------------------
  // Our coupling functors added dofs from neighboring elements to the
  // send list, but we may still need to add non-local dofs from local
  // elements.
  //-------------------------------------------------------------------------

  // Loop over the active local elements, adding all active elements
  // that neighbor an active local element to the send list.
  for ( ; local_elem_it != local_elem_end; ++local_elem_it)
    {
      const Elem * elem = *local_elem_it;

      std::vector<dof_id_type> di;
      this->dof_indices (elem, di);

      // Insert the remote DOF indices into the send list
      for (const auto & dof : di)
        if (!this->local_index(dof))
          _send_list.push_back(dof);
    }
}

// Not all templated methods will be included in this macro because there is no need to expicitly
// instantiate private methods
#define INSTANTIATE_DOF_MAP_REALTYPE_METHODS(RealType)                                             \
  template DofMap::DofMap(const unsigned int, MeshBaseTempl<RealType> &);                          \
  template void DofMap::create_dof_constraints(const MeshBaseTempl<RealType> &, Real time);        \
  template void DofMap::allgather_recursive_constraints(MeshBaseTempl<RealType> &);                \
  template void DofMap::local_variable_indices(std::vector<dof_id_type> & idx,                     \
                                               const MeshBaseTempl<RealType> & mesh,               \
                                               unsigned int var_num) const;                        \
  template void DofMap::reinit_send_list(MeshBaseTempl<RealType> &);                               \
  template void DofMap::compute_sparsity(const MeshBaseTempl<RealType> &);                               \
  template void DofMap::distribute_dofs(MeshBaseTempl<RealType> &);                                \
  template void DofMap::scatter_constraints(MeshBaseTempl<RealType> &);                            \
  template void DofMap::process_constraints(MeshBaseTempl<RealType> &);                            \
  template void DofMap::reinit(MeshBaseTempl<RealType> &)

// Below are private
// template void DofMap::invalidate_dofs(MeshBaseTempl<RealType> &);
// template DofObject * node_ptr(MeshBaseTempl<RealType> & mesh, dof_id_type i) const;
// template DofObject * elem_ptr(MeshBaseTempl<RealType> & mesh, dof_id_type i) const;
// template void DofMap::build_sparsity(MeshBaseTempl<RealType> &);

} // namespace libMesh

#endif // LIBMESH_DOF_MAP_IMPL_H
