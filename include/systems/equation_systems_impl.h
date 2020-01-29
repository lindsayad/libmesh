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

#ifndef LIBMESH_EQUATION_SYSTEMS_IMPL_H
#define LIBMESH_EQUATION_SYSTEMS_IMPL_H

// System includes
#include <sstream>
#include <cstdio>

// Local Includes
#include "libmesh/default_coupling.h" // For downconversion
#include "libmesh/explicit_system.h"
#include "libmesh/fe_interface.h"
#include "libmesh/frequency_system.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/newmark_system.h"
#include "libmesh/nonlinear_implicit_system.h"
#include "libmesh/rb_construction.h"
#include "libmesh/transient_rb_construction.h"
#include "libmesh/eigen_system.h"
#include "libmesh/parallel.h"
#include "libmesh/transient_system.h"
#include "libmesh/dof_map.h"
#include "libmesh/mesh_base.h"
#include "libmesh/elem.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/libmesh_version.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/xdr_cxx.h"

// Include the systems before this one to avoid
// overlapping forward declarations.
#include "libmesh/equation_systems.h"

namespace libMesh
{

// Forward Declarations




// ------------------------------------------------------------
// EquationSystems class implementation
template <typename RealType>
EquationSystemsTempl<RealType>::EquationSystemsTempl (MeshBase & m) :
  ParallelObject (m),
  _mesh          (m),
  _refine_in_reinit(true),
  _enable_default_ghosting(true)
{
  // Set default parameters
  this->parameters.template set<Real>        ("linear solver tolerance") = TOLERANCE * TOLERANCE;
  this->parameters.template set<unsigned int>("linear solver maximum iterations") = 5000;
}



template <typename RealType>
EquationSystemsTempl<RealType>::~EquationSystemsTempl ()
{
  this->clear ();
}



template <typename RealType>
void EquationSystemsTempl<RealType>::clear ()
{
  // Clear any additional parameters
  parameters.clear ();

  // clear the systems.  We must delete them
  // since we newed them!
  while (!_systems.empty())
    {
      system_iterator pos = _systems.begin();

      System * sys = pos->second;
      delete sys;
      sys = nullptr;

      _systems.erase (pos);
    }
}



template <typename RealType>
void EquationSystemsTempl<RealType>::init ()
{
  const unsigned int n_sys = this->n_systems();

  libmesh_assert_not_equal_to (n_sys, 0);

  // Tell all the \p DofObject entities how many systems
  // there are.
  for (auto & node : _mesh.node_ptr_range())
    node->set_n_systems(n_sys);

  for (auto & elem : _mesh.element_ptr_range())
    elem->set_n_systems(n_sys);

  for (unsigned int i=0; i != this->n_systems(); ++i)
    this->get_system(i).init();

#ifdef LIBMESH_ENABLE_AMR
  MeshRefinement mesh_refine(_mesh);
  mesh_refine.clean_refinement_flags();
#endif
}



template <typename RealType>
void EquationSystemsTempl<RealType>::reinit ()
{
  const bool mesh_changed = this->reinit_solutions();

  // If the mesh has changed, systems will need to reinitialize their
  // own data on the new mesh.
  if (mesh_changed)
    this->reinit_systems();
}



template <typename RealType>
bool EquationSystemsTempl<RealType>::reinit_solutions ()
{
  parallel_object_only();

  const unsigned int n_sys = this->n_systems();
  libmesh_assert_not_equal_to (n_sys, 0);

  // We may have added new systems since our last
  // EquationSystemsTempl<RealType>::(re)init call
  bool _added_new_systems = false;
  for (unsigned int i=0; i != n_sys; ++i)
    if (!this->get_system(i).is_initialized())
      _added_new_systems = true;

  if (_added_new_systems)
    {
      // Our DofObjects will need space for the additional systems
      for (auto & node : _mesh.node_ptr_range())
        node->set_n_systems(n_sys);

      for (auto & elem : _mesh.element_ptr_range())
        elem->set_n_systems(n_sys);

      // And any new systems will need initialization
      for (unsigned int i=0; i != n_sys; ++i)
        if (!this->get_system(i).is_initialized())
          this->get_system(i).init();
    }


  // We used to assert that all nodes and elements *already* had
  // n_systems() properly set; however this is false in the case where
  // user code has manually added nodes and/or elements to an
  // already-initialized system.

  // Make sure all the \p DofObject entities know how many systems
  // there are.
  {
    // All the nodes
    for (auto & node : _mesh.node_ptr_range())
      node->set_n_systems(this->n_systems());

    // All the elements
    for (auto & elem : _mesh.element_ptr_range())
      elem->set_n_systems(this->n_systems());
  }

  // Localize each system's vectors
  for (unsigned int i=0; i != this->n_systems(); ++i)
    this->get_system(i).re_update();

#ifdef LIBMESH_ENABLE_AMR

  bool mesh_changed = false;

  // FIXME: For backwards compatibility, assume
  // refine_and_coarsen_elements or refine_uniformly have already
  // been called
  {
    for (unsigned int i=0; i != this->n_systems(); ++i)
      {
        System & sys = this->get_system(i);

        // Even if the system doesn't have any variables in it we want
        // consistent behavior; e.g. distribute_dofs should have the
        // opportunity to count up zero dofs on each processor.
        //
        // Who's been adding zero-var systems anyway, outside of my
        // unit tests? - RHS
        // if (!sys.n_vars())
        // continue;

        sys.get_dof_map().distribute_dofs(_mesh);

        // Recreate any user or internal constraints
        sys.reinit_constraints();

        sys.prolong_vectors();
      }
    mesh_changed = true;
  }

  if (this->_refine_in_reinit)
    {
      // Don't override any user refinement settings
      MeshRefinement mesh_refine(_mesh);
      mesh_refine.face_level_mismatch_limit() = 0; // unlimited
      mesh_refine.overrefined_boundary_limit() = -1; // unlimited
      mesh_refine.underrefined_boundary_limit() = -1; // unlimited

      // Try to coarsen the mesh, then restrict each system's vectors
      // if necessary
      if (mesh_refine.coarsen_elements())
        {
          for (unsigned int i=0; i != this->n_systems(); ++i)
            {
              System & sys = this->get_system(i);
              sys.get_dof_map().distribute_dofs(_mesh);
              sys.reinit_constraints();
              sys.restrict_vectors();
            }
          mesh_changed = true;
        }

      // Once vectors are all restricted, we can delete
      // children of coarsened elements
      if (mesh_changed)
        this->get_mesh().contract();

      // Try to refine the mesh, then prolong each system's vectors
      // if necessary
      if (mesh_refine.refine_elements())
        {
          for (unsigned int i=0; i != this->n_systems(); ++i)
            {
              System & sys = this->get_system(i);
              sys.get_dof_map().distribute_dofs(_mesh);
              sys.reinit_constraints();
              sys.prolong_vectors();
            }
          mesh_changed = true;
        }
    }

  return mesh_changed;

#endif // #ifdef LIBMESH_ENABLE_AMR

  return false;
}



template <typename RealType>
void EquationSystemsTempl<RealType>::reinit_systems()
{
  for (unsigned int i=0; i != this->n_systems(); ++i)
    this->get_system(i).reinit();
}



template <typename RealType>
void EquationSystemsTempl<RealType>::allgather ()
{
  // A serial mesh means nothing needs to be done
  if (_mesh.is_serial())
    return;

  const unsigned int n_sys = this->n_systems();

  libmesh_assert_not_equal_to (n_sys, 0);

  // Gather the mesh
  _mesh.allgather();

  // Tell all the \p DofObject entities how many systems
  // there are.
  for (auto & node : _mesh.node_ptr_range())
    node->set_n_systems(n_sys);

  for (auto & elem : _mesh.element_ptr_range())
    elem->set_n_systems(n_sys);

  // And distribute each system's dofs
  for (unsigned int i=0; i != this->n_systems(); ++i)
    {
      System & sys = this->get_system(i);
      DofMap & dof_map = sys.get_dof_map();
      dof_map.distribute_dofs(_mesh);

      // The user probably won't need constraint equations or the
      // send_list after an allgather, but let's keep it in consistent
      // shape just in case.
      sys.reinit_constraints();
      dof_map.prepare_send_list();
    }
}



template <typename RealType>
void EquationSystemsTempl<RealType>::enable_default_ghosting (bool enable)
{
  _enable_default_ghosting = enable;
  MeshBase &mesh = this->get_mesh();

  if (enable)
    mesh.add_ghosting_functor(mesh.default_ghosting());
  else
    mesh.remove_ghosting_functor(mesh.default_ghosting());

  for (unsigned int i=0; i != this->n_systems(); ++i)
    {
      DofMap & dof_map = this->get_system(i).get_dof_map();
      if (enable)
        dof_map.add_default_ghosting();
      else
        dof_map.remove_default_ghosting();
    }
}



template <typename RealType>
void EquationSystemsTempl<RealType>::update ()
{
  LOG_SCOPE("update()", "EquationSystems");

  // Localize each system's vectors
  for (unsigned int i=0; i != this->n_systems(); ++i)
    this->get_system(i).update();
}



template <typename RealType>
System & EquationSystemsTempl<RealType>::add_system (const std::string & sys_type,
                                      const std::string & name)
{
  // If the user already built a system with this name, we'll
  // trust them and we'll use it.  That way they can pre-add
  // non-standard derived system classes, and if their restart file
  // has some non-standard sys_type we won't throw an error.
  if (_systems.count(name))
    {
      return this->get_system(name);
    }
  // Build a basic System
  else if (sys_type == "Basic")
    this->add_system<System> (name);

  // Build a Newmark system
  else if (sys_type == "Newmark")
    this->add_system<NewmarkSystem> (name);

  // Build an Explicit system
  else if ((sys_type == "Explicit"))
    this->add_system<ExplicitSystem> (name);

  // Build an Implicit system
  else if ((sys_type == "Implicit") ||
           (sys_type == "Steady"  ))
    this->add_system<ImplicitSystem> (name);

  // build a transient implicit linear system
  else if ((sys_type == "Transient") ||
           (sys_type == "TransientImplicit") ||
           (sys_type == "TransientLinearImplicit"))
    this->add_system<TransientLinearImplicitSystem> (name);

  // build a transient implicit nonlinear system
  else if (sys_type == "TransientNonlinearImplicit")
    this->add_system<TransientNonlinearImplicitSystem> (name);

  // build a transient explicit system
  else if (sys_type == "TransientExplicit")
    this->add_system<TransientExplicitSystem> (name);

  // build a linear implicit system
  else if (sys_type == "LinearImplicit")
    this->add_system<LinearImplicitSystem> (name);

  // build a nonlinear implicit system
  else if (sys_type == "NonlinearImplicit")
    this->add_system<NonlinearImplicitSystem> (name);

  // build a Reduced Basis Construction system
  else if (sys_type == "RBConstruction")
    this->add_system<RBConstruction> (name);

  // build a transient Reduced Basis Construction system
  else if (sys_type == "TransientRBConstruction")
    this->add_system<TransientRBConstruction> (name);

#ifdef LIBMESH_HAVE_SLEPC
  // build an eigen system
  else if (sys_type == "Eigen")
    this->add_system<EigenSystem> (name);
  else if (sys_type == "TransientEigenSystem")
    this->add_system<TransientEigenSystem> (name);
#endif

#if defined(LIBMESH_USE_COMPLEX_NUMBERS)
  // build a frequency system
  else if (sys_type == "Frequency")
    this->add_system<FrequencySystem> (name);
#endif

  else
    libmesh_error_msg("ERROR: Unknown system type: " << sys_type);

  // Return a reference to the new system
  //return (*this)(name);
  return this->get_system(name);
}






#ifdef LIBMESH_ENABLE_DEPRECATED
template <typename RealType>
void EquationSystemsTempl<RealType>::delete_system (const std::string & name)
{
  libmesh_deprecated();

  if (!_systems.count(name))
    libmesh_error_msg("ERROR: no system named " << name);

  delete _systems[name];

  _systems.erase (name);
}
#endif



template <typename RealType>
void EquationSystemsTempl<RealType>::solve ()
{
  libmesh_assert (this->n_systems());

  for (unsigned int i=0; i != this->n_systems(); ++i)
    this->get_system(i).solve();
}



template <typename RealType>
void EquationSystemsTempl<RealType>::sensitivity_solve (const ParameterVector & parameters_in)
{
  libmesh_assert (this->n_systems());

  for (unsigned int i=0; i != this->n_systems(); ++i)
    this->get_system(i).sensitivity_solve(parameters_in);
}



template <typename RealType>
void EquationSystemsTempl<RealType>::adjoint_solve (const QoISet & qoi_indices)
{
  libmesh_assert (this->n_systems());

  for (unsigned int i=this->n_systems(); i != 0; --i)
    this->get_system(i-1).adjoint_solve(qoi_indices);
}



template <typename RealType>
void EquationSystemsTempl<RealType>::build_variable_names (std::vector<std::string> & var_names,
                                            const FEType * type,
                                            const std::set<std::string> * system_names) const
{
  unsigned int var_num=0;

  const_system_iterator       pos = _systems.begin();
  const const_system_iterator end = _systems.end();

  // Need to size var_names by scalar variables plus all the
  // vector components for all the vector variables
  //Could this be replaced by a/some convenience methods?[PB]
  {
    unsigned int n_scalar_vars = 0;
    unsigned int n_vector_vars = 0;

    for (; pos != end; ++pos)
      {
        // Check current system is listed in system_names, and skip pos if not
        bool use_current_system = (system_names == nullptr);
        if (!use_current_system)
          use_current_system = system_names->count(pos->first);
        if (!use_current_system || pos->second->hide_output())
          continue;

        for (auto vn : IntRange<unsigned int>(0, pos->second->n_vars()))
          {
            if (FEInterface::field_type(pos->second->variable_type(vn)) == TYPE_VECTOR)
              n_vector_vars++;
            else
              n_scalar_vars++;
          }
      }

    // Here, we're assuming the number of vector components is the same
    // as the mesh dimension. Will break for mixed dimension meshes.
    unsigned int dim = this->get_mesh().mesh_dimension();
    unsigned int nv = n_scalar_vars + dim*n_vector_vars;

    // We'd better not have more than dim*his->n_vars() (all vector variables)
    libmesh_assert_less_equal ( nv, dim*this->n_vars() );

    // Here, we're assuming the number of vector components is the same
    // as the mesh dimension. Will break for mixed dimension meshes.

    var_names.resize( nv );
  }

  // reset
  pos = _systems.begin();

  for (; pos != end; ++pos)
    {
      // Check current system is listed in system_names, and skip pos if not
      bool use_current_system = (system_names == nullptr);
      if (!use_current_system)
        use_current_system = system_names->count(pos->first);
      if (!use_current_system || pos->second->hide_output())
        continue;

      for (auto vn : IntRange<unsigned int>(0, pos->second->n_vars()))
        {
          const std::string & var_name = pos->second->variable_name(vn);
          const FEType & fe_type = pos->second->variable_type(vn);

          unsigned int n_vec_dim = FEInterface::n_vec_dim( pos->second->get_mesh(), fe_type);

          // Filter on the type if requested
          if (type == nullptr || (type && *type == fe_type))
            {
              if (FEInterface::field_type(fe_type) == TYPE_VECTOR)
                {
                  switch(n_vec_dim)
                    {
                    case 0:
                    case 1:
                      var_names[var_num++] = var_name;
                      break;
                    case 2:
                      var_names[var_num++] = var_name+"_x";
                      var_names[var_num++] = var_name+"_y";
                      break;
                    case 3:
                      var_names[var_num++] = var_name+"_x";
                      var_names[var_num++] = var_name+"_y";
                      var_names[var_num++] = var_name+"_z";
                      break;
                    default:
                      libmesh_error_msg("Invalid dim in build_variable_names");
                    }
                }
              else
                var_names[var_num++] = var_name;
            }
        }
    }
  // Now resize again in case we filtered any names
  var_names.resize(var_num);
}



template <typename RealType>
void EquationSystemsTempl<RealType>::build_solution_vector (std::vector<Number> &,
                                             const std::string &,
                                             const std::string &) const
{
  // TODO:[BSK] re-implement this from the method below
  libmesh_not_implemented();
}




template <typename RealType>
std::unique_ptr<NumericVector<Number>>
EquationSystemsTempl<RealType>::build_parallel_solution_vector(const std::set<std::string> * system_names) const
{
  LOG_SCOPE("build_parallel_solution_vector()", "EquationSystems");

  // This function must be run on all processors at once
  parallel_object_only();

  const unsigned int dim = _mesh.mesh_dimension();
  const dof_id_type max_nn   = _mesh.max_node_id();

  // allocate vector storage to hold
  // (max_node_id)*(number_of_variables) entries.
  //
  // If node renumbering is disabled and adaptive coarsening has
  // created gaps between node numbers, then this vector will be
  // sparse.
  //
  // We have to differentiate between between scalar and vector
  // variables. We intercept vector variables and treat each
  // component as a scalar variable (consistently with build_solution_names).

  unsigned int nv = 0;

  //Could this be replaced by a/some convenience methods?[PB]
  {
    unsigned int n_scalar_vars = 0;
    unsigned int n_vector_vars = 0;
    const_system_iterator       pos = _systems.begin();
    const const_system_iterator end = _systems.end();

    for (; pos != end; ++pos)
      {
        // Check current system is listed in system_names, and skip pos if not
        bool use_current_system = (system_names == nullptr);
        if (!use_current_system)
          use_current_system = system_names->count(pos->first);
        if (!use_current_system || pos->second->hide_output())
          continue;

        for (auto vn : IntRange<unsigned int>(0, pos->second->n_vars()))
          {
            if (FEInterface::field_type(pos->second->variable_type(vn)) == TYPE_VECTOR)
              n_vector_vars++;
            else
              n_scalar_vars++;
          }
      }
    // Here, we're assuming the number of vector components is the same
    // as the mesh dimension. Will break for mixed dimension meshes.
    nv = n_scalar_vars + dim*n_vector_vars;
  }

  // Get the number of nodes to store locally.
  dof_id_type n_local_nodes = cast_int<dof_id_type>
    (std::distance(_mesh.local_nodes_begin(),
                   _mesh.local_nodes_end()));

  // If node renumbering has been disabled, nodes may not be numbered
  // contiguously, and the number of nodes might not match the
  // max_node_id.  In this case we just do our best.
  dof_id_type n_total_nodes = n_local_nodes;
  _mesh.comm().sum(n_total_nodes);

  const dof_id_type n_gaps = max_nn - n_total_nodes;
  const dof_id_type gaps_per_processor = n_gaps / _mesh.comm().size();
  const dof_id_type remainder_gaps = n_gaps % _mesh.comm().size();

  n_local_nodes = n_local_nodes +      // Actual nodes
                  gaps_per_processor + // Our even share of gaps
                  (_mesh.comm().rank() < remainder_gaps); // Leftovers

  // Create a NumericVector to hold the parallel solution
  std::unique_ptr<NumericVector<Number>> parallel_soln_ptr = NumericVector<Number>::build(_communicator);
  NumericVector<Number> & parallel_soln = *parallel_soln_ptr;
  parallel_soln.init(max_nn*nv, n_local_nodes*nv, false, PARALLEL);

  // Create a NumericVector to hold the "repeat_count" for each node - this is essentially
  // the number of elements contributing to that node's value
  std::unique_ptr<NumericVector<Number>> repeat_count_ptr = NumericVector<Number>::build(_communicator);
  NumericVector<Number> & repeat_count = *repeat_count_ptr;
  repeat_count.init(max_nn*nv, n_local_nodes*nv, false, PARALLEL);

  repeat_count.close();

  unsigned int var_num=0;

  // For each system in this EquationSystems object,
  // update the global solution and if we are on processor 0,
  // loop over the elements and build the nodal solution
  // from the element solution.  Then insert this nodal solution
  // into the vector passed to build_solution_vector.
  const_system_iterator       pos = _systems.begin();
  const const_system_iterator end = _systems.end();

  for (; pos != end; ++pos)
    {
      // Check current system is listed in system_names, and skip pos if not
      bool use_current_system = (system_names == nullptr);
      if (!use_current_system)
        use_current_system = system_names->count(pos->first);
      if (!use_current_system || pos->second->hide_output())
        continue;

      const System & system  = *(pos->second);
      const unsigned int nv_sys = system.n_vars();
      const unsigned int sys_num = system.number();

      //Could this be replaced by a/some convenience methods?[PB]
      unsigned int n_scalar_vars = 0;
      unsigned int n_vector_vars = 0;
      for (auto vn : IntRange<unsigned int>(0, pos->second->n_vars()))
        {
          if (FEInterface::field_type(pos->second->variable_type(vn)) == TYPE_VECTOR)
            n_vector_vars++;
          else
            n_scalar_vars++;
        }

      // Here, we're assuming the number of vector components is the same
      // as the mesh dimension. Will break for mixed dimension meshes.
      unsigned int nv_sys_split = n_scalar_vars + dim*n_vector_vars;

      // Update the current_local_solution
      {
        System & non_const_sys = const_cast<System &>(system);
        // We used to simply call non_const_sys.solution->close()
        // here, but that is not allowed when the solution vector is
        // locked read-only, for example when printing the solution
        // during the middle of a solve...  So try to be a bit
        // more careful about calling close() unnecessarily.
        libmesh_assert(this->comm().verify(non_const_sys.solution->closed()));
        if (!non_const_sys.solution->closed())
          non_const_sys.solution->close();
        non_const_sys.update();
      }

      NumericVector<Number> & sys_soln(*system.current_local_solution);

      std::vector<Number>      elem_soln;   // The finite element solution
      std::vector<Number>      nodal_soln;  // The FE solution interpolated to the nodes
      std::vector<dof_id_type> dof_indices; // The DOF indices for the finite element

      unsigned var_inc = 0;
      for (unsigned int var=0; var<nv_sys; var++)
        {
          const FEType & fe_type           = system.variable_type(var);
          const Variable & var_description = system.variable(var);
          const DofMap & dof_map           = system.get_dof_map();

          unsigned int n_vec_dim = FEInterface::n_vec_dim( pos->second->get_mesh(), fe_type );

          for (const auto & elem : _mesh.active_local_element_ptr_range())
            {
              if (var_description.active_on_subdomain(elem->subdomain_id()))
                {
                  dof_map.dof_indices (elem, dof_indices, var);

                  elem_soln.resize(dof_indices.size());

                  for (auto i : index_range(dof_indices))
                    elem_soln[i] = sys_soln(dof_indices[i]);

                  FEInterface::nodal_soln (dim,
                                           fe_type,
                                           elem,
                                           elem_soln,
                                           nodal_soln);

#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
                  // infinite elements should be skipped...
                  if (!elem->infinite())
#endif
                    {
                      libmesh_assert_equal_to (nodal_soln.size(), n_vec_dim*elem->n_nodes());

                      for (auto n : elem->node_index_range())
                        {
                          for (unsigned int d=0; d < n_vec_dim; d++)
                            {
                              // For vector-valued elements, all components are in nodal_soln. For each
                              // node, the components are stored in order, i.e. node_0 -> s0_x, s0_y, s0_z
                              parallel_soln.add(nv*(elem->node_id(n)) + (var_inc+d + var_num), nodal_soln[n_vec_dim*n+d]);

                              // Increment the repeat count for this position
                              repeat_count.add(nv*(elem->node_id(n)) + (var_inc+d + var_num), 1);
                            }
                        }
                    }
                }
              else // If this variable doesn't exist on this subdomain we have to still increment repeat_count so that we won't divide by 0 later:
                for (const Node & node : elem->node_ref_range())
                  // Only do this if this variable has NO DoFs at this node... it might have some from an adjoining element...
                  if (!node.n_dofs(sys_num, var))
                    for (unsigned int d=0; d < n_vec_dim; d++)
                      repeat_count.add(nv*node.id() + (var_inc+d + var_num), 1);

            } // end loop over elements
          var_inc += n_vec_dim;
        } // end loop on variables in this system

      var_num += nv_sys_split;
    } // end loop over systems

  // Sum the nodal solution values and repeat counts.
  parallel_soln.close();
  repeat_count.close();

  // If there were gaps in the node numbering, there will be
  // corresponding zeros in the parallel_soln and repeat_count
  // vectors.  We need to set those repeat_count entries to 1
  // in order to avoid dividing by zero.
  if (n_gaps)
    {
      for (numeric_index_type i=repeat_count.first_local_index();
           i<repeat_count.last_local_index(); ++i)
        {
          // repeat_count entries are integral values but let's avoid a
          // direct floating point comparison with 0 just in case some
          // roundoff noise crept in during vector assembly?
          if (std::abs(repeat_count(i)) < TOLERANCE)
            repeat_count.set(i, 1.);
        }

      // Make sure the repeat_count vector is up-to-date on all
      // processors.
      repeat_count.close();
    }

  // Divide to get the average value at the nodes
  parallel_soln /= repeat_count;

  return std::unique_ptr<NumericVector<Number>>(parallel_soln_ptr.release());
}



template <typename RealType>
void EquationSystemsTempl<RealType>::build_solution_vector (std::vector<Number> & soln,
                                             const std::set<std::string> * system_names) const
{
  LOG_SCOPE("build_solution_vector()", "EquationSystems");

  // Call the parallel implementation
  std::unique_ptr<NumericVector<Number>> parallel_soln =
    this->build_parallel_solution_vector(system_names);

  // Localize the NumericVector into the provided std::vector.
  parallel_soln->localize_to_one(soln);
}



template <typename RealType>
void EquationSystemsTempl<RealType>::get_vars_active_subdomains(const std::vector<std::string> & names,
                                                 std::vector<std::set<subdomain_id_type>> & vars_active_subdomains) const
{
  unsigned int var_num=0;

  vars_active_subdomains.clear();
  vars_active_subdomains.resize(names.size());

  const_system_iterator       pos = _systems.begin();
  const const_system_iterator end = _systems.end();

  for (; pos != end; ++pos)
    {
      for (auto vn : IntRange<unsigned int>(0, pos->second->n_vars()))
        {
          const std::string & var_name = pos->second->variable_name(vn);

          auto names_it = std::find(names.begin(), names.end(), var_name);
          if(names_it != names.end())
            {
              const Variable & variable = pos->second->variable(vn);
              const std::set<subdomain_id_type> & active_subdomains = variable.active_subdomains();
              vars_active_subdomains[var_num++] = active_subdomains;
            }
        }
    }

  libmesh_assert_equal_to(var_num, names.size());
}



template <typename RealType>
void EquationSystemsTempl<RealType>::get_solution (std::vector<Number> & soln,
                                    std::vector<std::string> & names) const
{
  libmesh_deprecated();
  this->build_elemental_solution_vector(soln, names);
}



template <typename RealType>
void
EquationSystemsTempl<RealType>::build_elemental_solution_vector (std::vector<Number> & soln,
                                                  std::vector<std::string> & names) const
{
  // Call the parallel version of this function
  std::unique_ptr<NumericVector<Number>> parallel_soln =
    this->build_parallel_elemental_solution_vector(names);

  // Localize into 'soln', provided that parallel_soln is not empty.
  // Note: parallel_soln will be empty in the event that none of the
  // input names were CONSTANT, MONOMIAL variables or there were
  // simply no CONSTANT, MONOMIAL variables in the EquationSystems
  // object.
  soln.clear();
  if (parallel_soln)
    parallel_soln->localize_to_one(soln);
}



template <typename RealType>
std::vector<std::pair<unsigned int, unsigned int>>
EquationSystemsTempl<RealType>::find_variable_numbers
  (std::vector<std::string> & names, const FEType * type) const
{
  // This function must be run on all processors at once
  parallel_object_only();

  libmesh_assert (this->n_systems());

  // If the names vector has entries, we will only populate the soln vector
  // with names included in that list.  Note: The names vector may be
  // reordered upon exiting this function
  std::vector<std::pair<unsigned int, unsigned int>> var_nums;
  std::vector<std::string> filter_names = names;
  bool is_filter_names = !filter_names.empty();

  names.clear();

  const_system_iterator       pos = _systems.begin();
  const const_system_iterator end = _systems.end();
  unsigned sys_ctr = 0;

  for (; pos != end; ++pos, ++sys_ctr)
    {
      const System & system = *(pos->second);
      const unsigned int nv_sys = system.n_vars();

      for (unsigned int var=0; var < nv_sys; ++var)
        {
          const std::string & name = system.variable_name(var);
          if ((type && system.variable_type(var) != *type) ||
              (is_filter_names && std::find(filter_names.begin(), filter_names.end(), name) == filter_names.end()))
            continue;

          // Otherwise, this variable should be output
          var_nums.push_back
            (std::make_pair(system.number(), var));
        }
    }

  std::sort(var_nums.begin(), var_nums.end());

  for (const auto & var_num : var_nums)
    {
      const std::string & name =
        this->get_system(var_num.first).variable_name(var_num.second);
      if (names.empty() || names.back() != name)
        names.push_back(name);
    }

  return var_nums;
}


template <typename RealType>
std::unique_ptr<NumericVector<Number>>
EquationSystemsTempl<RealType>::build_parallel_elemental_solution_vector (std::vector<std::string> & names) const
{
  FEType type(CONSTANT, MONOMIAL);
  std::vector<std::pair<unsigned int, unsigned int>> var_nums =
    this->find_variable_numbers(names, &type);

  const std::size_t nv = var_nums.size();
  const dof_id_type ne = _mesh.n_elem();
  libmesh_assert_equal_to (ne, _mesh.max_elem_id());

  // If there are no variables to write out don't do anything...
  if (!nv)
    return std::unique_ptr<NumericVector<Number>>(nullptr);

  // We can handle the case where there are nullptrs in the Elem vector
  // by just having extra zeros in the solution vector.
  numeric_index_type parallel_soln_global_size = ne*nv;

  numeric_index_type div = parallel_soln_global_size / this->n_processors();
  numeric_index_type mod = parallel_soln_global_size % this->n_processors();

  // Initialize all processors to the average size.
  numeric_index_type parallel_soln_local_size = div;

  // The first "mod" processors get an extra entry.
  if (this->processor_id() < mod)
    parallel_soln_local_size = div+1;

  // Create a NumericVector to hold the parallel solution
  std::unique_ptr<NumericVector<Number>> parallel_soln_ptr = NumericVector<Number>::build(_communicator);
  NumericVector<Number> & parallel_soln = *parallel_soln_ptr;
  parallel_soln.init(parallel_soln_global_size,
                     parallel_soln_local_size,
                     /*fast=*/false,
                     /*ParallelType=*/PARALLEL);

  unsigned int sys_ctr = 0;

  // For each system in this EquationSystems object,
  // update the global solution and collect the
  // CONSTANT MONOMIALs.  The entries are in variable-major
  // format.
  for (auto i : index_range(var_nums))
    {
      std::pair<unsigned int, unsigned int> var_num = var_nums[i];
      const System & system  = this->get_system(var_num.first);

      // Update the current_local_solution if necessary
      if (sys_ctr != var_num.first)
        {
          System & non_const_sys = const_cast<System &>(system);
          // We used to simply call non_const_sys.solution->close()
          // here, but that is not allowed when the solution vector is
          // locked read-only, for example when printing the solution
          // during during the middle of a solve...  So try to be a bit
          // more careful about calling close() unnecessarily.
          libmesh_assert(this->comm().verify(non_const_sys.solution->closed()));
          if (!non_const_sys.solution->closed())
            non_const_sys.solution->close();
          non_const_sys.update();
          sys_ctr = var_num.first;
        }

      NumericVector<Number> & sys_soln(*system.current_local_solution);

      // The DOF indices for the finite element
      std::vector<dof_id_type> dof_indices;

      const unsigned int var = var_num.second;

      const Variable & variable = system.variable(var);
      const DofMap & dof_map = system.get_dof_map();

      for (const auto & elem : _mesh.active_local_element_ptr_range())
        if (variable.active_on_subdomain(elem->subdomain_id()))
          {
            dof_map.dof_indices (elem, dof_indices, var);

            libmesh_assert_equal_to (1, dof_indices.size());

            parallel_soln.set((ne*i)+elem->id(), sys_soln(dof_indices[0]));
          }
    } // end loop over var_nums

  parallel_soln.close();
  return std::unique_ptr<NumericVector<Number>>(parallel_soln_ptr.release());
}



template <typename RealType>
void
EquationSystemsTempl<RealType>::build_discontinuous_solution_vector
(std::vector<Number> & soln,
 const std::set<std::string> * system_names,
 const std::vector<std::string> * var_names,
 bool vertices_only) const
{
  LOG_SCOPE("build_discontinuous_solution_vector()", "EquationSystems");

  libmesh_assert (this->n_systems());

  const unsigned int dim = _mesh.mesh_dimension();

  // Get the number of variables (nv) by counting the number of variables
  // in each system listed in system_names
  unsigned int nv = 0;

  for (const auto & pr : _systems)
    {
      // Check current system is listed in system_names, and skip pos if not
      bool use_current_system = (system_names == nullptr);
      if (!use_current_system)
        use_current_system = system_names->count(pr.first);
      if (!use_current_system || pr.second->hide_output())
        continue;

      const System * system  = pr.second;

      // Loop over all variables in this System and check whether we
      // are supposed to use each one.
      for (auto var_id : IntRange<unsigned int>(0, system->n_vars()))
        {
          bool use_current_var = (var_names == nullptr);
          if (!use_current_var)
            use_current_var = std::count(var_names->begin(),
                                         var_names->end(),
                                         system->variable_name(var_id));

          // Only increment the total number of vars if we are
          // supposed to use this one.
          if (use_current_var)
            nv++;
        }
    }

  // get the total weight
  unsigned int tw=0;
  for (const auto & elem : _mesh.active_element_ptr_range())
    tw += vertices_only ? elem->n_vertices() : elem->n_nodes();

  // Only if we are on processor zero, allocate the storage
  // to hold (number_of_nodes)*(number_of_variables) entries.
  if (_mesh.processor_id() == 0)
    soln.resize(tw*nv);

  std::vector<Number> sys_soln;

  // Keep track of the variable "offset". This is used for indexing
  // into the global solution vector.
  unsigned int var_offset = 0;

  // For each system in this EquationSystems object,
  // update the global solution and if we are on processor 0,
  // loop over the elements and build the nodal solution
  // from the element solution.  Then insert this nodal solution
  // into the vector passed to build_solution_vector.
  for (const auto & pr : _systems)
    {
      // Check current system is listed in system_names, and skip pos if not
      bool use_current_system = (system_names == nullptr);
      if (!use_current_system)
        use_current_system = system_names->count(pr.first);
      if (!use_current_system || pr.second->hide_output())
        continue;

      const System * system  = pr.second;
      const unsigned int nv_sys = system->n_vars();

      system->update_global_solution (sys_soln, 0);

      // Keep track of the number of vars actually written.
      unsigned int n_vars_written_current_system = 0;

      if (_mesh.processor_id() == 0)
        {
          std::vector<Number>       soln_coeffs; // The finite element solution coeffs
          std::vector<Number>       nodal_soln;  // The FE solution interpolated to the nodes
          std::vector<dof_id_type>  dof_indices; // The DOF indices for the finite element

          // For each variable, determine if we are supposed to
          // write it, then loop over the active elements, compute
          // the nodal_soln and store it to the "soln" vector. We
          // store zeros for subdomain-restricted variables on
          // elements where they are not active.
          for (unsigned int var=0; var<nv_sys; var++)
            {
              bool use_current_var = (var_names == nullptr);
              if (!use_current_var)
                use_current_var = std::count(var_names->begin(),
                                             var_names->end(),
                                             system->variable_name(var));

              // If we aren't supposed to write this var, go to the
              // next loop iteration.
              if (!use_current_var)
                continue;

              const FEType & fe_type = system->variable_type(var);
              const Variable & var_description = system->variable(var);

              unsigned int nn=0;

              for (auto & elem : _mesh.active_element_ptr_range())
                {
                  if (var_description.active_on_subdomain(elem->subdomain_id()))
                    {
                      system->get_dof_map().dof_indices (elem, dof_indices, var);

                      soln_coeffs.resize(dof_indices.size());

                      for (auto i : index_range(dof_indices))
                        soln_coeffs[i] = sys_soln[dof_indices[i]];

                      // Compute the FE solution at all the nodes, but
                      // only use the first n_vertices() entries if
                      // vertices_only == true.
                      FEInterface::nodal_soln (dim,
                                               fe_type,
                                               elem,
                                               soln_coeffs,
                                               nodal_soln);

#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
                      // infinite elements should be skipped...
                      if (!elem->infinite())
#endif
                        {
                          libmesh_assert_equal_to (nodal_soln.size(), elem->n_nodes());

                          const unsigned int n_vals =
                            vertices_only ? elem->n_vertices() : elem->n_nodes();

                          for (unsigned int n=0; n<n_vals; n++)
                            {
                              // Compute index into global solution vector.
                              std::size_t index =
                                nv * (nn++) + (n_vars_written_current_system + var_offset);

                              soln[index] += nodal_soln[n];
                            }
                        }
                    }
                  else
                    nn += vertices_only ? elem->n_vertices() : elem->n_nodes();
                } // end loop over active elements

              // If we made it here, we actually wrote a variable, so increment
              // the number of variables actually written for the current system.
              n_vars_written_current_system++;

            } // end loop over vars
        } // end if proc 0

      // Update offset for next loop iteration.
      var_offset += n_vars_written_current_system;
    } // end loop over systems
}



template <typename RealType>
bool EquationSystemsTempl<RealType>::compare (const EquationSystems & other_es,
                               const Real threshold,
                               const bool verbose) const
{
  // safety check, whether we handle at least the same number
  // of systems
  std::vector<bool> os_result;

  if (this->n_systems() != other_es.n_systems())
    {
      if (verbose)
        {
          libMesh::out << "  Fatal difference. This system handles "
                       << this->n_systems() << " systems," << std::endl
                       << "  while the other system handles "
                       << other_es.n_systems()
                       << " systems." << std::endl
                       << "  Aborting comparison." << std::endl;
        }
      return false;
    }
  else
    {
      // start comparing each system
      const_system_iterator       pos = _systems.begin();
      const const_system_iterator end = _systems.end();

      for (; pos != end; ++pos)
        {
          const std::string & sys_name = pos->first;
          const System &  system        = *(pos->second);

          // get the other system
          const System & other_system   = other_es.get_system (sys_name);

          os_result.push_back (system.compare (other_system, threshold, verbose));

        }

    }


  // sum up the results
  if (os_result.size()==0)
    return true;
  else
    {
      bool os_identical;
      unsigned int n = 0;
      do
        {
          os_identical = os_result[n];
          n++;
        }
      while (os_identical && n<os_result.size());
      return os_identical;
    }
}



template <typename RealType>
std::string EquationSystemsTempl<RealType>::get_info () const
{
  std::ostringstream oss;

  oss << " EquationSystems\n"
      << "  n_systems()=" << this->n_systems() << '\n';

  // Print the info for the individual systems
  const_system_iterator       pos = _systems.begin();
  const const_system_iterator end = _systems.end();

  for (; pos != end; ++pos)
    oss << pos->second->get_info();


  //   // Possibly print the parameters
  //   if (!this->parameters.empty())
  //     {
  //       oss << "  n_parameters()=" << this->n_parameters() << '\n';
  //       oss << "   Parameters:\n";

  //       for (const auto & pr : _parameters)
  //         oss << "    "
  //             << "\""
  //             << pr.first
  //             << "\""
  //             << "="
  //             << pr.second
  //             << '\n';
  //     }

  return oss.str();
}



template <typename RealType>
void EquationSystemsTempl<RealType>::print_info (std::ostream & os) const
{
  os << this->get_info()
     << std::endl;
}



template <typename RealType>
std::ostream & operator << (std::ostream & os,
                            const EquationSystems & es)
{
  es.print_info(os);
  return os;
}



template <typename RealType>
unsigned int EquationSystemsTempl<RealType>::n_vars () const
{
  unsigned int tot=0;

  const_system_iterator       pos = _systems.begin();
  const const_system_iterator end = _systems.end();

  for (; pos != end; ++pos)
    tot += pos->second->n_vars();

  return tot;
}



template <typename RealType>
std::size_t EquationSystemsTempl<RealType>::n_dofs () const
{
  std::size_t tot=0;

  const_system_iterator       pos = _systems.begin();
  const const_system_iterator end = _systems.end();

  for (; pos != end; ++pos)
    tot += pos->second->n_dofs();

  return tot;
}




template <typename RealType>
std::size_t EquationSystemsTempl<RealType>::n_active_dofs () const
{
  std::size_t tot=0;

  const_system_iterator       pos = _systems.begin();
  const const_system_iterator end = _systems.end();

  for (; pos != end; ++pos)
    tot += pos->second->n_active_dofs();

  return tot;
}


template <typename RealType>
void EquationSystemsTempl<RealType>::_add_system_to_nodes_and_elems()
{
  // All the nodes
  for (auto & node : _mesh.node_ptr_range())
    node->add_system();

  // All the elements
  for (auto & elem : _mesh.element_ptr_range())
    elem->add_system();
}

template <typename RealType>
void EquationSystemsTempl<RealType>::_remove_default_ghosting(unsigned int sys_num)
{
  this->get_system(sys_num).get_dof_map().remove_default_ghosting();
}

// Anonymous namespace for implementation details.
namespace {
std::string local_file_name (const unsigned int processor_id,
                             const std::string & name)
{
  std::string basename(name);
  char buf[256];

  if (basename.size() - basename.rfind(".bz2") == 4)
    {
      basename.erase(basename.end()-4, basename.end());
      std::sprintf(buf, "%s.%04u.bz2", basename.c_str(), processor_id);
    }
  else if (basename.size() - basename.rfind(".gz") == 3)
    {
      basename.erase(basename.end()-3, basename.end());
      std::sprintf(buf, "%s.%04u.gz", basename.c_str(), processor_id);
    }
  else
    std::sprintf(buf, "%s.%04u", basename.c_str(), processor_id);

  return std::string(buf);
}
}

template <typename RealType>
template <typename InValType>
void EquationSystemsTempl<RealType>::read (const std::string & name,
                            const unsigned int read_flags,
                            bool partition_agnostic)
{
  XdrMODE mode = READ;
  if (name.find(".xdr") != std::string::npos)
    mode = DECODE;
  this->read(name, mode, read_flags, partition_agnostic);

#ifdef LIBMESH_ENABLE_AMR
  MeshRefinement mesh_refine(_mesh);
  mesh_refine.clean_refinement_flags();
#endif
}



template <typename RealType>
template <typename InValType>
void EquationSystemsTempl<RealType>::read (const std::string & name,
                            const XdrMODE mode,
                            const unsigned int read_flags,
                            bool partition_agnostic)
{
  // If we have exceptions enabled we can be considerate and try
  // to read old restart files which contain infinite element
  // information but do not have the " with infinite elements"
  // string in the version information.

  // First try the read the user requested
  libmesh_try
    {
      this->_read_impl<InValType> (name, mode, read_flags, partition_agnostic);
    }

  // If that fails, try it again but explicitly request we look for infinite element info
  libmesh_catch (...)
    {
      libMesh::out << "\n*********************************************************************\n"
                   << "READING THE FILE \"" << name << "\" FAILED.\n"
                   << "It is possible this file contains infinite element information,\n"
                   << "but the version string does not contain \" with infinite elements\"\n"
                   << "Let's try this again, but looking for infinite element information...\n"
                   << "*********************************************************************\n"
                   << std::endl;

      libmesh_try
        {
          this->_read_impl<InValType> (name, mode, read_flags | EquationSystems::TRY_READ_IFEMS, partition_agnostic);
        }

      // If all that failed, we are out of ideas here...
      libmesh_catch (...)
        {
          libMesh::out << "\n*********************************************************************\n"
                       << "Well, at least we tried!\n"
                       << "Good Luck!!\n"
                       << "*********************************************************************\n"
                       << std::endl;
          LIBMESH_THROW();
        }
    }

#ifdef LIBMESH_ENABLE_AMR
  MeshRefinement mesh_refine(_mesh);
  mesh_refine.clean_refinement_flags();
#endif
}



template <typename RealType>
template <typename InValType>
void EquationSystemsTempl<RealType>::_read_impl (const std::string & name,
                                  const XdrMODE mode,
                                  const unsigned int read_flags,
                                  bool partition_agnostic)
{
  /**
   * This program implements the output of an
   * EquationSystems object.  This warrants some
   * documentation.  The output file essentially
   * consists of 11 sections:
   \verbatim
   1.) A version header (for non-'legacy' formats, libMesh-0.7.0 and greater).
   2.) The number of individual equation systems (unsigned int)

   for each system

   3.)  The name of the system (string)
   4.)  The type of the system (string)

   handled through System::read():

   +-------------------------------------------------------------+
   |  5.) The number of variables in the system (unsigned int)   |
   |                                                             |
   |   for each variable in the system                           |
   |                                                             |
   |    6.) The name of the variable (string)                    |
   |                                                             |
   |    7.) Combined in an FEType:                               |
   |         - The approximation order(s) of the variable (Order |
   |           Enum, cast to int/s)                              |
   |         - The finite element family/ies of the variable     |
   |           (FEFamily Enum, cast to int/s)                    |
   |                                                             |
   |   end variable loop                                         |
   |                                                             |
   | 8.) The number of additional vectors (unsigned int),        |
   |                                                             |
   |    for each additional vector in the equation system object |
   |                                                             |
   |    9.) the name of the additional vector  (string)          |
   +-------------------------------------------------------------+

   end system loop


   for each system, handled through System::read_{serialized,parallel}_data():

   +--------------------------------------------------------------+
   | 10.) The global solution vector, re-ordered to be node-major |
   |     (More on this later.)                                    |
   |                                                              |
   |    for each additional vector in the equation system object  |
   |                                                              |
   |    11.) The global additional vector, re-ordered to be       |
   |         node-major (More on this later.)                     |
   +--------------------------------------------------------------+

   end system loop
   \endverbatim
   *
   * Note that the actual IO is handled through the Xdr class
   * (to be renamed later?) which provides a uniform interface to
   * both the XDR (eXternal Data Representation) interface and standard
   * ASCII output.  Thus this one section of code will read XDR or ASCII
   * files with no changes.
   */

  // Set booleans from the read_flags argument
  const bool read_header          = read_flags & EquationSystems::READ_HEADER;
  const bool read_data            = read_flags & EquationSystems::READ_DATA;
  const bool read_additional_data = read_flags & EquationSystems::READ_ADDITIONAL_DATA;
  const bool read_legacy_format   = read_flags & EquationSystems::READ_LEGACY_FORMAT;
  const bool try_read_ifems       = read_flags & EquationSystems::TRY_READ_IFEMS;
  const bool read_basic_only      = read_flags & EquationSystems::READ_BASIC_ONLY;
  bool read_parallel_files  = false;

  std::vector<std::pair<std::string, System *>> xda_systems;

  // This will unzip a file with .bz2 as the extension, otherwise it
  // simply returns the name if the file need not be unzipped.
  Xdr io ((this->processor_id() == 0) ? name : "", mode);
  libmesh_assert (io.reading());

  {
    // 1.)
    // Read the version header.
    std::string version = "legacy";
    if (!read_legacy_format)
      {
        if (this->processor_id() == 0) io.data(version);
        this->comm().broadcast(version);

        // All processors have the version header, if it does not contain
        // the libMesh_label string then it is a legacy file.
        const std::string libMesh_label = "libMesh-";
        std::string::size_type lm_pos = version.find(libMesh_label);
        if (lm_pos==std::string::npos)
          {
            io.close();

            // Recursively call this read() function but with the
            // EquationSystems::READ_LEGACY_FORMAT bit set.
            this->read (name, mode, (read_flags | EquationSystems::READ_LEGACY_FORMAT), partition_agnostic);
            return;
          }

        // Figure out the libMesh version that created this file
        std::istringstream iss(version.substr(lm_pos + libMesh_label.size()));
        int ver_major = 0, ver_minor = 0, ver_patch = 0;
        char dot;
        iss >> ver_major >> dot >> ver_minor >> dot >> ver_patch;
        io.set_version(LIBMESH_VERSION_ID(ver_major, ver_minor, ver_patch));


        read_parallel_files = (version.rfind(" parallel") < version.size());

        // If requested that we try to read infinite element information,
        // and the string " with infinite elements" is not in the version,
        // then tack it on.  This is for compatibility reading ifem
        // files written prior to 11/10/2008 - BSK
        if (try_read_ifems)
          if (!(version.rfind(" with infinite elements") < version.size()))
            version += " with infinite elements";

      }
    else
      libmesh_deprecated();

    START_LOG("read()","EquationSystems");

    // 2.)
    // Read the number of equation systems
    unsigned int n_sys=0;
    if (this->processor_id() == 0) io.data (n_sys);
    this->comm().broadcast(n_sys);

    for (unsigned int sys=0; sys<n_sys; sys++)
      {
        // 3.)
        // Read the name of the sys-th equation system
        std::string sys_name;
        if (this->processor_id() == 0) io.data (sys_name);
        this->comm().broadcast(sys_name);

        // 4.)
        // Read the type of the sys-th equation system
        std::string sys_type;
        if (this->processor_id() == 0) io.data (sys_type);
        this->comm().broadcast(sys_type);

        if (read_header)
          this->add_system (sys_type, sys_name);

        // 5.) - 9.)
        // Let System::read_header() do the job
        System & new_system = this->get_system(sys_name);
        new_system.read_header (io,
                                version,
                                read_header,
                                read_additional_data,
                                read_legacy_format);

        xda_systems.push_back(std::make_pair(sys_name, &new_system));

        // If we're only creating "basic" systems, we need to tell
        // each system that before we call init() later.
        if (read_basic_only)
          new_system.set_basic_system_only();
      }
  }



  // Now we are ready to initialize the underlying data
  // structures. This will initialize the vectors for
  // storage, the dof_map, etc...
  if (read_header)
    this->init();

  // 10.) & 11.)
  // Read and set the numeric vector values
  if (read_data)
    {
      // the EquationSystems::read() method should look constant from the mesh
      // perspective, but we need to assign a temporary numbering to the nodes
      // and elements in the mesh, which requires that we abuse const_cast
      if (!read_legacy_format && partition_agnostic)
        {
          MeshBase & mesh = const_cast<MeshBase &>(this->get_mesh());
          MeshTools::Private::globally_renumber_nodes_and_elements(mesh);
        }

      Xdr local_io (read_parallel_files ? local_file_name(this->processor_id(),name) : "", mode);

      for (auto & pr : xda_systems)
        if (read_legacy_format)
          {
            libmesh_deprecated();
#ifdef LIBMESH_ENABLE_DEPRECATED
            pr.second->read_legacy_data (io, read_additional_data);
#endif
          }
        else
          if (read_parallel_files)
            pr.second->read_parallel_data<InValType>   (local_io, read_additional_data);
          else
            pr.second->read_serialized_data<InValType> (io, read_additional_data);


      // Undo the temporary numbering.
      if (!read_legacy_format && partition_agnostic)
        _mesh.fix_broken_node_and_element_numbering();
    }

  STOP_LOG("read()","EquationSystems");

  // Localize each system's data
  this->update();
}



template <typename RealType>
void EquationSystemsTempl<RealType>::write(const std::string & name,
                            const unsigned int write_flags,
                            bool partition_agnostic) const
{
  XdrMODE mode = WRITE;
  if (name.find(".xdr") != std::string::npos)
    mode = ENCODE;
  this->write(name, mode, write_flags, partition_agnostic);
}



template <typename RealType>
void EquationSystemsTempl<RealType>::write(const std::string & name,
                            const XdrMODE mode,
                            const unsigned int write_flags,
                            bool partition_agnostic) const
{
  /**
   * This program implements the output of an
   * EquationSystems object.  This warrants some
   * documentation.  The output file essentially
   * consists of 11 sections:
   \verbatim
   1.) The version header.
   2.) The number of individual equation systems (unsigned int)

   for each system

   3.)  The name of the system (string)
   4.)  The type of the system (string)

   handled through System::read():

   +-------------------------------------------------------------+
   |  5.) The number of variables in the system (unsigned int)   |
   |                                                             |
   |   for each variable in the system                           |
   |                                                             |
   |    6.) The name of the variable (string)                    |
   |                                                             |
   |    7.) Combined in an FEType:                               |
   |         - The approximation order(s) of the variable (Order |
   |           Enum, cast to int/s)                              |
   |         - The finite element family/ies of the variable     |
   |           (FEFamily Enum, cast to int/s)                    |
   |                                                             |
   |   end variable loop                                         |
   |                                                             |
   | 8.) The number of additional vectors (unsigned int),        |
   |                                                             |
   |    for each additional vector in the equation system object |
   |                                                             |
   |    9.) the name of the additional vector  (string)          |
   +-------------------------------------------------------------+

   end system loop


   for each system, handled through System::write_{serialized,parallel}_data():

   +--------------------------------------------------------------+
   | 10.) The global solution vector, re-ordered to be node-major |
   |     (More on this later.)                                    |
   |                                                              |
   |    for each additional vector in the equation system object  |
   |                                                              |
   |    11.) The global additional vector, re-ordered to be       |
   |         node-major (More on this later.)                     |
   +--------------------------------------------------------------+

   end system loop
   \endverbatim
   *
   * Note that the actual IO is handled through the Xdr class
   * (to be renamed later?) which provides a uniform interface to
   * both the XDR (eXternal Data Representation) interface and standard
   * ASCII output.  Thus this one section of code will write XDR or ASCII
   * files with no changes.
   */

  // the EquationSystems::write() method should look constant,
  // but we need to assign a temporary numbering to the nodes
  // and elements in the mesh, which requires that we abuse const_cast
  if (partition_agnostic)
    {
      MeshBase & mesh = const_cast<MeshBase &>(this->get_mesh());
      MeshTools::Private::globally_renumber_nodes_and_elements(mesh);
    }

  // set booleans from write_flags argument
  const bool write_data            = write_flags & EquationSystems::WRITE_DATA;
  const bool write_additional_data = write_flags & EquationSystems::WRITE_ADDITIONAL_DATA;

  // always write parallel files if we're instructed to write in
  // parallel
  const bool write_parallel_files  =
    (write_flags & EquationSystems::WRITE_PARALLEL_FILES)
    // Even if we're on a distributed mesh, we may or may not have a
    // consistent way of reconstructing the same mesh partitioning
    // later, but we need the same mesh partitioning if we want to
    // reread the parallel solution safely, so let's write a serial file
    // unless specifically requested not to.
    // ||
    // // but also write parallel files if we haven't been instructed to
    // // write in serial and we're on a distributed mesh
    // (!(write_flags & EquationSystems::WRITE_SERIAL_FILES) &&
    // !this->get_mesh().is_serial())
    ;

  // New scope so that io will close before we try to zip the file
  {
    Xdr io((this->processor_id()==0) ? name : "", mode);
    libmesh_assert (io.writing());

    LOG_SCOPE("write()", "EquationSystems");

    const unsigned int proc_id = this->processor_id();

    unsigned int n_sys = 0;
    for (auto & pr : _systems)
      if (!pr.second->hide_output())
        n_sys++;

    // set the version number in the Xdr object
    io.set_version(LIBMESH_VERSION_ID(LIBMESH_MAJOR_VERSION,
                                      LIBMESH_MINOR_VERSION,
                                      LIBMESH_MICRO_VERSION));

    // Only write the header information
    // if we are processor 0.
    if (proc_id == 0)
      {
        std::string comment;
        char buf[256];

        // 1.)
        // Write the version header
        std::string version("libMesh-" + libMesh::get_io_compatibility_version());
        if (write_parallel_files) version += " parallel";

#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
        version += " with infinite elements";
#endif
        io.data (version, "# File Format Identifier");

        // 2.)
        // Write the number of equation systems
        io.data (n_sys, "# No. of Equation Systems");

        for (auto & pr : _systems)
          {
            // Ignore this system if it has been marked as hidden
            if (pr.second->hide_output()) continue;

            // 3.)
            // Write the name of the sys_num-th system
            {
              const unsigned int sys_num = pr.second->number();
              std::string sys_name       = pr.first;

              comment =  "# Name, System No. ";
              std::sprintf(buf, "%u", sys_num);
              comment += buf;

              io.data (sys_name, comment.c_str());
            }

            // 4.)
            // Write the type of system handled
            {
              const unsigned int sys_num = pr.second->number();
              std::string sys_type       = pr.second->system_type();

              comment =  "# Type, System No. ";
              std::sprintf(buf, "%u", sys_num);
              comment += buf;

              io.data (sys_type, comment.c_str());
            }

            // 5.) - 9.)
            // Let System::write_header() do the job
            pr.second->write_header (io, version, write_additional_data);
          }
      }

    // Start from the first system, again,
    // to write vectors to disk, if wanted
    if (write_data)
      {
        // open a parallel buffer if warranted.
        Xdr local_io (write_parallel_files ? local_file_name(this->processor_id(),name) : "", mode);

        for (auto & pr : _systems)
          {
            // Ignore this system if it has been marked as hidden
            if (pr.second->hide_output()) continue;

            // 10.) + 11.)
            if (write_parallel_files)
              pr.second->write_parallel_data (local_io,write_additional_data);
            else
              pr.second->write_serialized_data (io,write_additional_data);
          }
      }
  }

  // the EquationSystems::write() method should look constant,
  // but we need to undo the temporary numbering of the nodes
  // and elements in the mesh, which requires that we abuse const_cast
  if (partition_agnostic)
    const_cast<MeshBase &>(_mesh).fix_broken_node_and_element_numbering();
}



// template instantiations

#ifdef LIBMESH_USE_COMPLEX_NUMBERS
#define ES_IO_INSTANTIATE(RealType)                                                                \
  template void EquationSystemsTempl<RealType>::read<Number>(                                      \
      const std::string & name, const unsigned int read_flags, bool partition_agnostic);           \
  template void EquationSystemsTempl<RealType>::read<Number>(const std::string & name,             \
                                                             const XdrMODE mode,                   \
                                                             const unsigned int read_flags,        \
                                                             bool partition_agnostic);             \
  template void EquationSystemsTempl<RealType>::_read_impl<Number>(const std::string & name,       \
                                                                   const XdrMODE mode,             \
                                                                   const unsigned int read_flags,  \
                                                                   bool partition_agnostic);       \
  template void EquationSystemsTempl<RealType>::read<Real>(                                        \
      const std::string & name, const unsigned int read_flags, bool partition_agnostic);           \
  template void EquationSystemsTempl<RealType>::read<Real>(const std::string & name,               \
                                                           const XdrMODE mode,                     \
                                                           const unsigned int read_flags,          \
                                                           bool partition_agnostic);               \
  template void EquationSystemsTempl<RealType>::_read_impl<Real>(const std::string & name,         \
                                                                 const XdrMODE mode,               \
                                                                 const unsigned int read_flags,    \
                                                                 bool partition_agnostic)
#else
#define ES_IO_INSTANTIATE(RealType)                                                                \
  template void EquationSystemsTempl<RealType>::read<Number>(                                      \
      const std::string & name, const unsigned int read_flags, bool partition_agnostic);           \
  template void EquationSystemsTempl<RealType>::read<Number>(const std::string & name,             \
                                                             const XdrMODE mode,                   \
                                                             const unsigned int read_flags,        \
                                                             bool partition_agnostic);             \
  template void EquationSystemsTempl<RealType>::_read_impl<Number>(const std::string & name,       \
                                                                   const XdrMODE mode,             \
                                                                   const unsigned int read_flags,  \
                                                                   bool partition_agnostic)
#endif // LIBMESH_USE_COMPLEX_NUMBERS

} // namespace libMesh

#endif // LIBMESH_EQUATION_SYSTEMS_IMPL_H
