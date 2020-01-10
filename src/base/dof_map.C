// The libMesh Finite Element Library.
// Copyright (C) 2002-2019 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public  License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA



// Local includes
#include "libmesh/dof_map_impl.h"

// libMesh includes
#include "libmesh/coupling_matrix.h"
#include "libmesh/default_coupling.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector_base.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/elem.h"
#include "libmesh/enum_to_string.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_type.h"
#include "libmesh/fe_base.h" // FEBase::build() for continuity test
#include "libmesh/ghosting_functor.h"
#include "libmesh/int_range.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/mesh_base.h"
#include "libmesh/mesh_subdivision_support.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/periodic_boundaries.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/sparsity_pattern.h"
#include "libmesh/threads.h"
#include "libmesh/auto_ptr.h" // libmesh_make_unique

// TIMPI includes
#include "timpi/parallel_implementation.h"
#include "timpi/parallel_sync.h"

// C++ Includes
#include <set>
#include <algorithm> // for std::fill, std::equal_range, std::max, std::lower_bound, etc.
#include <sstream>
#include <unordered_map>

namespace libMesh
{

// ------------------------------------------------------------
// DofMap member functions

// Destructor
DofMap::~DofMap()
{
  this->clear();

  // clear() resets all but the default DofMap-based functors.  We
  // need to remove those from the mesh too before we die.
  _mesh.remove_ghosting_functor(*_default_coupling);
  _mesh.remove_ghosting_functor(*_default_evaluating);

#ifdef LIBMESH_ENABLE_DIRICHLET
  for (auto & bnd : _adjoint_dirichlet_boundaries)
    delete bnd;
#endif
}


#ifdef LIBMESH_ENABLE_PERIODIC

bool DofMap::is_periodic_boundary (const boundary_id_type boundaryid) const
{
  if (_periodic_boundaries->count(boundaryid) != 0)
    return true;

  return false;
}

#endif



// void DofMap::add_variable (const Variable & var)
// {
//   libmesh_not_implemented();
//   _variables.push_back (var);
// }


void DofMap::set_error_on_cyclic_constraint(bool error_on_cyclic_constraint)
{
  // This function will eventually be officially libmesh_deprecated();
  // Call DofMap::set_error_on_constraint_loop() instead.
  set_error_on_constraint_loop(error_on_cyclic_constraint);
}

void DofMap::set_error_on_constraint_loop(bool error_on_constraint_loop)
{
  _error_on_constraint_loop = error_on_constraint_loop;
}



void DofMap::add_variable_group (const VariableGroup & var_group)
{
  const unsigned int vg = cast_int<unsigned int>(_variable_groups.size());

  _variable_groups.push_back(var_group);

  VariableGroup & new_var_group = _variable_groups.back();

  for (auto var : IntRange<unsigned int>(0, new_var_group.n_variables()))
    {
      _variables.push_back (new_var_group(var));
      _variable_group_numbers.push_back (vg);
    }
}



void DofMap::attach_matrix (SparseMatrix<Number> & matrix)
{
  parallel_object_only();

  // We shouldn't be trying to re-attach the same matrices repeatedly
  libmesh_assert (std::find(_matrices.begin(), _matrices.end(),
                            &matrix) == _matrices.end());

  _matrices.push_back(&matrix);

  matrix.attach_dof_map (*this);

  // If we've already computed sparsity, then it's too late
  // to wait for "compute_sparsity" to help with sparse matrix
  // initialization, and we need to handle this matrix individually
  bool computed_sparsity_already =
    ((_n_nz && !_n_nz->empty()) ||
     (_n_oz && !_n_oz->empty()));
  this->comm().max(computed_sparsity_already);
  if (computed_sparsity_already &&
      matrix.need_full_sparsity_pattern())
    {
      // We'd better have already computed the full sparsity pattern
      // if we need it here
      libmesh_assert(need_full_sparsity_pattern);
      libmesh_assert(_sp.get());

      matrix.update_sparsity_pattern (_sp->sparsity_pattern);
    }

  if (matrix.need_full_sparsity_pattern())
    need_full_sparsity_pattern = true;
}



bool DofMap::is_attached (SparseMatrix<Number> & matrix)
{
  return (std::find(_matrices.begin(), _matrices.end(),
                    &matrix) != _matrices.end());
}


void DofMap::clear()
{
  // we don't want to clear
  // the coupling matrix!
  // It should not change...
  //_dof_coupling->clear();
  //
  // But it would be inconsistent to leave our coupling settings
  // through a clear()...
  _dof_coupling = nullptr;

  // Reset ghosting functor statuses
  {
    for (const auto & gf : _coupling_functors)
      {
        libmesh_assert(gf);
        _mesh.remove_ghosting_functor(*gf);
      }
    this->_coupling_functors.clear();

    // Go back to default coupling

    _default_coupling->set_dof_coupling(this->_dof_coupling);
    _default_coupling->set_n_levels(this->use_coupled_neighbor_dofs(this->_mesh));

    this->add_coupling_functor(*_default_coupling);
  }


  {
    for (const auto & gf : _algebraic_ghosting_functors)
      {
        libmesh_assert(gf);
        _mesh.remove_ghosting_functor(*gf);
      }
    this->_algebraic_ghosting_functors.clear();

    // Go back to default send_list generation

    // _default_evaluating->set_dof_coupling(this->_dof_coupling);
    _default_evaluating->set_n_levels(1);
    this->add_algebraic_ghosting_functor(*_default_evaluating);
  }

  this->_shared_functors.clear();

  _variables.clear();
  _variable_groups.clear();
  _variable_group_numbers.clear();
  _first_df.clear();
  _end_df.clear();
  _first_scalar_df.clear();
  this->clear_send_list();
  this->clear_sparsity();
  need_full_sparsity_pattern = false;

#ifdef LIBMESH_ENABLE_AMR

  _dof_constraints.clear();
  _stashed_dof_constraints.clear();
  _primal_constraint_values.clear();
  _adjoint_constraint_values.clear();
  _n_old_dfs = 0;
  _first_old_df.clear();
  _end_old_df.clear();
  _first_old_scalar_df.clear();

#endif

  _matrices.clear();

  _n_dfs = 0;
}


void DofMap::prepare_send_list ()
{
  LOG_SCOPE("prepare_send_list()", "DofMap");

  // Return immediately if there's no ghost data
  if (this->n_processors() == 1)
    return;

  // Check to see if we have any extra stuff to add to the send_list
  if (_extra_send_list_function)
    {
      if (_augment_send_list)
        {
          libmesh_here();
          libMesh::out << "WARNING:  You have specified both an extra send list function and object.\n"
                       << "          Are you sure this is what you meant to do??"
                       << std::endl;
        }

      _extra_send_list_function(_send_list, _extra_send_list_context);
    }

  if (_augment_send_list)
    _augment_send_list->augment_send_list (_send_list);

  // First sort the send list.  After this
  // duplicated elements will be adjacent in the
  // vector
  std::sort(_send_list.begin(), _send_list.end());

  // Now use std::unique to remove duplicate entries
  std::vector<dof_id_type>::iterator new_end =
    std::unique (_send_list.begin(), _send_list.end());

  // Remove the end of the send_list.  Use the "swap trick"
  // from Effective STL
  std::vector<dof_id_type> (_send_list.begin(), new_end).swap (_send_list);
}

void DofMap::set_implicit_neighbor_dofs(bool implicit_neighbor_dofs)
{
  _implicit_neighbor_dofs_initialized = true;
  _implicit_neighbor_dofs = implicit_neighbor_dofs;
}


bool DofMap::use_coupled_neighbor_dofs(const MeshAbstract & mesh) const
{
  // If we were asked on the command line, then we need to
  // include sensitivities between neighbor degrees of freedom
  bool implicit_neighbor_dofs =
    libMesh::on_command_line ("--implicit-neighbor-dofs");

  // If the user specifies --implicit-neighbor-dofs 0, then
  // presumably he knows what he is doing and we won't try to
  // automatically turn it on even when all the variables are
  // discontinuous.
  if (implicit_neighbor_dofs)
    {
      // No flag provided defaults to 'true'
      int flag = 1;
      flag = libMesh::command_line_next ("--implicit-neighbor-dofs", flag);

      if (!flag)
        {
          // The user said --implicit-neighbor-dofs 0, so he knows
          // what he is doing and really doesn't want it.
          return false;
        }
    }

  // Possibly override the commandline option, if set_implicit_neighbor_dofs
  // has been called.
  if (_implicit_neighbor_dofs_initialized)
    {
      implicit_neighbor_dofs = _implicit_neighbor_dofs;

      // Again, if the user explicitly says implicit_neighbor_dofs = false,
      // then we return here.
      if (!implicit_neighbor_dofs)
        return false;
    }

  // Look at all the variables in this system.  If every one is
  // discontinuous then the user must be doing DG/FVM, so be nice
  // and force implicit_neighbor_dofs=true.
  {
    bool all_discontinuous_dofs = true;

    for (auto var : IntRange<unsigned int>(0, this->n_variables()))
      if (FEAbstract<>::build (mesh.mesh_dimension(),
                             this->variable_type(var))->get_continuity() !=  DISCONTINUOUS)
        all_discontinuous_dofs = false;

    if (all_discontinuous_dofs)
      implicit_neighbor_dofs = true;
  }

  return implicit_neighbor_dofs;
}


void DofMap::clear_sparsity()
{
  if (need_full_sparsity_pattern)
    {
      libmesh_assert(_sp.get());
      libmesh_assert(!_n_nz || _n_nz == &_sp->n_nz);
      libmesh_assert(!_n_oz || _n_oz == &_sp->n_oz);
      _sp.reset();
    }
  else
    {
      libmesh_assert(!_sp.get());
      delete _n_nz;
      delete _n_oz;
    }
  _n_nz = nullptr;
  _n_oz = nullptr;
}



void DofMap::remove_default_ghosting()
{
  this->remove_coupling_functor(this->default_coupling());
  this->remove_algebraic_ghosting_functor(this->default_algebraic_ghosting());
}



void DofMap::add_default_ghosting()
{
  this->add_coupling_functor(this->default_coupling());
  this->add_algebraic_ghosting_functor(this->default_algebraic_ghosting());
}



void
DofMap::add_coupling_functor(GhostingFunctorBase & coupling_functor,
                             bool to_mesh)
{
  _coupling_functors.insert(&coupling_functor);
  if (to_mesh)
    _mesh.add_ghosting_functor(coupling_functor);
}



void
DofMap::remove_coupling_functor(GhostingFunctorBase & coupling_functor)
{
  _coupling_functors.erase(&coupling_functor);
  _mesh.remove_ghosting_functor(coupling_functor);

  auto it = _shared_functors.find(&coupling_functor);
  if (it != _shared_functors.end())
    _shared_functors.erase(it);
}



void
DofMap::add_algebraic_ghosting_functor(GhostingFunctorBase & evaluable_functor,
                                       bool to_mesh)
{
  _algebraic_ghosting_functors.insert(&evaluable_functor);
  if (to_mesh)
    _mesh.add_ghosting_functor(evaluable_functor);
}



void
DofMap::remove_algebraic_ghosting_functor(GhostingFunctorBase & evaluable_functor)
{
  _algebraic_ghosting_functors.erase(&evaluable_functor);
  _mesh.remove_ghosting_functor(evaluable_functor);

  auto it = _shared_functors.find(&evaluable_functor);
  if (it != _shared_functors.end())
    _shared_functors.erase(it);
}



void DofMap::extract_local_vector (const NumericVector<Number> & Ug,
                                   const std::vector<dof_id_type> & dof_indices_in,
                                   DenseVectorBase<Number> & Ue) const
{
  const unsigned int n_original_dofs = dof_indices_in.size();

#ifdef LIBMESH_ENABLE_AMR

  // Trivial mapping
  libmesh_assert_equal_to (dof_indices_in.size(), Ue.size());
  bool has_constrained_dofs = false;

  for (unsigned int il=0; il != n_original_dofs; ++il)
    {
      const dof_id_type ig = dof_indices_in[il];

      if (this->is_constrained_dof (ig)) has_constrained_dofs = true;

      libmesh_assert_less (ig, Ug.size());

      Ue.el(il) = Ug(ig);
    }

  // If the element has any constrained DOFs then we need
  // to account for them in the mapping.  This will handle
  // the case that the input vector is not constrained.
  if (has_constrained_dofs)
    {
      // Copy the input DOF indices.
      std::vector<dof_id_type> constrained_dof_indices(dof_indices_in);

      DenseMatrix<Number> C;
      DenseVector<Number> H;

      this->build_constraint_matrix_and_vector (C, H, constrained_dof_indices);

      libmesh_assert_equal_to (dof_indices_in.size(), C.m());
      libmesh_assert_equal_to (constrained_dof_indices.size(), C.n());

      // zero-out Ue
      Ue.zero();

      // compute Ue = C Ug, with proper mapping.
      for (unsigned int i=0; i != n_original_dofs; i++)
        {
          Ue.el(i) = H(i);

          const unsigned int n_constrained =
            cast_int<unsigned int>(constrained_dof_indices.size());
          for (unsigned int j=0; j<n_constrained; j++)
            {
              const dof_id_type jg = constrained_dof_indices[j];

              //          If Ug is a serial or ghosted vector, then this assert is
              //          overzealous.  If Ug is a parallel vector, then this assert
              //          is redundant.
              //    libmesh_assert ((jg >= Ug.first_local_index()) &&
              //    (jg <  Ug.last_local_index()));

              Ue.el(i) += C(i,j)*Ug(jg);
            }
        }
    }

#else

  // Trivial mapping

  libmesh_assert_equal_to (n_original_dofs, Ue.size());

  for (unsigned int il=0; il<n_original_dofs; il++)
    {
      const dof_id_type ig = dof_indices_in[il];

      libmesh_assert ((ig >= Ug.first_local_index()) && (ig <  Ug.last_local_index()));

      Ue.el(il) = Ug(ig);
    }

#endif
}

void DofMap::dof_indices (const Elem * const elem,
                          std::vector<dof_id_type> & di) const
{
  // We now allow elem==nullptr to request just SCALAR dofs
  // libmesh_assert(elem);

  // If we are asking for current indices on an element, it ought to
  // be an active element (or a Side proxy, which also thinks it's
  // active)
  libmesh_assert(!elem || elem->active());

  LOG_SCOPE("dof_indices()", "DofMap");

  // Clear the DOF indices vector
  di.clear();

  const unsigned int n_var_groups  = this->n_variable_groups();

#ifdef DEBUG
  // Check that sizes match in DEBUG mode
  std::size_t tot_size = 0;
#endif

  if (elem && elem->type() == TRI3SUBDIVISION)
    {
      // Subdivision surface FE require the 1-ring around elem
      const Tri3Subdivision * sd_elem = static_cast<const Tri3Subdivision *>(elem);

      // Ghost subdivision elements have no real dofs
      if (!sd_elem->is_ghost())
        {
          // Determine the nodes contributing to element elem
          std::vector<const Node *> elem_nodes;
          MeshTools::Subdivision::find_one_ring(sd_elem, elem_nodes);

          // Get the dof numbers
          for (unsigned int vg=0; vg<n_var_groups; vg++)
            {
              const VariableGroup & var = this->variable_group(vg);
              const unsigned int vars_in_group = var.n_variables();

              if (var.type().family == SCALAR &&
                  var.active_on_subdomain(elem->subdomain_id()))
                {
                  for (unsigned int vig=0; vig != vars_in_group; ++vig)
                    {
#ifdef DEBUG
                      tot_size += var.type().order;
#endif
                      std::vector<dof_id_type> di_new;
                      this->SCALAR_dof_indices(di_new,var.number(vig));
                      di.insert( di.end(), di_new.begin(), di_new.end());
                    }
                }
              else
                for (unsigned int vig=0; vig != vars_in_group; ++vig)
                  {
                    _dof_indices(*elem, elem->p_level(), di, vg, vig,
                                 elem_nodes.data(),
                                 cast_int<unsigned int>(elem_nodes.size())
#ifdef DEBUG
                                 , var.number(vig), tot_size
#endif
                                 );
                  }
            }
        }

      return;
    }

  // Get the dof numbers for each variable
  const unsigned int n_nodes = elem ? elem->n_nodes() : 0;
  for (unsigned int vg=0; vg<n_var_groups; vg++)
    {
      const VariableGroup & var = this->variable_group(vg);
      const unsigned int vars_in_group = var.n_variables();

      if (var.type().family == SCALAR &&
          (!elem ||
           var.active_on_subdomain(elem->subdomain_id())))
        {
          for (unsigned int vig=0; vig != vars_in_group; ++vig)
            {
#ifdef DEBUG
              tot_size += var.type().order;
#endif
              std::vector<dof_id_type> di_new;
              this->SCALAR_dof_indices(di_new,var.number(vig));
              di.insert( di.end(), di_new.begin(), di_new.end());
            }
        }
      else if (elem)
        for (unsigned int vig=0; vig != vars_in_group; ++vig)
          {
            _dof_indices(*elem, elem->p_level(), di, vg, vig,
                         elem->get_nodes(), n_nodes
#ifdef DEBUG
                         , var.number(vig), tot_size
#endif
                     );
          }
    }

#ifdef DEBUG
  libmesh_assert_equal_to (tot_size, di.size());
#endif
}


void DofMap::dof_indices (const Elem * const elem,
                          std::vector<dof_id_type> & di,
                          const unsigned int vn,
                          int p_level) const
{
  // We now allow elem==nullptr to request just SCALAR dofs
  // libmesh_assert(elem);

  LOG_SCOPE("dof_indices()", "DofMap");

  // Clear the DOF indices vector
  di.clear();

  // Use the default p refinement level?
  if (p_level == -12345)
    p_level = elem ? elem->p_level() : 0;

  const unsigned int vg = this->_variable_group_numbers[vn];
  const VariableGroup & var = this->variable_group(vg);
  const unsigned int vig = vn - var.number();

#ifdef DEBUG
  // Check that sizes match in DEBUG mode
  std::size_t tot_size = 0;
#endif

  if (elem && elem->type() == TRI3SUBDIVISION)
    {
      // Subdivision surface FE require the 1-ring around elem
      const Tri3Subdivision * sd_elem = static_cast<const Tri3Subdivision *>(elem);

      // Ghost subdivision elements have no real dofs
      if (!sd_elem->is_ghost())
        {
          // Determine the nodes contributing to element elem
          std::vector<const Node *> elem_nodes;
          MeshTools::Subdivision::find_one_ring(sd_elem, elem_nodes);

          _dof_indices(*elem, p_level, di, vg, vig, elem_nodes.data(),
                       cast_int<unsigned int>(elem_nodes.size())
#ifdef DEBUG
                       , vn, tot_size
#endif
                       );
        }

      return;
    }

  // Get the dof numbers
  if (var.type().family == SCALAR &&
      (!elem ||
       var.active_on_subdomain(elem->subdomain_id())))
    {
#ifdef DEBUG
      tot_size += var.type().order;
#endif
      std::vector<dof_id_type> di_new;
      this->SCALAR_dof_indices(di_new,vn);
      di.insert( di.end(), di_new.begin(), di_new.end());
    }
  else if (elem)
    _dof_indices(*elem, p_level, di, vg, vig, elem->get_nodes(),
                 elem->n_nodes()
#ifdef DEBUG
                 , vn, tot_size
#endif
                 );

#ifdef DEBUG
  libmesh_assert_equal_to (tot_size, di.size());
#endif
}


void DofMap::dof_indices (const Node * const node,
                          std::vector<dof_id_type> & di) const
{
  // We allow node==nullptr to request just SCALAR dofs
  // libmesh_assert(elem);

  LOG_SCOPE("dof_indices(Node)", "DofMap");

  // Clear the DOF indices vector
  di.clear();

  const unsigned int n_var_groups  = this->n_variable_groups();
  const unsigned int sys_num = this->sys_number();

  // Get the dof numbers
  for (unsigned int vg=0; vg<n_var_groups; vg++)
    {
      const VariableGroup & var = this->variable_group(vg);
      const unsigned int vars_in_group = var.n_variables();

      if (var.type().family == SCALAR)
        {
          for (unsigned int vig=0; vig != vars_in_group; ++vig)
            {
              std::vector<dof_id_type> di_new;
              this->SCALAR_dof_indices(di_new,var.number(vig));
              di.insert( di.end(), di_new.begin(), di_new.end());
            }
        }
      else
        {
          const int n_comp = node->n_comp_group(sys_num,vg);
          for (unsigned int vig=0; vig != vars_in_group; ++vig)
            {
              for (int i=0; i != n_comp; ++i)
                {
                  const dof_id_type d =
                    node->dof_number(sys_num, vg, vig, i, n_comp);
                  libmesh_assert_not_equal_to
                    (d, DofObject::invalid_id);
                  di.push_back(d);
                }
            }
        }
    }
}


void DofMap::dof_indices (const Node * const node,
                          std::vector<dof_id_type> & di,
                          const unsigned int vn) const
{
  if (vn == libMesh::invalid_uint)
    {
      this->dof_indices(node, di);
      return;
    }

  // We allow node==nullptr to request just SCALAR dofs
  // libmesh_assert(elem);

  LOG_SCOPE("dof_indices(Node)", "DofMap");

  // Clear the DOF indices vector
  di.clear();

  const unsigned int sys_num = this->sys_number();

  // Get the dof numbers
  const unsigned int vg = this->_variable_group_numbers[vn];
  const VariableGroup & var = this->variable_group(vg);

  if (var.type().family == SCALAR)
    {
      std::vector<dof_id_type> di_new;
      this->SCALAR_dof_indices(di_new,vn);
      di.insert( di.end(), di_new.begin(), di_new.end());
    }
  else
    {
      const unsigned int vig = vn - var.number();
      const int n_comp = node->n_comp_group(sys_num,vg);
      for (int i=0; i != n_comp; ++i)
        {
          const dof_id_type d =
            node->dof_number(sys_num, vg, vig, i, n_comp);
          libmesh_assert_not_equal_to
            (d, DofObject::invalid_id);
          di.push_back(d);
        }
    }
}


void DofMap::dof_indices (const Elem & elem,
                          unsigned int n,
                          std::vector<dof_id_type> & di,
                          const unsigned int vn) const
{
  this->_node_dof_indices(elem, n, elem.node_ref(n), di, vn);
}



#ifdef LIBMESH_ENABLE_AMR

void DofMap::old_dof_indices (const Elem & elem,
                              unsigned int n,
                              std::vector<dof_id_type> & di,
                              const unsigned int vn) const
{
  const DofObject * old_obj = elem.node_ref(n).old_dof_object;
  libmesh_assert(old_obj);
  this->_node_dof_indices(elem, n, *old_obj, di, vn);
}

#endif // LIBMESH_ENABLE_AMR



void DofMap::_node_dof_indices (const Elem & elem,
                                unsigned int n,
                                const DofObject & obj,
                                std::vector<dof_id_type> & di,
                                const unsigned int vn) const
{
  // Half of this is a cut and paste of _dof_indices code below, but
  // duplication actually seems cleaner than creating a helper
  // function with a million arguments and hoping the compiler inlines
  // it properly into one of our most highly trafficked functions.

  LOG_SCOPE("_node_dof_indices()", "DofMap");

  const ElemType type = elem.type();
  const unsigned int dim = elem.dim();

  const unsigned int sys_num = this->sys_number();
  const std::pair<unsigned int, unsigned int>
    vg_and_offset = obj.var_to_vg_and_offset(sys_num,vn);
  const unsigned int vg = vg_and_offset.first;
  const unsigned int vig = vg_and_offset.second;
  const unsigned int n_comp = obj.n_comp_group(sys_num,vg);

  const VariableGroup & var = this->variable_group(vg);
  FEType fe_type = var.type();
  fe_type.order = static_cast<Order>(fe_type.order +
                                     elem.p_level());
  const bool extra_hanging_dofs =
    FEInterface::extra_hanging_dofs(fe_type);

  // There is a potential problem with h refinement.  Imagine a
  // quad9 that has a linear FE on it.  Then, on the hanging side,
  // it can falsely identify a DOF at the mid-edge node. This is why
  // we go through FEInterface instead of obj->n_comp() directly.
  const unsigned int nc =
    FEInterface::n_dofs_at_node(dim, fe_type, type, n);

  // If this is a non-vertex on a hanging node with extra
  // degrees of freedom, we use the non-vertex dofs (which
  // come in reverse order starting from the end, to
  // simplify p refinement)
  if (extra_hanging_dofs && nc && !elem.is_vertex(n))
    {
      const int dof_offset = n_comp - nc;

      // We should never have fewer dofs than necessary on a
      // node unless we're getting indices on a parent element,
      // and we should never need the indices on such a node
      if (dof_offset < 0)
        {
          libmesh_assert(!elem.active());
          di.resize(di.size() + nc, DofObject::invalid_id);
        }
      else
        for (unsigned int i = dof_offset; i != n_comp; ++i)
          {
            const dof_id_type d =
              obj.dof_number(sys_num, vg, vig, i, n_comp);
            libmesh_assert_not_equal_to (d, DofObject::invalid_id);
            di.push_back(d);
          }
    }
  // If this is a vertex or an element without extra hanging
  // dofs, our dofs come in forward order coming from the
  // beginning
  else
    for (unsigned int i=0; i<nc; i++)
      {
        const dof_id_type d =
          obj.dof_number(sys_num, vg, vig, i, n_comp);
        libmesh_assert_not_equal_to (d, DofObject::invalid_id);
        di.push_back(d);
      }
}



void DofMap::_dof_indices (const Elem & elem,
                           int p_level,
                           std::vector<dof_id_type> & di,
                           const unsigned int vg,
                           const unsigned int vig,
                           const Node * const * nodes,
                           unsigned int       n_nodes
#ifdef DEBUG
                           ,
                           const unsigned int v,
                           std::size_t & tot_size
#endif
                           ) const
{
  const VariableGroup & var = this->variable_group(vg);

  if (var.active_on_subdomain(elem.subdomain_id()))
    {
      const ElemType type        = elem.type();
      const unsigned int sys_num = this->sys_number();
      const unsigned int dim     = elem.dim();
#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
      const bool is_inf          = elem.infinite();
#endif

      // Increase the polynomial order on p refined elements
      FEType fe_type = var.type();
      fe_type.order = static_cast<Order>(fe_type.order + p_level);

      const bool extra_hanging_dofs =
        FEInterface::extra_hanging_dofs(fe_type);

#ifdef DEBUG
      // The number of dofs per element is non-static for subdivision FE
      if (fe_type.family == SUBDIVISION)
        tot_size += n_nodes;
      else
        tot_size += FEInterface::n_dofs(dim,fe_type,type);
#endif

      const FEInterface::n_dofs_at_node_ptr ndan =
        FEInterface::n_dofs_at_node_function(dim, fe_type);

      // Get the node-based DOF numbers
      for (unsigned int n=0; n != n_nodes; n++)
        {
          const Node & node = *nodes[n];

          // Cache the intermediate lookups that are common to every
          // component
#ifdef DEBUG
          const std::pair<unsigned int, unsigned int>
            vg_and_offset = node.var_to_vg_and_offset(sys_num,v);
          libmesh_assert_equal_to (vg, vg_and_offset.first);
          libmesh_assert_equal_to (vig, vg_and_offset.second);
#endif
          const unsigned int n_comp = node.n_comp_group(sys_num,vg);

          // There is a potential problem with h refinement.  Imagine a
          // quad9 that has a linear FE on it.  Then, on the hanging side,
          // it can falsely identify a DOF at the mid-edge node. This is why
          // we go through FEInterface instead of node.n_comp() directly.
          const unsigned int nc =
#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
            is_inf ?
            FEInterface::n_dofs_at_node(dim, fe_type, type, n) :
#endif
            ndan (type, fe_type.order, n);

          // If this is a non-vertex on a hanging node with extra
          // degrees of freedom, we use the non-vertex dofs (which
          // come in reverse order starting from the end, to
          // simplify p refinement)
          if (extra_hanging_dofs && !elem.is_vertex(n))
            {
              const int dof_offset = n_comp - nc;

              // We should never have fewer dofs than necessary on a
              // node unless we're getting indices on a parent element,
              // and we should never need the indices on such a node
              if (dof_offset < 0)
                {
                  libmesh_assert(!elem.active());
                  di.resize(di.size() + nc, DofObject::invalid_id);
                }
              else
                for (int i=n_comp-1; i>=dof_offset; i--)
                  {
                    const dof_id_type d =
                      node.dof_number(sys_num, vg, vig, i, n_comp);
                    libmesh_assert_not_equal_to (d, DofObject::invalid_id);
                    di.push_back(d);
                  }
            }
          // If this is a vertex or an element without extra hanging
          // dofs, our dofs come in forward order coming from the
          // beginning
          else
            for (unsigned int i=0; i<nc; i++)
              {
                const dof_id_type d =
                  node.dof_number(sys_num, vg, vig, i, n_comp);
                libmesh_assert_not_equal_to (d, DofObject::invalid_id);
                di.push_back(d);
              }
        }

      // If there are any element-based DOF numbers, get them
      const unsigned int nc = FEInterface::n_dofs_per_elem(dim,
                                                           fe_type,
                                                           type);
      // We should never have fewer dofs than necessary on an
      // element unless we're getting indices on a parent element,
      // and we should never need those indices
      if (nc != 0)
        {
          const unsigned int n_comp = elem.n_comp_group(sys_num,vg);
          if (elem.n_systems() > sys_num && nc <= n_comp)
            {
              for (unsigned int i=0; i<nc; i++)
                {
                  const dof_id_type d =
                    elem.dof_number(sys_num, vg, vig, i, n_comp);
                  libmesh_assert_not_equal_to (d, DofObject::invalid_id);

                  di.push_back(d);
                }
            }
          else
            {
              libmesh_assert(!elem.active() || fe_type.family == LAGRANGE || fe_type.family == SUBDIVISION);
              di.resize(di.size() + nc, DofObject::invalid_id);
            }
        }
    }
}



void DofMap::SCALAR_dof_indices (std::vector<dof_id_type> & di,
                                 const unsigned int vn,
#ifdef LIBMESH_ENABLE_AMR
                                 const bool old_dofs
#else
                                 const bool
#endif
                                 ) const
{
  LOG_SCOPE("SCALAR_dof_indices()", "DofMap");

  libmesh_assert(this->variable(vn).type().family == SCALAR);

#ifdef LIBMESH_ENABLE_AMR
  // If we're asking for old dofs then we'd better have some
  if (old_dofs)
    libmesh_assert_greater_equal(n_old_dofs(), n_SCALAR_dofs());

  dof_id_type my_idx = old_dofs ?
    this->_first_old_scalar_df[vn] : this->_first_scalar_df[vn];
#else
  dof_id_type my_idx = this->_first_scalar_df[vn];
#endif

  libmesh_assert_not_equal_to(my_idx, DofObject::invalid_id);

  // The number of SCALAR dofs comes from the variable order
  const int n_dofs_vn = this->variable(vn).type().order.get_order();

  di.resize(n_dofs_vn);
  for (int i = 0; i != n_dofs_vn; ++i)
    di[i] = my_idx++;
}



bool DofMap::semilocal_index (dof_id_type dof_index) const
{
  // If it's not in the local indices
  if (!this->local_index(dof_index))
    {
      // and if it's not in the ghost indices, then we're not
      // semilocal
      if (!std::binary_search(_send_list.begin(), _send_list.end(), dof_index))
        return false;
    }

  return true;
}



bool DofMap::all_semilocal_indices (const std::vector<dof_id_type> & dof_indices_in) const
{
  // We're all semilocal unless we find a counterexample
  for (const auto & di : dof_indices_in)
    if (!this->semilocal_index(di))
      return false;

  return true;
}



template <typename DofObjectSubclass>
bool DofMap::is_evaluable(const DofObjectSubclass & obj,
                          unsigned int var_num) const
{
  // Everything is evaluable on a local object
  if (obj.processor_id() == this->processor_id())
    return true;

  std::vector<dof_id_type> di;

  if (var_num == libMesh::invalid_uint)
    this->dof_indices(&obj, di);
  else
    this->dof_indices(&obj, di, var_num);

  return this->all_semilocal_indices(di);
}



#ifdef LIBMESH_ENABLE_AMR

void DofMap::old_dof_indices (const Elem * const elem,
                              std::vector<dof_id_type> & di,
                              const unsigned int vn) const
{
  LOG_SCOPE("old_dof_indices()", "DofMap");

  libmesh_assert(elem);

  const ElemType type              = elem->type();
  const unsigned int sys_num       = this->sys_number();
  const unsigned int n_var_groups  = this->n_variable_groups();
  const unsigned int dim           = elem->dim();
#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
  const bool is_inf                = elem->infinite();
#endif

  // If we have dof indices stored on the elem, and there's no chance
  // that we only have those indices because we were just p refined,
  // then we should have old dof indices too.
  libmesh_assert(!elem->has_dofs(sys_num) ||
                 elem->p_refinement_flag() == Elem::JUST_REFINED ||
                 elem->old_dof_object);

  // Clear the DOF indices vector.
  di.clear();

  // Determine the nodes contributing to element elem
  std::vector<const Node *> elem_nodes;
  const Node * const * nodes_ptr;
  unsigned int n_nodes;
  if (elem->type() == TRI3SUBDIVISION)
    {
      // Subdivision surface FE require the 1-ring around elem
      const Tri3Subdivision * sd_elem = static_cast<const Tri3Subdivision *>(elem);
      MeshTools::Subdivision::find_one_ring(sd_elem, elem_nodes);
      nodes_ptr = elem_nodes.data();
      n_nodes = cast_int<unsigned int>(elem_nodes.size());
    }
  else
    {
      // All other FE use only the nodes of elem itself
      nodes_ptr = elem->get_nodes();
      n_nodes = elem->n_nodes();
    }

  // Get the dof numbers
  for (unsigned int vg=0; vg<n_var_groups; vg++)
    {
      const VariableGroup & var = this->variable_group(vg);
      const unsigned int vars_in_group = var.n_variables();

      for (unsigned int vig=0; vig<vars_in_group; vig++)
        {
          const unsigned int v = var.number(vig);
          if ((vn == v) || (vn == libMesh::invalid_uint))
            {
              if (var.type().family == SCALAR &&
                  (!elem ||
                   var.active_on_subdomain(elem->subdomain_id())))
                {
                  // We asked for this variable, so add it to the vector.
                  std::vector<dof_id_type> di_new;
                  this->SCALAR_dof_indices(di_new,v,true);
                  di.insert( di.end(), di_new.begin(), di_new.end());
                }
              else
                if (var.active_on_subdomain(elem->subdomain_id()))
                  { // Do this for all the variables if one was not specified
                    // or just for the specified variable

                    // Increase the polynomial order on p refined elements,
                    // but make sure you get the right polynomial order for
                    // the OLD degrees of freedom
                    int p_adjustment = 0;
                    if (elem->p_refinement_flag() == Elem::JUST_REFINED)
                      {
                        libmesh_assert_greater (elem->p_level(), 0);
                        p_adjustment = -1;
                      }
                    else if (elem->p_refinement_flag() == Elem::JUST_COARSENED)
                      {
                        p_adjustment = 1;
                      }
                    FEType fe_type = var.type();
                    fe_type.order = static_cast<Order>(fe_type.order +
                                                       elem->p_level() +
                                                       p_adjustment);

                    const bool extra_hanging_dofs =
                      FEInterface::extra_hanging_dofs(fe_type);

                    const FEInterface::n_dofs_at_node_ptr ndan =
                      FEInterface::n_dofs_at_node_function(dim, fe_type);

                    // Get the node-based DOF numbers
                    for (unsigned int n=0; n<n_nodes; n++)
                      {
                        const Node * node = nodes_ptr[n];
                        const DofObject * old_dof_obj = node->old_dof_object;
                        libmesh_assert(old_dof_obj);

                        // There is a potential problem with h refinement.  Imagine a
                        // quad9 that has a linear FE on it.  Then, on the hanging side,
                        // it can falsely identify a DOF at the mid-edge node. This is why
                        // we call FEInterface instead of node->n_comp() directly.
                        const unsigned int nc =
#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
                          is_inf ?
                          FEInterface::n_dofs_at_node(dim, fe_type, type, n) :
#endif
                          ndan (type, fe_type.order, n);

                        const int n_comp = old_dof_obj->n_comp_group(sys_num,vg);

                        // If this is a non-vertex on a hanging node with extra
                        // degrees of freedom, we use the non-vertex dofs (which
                        // come in reverse order starting from the end, to
                        // simplify p refinement)
                        if (extra_hanging_dofs && !elem->is_vertex(n))
                          {
                            const int dof_offset = n_comp - nc;

                            // We should never have fewer dofs than necessary on a
                            // node unless we're getting indices on a parent element
                            // or a just-coarsened element
                            if (dof_offset < 0)
                              {
                                libmesh_assert(!elem->active() || elem->refinement_flag() ==
                                               Elem::JUST_COARSENED);
                                di.resize(di.size() + nc, DofObject::invalid_id);
                              }
                            else
                              for (int i=n_comp-1; i>=dof_offset; i--)
                                {
                                  const dof_id_type d =
                                    old_dof_obj->dof_number(sys_num, vg, vig, i, n_comp);
                                  libmesh_assert_not_equal_to (d, DofObject::invalid_id);
                                  di.push_back(d);
                                }
                          }
                        // If this is a vertex or an element without extra hanging
                        // dofs, our dofs come in forward order coming from the
                        // beginning
                        else
                          for (unsigned int i=0; i<nc; i++)
                            {
                              const dof_id_type d =
                                old_dof_obj->dof_number(sys_num, vg, vig, i, n_comp);
                              libmesh_assert_not_equal_to (d, DofObject::invalid_id);
                              di.push_back(d);
                            }
                      }

                    // If there are any element-based DOF numbers, get them
                    const unsigned int nc = FEInterface::n_dofs_per_elem(dim,
                                                                         fe_type,
                                                                         type);

                    // We should never have fewer dofs than necessary on an
                    // element unless we're getting indices on a parent element
                    // or a just-coarsened element
                    if (nc != 0)
                      {
                        const DofObject * old_dof_obj = elem->old_dof_object;
                        libmesh_assert(old_dof_obj);

                        const unsigned int n_comp =
                          old_dof_obj->n_comp_group(sys_num,vg);

                        if (old_dof_obj->n_systems() > sys_num &&
                            nc <= n_comp)
                          {

                            for (unsigned int i=0; i<nc; i++)
                              {
                                const dof_id_type d =
                                  old_dof_obj->dof_number(sys_num, vg, vig, i, n_comp);

                                di.push_back(d);
                              }
                          }
                        else
                          {
                            libmesh_assert(!elem->active() || fe_type.family == LAGRANGE ||
                                           elem->refinement_flag() == Elem::JUST_COARSENED);
                            di.resize(di.size() + nc, DofObject::invalid_id);
                          }
                      }
                  }
            }
        } // end loop over variables within group
    } // end loop over variable groups
}

#endif // LIBMESH_ENABLE_AMR


#ifdef LIBMESH_ENABLE_CONSTRAINTS

void DofMap::find_connected_dofs (std::vector<dof_id_type> & elem_dofs) const
{
  typedef std::set<dof_id_type> RCSet;

  // First insert the DOFS we already depend on into the set.
  RCSet dof_set (elem_dofs.begin(), elem_dofs.end());

  bool done = true;

  // Next insert any dofs those might be constrained in terms
  // of.  Note that in this case we may not be done:  Those may
  // in turn depend on others.  So, we need to repeat this process
  // in that case until the system depends only on unconstrained
  // degrees of freedom.
  for (const auto & dof : elem_dofs)
    if (this->is_constrained_dof(dof))
      {
        // If the DOF is constrained
        DofConstraints::const_iterator
          pos = _dof_constraints.find(dof);

        libmesh_assert (pos != _dof_constraints.end());

        const DofConstraintRow & constraint_row = pos->second;

        // adaptive p refinement currently gives us lots of empty constraint
        // rows - we should optimize those DoFs away in the future.  [RHS]
        //libmesh_assert (!constraint_row.empty());

        // Add the DOFs this dof is constrained in terms of.
        // note that these dofs might also be constrained, so
        // we will need to call this function recursively.
        for (const auto & pr : constraint_row)
          if (!dof_set.count (pr.first))
            {
              dof_set.insert (pr.first);
              done = false;
            }
      }


  // If not done then we need to do more work
  // (obviously :-) )!
  if (!done)
    {
      // Fill the vector with the contents of the set
      elem_dofs.clear();
      elem_dofs.insert (elem_dofs.end(),
                        dof_set.begin(), dof_set.end());


      // May need to do this recursively.  It is possible
      // that we just replaced a constrained DOF with another
      // constrained DOF.
      this->find_connected_dofs (elem_dofs);

    } // end if (!done)
}

#endif // LIBMESH_ENABLE_CONSTRAINTS



void DofMap::print_info(std::ostream & os) const
{
  os << this->get_info();
}



std::string DofMap::get_info() const
{
  std::ostringstream os;

  // If we didn't calculate the exact sparsity pattern, the threaded
  // sparsity pattern assembly may have just given us an upper bound
  // on sparsity.
  const char * may_equal = " <= ";

  // If we calculated the exact sparsity pattern, then we can report
  // exact bandwidth figures:
  for (const auto & mat : _matrices)
    if (mat->need_full_sparsity_pattern())
      may_equal = " = ";

  dof_id_type max_n_nz = 0, max_n_oz = 0;
  long double avg_n_nz = 0, avg_n_oz = 0;

  if (_n_nz)
    {
      for (const auto & val : *_n_nz)
        {
          max_n_nz = std::max(max_n_nz, val);
          avg_n_nz += val;
        }

      std::size_t n_nz_size = _n_nz->size();

      this->comm().max(max_n_nz);
      this->comm().sum(avg_n_nz);
      this->comm().sum(n_nz_size);

      avg_n_nz /= std::max(n_nz_size,std::size_t(1));

      libmesh_assert(_n_oz);

      for (const auto & val : *_n_oz)
        {
          max_n_oz = std::max(max_n_oz, val);
          avg_n_oz += val;
        }

      std::size_t n_oz_size = _n_oz->size();

      this->comm().max(max_n_oz);
      this->comm().sum(avg_n_oz);
      this->comm().sum(n_oz_size);

      avg_n_oz /= std::max(n_oz_size,std::size_t(1));
    }

  os << "    DofMap Sparsity\n      Average  On-Processor Bandwidth"
     << may_equal << avg_n_nz << '\n';

  os << "      Average Off-Processor Bandwidth"
     << may_equal << avg_n_oz << '\n';

  os << "      Maximum  On-Processor Bandwidth"
     << may_equal << max_n_nz << '\n';

  os << "      Maximum Off-Processor Bandwidth"
     << may_equal << max_n_oz << std::endl;

#ifdef LIBMESH_ENABLE_CONSTRAINTS

  std::size_t n_constraints = 0, max_constraint_length = 0,
    n_rhss = 0;
  long double avg_constraint_length = 0.;

  for (const auto & pr : _dof_constraints)
    {
      // Only count local constraints, then sum later
      const dof_id_type constrained_dof = pr.first;
      if (!this->local_index(constrained_dof))
        continue;

      const DofConstraintRow & row = pr.second;
      std::size_t rowsize = row.size();

      max_constraint_length = std::max(max_constraint_length,
                                       rowsize);
      avg_constraint_length += rowsize;
      n_constraints++;

      if (_primal_constraint_values.count(constrained_dof))
        n_rhss++;
    }

  this->comm().sum(n_constraints);
  this->comm().sum(n_rhss);
  this->comm().sum(avg_constraint_length);
  this->comm().max(max_constraint_length);

  os << "    DofMap Constraints\n      Number of DoF Constraints = "
     << n_constraints;
  if (n_rhss)
    os << '\n'
       << "      Number of Heterogenous Constraints= " << n_rhss;
  if (n_constraints)
    {
      avg_constraint_length /= n_constraints;

      os << '\n'
         << "      Average DoF Constraint Length= " << avg_constraint_length;
    }

#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
  std::size_t n_node_constraints = 0, max_node_constraint_length = 0,
    n_node_rhss = 0;
  long double avg_node_constraint_length = 0.;

  for (const auto & pr : _node_constraints)
    {
      // Only count local constraints, then sum later
      const Node * node = pr.first;
      if (node->processor_id() != this->processor_id())
        continue;

      const NodeConstraintRow & row = pr.second.first;
      std::size_t rowsize = row.size();

      max_node_constraint_length = std::max(max_node_constraint_length,
                                            rowsize);
      avg_node_constraint_length += rowsize;
      n_node_constraints++;

      if (pr.second.second != Point(0))
        n_node_rhss++;
    }

  this->comm().sum(n_node_constraints);
  this->comm().sum(n_node_rhss);
  this->comm().sum(avg_node_constraint_length);
  this->comm().max(max_node_constraint_length);

  os << "\n      Number of Node Constraints = " << n_node_constraints;
  if (n_node_rhss)
    os << '\n'
       << "      Number of Heterogenous Node Constraints= " << n_node_rhss;
  if (n_node_constraints)
    {
      avg_node_constraint_length /= n_node_constraints;
      os << "\n      Maximum Node Constraint Length= " << max_node_constraint_length
         << '\n'
         << "      Average Node Constraint Length= " << avg_node_constraint_length;
    }
#endif // LIBMESH_ENABLE_NODE_CONSTRAINTS

  os << std::endl;

#endif // LIBMESH_ENABLE_CONSTRAINTS

  return os.str();
}


template bool DofMap::is_evaluable<Elem>(const Elem &, unsigned int) const;
template bool DofMap::is_evaluable<Node>(const Node &, unsigned int) const;
INSTANTIATE_DOF_MAP_REALTYPE_METHODS(Real);

} // namespace libMesh
