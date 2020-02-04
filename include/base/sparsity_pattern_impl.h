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


#ifndef LIBMESH_SPARSITY_PATTERN_IMPL_H
#define LIBMESH_SPARSITY_PATTERN_IMPL_H

#include "libmesh/sparsity_pattern.h"
#include "libmesh/dof_map.h"
#include "libmesh/mesh_base.h"
#include "libmesh/coupling_matrix.h"

namespace libMesh
{

namespace SparsityPattern
{

template <typename RealType>
void Build::operator()(const ConstElemRangeTempl<RealType> & range)
{
  typedef ElemTempl<RealType> Elem;

  // Compute the sparsity structure of the global matrix.  This can be
  // fed into a PetscMatrix to allocate exactly the number of nonzeros
  // necessary to store the matrix.  This algorithm should be linear
  // in the (# of elements)*(# nodes per element)
  const processor_id_type proc_id           = mesh.processor_id();
  const dof_id_type n_dofs_on_proc    = dof_map.n_dofs_on_processor(proc_id);
  const dof_id_type first_dof_on_proc = dof_map.first_dof(proc_id);
  const dof_id_type end_dof_on_proc   = dof_map.end_dof(proc_id);

  sparsity_pattern.resize(n_dofs_on_proc);

  // Handle dof coupling specified by library and user coupling functors
  {
    const unsigned int n_var = dof_map.n_variables();

    std::vector<std::vector<dof_id_type> > element_dofs_i(n_var);

    std::vector<const Elem *> coupled_neighbors;
    for (const auto & elem : range)
      {
        // Make some fake element iterators defining a range
        // pointing to only this element.
        Elem * const * elempp = const_cast<Elem * const *>(&elem);
        Elem * const * elemend = elempp+1;

        const auto fake_elem_it =
          const_element_iterator_templ<RealType>(elempp,
                                                 elemend,
                                                 Predicates::NotNull<Elem * const *>());

        const auto fake_elem_end =
          const_element_iterator_templ<RealType>(elemend,
                                                 elemend,
                                                 Predicates::NotNull<Elem * const *>());

        typename GhostingFunctorTempl<RealType>::map_type elements_to_couple;

        // Man, I wish we had guaranteed unique_ptr availability...
        std::set<CouplingMatrix *> temporary_coupling_matrices;

        dof_map.merge_ghost_functor_outputs(elements_to_couple,
                                            temporary_coupling_matrices,
                                            dof_map.coupling_functors_begin(),
                                            dof_map.coupling_functors_end(),
                                            fake_elem_it,
                                            fake_elem_end,
                                            DofObject::invalid_processor_id);
        for (unsigned int vi=0; vi<n_var; vi++)
          this->sorted_connected_dofs(elem, element_dofs_i[vi], vi);

        for (unsigned int vi=0; vi<n_var; vi++)
          for (const auto & pr : elements_to_couple)
            {
              const Elem * const partner = pr.first;
              const CouplingMatrix * ghost_coupling = pr.second;

              // Loop over coupling matrix row variables if we have a
              // coupling matrix, or all variables if not.
              if (ghost_coupling)
                {
                  libmesh_assert_equal_to (ghost_coupling->size(), n_var);
                  ConstCouplingRow ccr(vi, *ghost_coupling);

                  for (const auto & idx : ccr)
                    {
                      if (partner == elem)
                        this->handle_vi_vj(element_dofs_i[vi], element_dofs_i[idx]);
                      else
                        {
                          std::vector<dof_id_type> partner_dofs;
                          this->sorted_connected_dofs(partner, partner_dofs, idx);
                          this->handle_vi_vj(element_dofs_i[vi], partner_dofs);
                        }
                    }
                }
              else
                {
                  for (unsigned int vj = 0; vj != n_var; ++vj)
                    {
                      if (partner == elem)
                        this->handle_vi_vj(element_dofs_i[vi], element_dofs_i[vj]);
                      else
                        {
                          std::vector<dof_id_type> partner_dofs;
                          this->sorted_connected_dofs(partner, partner_dofs, vj);
                          this->handle_vi_vj(element_dofs_i[vi], partner_dofs);
                        }
                    }
                }
            } // End ghosted element loop

        for (auto & mat : temporary_coupling_matrices)
          delete mat;

      } // End range element loop
  } // End ghosting functor section

  // Now a new chunk of sparsity structure is built for all of the
  // DOFs connected to our rows of the matrix.

  // If we're building a full sparsity pattern, then we've got
  // complete rows to work with, so we can just count them from
  // scratch.
  if (need_full_sparsity_pattern)
    {
      n_nz.clear();
      n_oz.clear();
    }

  n_nz.resize (n_dofs_on_proc, 0);
  n_oz.resize (n_dofs_on_proc, 0);

  for (dof_id_type i=0; i<n_dofs_on_proc; i++)
    {
      // Get the row of the sparsity pattern
      SparsityPattern::Row & row = sparsity_pattern[i];

      for (const auto & df : row)
        if ((df < first_dof_on_proc) || (df >= end_dof_on_proc))
          n_oz[i]++;
        else
          n_nz[i]++;

      // If we're not building a full sparsity pattern, then we want
      // to avoid overcounting these entries as much as possible.
      if (!need_full_sparsity_pattern)
        row.clear();
    }
}

} // namespace SparsityPattern
} // namespace libMesh
#endif // LIBMESH_SPARSITY_PATTERN_IMPL_H
