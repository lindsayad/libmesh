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

// Local includes
#include "libmesh/dof_map.h"

// libMesh includes
#include "libmesh/boundary_info.h" // needed for dirichlet constraints
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/elem.h"
#include "libmesh/elem_range.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_type.h"
#include "libmesh/function_base.h"
#include "libmesh/int_range.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/mesh_base.h"
#include "libmesh/mesh_inserter_iterator.h"
#include "libmesh/mesh_tools.h" // for libmesh_assert_valid_boundary_ids()
#include "libmesh/nonlinear_implicit_system.h"
#include "libmesh/numeric_vector.h" // for enforce_constraints_exactly()
#include "libmesh/parallel_algebra.h"
#include "libmesh/parallel_elem.h"
#include "libmesh/parallel_node.h"
#include "libmesh/periodic_boundaries.h"
#include "libmesh/periodic_boundary.h"
#include "libmesh/periodic_boundary_base.h"
#include "libmesh/point_locator_base.h"
#include "libmesh/quadrature.h" // for dirichlet constraints
#include "libmesh/raw_accessor.h"
#include "libmesh/sparse_matrix.h" // needed to constrain adjoint rhs
#include "libmesh/system.h" // needed by enforce_constraints_exactly()
#include "libmesh/tensor_tools.h"
#include "libmesh/threads.h"

// TIMPI includes
#include "timpi/parallel_implementation.h"
#include "timpi/parallel_sync.h"

// C++ Includes
#include <set>
#include <algorithm> // for std::count, std::fill
#include <sstream>
#include <cstdlib> // *must* precede <cmath> for proper std:abs() on PGI, Sun Studio CC
#include <cmath>
#include <numeric>

namespace libMesh
{

// ------------------------------------------------------------
// DofMap member functions

#ifdef LIBMESH_ENABLE_CONSTRAINTS


dof_id_type DofMap::n_constrained_dofs() const
{
  parallel_object_only();

  dof_id_type nc_dofs = this->n_local_constrained_dofs();
  this->comm().sum(nc_dofs);
  return nc_dofs;
}


dof_id_type DofMap::n_local_constrained_dofs() const
{
  const DofConstraints::const_iterator lower =
    _dof_constraints.lower_bound(this->first_dof()),
    upper =
    _dof_constraints.upper_bound(this->end_dof()-1);

  return cast_int<dof_id_type>(std::distance(lower, upper));
}


void DofMap::add_constraint_row (const dof_id_type dof_number,
                                 const DofConstraintRow & constraint_row,
                                 const Number constraint_rhs,
                                 const bool forbid_constraint_overwrite)
{
  // Optionally allow the user to overwrite constraints.  Defaults to false.
  if (forbid_constraint_overwrite)
    if (this->is_constrained_dof(dof_number))
      libmesh_error_msg("ERROR: DOF " << dof_number << " was already constrained!");

  libmesh_assert_less(dof_number, this->n_dofs());
#ifndef NDEBUG
  for (const auto & pr : constraint_row)
    libmesh_assert_less(pr.first, this->n_dofs());
#endif

  // We don't get insert_or_assign until C++17 so we make do.
  std::pair<DofConstraints::iterator, bool> it =
    _dof_constraints.insert(std::make_pair(dof_number, constraint_row));
  if (!it.second)
    it.first->second = constraint_row;

  std::pair<DofConstraintValueMap::iterator, bool> rhs_it =
    _primal_constraint_values.insert(std::make_pair(dof_number, constraint_rhs));
  if (!rhs_it.second)
    rhs_it.first->second = constraint_rhs;
}


void DofMap::add_adjoint_constraint_row (const unsigned int qoi_index,
                                         const dof_id_type dof_number,
                                         const DofConstraintRow & /*constraint_row*/,
                                         const Number constraint_rhs,
                                         const bool forbid_constraint_overwrite)
{
  // Optionally allow the user to overwrite constraints.  Defaults to false.
  if (forbid_constraint_overwrite)
    {
      if (!this->is_constrained_dof(dof_number))
        libmesh_error_msg("ERROR: DOF " << dof_number << " has no corresponding primal constraint!");
#ifndef NDEBUG
      // No way to do this without a non-normalized tolerance?

      // // If the user passed in more than just the rhs, let's check the
      // // coefficients for consistency
      // if (!constraint_row.empty())
      //   {
      //     DofConstraintRow row = _dof_constraints[dof_number];
      //     for (const auto & pr : row)
      //       libmesh_assert(constraint_row.find(pr.first)->second == pr.second);
      //   }
      //
      // if (_adjoint_constraint_values[qoi_index].find(dof_number) !=
      //     _adjoint_constraint_values[qoi_index].end())
      //   libmesh_assert_equal_to(_adjoint_constraint_values[qoi_index][dof_number],
      //                           constraint_rhs);

#endif
    }

  // Creates the map of rhs values if it doesn't already exist; then
  // adds the current value to that map

  // We don't get insert_or_assign until C++17 so we make do.
  std::pair<DofConstraintValueMap::iterator, bool> rhs_it =
    _adjoint_constraint_values[qoi_index].insert(std::make_pair(dof_number,
                                                                constraint_rhs));
  if (!rhs_it.second)
    rhs_it.first->second = constraint_rhs;
}




void DofMap::print_dof_constraints(std::ostream & os,
                                   bool print_nonlocal) const
{
  parallel_object_only();

  std::string local_constraints =
    this->get_local_constraints(print_nonlocal);

  if (this->processor_id())
    {
      this->comm().send(0, local_constraints);
    }
  else
    {
      os << "Processor 0:\n";
      os << local_constraints;

      for (auto p : IntRange<processor_id_type>(1, this->n_processors()))
        {
          this->comm().receive(p, local_constraints);
          os << "Processor " << p << ":\n";
          os << local_constraints;
        }
    }
}

std::string DofMap::get_local_constraints(bool print_nonlocal) const
{
  std::ostringstream os;
#ifdef LIBMESH_ENABLE_NODE_CONSTRAINTS
  if (print_nonlocal)
    os << "All ";
  else
    os << "Local ";

  os << "Node Constraints:"
     << std::endl;

  for (const auto & pr : _node_constraints)
    {
      const Node * node = pr.first;

      // Skip non-local nodes if requested
      if (!print_nonlocal &&
          node->processor_id() != this->processor_id())
        continue;

      const NodeConstraintRow & row = pr.second.first;
      const Point & offset = pr.second.second;

      os << "Constraints for Node id " << node->id()
         << ": \t";

      for (const auto & item : row)
        os << " (" << item.first->id() << "," << item.second << ")\t";

      os << "rhs: " << offset;

      os << std::endl;
    }
#endif // LIBMESH_ENABLE_NODE_CONSTRAINTS

  if (print_nonlocal)
    os << "All ";
  else
    os << "Local ";

  os << "DoF Constraints:"
     << std::endl;

  for (const auto & pr : _dof_constraints)
    {
      const dof_id_type i = pr.first;

      // Skip non-local dofs if requested
      if (!print_nonlocal && !this->local_index(i))
        continue;

      const DofConstraintRow & row = pr.second;
      DofConstraintValueMap::const_iterator rhsit =
        _primal_constraint_values.find(i);
      const Number rhs = (rhsit == _primal_constraint_values.end()) ?
        0 : rhsit->second;

      os << "Constraints for DoF " << i
         << ": \t";

      for (const auto & item : row)
        os << " (" << item.first << "," << item.second << ")\t";

      os << "rhs: " << rhs;
      os << std::endl;
    }

  for (unsigned int qoi_index = 0,
       n_qois = cast_int<unsigned int>(_adjoint_dirichlet_boundaries.size());
       qoi_index != n_qois; ++qoi_index)
    {
      os << "Adjoint " << qoi_index << " DoF rhs values:"
         << std::endl;

      AdjointDofConstraintValues::const_iterator adjoint_map_it =
        _adjoint_constraint_values.find(qoi_index);

      if (adjoint_map_it != _adjoint_constraint_values.end())
        for (const auto & pr : adjoint_map_it->second)
          {
            const dof_id_type i = pr.first;

            // Skip non-local dofs if requested
            if (!print_nonlocal && !this->local_index(i))
              continue;

            const Number rhs = pr.second;

            os << "RHS for DoF " << i
               << ": " << rhs;

            os << std::endl;
          }
    }

  return os.str();
}



void DofMap::constrain_element_matrix (DenseMatrix<Number> & matrix,
                                       std::vector<dof_id_type> & elem_dofs,
                                       bool asymmetric_constraint_rows) const
{
  libmesh_assert_equal_to (elem_dofs.size(), matrix.m());
  libmesh_assert_equal_to (elem_dofs.size(), matrix.n());

  // check for easy return
  if (this->_dof_constraints.empty())
    return;

  // The constrained matrix is built up as C^T K C.
  DenseMatrix<Number> C;


  this->build_constraint_matrix (C, elem_dofs);

  LOG_SCOPE("constrain_elem_matrix()", "DofMap");

  // It is possible that the matrix is not constrained at all.
  if ((C.m() == matrix.m()) &&
      (C.n() == elem_dofs.size())) // It the matrix is constrained
    {
      // Compute the matrix-matrix-matrix product C^T K C
      matrix.left_multiply_transpose  (C);
      matrix.right_multiply (C);


      libmesh_assert_equal_to (matrix.m(), matrix.n());
      libmesh_assert_equal_to (matrix.m(), elem_dofs.size());
      libmesh_assert_equal_to (matrix.n(), elem_dofs.size());


      for (unsigned int i=0,
           n_elem_dofs = cast_int<unsigned int>(elem_dofs.size());
           i != n_elem_dofs; i++)
        // If the DOF is constrained
        if (this->is_constrained_dof(elem_dofs[i]))
          {
            for (auto j : IntRange<unsigned int>(0, matrix.n()))
              matrix(i,j) = 0.;

            matrix(i,i) = 1.;

            if (asymmetric_constraint_rows)
              {
                DofConstraints::const_iterator
                  pos = _dof_constraints.find(elem_dofs[i]);

                libmesh_assert (pos != _dof_constraints.end());

                const DofConstraintRow & constraint_row = pos->second;

                // This is an overzealous assertion in the presence of
                // heterogenous constraints: we now can constrain "u_i = c"
                // with no other u_j terms involved.
                //
                // libmesh_assert (!constraint_row.empty());

                for (const auto & item : constraint_row)
                  for (unsigned int j=0; j != n_elem_dofs; j++)
                    if (elem_dofs[j] == item.first)
                      matrix(i,j) = -item.second;
              }
          }
    } // end if is constrained...
}



void DofMap::constrain_element_matrix_and_vector (DenseMatrix<Number> & matrix,
                                                  DenseVector<Number> & rhs,
                                                  std::vector<dof_id_type> & elem_dofs,
                                                  bool asymmetric_constraint_rows) const
{
  libmesh_assert_equal_to (elem_dofs.size(), matrix.m());
  libmesh_assert_equal_to (elem_dofs.size(), matrix.n());
  libmesh_assert_equal_to (elem_dofs.size(), rhs.size());

  // check for easy return
  if (this->_dof_constraints.empty())
    return;

  // The constrained matrix is built up as C^T K C.
  // The constrained RHS is built up as C^T F
  DenseMatrix<Number> C;

  this->build_constraint_matrix (C, elem_dofs);

  LOG_SCOPE("cnstrn_elem_mat_vec()", "DofMap");

  // It is possible that the matrix is not constrained at all.
  if ((C.m() == matrix.m()) &&
      (C.n() == elem_dofs.size())) // It the matrix is constrained
    {
      // Compute the matrix-matrix-matrix product C^T K C
      matrix.left_multiply_transpose  (C);
      matrix.right_multiply (C);


      libmesh_assert_equal_to (matrix.m(), matrix.n());
      libmesh_assert_equal_to (matrix.m(), elem_dofs.size());
      libmesh_assert_equal_to (matrix.n(), elem_dofs.size());


      for (unsigned int i=0,
           n_elem_dofs = cast_int<unsigned int>(elem_dofs.size());
           i != n_elem_dofs; i++)
        if (this->is_constrained_dof(elem_dofs[i]))
          {
            for (auto j : IntRange<unsigned int>(0, matrix.n()))
              matrix(i,j) = 0.;

            // If the DOF is constrained
            matrix(i,i) = 1.;

            // This will put a nonsymmetric entry in the constraint
            // row to ensure that the linear system produces the
            // correct value for the constrained DOF.
            if (asymmetric_constraint_rows)
              {
                DofConstraints::const_iterator
                  pos = _dof_constraints.find(elem_dofs[i]);

                libmesh_assert (pos != _dof_constraints.end());

                const DofConstraintRow & constraint_row = pos->second;

                // p refinement creates empty constraint rows
                //    libmesh_assert (!constraint_row.empty());

                for (const auto & item : constraint_row)
                  for (unsigned int j=0; j != n_elem_dofs; j++)
                    if (elem_dofs[j] == item.first)
                      matrix(i,j) = -item.second;
              }
          }


      // Compute the matrix-vector product C^T F
      DenseVector<Number> old_rhs(rhs);

      // compute matrix/vector product
      C.vector_mult_transpose(rhs, old_rhs);
    } // end if is constrained...
}



void DofMap::heterogenously_constrain_element_matrix_and_vector (DenseMatrix<Number> & matrix,
                                                                 DenseVector<Number> & rhs,
                                                                 std::vector<dof_id_type> & elem_dofs,
                                                                 bool asymmetric_constraint_rows,
                                                                 int qoi_index) const
{
  libmesh_assert_equal_to (elem_dofs.size(), matrix.m());
  libmesh_assert_equal_to (elem_dofs.size(), matrix.n());
  libmesh_assert_equal_to (elem_dofs.size(), rhs.size());

  // check for easy return
  if (this->_dof_constraints.empty())
    return;

  // The constrained matrix is built up as C^T K C.
  // The constrained RHS is built up as C^T (F - K H)
  DenseMatrix<Number> C;
  DenseVector<Number> H;

  this->build_constraint_matrix_and_vector (C, H, elem_dofs, qoi_index);

  LOG_SCOPE("hetero_cnstrn_elem_mat_vec()", "DofMap");

  // It is possible that the matrix is not constrained at all.
  if ((C.m() == matrix.m()) &&
      (C.n() == elem_dofs.size())) // It the matrix is constrained
    {
      // We may have rhs values to use later
      const DofConstraintValueMap * rhs_values = nullptr;
      if (qoi_index < 0)
        rhs_values = &_primal_constraint_values;
      else
        {
          const AdjointDofConstraintValues::const_iterator
            it = _adjoint_constraint_values.find(qoi_index);
          if (it != _adjoint_constraint_values.end())
            rhs_values = &it->second;
        }

      // Compute matrix/vector product K H
      DenseVector<Number> KH;
      matrix.vector_mult(KH, H);

      // Compute the matrix-vector product C^T (F - KH)
      DenseVector<Number> F_minus_KH(rhs);
      F_minus_KH -= KH;
      C.vector_mult_transpose(rhs, F_minus_KH);

      // Compute the matrix-matrix-matrix product C^T K C
      matrix.left_multiply_transpose  (C);
      matrix.right_multiply (C);

      libmesh_assert_equal_to (matrix.m(), matrix.n());
      libmesh_assert_equal_to (matrix.m(), elem_dofs.size());
      libmesh_assert_equal_to (matrix.n(), elem_dofs.size());

      for (unsigned int i=0,
           n_elem_dofs = cast_int<unsigned int>(elem_dofs.size());
           i != n_elem_dofs; i++)
        {
          const dof_id_type dof_id = elem_dofs[i];

          if (this->is_constrained_dof(dof_id))
            {
              for (auto j : IntRange<unsigned int>(0, matrix.n()))
                matrix(i,j) = 0.;

              // If the DOF is constrained
              matrix(i,i) = 1.;

              // This will put a nonsymmetric entry in the constraint
              // row to ensure that the linear system produces the
              // correct value for the constrained DOF.
              if (asymmetric_constraint_rows)
                {
                  DofConstraints::const_iterator
                    pos = _dof_constraints.find(dof_id);

                  libmesh_assert (pos != _dof_constraints.end());

                  const DofConstraintRow & constraint_row = pos->second;

                  for (const auto & item : constraint_row)
                    for (unsigned int j=0; j != n_elem_dofs; j++)
                      if (elem_dofs[j] == item.first)
                        matrix(i,j) = -item.second;

                  if (rhs_values)
                    {
                      const DofConstraintValueMap::const_iterator valpos =
                        rhs_values->find(dof_id);

                      rhs(i) = (valpos == rhs_values->end()) ?
                        0 : valpos->second;
                    }
                }
              else
                rhs(i) = 0.;
            }
        }

    } // end if is constrained...
}



void DofMap::heterogenously_constrain_element_vector (const DenseMatrix<Number> & matrix,
                                                      DenseVector<Number> & rhs,
                                                      std::vector<dof_id_type> & elem_dofs,
                                                      bool asymmetric_constraint_rows,
                                                      int qoi_index) const
{
  libmesh_assert_equal_to (elem_dofs.size(), matrix.m());
  libmesh_assert_equal_to (elem_dofs.size(), matrix.n());
  libmesh_assert_equal_to (elem_dofs.size(), rhs.size());

  // check for easy return
  if (this->_dof_constraints.empty())
    return;

  // The constrained matrix is built up as C^T K C.
  // The constrained RHS is built up as C^T (F - K H)
  DenseMatrix<Number> C;
  DenseVector<Number> H;

  this->build_constraint_matrix_and_vector (C, H, elem_dofs, qoi_index);

  LOG_SCOPE("hetero_cnstrn_elem_vec()", "DofMap");

  // It is possible that the matrix is not constrained at all.
  if ((C.m() == matrix.m()) &&
      (C.n() == elem_dofs.size())) // It the matrix is constrained
    {
      // We may have rhs values to use later
      const DofConstraintValueMap * rhs_values = nullptr;
      if (qoi_index < 0)
        rhs_values = &_primal_constraint_values;
      else
        {
          const AdjointDofConstraintValues::const_iterator
            it = _adjoint_constraint_values.find(qoi_index);
          if (it != _adjoint_constraint_values.end())
            rhs_values = &it->second;
        }

      // Compute matrix/vector product K H
      DenseVector<Number> KH;
      matrix.vector_mult(KH, H);

      // Compute the matrix-vector product C^T (F - KH)
      DenseVector<Number> F_minus_KH(rhs);
      F_minus_KH -= KH;
      C.vector_mult_transpose(rhs, F_minus_KH);

      for (unsigned int i=0,
           n_elem_dofs = cast_int<unsigned int>(elem_dofs.size());
           i != n_elem_dofs; i++)
        {
          const dof_id_type dof_id = elem_dofs[i];

          if (this->is_constrained_dof(dof_id))
            {
              // This will put a nonsymmetric entry in the constraint
              // row to ensure that the linear system produces the
              // correct value for the constrained DOF.
              if (asymmetric_constraint_rows && rhs_values)
                {
                  const DofConstraintValueMap::const_iterator valpos =
                    rhs_values->find(dof_id);

                  rhs(i) = (valpos == rhs_values->end()) ?
                    0 : valpos->second;
                }
              else
                rhs(i) = 0.;
            }
        }

    } // end if is constrained...
}




void DofMap::constrain_element_matrix (DenseMatrix<Number> & matrix,
                                       std::vector<dof_id_type> & row_dofs,
                                       std::vector<dof_id_type> & col_dofs,
                                       bool asymmetric_constraint_rows) const
{
  libmesh_assert_equal_to (row_dofs.size(), matrix.m());
  libmesh_assert_equal_to (col_dofs.size(), matrix.n());

  // check for easy return
  if (this->_dof_constraints.empty())
    return;

  // The constrained matrix is built up as R^T K C.
  DenseMatrix<Number> R;
  DenseMatrix<Number> C;

  // Safeguard against the user passing us the same
  // object for row_dofs and col_dofs.  If that is done
  // the calls to build_matrix would fail
  std::vector<dof_id_type> orig_row_dofs(row_dofs);
  std::vector<dof_id_type> orig_col_dofs(col_dofs);

  this->build_constraint_matrix (R, orig_row_dofs);
  this->build_constraint_matrix (C, orig_col_dofs);

  LOG_SCOPE("constrain_elem_matrix()", "DofMap");

  row_dofs = orig_row_dofs;
  col_dofs = orig_col_dofs;

  bool constraint_found = false;

  // K_constrained = R^T K C

  if ((R.m() == matrix.m()) &&
      (R.n() == row_dofs.size()))
    {
      matrix.left_multiply_transpose  (R);
      constraint_found = true;
    }

  if ((C.m() == matrix.n()) &&
      (C.n() == col_dofs.size()))
    {
      matrix.right_multiply (C);
      constraint_found = true;
    }

  // It is possible that the matrix is not constrained at all.
  if (constraint_found)
    {
      libmesh_assert_equal_to (matrix.m(), row_dofs.size());
      libmesh_assert_equal_to (matrix.n(), col_dofs.size());


      for (unsigned int i=0,
           n_row_dofs = cast_int<unsigned int>(row_dofs.size());
           i != n_row_dofs; i++)
        if (this->is_constrained_dof(row_dofs[i]))
          {
            for (auto j : IntRange<unsigned int>(0, matrix.n()))
              {
                if (row_dofs[i] != col_dofs[j])
                  matrix(i,j) = 0.;
                else // If the DOF is constrained
                  matrix(i,j) = 1.;
              }

            if (asymmetric_constraint_rows)
              {
                DofConstraints::const_iterator
                  pos = _dof_constraints.find(row_dofs[i]);

                libmesh_assert (pos != _dof_constraints.end());

                const DofConstraintRow & constraint_row = pos->second;

                libmesh_assert (!constraint_row.empty());

                for (const auto & item : constraint_row)
                  for (unsigned int j=0,
                       n_col_dofs = cast_int<unsigned int>(col_dofs.size());
                       j != n_col_dofs; j++)
                    if (col_dofs[j] == item.first)
                      matrix(i,j) = -item.second;
              }
          }
    } // end if is constrained...
}



void DofMap::constrain_element_vector (DenseVector<Number> & rhs,
                                       std::vector<dof_id_type> & row_dofs,
                                       bool) const
{
  libmesh_assert_equal_to (rhs.size(), row_dofs.size());

  // check for easy return
  if (this->_dof_constraints.empty())
    return;

  // The constrained RHS is built up as R^T F.
  DenseMatrix<Number> R;

  this->build_constraint_matrix (R, row_dofs);

  LOG_SCOPE("constrain_elem_vector()", "DofMap");

  // It is possible that the vector is not constrained at all.
  if ((R.m() == rhs.size()) &&
      (R.n() == row_dofs.size())) // if the RHS is constrained
    {
      // Compute the matrix-vector product
      DenseVector<Number> old_rhs(rhs);
      R.vector_mult_transpose(rhs, old_rhs);

      libmesh_assert_equal_to (row_dofs.size(), rhs.size());

      for (unsigned int i=0,
           n_row_dofs = cast_int<unsigned int>(row_dofs.size());
           i != n_row_dofs; i++)
        if (this->is_constrained_dof(row_dofs[i]))
          {
            // If the DOF is constrained
            libmesh_assert (_dof_constraints.find(row_dofs[i]) != _dof_constraints.end());

            rhs(i) = 0;
          }
    } // end if the RHS is constrained.
}



void DofMap::constrain_element_dyad_matrix (DenseVector<Number> & v,
                                            DenseVector<Number> & w,
                                            std::vector<dof_id_type> & row_dofs,
                                            bool) const
{
  libmesh_assert_equal_to (v.size(), row_dofs.size());
  libmesh_assert_equal_to (w.size(), row_dofs.size());

  // check for easy return
  if (this->_dof_constraints.empty())
    return;

  // The constrained RHS is built up as R^T F.
  DenseMatrix<Number> R;

  this->build_constraint_matrix (R, row_dofs);

  LOG_SCOPE("cnstrn_elem_dyad_mat()", "DofMap");

  // It is possible that the vector is not constrained at all.
  if ((R.m() == v.size()) &&
      (R.n() == row_dofs.size())) // if the RHS is constrained
    {
      // Compute the matrix-vector products
      DenseVector<Number> old_v(v);
      DenseVector<Number> old_w(w);

      // compute matrix/vector product
      R.vector_mult_transpose(v, old_v);
      R.vector_mult_transpose(w, old_w);

      libmesh_assert_equal_to (row_dofs.size(), v.size());
      libmesh_assert_equal_to (row_dofs.size(), w.size());

      /* Constrain only v, not w.  */

      for (unsigned int i=0,
           n_row_dofs = cast_int<unsigned int>(row_dofs.size());
           i != n_row_dofs; i++)
        if (this->is_constrained_dof(row_dofs[i]))
          {
            // If the DOF is constrained
            libmesh_assert (_dof_constraints.find(row_dofs[i]) != _dof_constraints.end());

            v(i) = 0;
          }
    } // end if the RHS is constrained.
}



void DofMap::constrain_nothing (std::vector<dof_id_type> & dofs) const
{
  // check for easy return
  if (this->_dof_constraints.empty())
    return;

  // All the work is done by \p build_constraint_matrix.  We just need
  // a dummy matrix.
  DenseMatrix<Number> R;
  this->build_constraint_matrix (R, dofs);
}



void DofMap::enforce_constraints_exactly (const System & system,
                                          NumericVector<Number> * v,
                                          bool homogeneous) const
{
  parallel_object_only();

  if (!this->n_constrained_dofs())
    return;

  LOG_SCOPE("enforce_constraints_exactly()","DofMap");

  if (!v)
    v = system.solution.get();

  NumericVector<Number> * v_local  = nullptr; // will be initialized below
  NumericVector<Number> * v_global = nullptr; // will be initialized below
  std::unique_ptr<NumericVector<Number>> v_built;
  if (v->type() == SERIAL)
    {
      v_built = NumericVector<Number>::build(this->comm());
      v_built->init(this->n_dofs(), this->n_local_dofs(), true, PARALLEL);
      v_built->close();

      for (dof_id_type i=v_built->first_local_index();
           i<v_built->last_local_index(); i++)
        v_built->set(i, (*v)(i));
      v_built->close();
      v_global = v_built.get();

      v_local = v;
      libmesh_assert (v_local->closed());
    }
  else if (v->type() == PARALLEL)
    {
      v_built = NumericVector<Number>::build(this->comm());
      v_built->init (v->size(), v->size(), true, SERIAL);
      v->localize(*v_built);
      v_built->close();
      v_local = v_built.get();

      v_global = v;
    }
  else if (v->type() == GHOSTED)
    {
      v_local = v;
      v_global = v;
    }
  else // unknown v->type()
    libmesh_error_msg("ERROR: Unknown v->type() == " << v->type());

  // We should never hit these asserts because we should error-out in
  // else clause above.  Just to be sure we don't try to use v_local
  // and v_global uninitialized...
  libmesh_assert(v_local);
  libmesh_assert(v_global);
  libmesh_assert_equal_to (this, &(system.get_dof_map()));

  for (const auto & pr : _dof_constraints)
    {
      dof_id_type constrained_dof = pr.first;
      if (!this->local_index(constrained_dof))
        continue;

      const DofConstraintRow constraint_row = pr.second;

      Number exact_value = 0;
      if (!homogeneous)
        {
          DofConstraintValueMap::const_iterator rhsit =
            _primal_constraint_values.find(constrained_dof);
          if (rhsit != _primal_constraint_values.end())
            exact_value = rhsit->second;
        }
      for (const auto & j : constraint_row)
        exact_value += j.second * (*v_local)(j.first);

      v_global->set(constrained_dof, exact_value);
    }

  // If the old vector was serial, we probably need to send our values
  // to other processors
  if (v->type() == SERIAL)
    {
#ifndef NDEBUG
      v_global->close();
#endif
      v_global->localize (*v);
    }
  v->close();
}

void DofMap::enforce_constraints_on_residual (const NonlinearImplicitSystem & system,
                                              NumericVector<Number> * rhs,
                                              NumericVector<Number> const * solution,
                                              bool homogeneous) const
{
  parallel_object_only();

  if (!this->n_constrained_dofs())
    return;

  if (!rhs)
    rhs = system.rhs;
  if (!solution)
    solution = system.solution.get();

  NumericVector<Number> const * solution_local  = nullptr; // will be initialized below
  std::unique_ptr<NumericVector<Number>> solution_built;
  if (solution->type() == SERIAL || solution->type() == GHOSTED)
      solution_local = solution;
  else if (solution->type() == PARALLEL)
    {
      solution_built = NumericVector<Number>::build(this->comm());
      solution_built->init (solution->size(), solution->size(), true, SERIAL);
      solution->localize(*solution_built);
      solution_built->close();
      solution_local = solution_built.get();
    }
  else // unknown solution->type()
    libmesh_error_msg("ERROR: Unknown solution->type() == " << solution->type());

  // We should never hit these asserts because we should error-out in
  // else clause above.  Just to be sure we don't try to use solution_local
  libmesh_assert(solution_local);
  libmesh_assert_equal_to (this, &(system.get_dof_map()));

  for (const auto & pr : _dof_constraints)
    {
      dof_id_type constrained_dof = pr.first;
      if (!this->local_index(constrained_dof))
        continue;

      const DofConstraintRow constraint_row = pr.second;

      Number exact_value = 0;
      for (const auto & j : constraint_row)
        exact_value -= j.second * (*solution_local)(j.first);
      exact_value += (*solution_local)(constrained_dof);
      if (!homogeneous)
        {
          DofConstraintValueMap::const_iterator rhsit =
            _primal_constraint_values.find(constrained_dof);
          if (rhsit != _primal_constraint_values.end())
            exact_value += rhsit->second;
        }

      rhs->set(constrained_dof, exact_value);
    }
}

void DofMap::enforce_constraints_on_jacobian (const NonlinearImplicitSystem & system,
                                              SparseMatrix<Number> * jac) const
{
  parallel_object_only();

  if (!this->n_constrained_dofs())
    return;

  if (!jac)
    jac = system.matrix;

  libmesh_assert_equal_to (this, &(system.get_dof_map()));

  for (const auto & pr : _dof_constraints)
    {
      dof_id_type constrained_dof = pr.first;
      if (!this->local_index(constrained_dof))
        continue;

      const DofConstraintRow constraint_row = pr.second;

      for (const auto & j : constraint_row)
        jac->set(constrained_dof, j.first, -j.second);
      jac->set(constrained_dof, constrained_dof, 1);
    }
}


void DofMap::enforce_adjoint_constraints_exactly (NumericVector<Number> & v,
                                                  unsigned int q) const
{
  parallel_object_only();

  if (!this->n_constrained_dofs())
    return;

  LOG_SCOPE("enforce_adjoint_constraints_exactly()", "DofMap");

  NumericVector<Number> * v_local  = nullptr; // will be initialized below
  NumericVector<Number> * v_global = nullptr; // will be initialized below
  std::unique_ptr<NumericVector<Number>> v_built;
  if (v.type() == SERIAL)
    {
      v_built = NumericVector<Number>::build(this->comm());
      v_built->init(this->n_dofs(), this->n_local_dofs(), true, PARALLEL);
      v_built->close();

      for (dof_id_type i=v_built->first_local_index();
           i<v_built->last_local_index(); i++)
        v_built->set(i, v(i));
      v_built->close();
      v_global = v_built.get();

      v_local = &v;
      libmesh_assert (v_local->closed());
    }
  else if (v.type() == PARALLEL)
    {
      v_built = NumericVector<Number>::build(this->comm());
      v_built->init (v.size(), v.size(), true, SERIAL);
      v.localize(*v_built);
      v_built->close();
      v_local = v_built.get();

      v_global = &v;
    }
  else if (v.type() == GHOSTED)
    {
      v_local = &v;
      v_global = &v;
    }
  else // unknown v.type()
    libmesh_error_msg("ERROR: Unknown v.type() == " << v.type());

  // We should never hit these asserts because we should error-out in
  // else clause above.  Just to be sure we don't try to use v_local
  // and v_global uninitialized...
  libmesh_assert(v_local);
  libmesh_assert(v_global);

  // Do we have any non_homogeneous constraints?
  const AdjointDofConstraintValues::const_iterator
    adjoint_constraint_map_it = _adjoint_constraint_values.find(q);
  const DofConstraintValueMap * constraint_map =
    (adjoint_constraint_map_it == _adjoint_constraint_values.end()) ?
    nullptr : &adjoint_constraint_map_it->second;

  for (const auto & pr : _dof_constraints)
    {
      dof_id_type constrained_dof = pr.first;
      if (!this->local_index(constrained_dof))
        continue;

      const DofConstraintRow constraint_row = pr.second;

      Number exact_value = 0;
      if (constraint_map)
        {
          const DofConstraintValueMap::const_iterator
            adjoint_constraint_it =
            constraint_map->find(constrained_dof);
          if (adjoint_constraint_it != constraint_map->end())
            exact_value = adjoint_constraint_it->second;
        }

      for (const auto & j : constraint_row)
        exact_value += j.second * (*v_local)(j.first);

      v_global->set(constrained_dof, exact_value);
    }

  // If the old vector was serial, we probably need to send our values
  // to other processors
  if (v.type() == SERIAL)
    {
#ifndef NDEBUG
      v_global->close();
#endif
      v_global->localize (v);
    }
  v.close();
}



std::pair<Real, Real>
DofMap::max_constraint_error (const System & system,
                              NumericVector<Number> * v) const
{
  if (!v)
    v = system.solution.get();
  NumericVector<Number> & vec = *v;

  // We'll assume the vector is closed
  libmesh_assert (vec.closed());

  Real max_absolute_error = 0., max_relative_error = 0.;

  const MeshBase & mesh = system.get_mesh();

  libmesh_assert_equal_to (this, &(system.get_dof_map()));

  // indices on each element
  std::vector<dof_id_type> local_dof_indices;

  for (const auto & elem : mesh.active_local_element_ptr_range())
    {
      this->dof_indices(elem, local_dof_indices);
      std::vector<dof_id_type> raw_dof_indices = local_dof_indices;

      // Constraint matrix for each element
      DenseMatrix<Number> C;

      this->build_constraint_matrix (C, local_dof_indices);

      // Continue if the element is unconstrained
      if (!C.m())
        continue;

      libmesh_assert_equal_to (C.m(), raw_dof_indices.size());
      libmesh_assert_equal_to (C.n(), local_dof_indices.size());

      for (auto i : IntRange<unsigned int>(0, C.m()))
        {
          // Recalculate any constrained dof owned by this processor
          dof_id_type global_dof = raw_dof_indices[i];
          if (this->is_constrained_dof(global_dof) &&
              global_dof >= vec.first_local_index() &&
              global_dof < vec.last_local_index())
            {
#ifndef NDEBUG
              DofConstraints::const_iterator
                pos = _dof_constraints.find(global_dof);

              libmesh_assert (pos != _dof_constraints.end());
#endif

              Number exact_value = 0;
              DofConstraintValueMap::const_iterator rhsit =
                _primal_constraint_values.find(global_dof);
              if (rhsit != _primal_constraint_values.end())
                exact_value = rhsit->second;

              for (auto j : IntRange<unsigned int>(0, C.n()))
                {
                  if (local_dof_indices[j] != global_dof)
                    exact_value += C(i,j) *
                      vec(local_dof_indices[j]);
                }

              max_absolute_error = std::max(max_absolute_error,
                                            std::abs(vec(global_dof) - exact_value));
              max_relative_error = std::max(max_relative_error,
                                            std::abs(vec(global_dof) - exact_value)
                                            / std::abs(exact_value));
            }
        }
    }

  return std::pair<Real, Real>(max_absolute_error, max_relative_error);
}



void DofMap::build_constraint_matrix (DenseMatrix<Number> & C,
                                      std::vector<dof_id_type> & elem_dofs,
                                      const bool called_recursively) const
{
  LOG_SCOPE_IF("build_constraint_matrix()", "DofMap", !called_recursively);

  // Create a set containing the DOFs we already depend on
  typedef std::set<dof_id_type> RCSet;
  RCSet dof_set;

  bool we_have_constraints = false;

  // Next insert any other dofs the current dofs might be constrained
  // in terms of.  Note that in this case we may not be done: Those
  // may in turn depend on others.  So, we need to repeat this process
  // in that case until the system depends only on unconstrained
  // degrees of freedom.
  for (const auto & dof : elem_dofs)
    if (this->is_constrained_dof(dof))
      {
        we_have_constraints = true;

        // If the DOF is constrained
        DofConstraints::const_iterator
          pos = _dof_constraints.find(dof);

        libmesh_assert (pos != _dof_constraints.end());

        const DofConstraintRow & constraint_row = pos->second;

        // Constraint rows in p refinement may be empty
        //libmesh_assert (!constraint_row.empty());

        for (const auto & item : constraint_row)
          dof_set.insert (item.first);
      }

  // May be safe to return at this point
  // (but remember to stop the perflog)
  if (!we_have_constraints)
    return;

  for (const auto & dof : elem_dofs)
    dof_set.erase (dof);

  // If we added any DOFS then we need to do this recursively.
  // It is possible that we just added a DOF that is also
  // constrained!
  //
  // Also, we need to handle the special case of an element having DOFs
  // constrained in terms of other, local DOFs
  if (!dof_set.empty() ||  // case 1: constrained in terms of other DOFs
      !called_recursively) // case 2: constrained in terms of our own DOFs
    {
      const unsigned int old_size =
        cast_int<unsigned int>(elem_dofs.size());

      // Add new dependency dofs to the end of the current dof set
      elem_dofs.insert(elem_dofs.end(),
                       dof_set.begin(), dof_set.end());

      // Now we can build the constraint matrix.
      // Note that resize also zeros for a DenseMatrix<Number>.
      C.resize (old_size,
                cast_int<unsigned int>(elem_dofs.size()));

      // Create the C constraint matrix.
      for (unsigned int i=0; i != old_size; i++)
        if (this->is_constrained_dof(elem_dofs[i]))
          {
            // If the DOF is constrained
            DofConstraints::const_iterator
              pos = _dof_constraints.find(elem_dofs[i]);

            libmesh_assert (pos != _dof_constraints.end());

            const DofConstraintRow & constraint_row = pos->second;

            // p refinement creates empty constraint rows
            //    libmesh_assert (!constraint_row.empty());

            for (const auto & item : constraint_row)
              for (unsigned int j=0,
                   n_elem_dofs = cast_int<unsigned int>(elem_dofs.size());
                   j != n_elem_dofs; j++)
                if (elem_dofs[j] == item.first)
                  C(i,j) = item.second;
          }
        else
          {
            C(i,i) = 1.;
          }

      // May need to do this recursively.  It is possible
      // that we just replaced a constrained DOF with another
      // constrained DOF.
      DenseMatrix<Number> Cnew;

      this->build_constraint_matrix (Cnew, elem_dofs, true);

      if ((C.n() == Cnew.m()) &&
          (Cnew.n() == elem_dofs.size())) // If the constraint matrix
        C.right_multiply(Cnew);           // is constrained...

      libmesh_assert_equal_to (C.n(), elem_dofs.size());
    }
}



void DofMap::build_constraint_matrix_and_vector (DenseMatrix<Number> & C,
                                                 DenseVector<Number> & H,
                                                 std::vector<dof_id_type> & elem_dofs,
                                                 int qoi_index,
                                                 const bool called_recursively) const
{
  LOG_SCOPE_IF("build_constraint_matrix_and_vector()", "DofMap", !called_recursively);

  // Create a set containing the DOFs we already depend on
  typedef std::set<dof_id_type> RCSet;
  RCSet dof_set;

  bool we_have_constraints = false;

  // Next insert any other dofs the current dofs might be constrained
  // in terms of.  Note that in this case we may not be done: Those
  // may in turn depend on others.  So, we need to repeat this process
  // in that case until the system depends only on unconstrained
  // degrees of freedom.
  for (const auto & dof : elem_dofs)
    if (this->is_constrained_dof(dof))
      {
        we_have_constraints = true;

        // If the DOF is constrained
        DofConstraints::const_iterator
          pos = _dof_constraints.find(dof);

        libmesh_assert (pos != _dof_constraints.end());

        const DofConstraintRow & constraint_row = pos->second;

        // Constraint rows in p refinement may be empty
        //libmesh_assert (!constraint_row.empty());

        for (const auto & item : constraint_row)
          dof_set.insert (item.first);
      }

  // May be safe to return at this point
  // (but remember to stop the perflog)
  if (!we_have_constraints)
    return;

  for (const auto & dof : elem_dofs)
    dof_set.erase (dof);

  // If we added any DOFS then we need to do this recursively.
  // It is possible that we just added a DOF that is also
  // constrained!
  //
  // Also, we need to handle the special case of an element having DOFs
  // constrained in terms of other, local DOFs
  if (!dof_set.empty() ||  // case 1: constrained in terms of other DOFs
      !called_recursively) // case 2: constrained in terms of our own DOFs
    {
      const DofConstraintValueMap * rhs_values = nullptr;
      if (qoi_index < 0)
        rhs_values = &_primal_constraint_values;
      else
        {
          const AdjointDofConstraintValues::const_iterator
            it = _adjoint_constraint_values.find(qoi_index);
          if (it != _adjoint_constraint_values.end())
            rhs_values = &it->second;
        }

      const unsigned int old_size =
        cast_int<unsigned int>(elem_dofs.size());

      // Add new dependency dofs to the end of the current dof set
      elem_dofs.insert(elem_dofs.end(),
                       dof_set.begin(), dof_set.end());

      // Now we can build the constraint matrix and vector.
      // Note that resize also zeros for a DenseMatrix and DenseVector
      C.resize (old_size,
                cast_int<unsigned int>(elem_dofs.size()));
      H.resize (old_size);

      // Create the C constraint matrix.
      for (unsigned int i=0; i != old_size; i++)
        if (this->is_constrained_dof(elem_dofs[i]))
          {
            // If the DOF is constrained
            DofConstraints::const_iterator
              pos = _dof_constraints.find(elem_dofs[i]);

            libmesh_assert (pos != _dof_constraints.end());

            const DofConstraintRow & constraint_row = pos->second;

            // p refinement creates empty constraint rows
            //    libmesh_assert (!constraint_row.empty());

            for (const auto & item : constraint_row)
              for (unsigned int j=0,
                   n_elem_dofs = cast_int<unsigned int>(elem_dofs.size());
                   j != n_elem_dofs; j++)
                if (elem_dofs[j] == item.first)
                  C(i,j) = item.second;

            if (rhs_values)
              {
                DofConstraintValueMap::const_iterator rhsit =
                  rhs_values->find(elem_dofs[i]);
                if (rhsit != rhs_values->end())
                  H(i) = rhsit->second;
              }
          }
        else
          {
            C(i,i) = 1.;
          }

      // May need to do this recursively.  It is possible
      // that we just replaced a constrained DOF with another
      // constrained DOF.
      DenseMatrix<Number> Cnew;
      DenseVector<Number> Hnew;

      this->build_constraint_matrix_and_vector (Cnew, Hnew, elem_dofs,
                                                qoi_index, true);

      if ((C.n() == Cnew.m()) &&          // If the constraint matrix
          (Cnew.n() == elem_dofs.size())) // is constrained...
        {
          // If x = Cy + h and y = Dz + g
          // Then x = (CD)z + (Cg + h)
          C.vector_mult_add(H, 1, Hnew);

          C.right_multiply(Cnew);
        }

      libmesh_assert_equal_to (C.n(), elem_dofs.size());
    }
}


#ifdef LIBMESH_ENABLE_CONSTRAINTS
void DofMap::check_for_cyclic_constraints()
{
  // Eventually make this officially libmesh_deprecated();
  check_for_constraint_loops();
}

void DofMap::check_for_constraint_loops()
{
  // Create a set containing the DOFs we already depend on
  typedef std::set<dof_id_type> RCSet;
  RCSet unexpanded_set;

  // Use dof_constraints_copy in this method so that we don't
  // mess with _dof_constraints.
  DofConstraints dof_constraints_copy = _dof_constraints;

  for (const auto & i : dof_constraints_copy)
    unexpanded_set.insert(i.first);

  while (!unexpanded_set.empty())
    for (RCSet::iterator i = unexpanded_set.begin();
         i != unexpanded_set.end(); /* nothing */)
      {
        // If the DOF is constrained
        DofConstraints::iterator
          pos = dof_constraints_copy.find(*i);

        libmesh_assert (pos != dof_constraints_copy.end());

        DofConstraintRow & constraint_row = pos->second;

        // Comment out "rhs" parts of this method copied from process_constraints
        // DofConstraintValueMap::iterator rhsit =
        //   _primal_constraint_values.find(*i);
        // Number constraint_rhs = (rhsit == _primal_constraint_values.end()) ?
        //   0 : rhsit->second;

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
              subpos = dof_constraints_copy.find(expandable);

            libmesh_assert (subpos != dof_constraints_copy.end());

            const DofConstraintRow & subconstraint_row = subpos->second;

            for (const auto & item : subconstraint_row)
              {
                if (item.first == expandable)
                  libmesh_error_msg("Constraint loop detected");

                constraint_row[item.first] += item.second * this_coef;
              }

            // Comment out "rhs" parts of this method copied from process_constraints
            // DofConstraintValueMap::const_iterator subrhsit =
            //   _primal_constraint_values.find(expandable);
            // if (subrhsit != _primal_constraint_values.end())
            //   constraint_rhs += subrhsit->second * this_coef;

            constraint_row.erase(expandable);
          }

        // Comment out "rhs" parts of this method copied from process_constraints
        // if (rhsit == _primal_constraint_values.end())
        //   {
        //     if (constraint_rhs != Number(0))
        //       _primal_constraint_values[*i] = constraint_rhs;
        //     else
        //       _primal_constraint_values.erase(*i);
        //   }
        // else
        //   {
        //     if (constraint_rhs != Number(0))
        //       rhsit->second = constraint_rhs;
        //     else
        //       _primal_constraint_values.erase(rhsit);
        //   }

        if (constraints_to_expand.empty())
          i = unexpanded_set.erase(i);
        else
          ++i;
      }
}
#else
void DofMap::check_for_constraint_loops() {}
void DofMap::check_for_cyclic_constraints()
{
  // Do nothing
}
#endif


void DofMap::gather_constraints (MeshAbstract & /*mesh*/,
                                 std::set<dof_id_type> & unexpanded_dofs,
                                 bool /*look_for_constrainees*/)
{
  typedef std::set<dof_id_type> DoF_RCSet;

  // If we have heterogenous adjoint constraints we need to
  // communicate those too.
  const unsigned int max_qoi_num =
    _adjoint_constraint_values.empty() ?
    0 : _adjoint_constraint_values.rbegin()->first;

  // We have to keep recursing while the unexpanded set is
  // nonempty on *any* processor
  bool unexpanded_set_nonempty = !unexpanded_dofs.empty();
  this->comm().max(unexpanded_set_nonempty);

  while (unexpanded_set_nonempty)
    {
      // Let's make sure we don't lose sync in this loop.
      parallel_object_only();

      // Request sets
      DoF_RCSet   dof_request_set;

      // Request sets to send to each processor
      std::map<processor_id_type, std::vector<dof_id_type>>
        requested_dof_ids;

      // And the sizes of each
      std::map<processor_id_type, dof_id_type>
        dof_ids_on_proc;

      // Fill (and thereby sort and uniq!) the main request sets
      for (const auto & unexpanded_dof : unexpanded_dofs)
        {
          DofConstraints::const_iterator
            pos = _dof_constraints.find(unexpanded_dof);

          // If we were asked for a DoF and we don't already have a
          // constraint for it, then we need to check for one.
          if (pos == _dof_constraints.end())
            {
              if (!this->local_index(unexpanded_dof) &&
                  !_dof_constraints.count(unexpanded_dof) )
                dof_request_set.insert(unexpanded_dof);
            }
          // If we were asked for a DoF and we already have a
          // constraint for it, then we need to check if the
          // constraint is recursive.
          else
            {
              const DofConstraintRow & row = pos->second;
              for (const auto & j : row)
                {
                  const dof_id_type constraining_dof = j.first;

                  // If it's non-local and we haven't already got a
                  // constraint for it, we might need to ask for one
                  if (!this->local_index(constraining_dof) &&
                      !_dof_constraints.count(constraining_dof))
                    dof_request_set.insert(constraining_dof);
                }
            }
        }

      // Clear the unexpanded constraint set; we're about to expand it
      unexpanded_dofs.clear();

      // Count requests by processor
      processor_id_type proc_id = 0;
      for (const auto & i : dof_request_set)
        {
          while (i >= _end_df[proc_id])
            proc_id++;
          dof_ids_on_proc[proc_id]++;
        }

      for (auto & pair : dof_ids_on_proc)
        {
          requested_dof_ids[pair.first].reserve(pair.second);
        }

      // Prepare each processor's request set
      proc_id = 0;
      for (const auto & i : dof_request_set)
        {
          while (i >= _end_df[proc_id])
            proc_id++;
          requested_dof_ids[proc_id].push_back(i);
        }

      typedef std::vector<std::pair<dof_id_type, Real>> row_datum;

      typedef std::vector<Number> rhss_datum;

      auto row_gather_functor =
        [this]
        (processor_id_type,
         const std::vector<dof_id_type> & ids,
         std::vector<row_datum> & data)
        {
          // Fill those requests
          const std::size_t query_size = ids.size();

          data.resize(query_size);
          for (std::size_t i=0; i != query_size; ++i)
            {
              dof_id_type constrained = ids[i];
              if (_dof_constraints.count(constrained))
                {
                  DofConstraintRow & row = _dof_constraints[constrained];
                  std::size_t row_size = row.size();
                  data[i].reserve(row_size);
                  for (const auto & j : row)
                    {
                      data[i].push_back(j);

                      // We should never have an invalid constraining
                      // dof id
                      libmesh_assert(j.first != DofObject::invalid_id);

                      // We should never have a 0 constraint
                      // coefficient; that's implicit via sparse
                      // constraint storage
                      //
                      // But we can't easily control how users add
                      // constraints, so we can't safely assert that
                      // we're being efficient here.
                      //
                      // libmesh_assert(j.second);
                    }
                }
              else
                {
                  // We have to distinguish "constraint with no
                  // constraining dofs" (e.g. due to Dirichlet
                  // constraint equations) from "no constraint".
                  // We'll use invalid_id for the latter.
                  data[i].push_back
                    (std::make_pair(DofObject::invalid_id, Real(0)));
                }
            }
        };

      auto rhss_gather_functor =
        [this,
         max_qoi_num]
        (processor_id_type,
         const std::vector<dof_id_type> & ids,
         std::vector<rhss_datum> & data)
        {
          // Fill those requests
          const std::size_t query_size = ids.size();

          data.resize(query_size);
          for (std::size_t i=0; i != query_size; ++i)
            {
              dof_id_type constrained = ids[i];
              data[i].clear();
              if (_dof_constraints.count(constrained))
                {
                  DofConstraintValueMap::const_iterator rhsit =
                    _primal_constraint_values.find(constrained);
                  data[i].push_back
                    ((rhsit == _primal_constraint_values.end()) ?
                     0 : rhsit->second);

                  for (unsigned int q = 0; q != max_qoi_num; ++q)
                    {
                      AdjointDofConstraintValues::const_iterator adjoint_map_it =
                        _adjoint_constraint_values.find(q);

                      if (adjoint_map_it == _adjoint_constraint_values.end())
                        {
                          data[i].push_back(0);
                          continue;
                        }

                      const DofConstraintValueMap & constraint_map =
                        adjoint_map_it->second;

                      DofConstraintValueMap::const_iterator adj_rhsit =
                        constraint_map.find(constrained);
                      data[i].push_back
                        ((adj_rhsit == constraint_map.end()) ?
                         0 : adj_rhsit->second);
                    }
                }
            }
        };

      auto row_action_functor =
        [this,
         & unexpanded_dofs]
        (processor_id_type,
         const std::vector<dof_id_type> & ids,
         const std::vector<row_datum> & data)
        {
          // Add any new constraint rows we've found
          const std::size_t query_size = ids.size();

          for (std::size_t i=0; i != query_size; ++i)
            {
              const dof_id_type constrained = ids[i];

              // An empty row is an constraint with an empty row; for
              // no constraint we use a "no row" placeholder
              if (data[i].empty())
                {
                  DofConstraintRow & row = _dof_constraints[constrained];
                  row.clear();
                }
              else if (data[i][0].first != DofObject::invalid_id)
                {
                  DofConstraintRow & row = _dof_constraints[constrained];
                  row.clear();
                  for (auto & pair : data[i])
                    {
                      libmesh_assert_less(pair.first, this->n_dofs());
                      row[pair.first] = pair.second;
                    }

                  // And prepare to check for more recursive constraints
                  unexpanded_dofs.insert(constrained);
                }
            }
        };

      auto rhss_action_functor =
        [this,
         max_qoi_num]
        (processor_id_type,
         const std::vector<dof_id_type> & ids,
         const std::vector<rhss_datum> & data)
        {
          // Add rhs data for any new constraint rows we've found
          const std::size_t query_size = ids.size();

          for (std::size_t i=0; i != query_size; ++i)
            {
              if (!data[i].empty())
                {
                  dof_id_type constrained = ids[i];
                  if (data[i][0] != Number(0))
                    _primal_constraint_values[constrained] = data[i][0];
                  else
                    _primal_constraint_values.erase(constrained);

                  for (unsigned int q = 0; q != max_qoi_num; ++q)
                    {
                      AdjointDofConstraintValues::iterator adjoint_map_it =
                        _adjoint_constraint_values.find(q);

                      if ((adjoint_map_it == _adjoint_constraint_values.end()) &&
                          data[i][q+1] == Number(0))
                        continue;

                      if (adjoint_map_it == _adjoint_constraint_values.end())
                        adjoint_map_it = _adjoint_constraint_values.insert
                          (std::make_pair(q,DofConstraintValueMap())).first;

                      DofConstraintValueMap & constraint_map =
                        adjoint_map_it->second;

                      if (data[i][q+1] != Number(0))
                        constraint_map[constrained] =
                          data[i][q+1];
                      else
                        constraint_map.erase(constrained);
                    }
                }
            }

        };

      // Now request constraint rows from other processors
      row_datum * row_ex = nullptr;
      Parallel::pull_parallel_vector_data
        (this->comm(), requested_dof_ids, row_gather_functor,
         row_action_functor, row_ex);

      // And request constraint right hand sides from other procesors
      rhss_datum * rhs_ex = nullptr;
      Parallel::pull_parallel_vector_data
        (this->comm(), requested_dof_ids, rhss_gather_functor,
         rhss_action_functor, rhs_ex);

      // We have to keep recursing while the unexpanded set is
      // nonempty on *any* processor
      unexpanded_set_nonempty = !unexpanded_dofs.empty();
      this->comm().max(unexpanded_set_nonempty);
    }
}

void DofMap::add_constraints_to_send_list()
{
  // This function must be run on all processors at once
  parallel_object_only();

  // Return immediately if there's nothing to gather
  if (this->n_processors() == 1)
    return;

  // We might get to return immediately if none of the processors
  // found any constraints
  unsigned int has_constraints = !_dof_constraints.empty();
  this->comm().max(has_constraints);
  if (!has_constraints)
    return;

  for (const auto & i : _dof_constraints)
    {
      dof_id_type constrained_dof = i.first;

      // We only need the dependencies of our own constrained dofs
      if (!this->local_index(constrained_dof))
        continue;

      const DofConstraintRow & constraint_row = i.second;
      for (const auto & j : constraint_row)
        {
          dof_id_type constraint_dependency = j.first;

          // No point in adding one of our own dofs to the send_list
          if (this->local_index(constraint_dependency))
            continue;

          _send_list.push_back(constraint_dependency);
        }
    }
}



#endif // LIBMESH_ENABLE_CONSTRAINTS


#ifdef LIBMESH_ENABLE_AMR

void DofMap::constrain_p_dofs (unsigned int var,
                               const Elem * elem,
                               unsigned int s,
                               unsigned int p)
{
  // We're constraining dofs on elem which correspond to p refinement
  // levels above p - this only makes sense if elem's p refinement
  // level is above p.
  libmesh_assert_greater (elem->p_level(), p);
  libmesh_assert_less (s, elem->n_sides());

  const unsigned int sys_num = this->sys_number();
  const unsigned int dim = elem->dim();
  ElemType type = elem->type();
  FEType low_p_fe_type = this->variable_type(var);
  FEType high_p_fe_type = this->variable_type(var);
  low_p_fe_type.order = static_cast<Order>(low_p_fe_type.order + p);
  high_p_fe_type.order = static_cast<Order>(high_p_fe_type.order +
                                            elem->p_level());

  const unsigned int n_nodes = elem->n_nodes();
  for (unsigned int n = 0; n != n_nodes; ++n)
    if (elem->is_node_on_side(n, s))
      {
        const Node & node = elem->node_ref(n);
        const unsigned int low_nc =
          FEInterface::n_dofs_at_node (dim, low_p_fe_type, type, n);
        const unsigned int high_nc =
          FEInterface::n_dofs_at_node (dim, high_p_fe_type, type, n);

        // since we may be running this method concurrently
        // on multiple threads we need to acquire a lock
        // before modifying the _dof_constraints object.
        Threads::spin_mutex::scoped_lock lock(Threads::spin_mtx);

        if (elem->is_vertex(n))
          {
            // Add "this is zero" constraint rows for high p vertex
            // dofs
            for (unsigned int i = low_nc; i != high_nc; ++i)
              {
                _dof_constraints[node.dof_number(sys_num,var,i)].clear();
                _primal_constraint_values.erase(node.dof_number(sys_num,var,i));
              }
          }
        else
          {
            const unsigned int total_dofs = node.n_comp(sys_num, var);
            libmesh_assert_greater_equal (total_dofs, high_nc);
            // Add "this is zero" constraint rows for high p
            // non-vertex dofs, which are numbered in reverse
            for (unsigned int j = low_nc; j != high_nc; ++j)
              {
                const unsigned int i = total_dofs - j - 1;
                _dof_constraints[node.dof_number(sys_num,var,i)].clear();
                _primal_constraint_values.erase(node.dof_number(sys_num,var,i));
              }
          }
      }
}

#endif // LIBMESH_ENABLE_AMR


#ifdef LIBMESH_ENABLE_DIRICHLET
void DofMap::add_dirichlet_boundary (const DirichletBoundary & dirichlet_boundary)
{
  _dirichlet_boundaries->push_back(new DirichletBoundary(dirichlet_boundary));
}


void DofMap::add_adjoint_dirichlet_boundary (const DirichletBoundary & dirichlet_boundary,
                                             unsigned int qoi_index)
{
  unsigned int old_size = cast_int<unsigned int>
    (_adjoint_dirichlet_boundaries.size());
  for (unsigned int i = old_size; i <= qoi_index; ++i)
    _adjoint_dirichlet_boundaries.push_back(new DirichletBoundaries());

  _adjoint_dirichlet_boundaries[qoi_index]->push_back
    (new DirichletBoundary(dirichlet_boundary));
}


bool DofMap::has_adjoint_dirichlet_boundaries(unsigned int q) const
{
  if (_adjoint_dirichlet_boundaries.size() > q)
    return true;

  return false;
}


const DirichletBoundaries *
DofMap::get_adjoint_dirichlet_boundaries(unsigned int q) const
{
  libmesh_assert_greater(_adjoint_dirichlet_boundaries.size(),q);
  return _adjoint_dirichlet_boundaries[q];
}


DirichletBoundaries *
DofMap::get_adjoint_dirichlet_boundaries(unsigned int q)
{
  unsigned int old_size = cast_int<unsigned int>
    (_adjoint_dirichlet_boundaries.size());
  for (unsigned int i = old_size; i <= q; ++i)
    _adjoint_dirichlet_boundaries.push_back(new DirichletBoundaries());

  return _adjoint_dirichlet_boundaries[q];
}


void DofMap::remove_dirichlet_boundary (const DirichletBoundary & boundary_to_remove)
{
  // Find a boundary condition matching the one to be removed
  auto lam = [&boundary_to_remove](const DirichletBoundary * bdy)
    {return bdy->b == boundary_to_remove.b && bdy->variables == boundary_to_remove.variables;};

  auto it = std::find_if(_dirichlet_boundaries->begin(), _dirichlet_boundaries->end(), lam);

  // Delete it and remove it
  libmesh_assert (it != _dirichlet_boundaries->end());
  delete *it;
  _dirichlet_boundaries->erase(it);
}


void DofMap::remove_adjoint_dirichlet_boundary (const DirichletBoundary & boundary_to_remove,
                                                unsigned int qoi_index)
{
  libmesh_assert_greater(_adjoint_dirichlet_boundaries.size(),
                         qoi_index);

  auto lam = [&boundary_to_remove](const DirichletBoundary * bdy)
    {return bdy->b == boundary_to_remove.b && bdy->variables == boundary_to_remove.variables;};

  auto it = std::find_if(_adjoint_dirichlet_boundaries[qoi_index]->begin(),
                         _adjoint_dirichlet_boundaries[qoi_index]->end(),
                         lam);

  // Delete it and remove it
  libmesh_assert (it != _adjoint_dirichlet_boundaries[qoi_index]->end());
  delete *it;
  _adjoint_dirichlet_boundaries[qoi_index]->erase(it);
}


DirichletBoundaries::~DirichletBoundaries()
{
  for (auto & item : *this)
    delete item;
}

void DofMap::check_dirichlet_bcid_consistency (const MeshBase & mesh,
                                               const DirichletBoundary & boundary) const
{
  const std::set<boundary_id_type>& mesh_bcids = mesh.get_boundary_info().get_boundary_ids();
  const std::set<boundary_id_type>& dbc_bcids = boundary.b;

  // DirichletBoundary id sets should be consistent across all ranks
  libmesh_assert(mesh.comm().verify(dbc_bcids.size()));

  for (const auto & bc_id : dbc_bcids)
    {
      // DirichletBoundary id sets should be consistent across all ranks
      libmesh_assert(mesh.comm().verify(bc_id));

      bool found_bcid = (mesh_bcids.find(bc_id) != mesh_bcids.end());

      // On a distributed mesh, boundary id sets may *not* be
      // consistent across all ranks, since not all ranks see all
      // boundaries
      mesh.comm().max(found_bcid);

      if (!found_bcid)
        libmesh_error_msg("Could not find Dirichlet boundary id " << bc_id << " in mesh!");
    }
}

#endif // LIBMESH_ENABLE_DIRICHLET


#ifdef LIBMESH_ENABLE_PERIODIC

void DofMap::add_periodic_boundary (const PeriodicBoundaryBase & periodic_boundary)
{
  // See if we already have a periodic boundary associated myboundary...
  PeriodicBoundaryBase * existing_boundary = _periodic_boundaries->boundary(periodic_boundary.myboundary);

  if (existing_boundary == nullptr)
    {
      // ...if not, clone the input (and its inverse) and add them to the PeriodicBoundaries object
      PeriodicBoundaryBase * boundary = periodic_boundary.clone().release();
      PeriodicBoundaryBase * inverse_boundary = periodic_boundary.clone(PeriodicBoundaryBase::INVERSE).release();

      // _periodic_boundaries takes ownership of the pointers
      _periodic_boundaries->insert(std::make_pair(boundary->myboundary, boundary));
      _periodic_boundaries->insert(std::make_pair(inverse_boundary->myboundary, inverse_boundary));
    }
  else
    {
      // ...otherwise, merge this object's variable IDs with the existing boundary object's.
      existing_boundary->merge(periodic_boundary);

      // Do the same merging process for the inverse boundary.  Note: the inverse better already exist!
      PeriodicBoundaryBase * inverse_boundary = _periodic_boundaries->boundary(periodic_boundary.pairedboundary);
      libmesh_assert(inverse_boundary);
      inverse_boundary->merge(periodic_boundary);
    }
}




void DofMap::add_periodic_boundary (const PeriodicBoundaryBase & boundary,
                                    const PeriodicBoundaryBase & inverse_boundary)
{
  libmesh_assert_equal_to (boundary.myboundary, inverse_boundary.pairedboundary);
  libmesh_assert_equal_to (boundary.pairedboundary, inverse_boundary.myboundary);

  // Allocate copies on the heap.  The _periodic_boundaries object will manage this memory.
  // Note: this also means that the copy constructor for the PeriodicBoundary (or user class
  // derived therefrom) must be implemented!
  // PeriodicBoundary * p_boundary = new PeriodicBoundary(boundary);
  // PeriodicBoundary * p_inverse_boundary = new PeriodicBoundary(inverse_boundary);

  // We can't use normal copy construction since this leads to slicing with derived classes.
  // Use clone() "virtual constructor" instead.  But, this *requires* user to override the clone()
  // method.  Note also that clone() allocates memory.  In this case, the _periodic_boundaries object
  // takes responsibility for cleanup.
  PeriodicBoundaryBase * p_boundary = boundary.clone().release();
  PeriodicBoundaryBase * p_inverse_boundary = inverse_boundary.clone().release();

  // Add the periodic boundary and its inverse to the PeriodicBoundaries data structure.  The
  // PeriodicBoundaries data structure takes ownership of the pointers.
  _periodic_boundaries->insert(std::make_pair(p_boundary->myboundary, p_boundary));
  _periodic_boundaries->insert(std::make_pair(p_inverse_boundary->myboundary, p_inverse_boundary));
}


#endif


} // namespace libMesh
