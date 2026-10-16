// The libMesh Finite Element Library.
// Copyright (C) 2002-2026 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

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

#ifndef LIBMESH_FE_DUAL_COEFFS_H
#define LIBMESH_FE_DUAL_COEFFS_H

// The biorthogonal dual basis coefficients, written against matrices
// with operator()(i,j) so that FEGenericBase and device code compute
// them with the same routine, and templated on the scalar type so that
// their derivatives can be carried by dual numbers.

#include "libmesh/dense_cholesky.h"
#include "libmesh/enum_elem_type.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/libmesh_device.h"

namespace libMesh
{

/**
 * \returns The number of vertices of an element whose \p n_shapes
 * Lagrange shape functions are biorthogonalized against a transformed
 * basis in compute_dual_coeffs(), and zero for every other basis.  The
 * mid-edge nodes of such an element follow its vertices, one per edge.
 */
LIBMESH_DEVICE_INLINE
unsigned int dual_transform_n_vertices (const FEFamily family,
                                        const ElemType type,
                                        const unsigned int n_shapes)
{
  if (family != LAGRANGE && family != L2_LAGRANGE)
    return 0;

  // SECOND order on TRI7 is the TRI6 basis, with no degree of freedom
  // at the centroid node, and L2_LAGRANGE shares the LAGRANGE shape
  // functions.
  if ((type == TRI6 || type == TRI7) && n_shapes == 6)
    return 3;

  if (type == QUAD8 && n_shapes == 8)
    return 4;

  return 0;
}

/**
 * \returns The element-local number of vertex \p v, 0 or 1, of the edge
 * of mid-edge node \p n_vertices + \p e, as in
 * Elem::second_order_adjacent_vertex() for the elements accepted by
 * dual_transform_n_vertices().
 */
LIBMESH_DEVICE_INLINE
unsigned int dual_transform_adjacent_vertex (const unsigned int n_vertices,
                                             const unsigned int e,
                                             const unsigned int v)
{
  // The last edge of a triangle or quadrilateral joins vertex 0 to the
  // last vertex; edge e otherwise joins vertices e and e + 1.
  return (e + 1 == n_vertices) ? (v ? e : 0) : e + v;
}

/**
 * Computes the coefficients of the dual basis biorthogonal to the \p
 * n_shapes shape functions \p phi of an element of \p type, from the
 * shape function values \p phi(i, qp) and the weights \p JxW(qp) at
 * \p n_qp quadrature points.  \p dual_coeff(i, j) is the coefficient of
 * shape function i in dual shape function j.  \p A and \p D are
 * workspaces of at least \p n_shapes by \p n_shapes entries, and \p
 * rhs and \p x of at least \p n_shapes, all of the scalar type of \p
 * JxW.
 */
template <typename Weights, typename Shapes, typename Matrix, typename Vector, typename Coeffs>
LIBMESH_DEVICE_INLINE
void compute_dual_coeffs (const FEFamily family,
                          const ElemType type,
                          const unsigned int n_shapes,
                          const unsigned int n_qp,
                          const Weights & JxW,
                          const Shapes & phi,
                          Matrix & A,
                          Matrix & D,
                          Vector & rhs,
                          Vector & x,
                          Coeffs & dual_coeff)
{
  for (unsigned int i=0; i<n_shapes; ++i)
    for (unsigned int j=0; j<n_shapes; ++j)
      {
        A(i,j) = 0;
        D(i,j) = 0;
      }

  for (unsigned int i=0; i<n_shapes; ++i)
    for (unsigned int qp=0; qp<n_qp; ++qp)
      {
        D(i,i) += JxW(qp)*phi(i,qp);
        for (unsigned int j=0; j<n_shapes; ++j)
          A(i,j) += JxW(qp)*phi(i,qp)*phi(j,qp);
      }

  // The integral positivity condition on the dual basis (Popp, Wohlmuth, Gee and Wall, SIAM J. Sci.
  // Comput. 34(4):B421-B446, 2012, Eq. (4.2)) fails on QUAD8 and TRI6: D(k,k) = \int N_k is -1/3 at a
  // QUAD8 corner and exactly 0 at a TRI6 vertex (their Eqs. (4.3) and (4.4)), the latter leaving those
  // dual shape functions identically zero. Biorthogonalize instead against the locally quadratic
  // transformed basis Ntilde = T N of their Sec. 4.4.1, in which each vertex absorbs a fraction alpha
  // of its adjacent mid-edge shapes. That replaces D by T^-1 diag(T d), which for this T is the sparse
  // update below. alpha = 1/5 is recommended there, and makes the weights strictly positive (QUAD8
  // 1/5 and 4/5, TRI6 1/15 and 1/10) while preserving the partition of unity. T = I elsewhere.
  const unsigned int n_vertices = dual_transform_n_vertices(family, type, n_shapes);
  if (n_vertices)
    {
      const Real alpha = Real(1)/5;
      // Mid-edge nodes are the trailing indices, and only vertex entries are written, so D(m,m) here
      // is never a value an earlier iteration modified.
      for (unsigned int e=0; e<n_vertices; ++e)
        {
          const unsigned int m = n_vertices + e;
          for (unsigned int v=0; v<2; ++v)
            {
              const unsigned int vertex = dual_transform_adjacent_vertex(n_vertices, e, v);
              D(vertex, vertex) += alpha*D(m,m);
              D(vertex, m)      -= alpha*D(m,m);
            }
        }
    }

  // dual_coeff = A^-1*D
  cholesky_decompose(n_shapes, A);

  for (unsigned int j=0; j<n_shapes; ++j)
    {
      for (unsigned int i=0; i<n_shapes; ++i)
        rhs(i) = D(i, j);

      cholesky_back_substitute(n_shapes, A, rhs, x);

      for (unsigned int row=0; row<n_shapes; ++row)
        dual_coeff(row, j) = x(row);
    }
}

} // namespace libMesh

#endif // LIBMESH_FE_DUAL_COEFFS_H
