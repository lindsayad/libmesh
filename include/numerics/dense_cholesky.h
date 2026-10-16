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

#ifndef LIBMESH_DENSE_CHOLESKY_H
#define LIBMESH_DENSE_CHOLESKY_H

// The Cholesky factorization and solve that DenseMatrix uses, written
// against any matrix with operator()(i,j) and vector with operator()(i)
// so that device code, and code with small fixed-size storage, can use
// the same routines.

#include "libmesh/libmesh_common.h"
#include "libmesh/libmesh_device.h"

#include <cmath>

namespace libMesh
{

/**
 * Overwrites the lower triangle of the symmetric positive definite
 * \p n by \p n matrix \p A with its Cholesky factor L, A = L L^T.  The
 * algorithm follows the Cholesky decomposition of Numerical Recipes in
 * C.
 */
template <typename Matrix>
LIBMESH_DEVICE_INLINE
void cholesky_decompose (const unsigned int n,
                         Matrix & A)
{
  using std::sqrt;

  for (unsigned int i=0; i<n; ++i)
    for (unsigned int j=i; j<n; ++j)
      {
        for (unsigned int k=0; k<i; ++k)
          A(i,j) -= A(i,k) * A(j,k);

        if (i == j)
          {
#ifndef LIBMESH_USE_COMPLEX_NUMBERS
            libmesh_error_msg_if(A(i,j) <= 0.0,
                                 "Error! Can only use Cholesky decomposition with symmetric positive definite matrices.");
#endif

            A(i,i) = sqrt(A(i,j));
          }
        else
          A(j,i) = A(i,j) / A(i,i);
      }
}

/**
 * Solves L L^T x = b for \p x, given the Cholesky factor L of an \p n
 * by \p n matrix in the lower triangle of \p A, as left by
 * cholesky_decompose().
 */
template <typename Matrix, typename RHS, typename Solution>
LIBMESH_DEVICE_INLINE
void cholesky_back_substitute (const unsigned int n,
                               const Matrix & A,
                               const RHS & b,
                               Solution & x)
{
  // Solve for Ly=b
  for (unsigned int i=0; i<n; ++i)
    {
      auto temp = b(i);

      for (unsigned int k=0; k<i; ++k)
        temp -= A(i,k)*x(k);

      x(i) = temp / A(i,i);
    }

  // Solve for L^T x = y
  for (unsigned int i=0; i<n; ++i)
    {
      const unsigned int ib = (n-1)-i;

      for (unsigned int k=(ib+1); k<n; ++k)
        x(ib) -= A(k,ib) * x(k);

      x(ib) /= A(ib,ib);
    }
}

} // namespace libMesh

#endif // LIBMESH_DENSE_CHOLESKY_H
