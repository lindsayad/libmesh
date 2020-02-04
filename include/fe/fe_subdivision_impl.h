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

#ifndef LIBMESH_FE_SUBDIVISION_IMPL_H
#define LIBMESH_FE_SUBDIVISION_IMPL_H

// Local includes
#include "libmesh/fe.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/fe_type.h"
#include "libmesh/quadrature.h"
#include "libmesh/face_tri3_subdivision.h"
#include "libmesh/fe_macro.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/utility.h"


namespace libMesh
{

template <typename RealType>
FESubdivisionTempl<RealType>::FESubdivisionTempl(const FEType & fet) :
    FE<2,SUBDIVISION,RealType>(fet)
{
  // Only 2D meshes in 3D space are supported
  libmesh_assert_equal_to(LIBMESH_DIM, 3);
}



template <typename RealType>
void FESubdivisionTempl<RealType>::init_subdivision_matrix(DenseMatrix<Real> & A,
                                                           unsigned int valence)
{
  A.resize(valence + 12, valence + 12);

  // A = (S11 0; S21 S22), see Cirak et al.,
  // Int. J. Numer. Meth. Engng. 2000; 47:2039-2072, Appendix A.2.

  // First, set the static S21 part
  A(valence+ 1,0        ) = 0.125;
  A(valence+ 1,1        ) = 0.375;
  A(valence+ 1,valence  ) = 0.375;
  A(valence+ 2,0        ) = 0.0625;
  A(valence+ 2,1        ) = 0.625;
  A(valence+ 2,2        ) = 0.0625;
  A(valence+ 2,valence  ) = 0.0625;
  A(valence+ 3,0        ) = 0.125;
  A(valence+ 3,1        ) = 0.375;
  A(valence+ 3,2        ) = 0.375;
  A(valence+ 4,0        ) = 0.0625;
  A(valence+ 4,1        ) = 0.0625;
  A(valence+ 4,valence-1) = 0.0625;
  A(valence+ 4,valence  ) = 0.625;
  A(valence+ 5,0        ) = 0.125;
  A(valence+ 5,valence-1) = 0.375;
  A(valence+ 5,valence  ) = 0.375;
  A(valence+ 6,1        ) = 0.375;
  A(valence+ 6,valence  ) = 0.125;
  A(valence+ 7,1        ) = 0.375;
  A(valence+ 8,1        ) = 0.375;
  A(valence+ 8,2        ) = 0.125;
  A(valence+ 9,1        ) = 0.125;
  A(valence+ 9,valence  ) = 0.375;
  A(valence+10,valence  ) = 0.375;
  A(valence+11,valence-1) = 0.125;
  A(valence+11,valence  ) = 0.375;

  // Next, set the static S22 part
  A(valence+ 1,valence+1) = 0.125;
  A(valence+ 2,valence+1) = 0.0625;
  A(valence+ 2,valence+2) = 0.0625;
  A(valence+ 2,valence+3) = 0.0625;
  A(valence+ 3,valence+3) = 0.125;
  A(valence+ 4,valence+1) = 0.0625;
  A(valence+ 4,valence+4) = 0.0625;
  A(valence+ 4,valence+5) = 0.0625;
  A(valence+ 5,valence+5) = 0.125;
  A(valence+ 6,valence+1) = 0.375;
  A(valence+ 6,valence+2) = 0.125;
  A(valence+ 7,valence+1) = 0.125;
  A(valence+ 7,valence+2) = 0.375;
  A(valence+ 7,valence+3) = 0.125;
  A(valence+ 8,valence+2) = 0.125;
  A(valence+ 8,valence+3) = 0.375;
  A(valence+ 9,valence+1) = 0.375;
  A(valence+ 9,valence+4) = 0.125;
  A(valence+10,valence+1) = 0.125;
  A(valence+10,valence+4) = 0.375;
  A(valence+10,valence+5) = 0.125;
  A(valence+11,valence+4) = 0.125;
  A(valence+11,valence+5) = 0.375;

  // Last, set the S11 part: first row
  std::vector<Real> weights;
  loop_subdivision_mask(weights, valence);
  for (unsigned int i = 0; i <= valence; ++i)
    A(0,i) = weights[i];

  // second row
  A(1,0) = 0.375;
  A(1,1) = 0.375;
  A(1,2) = 0.125;
  A(1,valence) = 0.125;

  // third to second-to-last rows
  for (unsigned int i = 2; i < valence; ++i)
    {
      A(i,0  ) = 0.375;
      A(i,i-1) = 0.125;
      A(i,i  ) = 0.375;
      A(i,i+1) = 0.125;
    }

  // last row
  A(valence,0) = 0.375;
  A(valence,1) = 0.125;
  A(valence,valence-1) = 0.125;
  A(valence,valence  ) = 0.375;
}



template <typename RealType>
RealType FESubdivisionTempl<RealType>::regular_shape(const unsigned int i,
                                                     const RealType & v,
                                                     const RealType & w)
{
  // These are the 12 quartic box splines, see Cirak et al.,
  // Int. J. Numer. Meth. Engng. 2000; 47:2039-2072, Appendix A.1.

  const auto u = 1 - v - w;
  libmesh_assert_less_equal(0, v);
  libmesh_assert_less_equal(0, w);
  libmesh_assert_less_equal(0, u);

  using Utility::pow;
  const Real factor = 1. / 12;

  switch (i)
    {
    case 0:
      return factor*(pow<4>(u) + 2*u*u*u*v);
    case 1:
      return factor*(pow<4>(u) + 2*u*u*u*w);
    case 2:
      return factor*(pow<4>(u) + 2*u*u*u*w + 6*u*u*u*v + 6*u*u*v*w + 12*u*u*v*v + 6*u*v*v*w + 6*u*v*v*v +
                     2*v*v*v*w + pow<4>(v));
    case 3:
      return factor*(6*pow<4>(u) + 24*u*u*u*w + 24*u*u*w*w + 8*u*w*w*w + pow<4>(w) + 24*u*u*u*v +
                     60*u*u*v*w + 36*u*v*w*w + 6*v*w*w*w + 24*u*u*v*v + 36*u*v*v*w + 12*v*v*w*w + 8*u*v*v*v +
                     6*v*v*v*w + pow<4>(v));
    case 4:
      return factor*(pow<4>(u) + 6*u*u*u*w + 12*u*u*w*w + 6*u*w*w*w + pow<4>(w) + 2*u*u*u*v + 6*u*u*v*w +
                     6*u*v*w*w + 2*v*w*w*w);
    case 5:
      return factor*(2*u*v*v*v + pow<4>(v));
    case 6:
      return factor*(pow<4>(u) + 6*u*u*u*w + 12*u*u*w*w + 6*u*w*w*w + pow<4>(w) + 8*u*u*u*v + 36*u*u*v*w +
                     36*u*v*w*w + 8*v*w*w*w + 24*u*u*v*v + 60*u*v*v*w + 24*v*v*w*w + 24*u*v*v*v + 24*v*v*v*w + 6*pow<4>(v));
    case 7:
      return factor*(pow<4>(u) + 8*u*u*u*w + 24*u*u*w*w + 24*u*w*w*w + 6*pow<4>(w) + 6*u*u*u*v + 36*u*u*v*w +
                     60*u*v*w*w + 24*v*w*w*w + 12*u*u*v*v + 36*u*v*v*w + 24*v*v*w*w + 6*u*v*v*v + 8*v*v*v*w + pow<4>(v));
    case 8:
      return factor*(2*u*w*w*w + pow<4>(w));
    case 9:
      return factor*(2*v*v*v*w + pow<4>(v));
    case 10:
      return factor*(2*u*w*w*w + pow<4>(w) + 6*u*v*w*w + 6*v*w*w*w + 6*u*v*v*w + 12*v*v*w*w + 2*u*v*v*v +
                     6*v*v*v*w + pow<4>(v));
    case 11:
      return factor*(pow<4>(w) + 2*v*w*w*w);

    default:
      libmesh_error_msg("Invalid i = " << i);
    }
}



template <typename RealType>
RealType FESubdivisionTempl<RealType>::regular_shape_deriv(const unsigned int i,
                                                           const unsigned int j,
                                                           const RealType & v,
                                                           const RealType & w)
{
  const auto u = 1 - v - w;
  const Real factor = 1. / 12;

  switch (j) // j=0: xi-directional derivative, j=1: eta-directional derivative
    {
    case 0: // xi derivatives
      {
        switch (i) // shape function number
          {
          case 0:
            return factor*(-6*v*u*u - 2*u*u*u);
          case 1:
            return factor*(-4*u*u*u - 6*u*u*w);
          case 2:
            return factor*(-2*v*v*v - 6*v*v*u + 6*v*u*u + 2*u*u*u);
          case 3:
            return factor*(-4*v*v*v - 24*v*v*u - 24*v*u*u - 18*v*v*w - 48*v*u*w - 12*u*u*w -
                           12*v*w*w - 12*u*w*w - 2*w*w*w);
          case 4:
            return factor*(-6*v*u*u - 2*u*u*u - 12*v*u*w-12*u*u*w - 6*v*w*w - 18*u*w*w - 4*w*w*w);
          case 5:
            return factor*(2*v*v*v + 6*v*v*u);
          case 6:
            return factor*(24*v*v*u + 24*v*u*u + 4*u*u*u + 12*v*v*w + 48*v*u*w + 18*u*u*w +
                           12*v*w*w + 12*u*w*w + 2*w*w*w);
          case 7:
            return factor*(-2*v*v*v - 6*v*v*u + 6*v*u*u + 2*u*u*u - 12*v*v*w + 12*u*u*w -
                           12*v*w*w + 12*u*w*w);
          case 8:
            return -w*w*w/6;
          case 9:
            return factor*(4*v*v*v + 6*v*v*w);
          case 10:
            return factor*(2*v*v*v + 6*v*v*u + 12*v*v*w + 12*v*u*w + 18*v*w*w + 6*u*w*w + 4*w*w*w);
          case 11:
            return w*w*w/6;
          default:
            libmesh_error_msg("Invalid i = " << i);
          }
      }
    case 1: // eta derivatives
      {
        switch (i) // shape function number
          {
          case 0:
            return factor*(-6*v*u*u - 4*u*u*u);
          case 1:
            return factor*(-2*u*u*u - 6*u*u*w);
          case 2:
            return factor*(-4*v*v*v - 18*v*v*u - 12*v*u*u - 2*u*u*u - 6*v*v*w - 12*v*u*w -
                           6*u*u*w);
          case 3:
            return factor*(-2*v*v*v-12*v*v*u - 12*v*u*u - 12*v*v*w - 48*v*u*w - 24*u*u*w -
                           18*v*w*w - 24*u*w*w - 4*w*w*w);
          case 4:
            return factor*(2*u*u*u + 6*u*u*w - 6*u*w*w - 2*w*w*w);
          case 5:
            return -v*v*v/6;
          case 6:
            return factor*(12*v*v*u + 12*v*u*u + 2*u*u*u - 12*v*v*w + 6*u*u*w - 12*v*w*w -
                           6*u*w*w - 2*w*w*w);
          case 7:
            return factor*(2*v*v*v + 12*v*v*u + 18*v*u*u + 4*u*u*u + 12*v*v*w + 48*v*u*w +
                           24*u*u*w + 12*v*w*w + 24*u*w*w);
          case 8:
            return factor*(6*u*w*w + 2*w*w*w);
          case 9:
            return v*v*v/6;
          case 10:
            return factor*(4*v*v*v + 6*v*v*u + 18*v*v*w + 12*v*u*w + 12*v*w*w + 6*u*w*w +
                           2*w*w*w);
          case 11:
            return factor*(6*v*w*w + 4*w*w*w);
          default:
            libmesh_error_msg("Invalid i = " << i);
          }
      }
    default:
      libmesh_error_msg("Invalid j = " << j);
    }
}



#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
template <typename RealType>
RealType FESubdivisionTempl<RealType>::regular_shape_second_deriv(const unsigned int i,
                                                                  const unsigned int j,
                                                                  const RealType & v,
                                                                  const RealType & w)
{
  const auto u = 1 - v - w;
  const Real factor = 1. / 12;

  switch (j)
    {
    case 0: // xi-xi derivative
      {
        switch (i) // shape function number
          {
          case 0:
            return v*u;
          case 1:
            return u*u + u*w;
          case 2:
            return -2*v*u;
          case 3:
            return v*v - 2*u*u + v*w - 2*u*w;
          case 4:
            return v*u + v*w + u*w + w*w;
          case 5:
            return v*u;
          case 6:
            return factor*(-24*v*v + 12*u*u - 24*v*w + 12*u*w);
          case 7:
            return -2*v*u - 2*v*w - 2*u*w - 2*w*w;
          case 8:
            return 0.;
          case 9:
            return v*v + v*w;
          case 10:
            return v*u + v*w + u*w + w*w;
          case 11:
            return 0.;
          default:
            libmesh_error_msg("Invalid i = " << i);
          }
      }
    case 1: //eta-xi derivative
      {
        switch (i)
          {
          case 0:
            return factor*(12*v*u + 6*u*u);
          case 1:
            return factor*(6*u*u + 12*u*w);
          case 2:
            return factor*(6*v*v - 12*v*u - 6*u*u);
          case 3:
            return factor*(6*v*v - 12*u*u + 24*v*w + 6*w*w);
          case 4:
            return factor*(-6*u*u - 12*u*w + 6*w*w);
          case 5:
            return -v*v/2.;
          case 6:
            return factor*(-12*v*v + 6*u*u - 24*v*w - 12*u*w - 6*w*w);
          case 7:
            return factor*(-6*v*v - 12*v*u + 6*u*u - 24*v*w - 12*w*w);
          case 8:
            return -w*w/2.;
          case 9:
            return v*v/2.;
          case 10:
            return factor*(6*v*v + 12*v*u + 24*v*w + 12*u*w + 6*w*w);
          case 11:
            return w*w/2.;
          default:
            libmesh_error_msg("Invalid i = " << i);
          }
      }
    case 2: // eta-eta derivative
      {
        switch (i)
          {
          case 0:
            return v*u + u*u;
          case 1:
            return u*w;
          case 2:
            return v*v + v*u + v*w + u*w;
          case 3:
            return -2*v*u - 2*u*u + v*w + w*w;
          case 4:
            return -2*u*w;
          case 5:
            return 0.;
          case 6:
            return -2*v*v - 2*v*u - 2*v*w - 2*u*w;
          case 7:
            return v*u + u*u - 2*v*w - 2*w*w;
          case 8:
            return u*w;
          case 9:
            return 0.;
          case 10:
            return v*v + v*u + v*w + u*w;
          case 11:
            return v*w + w*w;
          default:
            libmesh_error_msg("Invalid i = " << i);
          }
      }
    default:
      libmesh_error_msg("Invalid j = " << j);
    }
}

#endif //LIBMESH_ENABLE_SECOND_DERIVATIVES


template <typename RealType>
void FESubdivisionTempl<RealType>::loop_subdivision_mask(std::vector<Real> & weights,
                                                         const unsigned int valence)
{
  libmesh_assert_greater(valence, 0);
  const Real cs = std::cos(2 * libMesh::pi / valence);
  const Real nb_weight = (0.625 - Utility::pow<2>(0.375 + 0.25 * cs)) / valence;
  weights.resize(1 + valence, nb_weight);
  weights[0] = 1.0 - valence * nb_weight;
}



template <typename RealType>
void FESubdivisionTempl<RealType>::init_shape_functions(const std::vector<Point> & qp,
                                                        const Elem * elem)
{
  libmesh_assert(elem);
  libmesh_assert_equal_to(elem->type(), TRI3SUBDIVISION);
  const Tri3Subdivision * sd_elem = static_cast<const Tri3Subdivision *>(elem);

  LOG_SCOPE("init_shape_functions()", "FESubdivision");

  this->calculations_started = true;

  const unsigned int valence = sd_elem->get_ordered_valence(0);
  const unsigned int n_qp = cast_int<unsigned int>(qp.size());
  const unsigned int n_approx_shape_functions = valence + 6;

  // resize the vectors to hold current data
  this->phi.resize         (n_approx_shape_functions);
  this->dphi.resize        (n_approx_shape_functions);
  this->dphidxi.resize     (n_approx_shape_functions);
  this->dphideta.resize    (n_approx_shape_functions);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
  this->d2phi.resize       (n_approx_shape_functions);
  this->d2phidxi2.resize   (n_approx_shape_functions);
  this->d2phidxideta.resize(n_approx_shape_functions);
  this->d2phideta2.resize  (n_approx_shape_functions);
#endif

  for (unsigned int i = 0; i < n_approx_shape_functions; ++i)
    {
      this->phi[i].resize         (n_qp);
      this->dphi[i].resize        (n_qp);
      this->dphidxi[i].resize     (n_qp);
      this->dphideta[i].resize    (n_qp);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
      this->d2phi[i].resize       (n_qp);
      this->d2phidxi2[i].resize   (n_qp);
      this->d2phidxideta[i].resize(n_qp);
      this->d2phideta2[i].resize  (n_qp);
#endif
    }

  // Renumbering of the shape functions
  static const unsigned int cvi[12] = {3,6,2,0,1,4,7,10,9,5,11,8};

  if (valence == 6) // This means that all vertices are regular, i.e. we have 12 shape functions
    {
      for (unsigned int i = 0; i < n_approx_shape_functions; ++i)
        {
          for (unsigned int p = 0; p < n_qp; ++p)
            {
              this->phi[i][p]          = FEShim<2,SUBDIVISION,RealType>::shape             (elem, this->fe_type.order, cvi[i],    qp[p]);
              this->dphidxi[i][p]      = FEShim<2,SUBDIVISION,RealType>::shape_deriv       (elem, this->fe_type.order, cvi[i], 0, qp[p]);
              this->dphideta[i][p]     = FEShim<2,SUBDIVISION,RealType>::shape_deriv       (elem, this->fe_type.order, cvi[i], 1, qp[p]);
              this->dphi[i][p](0)      = this->dphidxi[i][p];
              this->dphi[i][p](1)      = this->dphideta[i][p];
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
              this->d2phidxi2[i][p]    = FEShim<2,SUBDIVISION,RealType>::shape_second_deriv(elem, this->fe_type.order, cvi[i], 0, qp[p]);
              this->d2phidxideta[i][p] = FEShim<2,SUBDIVISION,RealType>::shape_second_deriv(elem, this->fe_type.order, cvi[i], 1, qp[p]);
              this->d2phideta2[i][p]   = FEShim<2,SUBDIVISION,RealType>::shape_second_deriv(elem, this->fe_type.order, cvi[i], 2, qp[p]);
              this->d2phi[i][p](0,0)   = this->d2phidxi2[i][p];
              this->d2phi[i][p](0,1)   = this->d2phi[i][p](1,0) = this->d2phidxideta[i][p];
              this->d2phi[i][p](1,1)   = this->d2phideta2[i][p];
#endif
            }
        }
    }
  else // vertex 0 is irregular by construction of the mesh
    {
      static const Real eps = 1e-10;

      // temporary values
      std::vector<RealType> tphi(12);
      std::vector<RealType> tdphidxi(12);
      std::vector<RealType> tdphideta(12);
#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
      std::vector<RealType> td2phidxi2(12);
      std::vector<RealType> td2phidxideta(12);
      std::vector<RealType> td2phideta2(12);
#endif

      for (unsigned int p = 0; p < n_qp; ++p)
        {
          // evaluate the number of the required subdivisions
          auto v = qp[p](0);
          auto w = qp[p](1);
          auto u = 1 - v - w;
          Real min = 0, max = 0.5;
          int n = 0;
          while (!(u > min-eps && u < max+eps))
            {
              ++n;
              min = max;
              max += std::pow((Real)(2), -n-1);
            }

          // transform u, v and w according to the number of subdivisions required.
          const auto pow2 = std::pow((Real)(2), n);
          v *= pow2;
          w *= pow2;
          u = 1 - v - w;
          libmesh_assert_less(u, 0.5 + eps);
          libmesh_assert_greater(u, -eps);

          // find out in which subdivided patch we are and setup the "selection matrix" P and the transformation Jacobian
          // (see Int. J. Numer. Meth. Engng. 2000; 47:2039-2072, Appendix A.2.)
          const int k = n+1;
          Real jfac; // the additional factor per derivative order
          DenseMatrix<Real> P(12, valence+12);
          if (v > 0.5 - eps)
            {
              v = 2*v - 1;
              w = 2*w;
              jfac = std::pow((Real)(2), k);
              P( 0,2        ) = 1;
              P( 1,0        ) = 1;
              P( 2,valence+3) = 1;
              P( 3,1        ) = 1;
              P( 4,valence  ) = 1;
              P( 5,valence+8) = 1;
              P( 6,valence+2) = 1;
              P( 7,valence+1) = 1;
              P( 8,valence+4) = 1;
              P( 9,valence+7) = 1;
              P(10,valence+6) = 1;
              P(11,valence+9) = 1;
            }
          else if (w > 0.5 - eps)
            {
              v = 2*v;
              w = 2*w - 1;
              jfac = std::pow((Real)(2), k);
              P( 0,0         ) = 1;
              P( 1,valence- 1) = 1;
              P( 2,1         ) = 1;
              P( 3,valence   ) = 1;
              P( 4,valence+ 5) = 1;
              P( 5,valence+ 2) = 1;
              P( 6,valence+ 1) = 1;
              P( 7,valence+ 4) = 1;
              P( 8,valence+11) = 1;
              P( 9,valence+ 6) = 1;
              P(10,valence+ 9) = 1;
              P(11,valence+10) = 1;
            }
          else
            {
              v = 1 - 2*v;
              w = 1 - 2*w;
              jfac = std::pow((Real)(-2), k);
              P( 0,valence+9) = 1;
              P( 1,valence+6) = 1;
              P( 2,valence+4) = 1;
              P( 3,valence+1) = 1;
              P( 4,valence+2) = 1;
              P( 5,valence+5) = 1;
              P( 6,valence  ) = 1;
              P( 7,1        ) = 1;
              P( 8,valence+3) = 1;
              P( 9,valence-1) = 1;
              P(10,0        ) = 1;
              P(11,2        ) = 1;
            }

          u = 1 - v - w;
          if ((u > 1 + eps) || (u < -eps))
            libmesh_error_msg("SUBDIVISION irregular patch: u is outside valid range!");

          DenseMatrix<Real> A;
          init_subdivision_matrix(A, valence);

          // compute P*A^k
          if (k > 1)
            {
              DenseMatrix<Real> Acopy(A);
              for (int e = 1; e < k; ++e)
                A.right_multiply(Acopy);
            }
          P.right_multiply(A);

          const PointTempl<RealType> transformed_p(v,w);

          for (unsigned int i = 0; i < 12; ++i)
            {
              tphi[i]          = FEShim<2,SUBDIVISION,RealType>::shape             (elem, this->fe_type.order, i,    transformed_p);
              tdphidxi[i]      = FEShim<2,SUBDIVISION,RealType>::shape_deriv       (elem, this->fe_type.order, i, 0, transformed_p);
              tdphideta[i]     = FEShim<2,SUBDIVISION,RealType>::shape_deriv       (elem, this->fe_type.order, i, 1, transformed_p);

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
              td2phidxi2[i]    = FEShim<2,SUBDIVISION,RealType>::shape_second_deriv(elem, this->fe_type.order, i, 0, transformed_p);
              td2phidxideta[i] = FEShim<2,SUBDIVISION,RealType>::shape_second_deriv(elem, this->fe_type.order, i, 1, transformed_p);
              td2phideta2[i]   = FEShim<2,SUBDIVISION,RealType>::shape_second_deriv(elem, this->fe_type.order, i, 2, transformed_p);
#endif
            }

          // Finally, we can compute the irregular shape functions as the product of P
          // and the regular shape functions:
          RealType sum1, sum2, sum3, sum4, sum5, sum6;
          for (unsigned int j = 0; j < n_approx_shape_functions; ++j)
            {
              sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = 0;
              for (unsigned int i = 0; i < 12; ++i)
                {
                  sum1 += P(i,j) * tphi[i];
                  sum2 += P(i,j) * tdphidxi[i];
                  sum3 += P(i,j) * tdphideta[i];

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
                  sum4 += P(i,j) * td2phidxi2[i];
                  sum5 += P(i,j) * td2phidxideta[i];
                  sum6 += P(i,j) * td2phideta2[i];
#endif
                }
              this->phi[j][p]          = sum1;
              this->dphidxi[j][p]      = sum2 * jfac;
              this->dphideta[j][p]     = sum3 * jfac;
              this->dphi[j][p](0)      = this->dphidxi[j][p];
              this->dphi[j][p](1)      = this->dphideta[j][p];

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
              this->d2phidxi2[j][p]    = sum4 * jfac * jfac;
              this->d2phidxideta[j][p] = sum5 * jfac * jfac;
              this->d2phideta2[j][p]   = sum6 * jfac * jfac;
              this->d2phi[j][p](0,0)   = this->d2phidxi2[j][p];
              this->d2phi[j][p](0,1)   = this->d2phi[j][p](1,0) = this->d2phidxideta[j][p];
              this->d2phi[j][p](1,1)   = this->d2phideta2[j][p];
#endif
            }
        } // end quadrature loop
    } // end irregular vertex

  // Let the FEMap use the same initialized shape functions
  this->_fe_map->get_phi_map()          = this->phi;
  this->_fe_map->get_dphidxi_map()      = this->dphidxi;
  this->_fe_map->get_dphideta_map()     = this->dphideta;

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
  this->_fe_map->get_d2phidxi2_map()    = this->d2phidxi2;
  this->_fe_map->get_d2phideta2_map()   = this->d2phideta2;
  this->_fe_map->get_d2phidxideta_map() = this->d2phidxideta;
#endif
}



template <typename RealType>
void FESubdivisionTempl<RealType>::attach_quadrature_rule(QBase * q)
{
  libmesh_assert(q);

  this->qrule = q;
  // make sure we don't cache results from a previous quadrature rule
  this->elem_type = INVALID_ELEM;
  return;
}



template <typename RealType>
void FESubdivisionTempl<RealType>::reinit(const Elem * elem,
                                          const std::vector<Point> * const libmesh_dbg_var(pts),
                                          const std::vector<Real> * const)
{
  libmesh_assert(elem);
  libmesh_assert_equal_to(elem->type(), TRI3SUBDIVISION);
#ifndef NDEBUG
  const Tri3Subdivision * sd_elem = static_cast<const Tri3Subdivision *>(elem);
#endif

  LOG_SCOPE("reinit()", "FESubdivision");

  libmesh_assert(!sd_elem->is_ghost());
  libmesh_assert(sd_elem->is_subdivision_updated());

  // check if vertices 1 and 2 are regular
  libmesh_assert_equal_to(sd_elem->get_ordered_valence(1), 6);
  libmesh_assert_equal_to(sd_elem->get_ordered_valence(2), 6);

  // We're calculating now!  Time to determine what.
  this->determine_calculations();

  // no custom quadrature support
  libmesh_assert(pts == nullptr);
  libmesh_assert(this->qrule);
  this->qrule->init(elem->type());

  // Initialize the shape functions
  this->init_shape_functions(this->qrule->get_points(), elem);

  // The shape functions correspond to the qrule
  this->shapes_on_quadrature = true;

  // Compute the map for this element.
  this->_fe_map->compute_map (this->dim, this->qrule->get_weights(), elem, this->calculate_d2phi);
}



template <typename RealType>
typename FEShim<2,SUBDIVISION,RealType>::OutputShape FEShim<2,SUBDIVISION,RealType>::shape(const ElemType type,
                                               const Order order,
                                               const unsigned int i,
                                               const Point & p)
{
  switch (order)
    {
    case FOURTH:
      {
        switch (type)
          {
          case TRI3SUBDIVISION:
            libmesh_assert_less(i, 12);
            return FESubdivisionTempl<RealType>::regular_shape(i,p(0),p(1));
          default:
            libmesh_error_msg("ERROR: Unsupported element type!");
          }
      }
    default:
      libmesh_error_msg("ERROR: Unsupported polynomial order!");
    }
}



template <typename RealType>
typename FEShim<2,SUBDIVISION,RealType>::OutputShape FEShim<2,SUBDIVISION,RealType>::shape(const Elem * elem,
                                               const Order order,
                                               const unsigned int i,
                                               const Point & p,
                                               const bool add_p_level)
{
  libmesh_assert(elem);
  const Order totalorder =
    static_cast<Order>(order+add_p_level*elem->p_level());
  return FEShim<2,SUBDIVISION,RealType>::shape(elem->type(), totalorder, i, p);
}



template <typename RealType>
typename FEShim<2,SUBDIVISION,RealType>::OutputShape FEShim<2,SUBDIVISION,RealType>::shape_deriv(const ElemType type,
                                                     const Order order,
                                                     const unsigned int i,
                                                     const unsigned int j,
                                                     const Point & p)
{
  switch (order)
    {
    case FOURTH:
      {
        switch (type)
          {
          case TRI3SUBDIVISION:
            libmesh_assert_less(i, 12);
            return FESubdivisionTempl<RealType>::regular_shape_deriv(i,j,p(0),p(1));
          default:
            libmesh_error_msg("ERROR: Unsupported element type!");
          }
      }
    default:
      libmesh_error_msg("ERROR: Unsupported polynomial order!");
    }
}



template <typename RealType>
typename FEShim<2,SUBDIVISION,RealType>::OutputShape FEShim<2,SUBDIVISION,RealType>::shape_deriv(const Elem * elem,
                                                     const Order order,
                                                     const unsigned int i,
                                                     const unsigned int j,
                                                     const Point & p,
                                                     const bool add_p_level)
{
  libmesh_assert(elem);
  const Order totalorder =
    static_cast<Order>(order+add_p_level*elem->p_level());
  return FEShim<2,SUBDIVISION,RealType>::shape_deriv(elem->type(), totalorder, i, j, p);
}


#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES

template <typename RealType>
typename FEShim<2,SUBDIVISION,RealType>::OutputShape FEShim<2,SUBDIVISION,RealType>::shape_second_deriv(const ElemType type,
                                                            const Order order,
                                                            const unsigned int i,
                                                            const unsigned int j,
                                                            const Point & p)
{
  switch (order)
    {
    case FOURTH:
      {
        switch (type)
          {
          case TRI3SUBDIVISION:
            libmesh_assert_less(i, 12);
            return FESubdivisionTempl<RealType>::regular_shape_second_deriv(i,j,p(0),p(1));
          default:
            libmesh_error_msg("ERROR: Unsupported element type!");
          }
      }
    default:
      libmesh_error_msg("ERROR: Unsupported polynomial order!");
    }
}



template <typename RealType>
typename FEShim<2,SUBDIVISION,RealType>::OutputShape FEShim<2,SUBDIVISION,RealType>::shape_second_deriv(const Elem * elem,
                                                            const Order order,
                                                            const unsigned int i,
                                                            const unsigned int j,
                                                            const Point & p,
                                                            const bool add_p_level)
{
  libmesh_assert(elem);
  const Order totalorder =
    static_cast<Order>(order+add_p_level*elem->p_level());
  return FEShim<2,SUBDIVISION,RealType>::shape_second_deriv(elem->type(), totalorder, i, j, p);
}

#endif // LIBMESH_ENABLE_SECOND_DERIVATIVES

template <typename RealType>
void FEShim<2,SUBDIVISION,RealType>::nodal_soln(const Elem * elem,
                                                const Order,
                                                const std::vector<Number> & elem_soln,
                                                std::vector<Number> & nodal_soln)
{
  libmesh_assert(elem);
  libmesh_assert_equal_to(elem->type(), TRI3SUBDIVISION);
  const Tri3Subdivision * sd_elem = static_cast<const Tri3Subdivision *>(elem);

  nodal_soln.resize(3); // three nodes per element

  // Ghost nodes are auxiliary.
  if (sd_elem->is_ghost())
    {
      nodal_soln[0] = 0;
      nodal_soln[1] = 0;
      nodal_soln[2] = 0;
      return;
    }

  // First node (node 0 in the element patch):
  unsigned int j = sd_elem->local_node_number(sd_elem->get_ordered_node(0)->id());
  nodal_soln[j] = elem_soln[0];

  // Second node (node 1 in the element patch):
  j = sd_elem->local_node_number(sd_elem->get_ordered_node(1)->id());
  nodal_soln[j] = elem_soln[1];

  // Third node (node 'valence' in the element patch):
  j = sd_elem->local_node_number(sd_elem->get_ordered_node(2)->id());
  nodal_soln[j] = elem_soln[sd_elem->get_ordered_valence(0)];
}



template <typename RealType>
void FESideMapShim<2,SUBDIVISION,RealType>::side_map(FE<2,SUBDIVISION,RealType> &,
                                                     const Elem *,
                                                     const Elem *,
                                                     const unsigned int,
                                                     const std::vector<Point> &,
                                                     std::vector<Point> &)
{
  libmesh_not_implemented();
}

template <typename RealType>
void FEEdgeReinitShim<2,SUBDIVISION,RealType>::edge_reinit(FE<2,SUBDIVISION,RealType> &,
                                                           Elem const *,
                                                           unsigned int,
                                                           Real,
                                                           const std::vector<Point> * const,
                                                           const std::vector<Real> * const)
{
  libmesh_not_implemented();
}

template <typename RealType>
PointTempl<RealType> FEInverseMapShim<2,SUBDIVISION,RealType>::inverse_map(const Elem *,
                                                                           const Point &,
                                                                           const Real,
                                                                           const bool)
{
  libmesh_not_implemented();
}

template <typename RealType>
void FEInverseMapShim<2,SUBDIVISION,RealType>::inverse_map(const Elem *,
                                                           const std::vector<Point> &,
                                                           std::vector<Point> &,
                                                           Real,
                                                           bool)
{
  libmesh_not_implemented();
}



// The number of dofs per subdivision element depends on the valence of its nodes, so it is non-static
template <typename RealType> unsigned int FEShim<2,SUBDIVISION,RealType>::n_dofs(const ElemType, const Order) { libmesh_not_implemented(); return 0; }

// Loop subdivision elements have only a single dof per node
template <typename RealType> unsigned int FEShim<2,SUBDIVISION,RealType>::n_dofs_at_node(const ElemType, const Order, const unsigned int) { return 1; }

// Subdivision FEMs have dofs only at the nodes
template <typename RealType> unsigned int FEShim<2,SUBDIVISION,RealType>::n_dofs_per_elem(const ElemType, const Order) { return 0; }

// Subdivision FEMs have dofs only at the nodes
template <typename RealType> void FEDofsOnSideShim<2,SUBDIVISION,RealType>::dofs_on_side(const Elem * const, const Order, unsigned int, std::vector<unsigned int> & di) { di.resize(0); }
template <typename RealType> void FEDofsOnSideShim<2,SUBDIVISION,RealType>::dofs_on_edge(const Elem * const, const Order, unsigned int, std::vector<unsigned int> & di) { di.resize(0); }

// Subdivision FEMs are C^1 continuous
template <typename RealType> FEContinuity FEShim<2,SUBDIVISION,RealType>::get_continuity() { return C_ONE; }

// Subdivision FEMs are not hierarchic
template <typename RealType> bool FEShim<2,SUBDIVISION,RealType>::is_hierarchic() { return false; }

// Subdivision FEM shapes need reinit
template <typename RealType> bool FEShim<2,SUBDIVISION,RealType>::shapes_need_reinit() { return true; }

} // namespace libMesh

#endif // LIBMESH_FE_SUBDIVISION_IMPL_H
