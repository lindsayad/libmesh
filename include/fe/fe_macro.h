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


#ifndef LIBMESH_FE_MACRO_H
#define LIBMESH_FE_MACRO_H




// These macros help in instantiating specific versions
// of the \p FE class.  Simply include this file, and
// instantiate at the end for the desired dimension(s).
#define INSTANTIATE_MAPS(_dim,_type,RealType)                                   \
  template PointTempl<RealType> FE<_dim, _type, RealType>::map(const ElemTempl<RealType> *, const PointTempl<RealType> &);     \
  template PointTempl<RealType> FE<_dim, _type, RealType>::map_xi(const ElemTempl<RealType> *, const PointTempl<RealType> &);  \
  template PointTempl<RealType> FE<_dim, _type, RealType>::map_eta(const ElemTempl<RealType> *, const PointTempl<RealType> &); \
  template PointTempl<RealType> FE<_dim, _type, RealType>::map_zeta(const ElemTempl<RealType> *, const PointTempl<RealType> &); \
  template void Templ<RealType> FE<_dim, _type, RealType>::inverse_map(const ElemTempl<RealType> *, const std::vector<PointTempl<RealType>> &, std::vector<PointTempl<RealType>> &, Real, bool); \
  template PointTempl<RealType> FE<_dim, _type, RealType>::inverse_map(const ElemTempl<RealType> *, const PointTempl<RealType> &, Real, bool)

#define INSTANTIATE_SUBDIVISION_MAPS(RealType)                          \
  template PointTempl<RealType> FE<2, SUBDIVISION, RealType>::map(const ElemTempl<RealType> *, const PointTempl<RealType> &);  \
  template PointTempl<RealType> FE<2, SUBDIVISION, RealType>::map_xi(const ElemTempl<RealType> *, const PointTempl<RealType> &); \
  template PointTempl<RealType> FE<2, SUBDIVISION, RealType>::map_eta(const ElemTempl<RealType> *, const PointTempl<RealType> &); \
  template PointTempl<RealType> FE<2, SUBDIVISION, RealType>::map_zeta(const ElemTempl<RealType> *, const PointTempl<RealType> &)

#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS

#define INSTANTIATE_SUBDIVISION_FE                                      \
  template FE<2,SUBDIVISION>::FE(const FEType & fet); \
  template unsigned int FE<2,SUBDIVISION>::n_shape_functions () const;  \
  template void         FE<2,SUBDIVISION>::attach_quadrature_rule (QBase *); \
  template unsigned int FE<2,SUBDIVISION>::n_quadrature_points () const; \
  template void         FE<2,SUBDIVISION>::reinit(const Elem *,const std::vector<Point> * const,const std::vector<Real> * const); \
  template void         FE<2,SUBDIVISION>::init_base_shape_functions(const std::vector<Point> &, const Elem *); \
  template void         FE<2,SUBDIVISION>::init_shape_functions(const std::vector<Point> &, const Elem *)

#else // LIBMESH_ENABLE_INFINITE_ELEMENTS

#define INSTANTIATE_SUBDIVISION_FE                                      \
  template FE<2,SUBDIVISION>::FE(const FEType & fet);                     \
  template unsigned int FE<2,SUBDIVISION>::n_shape_functions () const;  \
  template void         FE<2,SUBDIVISION>::attach_quadrature_rule (QBase *); \
  template unsigned int FE<2,SUBDIVISION>::n_quadrature_points () const; \
  template void         FE<2,SUBDIVISION>::reinit(const Elem *,const std::vector<Point> * const,const std::vector<Real> * const); \
  template void         FE<2,SUBDIVISION>::init_shape_functions(const std::vector<Point> &, const Elem *)

#endif // LIBMESH_ENABLE_INFINITE_ELEMENTS


#ifndef LIBMESH_ENABLE_HIGHER_ORDER_SHAPES

#define INSTANTIATE_FE(_dim)   template class FE< (_dim), CLOUGH, Real>;      \
  template class FE< (_dim), HERMITE, Real>;                                  \
  template class FE< (_dim), HIERARCHIC, Real>;                               \
  template class FE< (_dim), L2_HIERARCHIC, Real>;                            \
  template class FE< (_dim), LAGRANGE, Real>;                                 \
  template class FE< (_dim), LAGRANGE_VEC, Real>;                             \
  template class FE< (_dim), L2_LAGRANGE, Real>;                              \
  template class FE< (_dim), MONOMIAL, Real>;                                 \
  template class FE< (_dim), SCALAR, Real>;                                   \
  template class FE< (_dim), XYZ, Real>;                                      \
  template class FE< (_dim), NEDELEC_ONE, Real>;                              \
  template class FE< (_dim), MONOMIAL_VEC, Real>

#else //LIBMESH_ENABLE_HIGHER_ORDER_SHAPES

#define INSTANTIATE_FE(_dim)   template class FE< (_dim), CLOUGH, Real>;      \
  template class FE< (_dim), HERMITE, Real>;                                  \
  template class FE< (_dim), HIERARCHIC, Real>;                               \
  template class FE< (_dim), L2_HIERARCHIC, Real>;                            \
  template class FE< (_dim), LAGRANGE, Real>;                                 \
  template class FE< (_dim), LAGRANGE_VEC, Real>;                             \
  template class FE< (_dim), L2_LAGRANGE, Real>;                              \
  template class FE< (_dim), MONOMIAL, Real>;                                 \
  template class FE< (_dim), SCALAR, Real>;                                   \
  template class FE< (_dim), BERNSTEIN, Real>;                                \
  template class FE< (_dim), SZABAB, Real>;                                   \
  template class FE< (_dim), XYZ, Real>;                                      \
  template class FE< (_dim), RATIONAL_BERNSTEIN, Real>;                       \
  template class FE< (_dim), NEDELEC_ONE, Real>;                              \
  template class FE< (_dim), MONOMIAL_VEC, Real>

#endif //LIBMESH_ENABLE_HIGHER_ORDER_SHAPES

#endif // LIBMESH_FE_MACRO_H
