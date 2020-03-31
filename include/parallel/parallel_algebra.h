// The libMesh Finite Element Library.
// Copyright (C) 2002-2020 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

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


#ifndef LIBMESH_PARALLEL_ALGEBRA_H
#define LIBMESH_PARALLEL_ALGEBRA_H

// This class contains all the functionality for bin sorting
// Templated on the type of keys you will be sorting and the
// type of iterator you will be using.


// libMesh includes
#include "libmesh/libmesh_config.h"
#include "libmesh/point.h"
#include "libmesh/tensor_value.h"
#include "libmesh/vector_value.h"
#include "libmesh/auto_ptr.h" // libmesh_make_unique

// TIMPI includes
#include "timpi/op_function.h"
#include "timpi/standard_type.h"
#include "timpi/packing.h"

// C++ includes
#include <cstddef>
#include <memory>
#include <type_traits>

#ifdef LIBMESH_HAVE_METAPHYSICL
// For parallel GeomReal
#include "metaphysicl/parallel_dualnumber.h"
#include "metaphysicl/parallel_dynamicsparsenumberarray.h"
#endif

namespace TIMPI {

using libMesh::TypeVector;
using libMesh::TypeTensor;
using libMesh::VectorValue;
using libMesh::TensorValue;
using libMesh::Point;

// StandardType<> specializations to return a derived MPI datatype
// to handle communication of LIBMESH_DIM-vectors.
//
// We use MPI_Create_struct here because our vector classes might
// have vptrs, and I'd rather not have the datatype break in those
// cases.
template <typename T>
class StandardType<TypeVector<T>, typename std::enable_if<StandardType<T>::is_fixed_type>::type>
  : public DataType
{
public:
  explicit
  StandardType(const TypeVector<T> * example=nullptr) {
    // We need an example for MPI_Address to use
    TypeVector<T> * ex;
    std::unique_ptr<TypeVector<T>> temp;
    if (example)
      ex = const_cast<TypeVector<T> *>(example);
    else
      {
        temp = libmesh_make_unique<TypeVector<T>>();
        ex = temp.get();
      }

#ifdef LIBMESH_HAVE_MPI
    StandardType<T> T_type(&((*ex)(0)));

    // We require MPI-2 here:
    int blocklength = LIBMESH_DIM;
    MPI_Aint displs, start;
    MPI_Datatype tmptype, type = T_type;

    timpi_call_mpi
      (MPI_Get_address (ex, &start));
    timpi_call_mpi
      (MPI_Get_address (&((*ex)(0)), &displs));

    // subtract off offset to first value from the beginning of the structure
    displs -= start;

    // create a prototype structure
    timpi_call_mpi
      (MPI_Type_create_struct (1, &blocklength, &displs, &type,
                               &tmptype));
    timpi_call_mpi
      (MPI_Type_commit (&tmptype));

    // resize the structure type to account for padding, if any
    timpi_call_mpi
      (MPI_Type_create_resized (tmptype, 0, sizeof(TypeVector<T>),
                                &_datatype));

    timpi_call_mpi
      (MPI_Type_commit (&_datatype));

    timpi_call_mpi
      (MPI_Type_free (&tmptype));
#endif // #ifdef LIBMESH_HAVE_MPI
  }

  StandardType(const StandardType<TypeVector<T>> & timpi_mpi_var(t))
    : DataType()
  {
    timpi_call_mpi (MPI_Type_dup (t._datatype, &_datatype));
  }

  ~StandardType() { this->free(); }

  static const bool is_fixed_type = true;
};


template <typename T>
class StandardType<VectorValue<T>, typename std::enable_if<StandardType<T>::is_fixed_type>::type>
  : public DataType
{
public:
  explicit
  StandardType(const VectorValue<T> * example=nullptr) {
    // We need an example for MPI_Address to use
    VectorValue<T> * ex;
    std::unique_ptr<VectorValue<T>> temp;
    if (example)
      ex = const_cast<VectorValue<T> *>(example);
    else
      {
        temp = libmesh_make_unique<VectorValue<T>>();
        ex = temp.get();
      }

#ifdef LIBMESH_HAVE_MPI
    StandardType<T> T_type(&((*ex)(0)));

    int blocklength = LIBMESH_DIM;
    MPI_Aint displs, start;
    MPI_Datatype tmptype, type = T_type;

    timpi_call_mpi
      (MPI_Get_address (ex, &start));
    timpi_call_mpi
      (MPI_Get_address (&((*ex)(0)), &displs));

    // subtract off offset to first value from the beginning of the structure
    displs -= start;

    // create a prototype structure
    timpi_call_mpi
      (MPI_Type_create_struct (1, &blocklength, &displs, &type,
                               &tmptype));
    timpi_call_mpi
      (MPI_Type_commit (&tmptype));

    // resize the structure type to account for padding, if any
    timpi_call_mpi
      (MPI_Type_create_resized (tmptype, 0,
                                sizeof(VectorValue<T>),
                                &_datatype));

    timpi_call_mpi
      (MPI_Type_commit (&_datatype));

    timpi_call_mpi
      (MPI_Type_free (&tmptype));
#endif // #ifdef LIBMESH_HAVE_MPI
  }

  StandardType(const StandardType<VectorValue<T>> & timpi_mpi_var(t))
    : DataType()
  {
#ifdef LIBMESH_HAVE_MPI
    timpi_call_mpi (MPI_Type_dup (t._datatype, &_datatype));
#endif
  }

  ~StandardType() { this->free(); }

  static const bool is_fixed_type = true;
};

template <>
class StandardType<Point> : public DataType
{
public:
  template <typename T = libMesh::GeomReal, typename std::enable_if<StandardType<T>::is_fixed_type,
                                                           int>::type = 0>
  explicit
  StandardType(const Point * example=nullptr)
  {
    // Prevent unused variable warnings when !LIBMESH_HAVE_MPI
    libmesh_ignore(example);

#ifdef LIBMESH_HAVE_MPI

    // We need an example for MPI_Address to use
    Point * ex;

    std::unique_ptr<Point> temp;
    if (example)
      ex = const_cast<Point *>(example);
    else
      {
        temp = libmesh_make_unique<Point>();
        ex = temp.get();
      }

    StandardType<T> T_type(&((*ex)(0)));

    int blocklength = LIBMESH_DIM;
    MPI_Aint displs, start;
    MPI_Datatype tmptype, type = T_type;

    timpi_call_mpi
      (MPI_Get_address (ex, &start));
    timpi_call_mpi
      (MPI_Get_address (&((*ex)(0)), &displs));

    // subtract off offset to first value from the beginning of the structure
    displs -= start;

    // create a prototype structure
    timpi_call_mpi
      (MPI_Type_create_struct (1, &blocklength, &displs, &type,
                               &tmptype));
    timpi_call_mpi
      (MPI_Type_commit (&tmptype));

    // resize the structure type to account for padding, if any
    timpi_call_mpi
      (MPI_Type_create_resized (tmptype, 0, sizeof(Point),
                                &_datatype));

    timpi_call_mpi
      (MPI_Type_commit (&_datatype));

    timpi_call_mpi
      (MPI_Type_free (&tmptype));
#endif // #ifdef LIBMESH_HAVE_MPI
  }

  template <typename T = libMesh::GeomReal, typename std::enable_if<StandardType<T>::is_fixed_type,
                                                           int>::type = 0>
  StandardType(const StandardType<Point> & timpi_mpi_var(t))
    : DataType()
  {
    timpi_call_mpi (MPI_Type_dup (t._datatype, &_datatype));
  }

  // GeomReal is_fixed_type free
  template <
      typename T = libMesh::GeomReal,
      typename std::enable_if<StandardType<T>::is_fixed_type, int>::type = 0>
  void free() {
    DataType::free();
  }

  // GeomReal !is_fixed_type free
  template <
      typename T = libMesh::GeomReal,
      typename std::enable_if<!StandardType<T>::is_fixed_type, int>::type = 0>
  void free() {}

  ~StandardType() { this->free(); }

  static const bool is_fixed_type = StandardType<libMesh::GeomReal>::is_fixed_type;
};

// OpFunction<> specializations to return an MPI_Op version of the
// reduction operations on LIBMESH_DIM-vectors.
//
// We use static variables to minimize the number of MPI datatype
// construction calls executed over the course of the program.
//
// We use a singleton pattern because a global variable would
// have tried to call MPI functions before MPI got initialized.
//
// min() and max() are applied component-wise; this makes them useful
// for bounding box reduction operations.
template <typename V>
class TypeVectorOpFunction
{
public:
#ifdef LIBMESH_HAVE_MPI
  static void vector_max (void * invec, void * inoutvec, int * len, MPI_Datatype *)
  {
    V *in = static_cast<V *>(invec);
    V *inout = static_cast<V *>(inoutvec);
    for (int i=0; i != *len; ++i)
      for (int d=0; d != LIBMESH_DIM; ++d)
        inout[i](d) = std::max(in[i](d), inout[i](d));
  }

  static void vector_min (void * invec, void * inoutvec, int * len, MPI_Datatype *)
  {
    V *in = static_cast<V *>(invec);
    V *inout = static_cast<V *>(inoutvec);
    for (int i=0; i != *len; ++i)
      for (int d=0; d != LIBMESH_DIM; ++d)
        inout[i](d) = std::min(in[i](d), inout[i](d));
  }

  static void vector_sum (void * invec, void * inoutvec, int * len, MPI_Datatype *)
  {
    V *in = static_cast<V *>(invec);
    V *inout = static_cast<V *>(inoutvec);
    for (int i=0; i != *len; ++i)
      for (int d=0; d != LIBMESH_DIM; ++d)
        inout[i](d) += in[i](d);
  }

  static MPI_Op max()
  {
    // _static_op never gets freed, but it only gets committed once
    // per T, so it's not a *huge* memory leak...
    static MPI_Op _static_op;
    static bool _is_initialized = false;
    if (!_is_initialized)
      {
        timpi_call_mpi
          (MPI_Op_create (vector_max, /*commute=*/ true,
                          &_static_op));

        _is_initialized = true;
      }

    return _static_op;
  }
  static MPI_Op min()
  {
    // _static_op never gets freed, but it only gets committed once
    // per T, so it's not a *huge* memory leak...
    static MPI_Op _static_op;
    static bool _is_initialized = false;
    if (!_is_initialized)
      {
        timpi_call_mpi
          (MPI_Op_create (vector_min, /*commute=*/ true,
                          &_static_op));

        _is_initialized = true;
      }

    return _static_op;
  }
  static MPI_Op sum()
  {
    // _static_op never gets freed, but it only gets committed once
    // per T, so it's not a *huge* memory leak...
    static MPI_Op _static_op;
    static bool _is_initialized = false;
    if (!_is_initialized)
      {
        timpi_call_mpi
          (MPI_Op_create (vector_sum, /*commute=*/ true,
                          &_static_op));

        _is_initialized = true;
      }

    return _static_op;
  }

#endif // LIBMESH_HAVE_MPI
};

template <typename T>
class OpFunction<TypeVector<T>> : public TypeVectorOpFunction<TypeVector<T>> {};

template <typename T>
class OpFunction<VectorValue<T>> : public TypeVectorOpFunction<VectorValue<T>> {};

template <>
class OpFunction<Point> : public TypeVectorOpFunction<Point> {};

// StandardType<> specializations to return a derived MPI datatype
// to handle communication of LIBMESH_DIM*LIBMESH_DIM-tensors.
//
// We assume contiguous storage here
template <typename T>
class StandardType<TypeTensor<T>, typename std::enable_if<StandardType<T>::is_fixed_type>::type>
  : public DataType
{
public:
  explicit
  StandardType(const TypeTensor<T> * example=nullptr) :
    DataType(StandardType<T>(example ?  &((*example)(0,0)) : nullptr), LIBMESH_DIM*LIBMESH_DIM) {}

  inline ~StandardType() { this->free(); }

  static const bool is_fixed_type = true;
};

template <typename T>
class StandardType<TensorValue<T>, typename std::enable_if<StandardType<T>::is_fixed_type>::type>
  : public DataType
{
public:
  explicit
  StandardType(const TensorValue<T> * example=nullptr) :
    DataType(StandardType<T>(example ?  &((*example)(0,0)) : nullptr), LIBMESH_DIM*LIBMESH_DIM) {}

  inline ~StandardType() { this->free(); }

  static const bool is_fixed_type = true;
};
} // namespace TIMPI

namespace libMesh
{
namespace Parallel
{
template <>
class Packing<Point>
{
public:
  typedef std::size_t buffer_type;

  template <typename OutputIter,
            typename Context>
  static void pack(const Point & point, OutputIter data_out, const Context * context);

  template <typename Context>
  static unsigned int packable_size(const Point & point, const Context * context);

  template <typename BufferIter>
  static unsigned int packed_size(BufferIter iter);

  template <typename BufferIter, typename Context>
  static Point unpack(BufferIter in, Context * ctx);
};

template <>
template <typename Context>
unsigned int
Packing<Point>::
packable_size(const Point & point,
              const Context * ctx)
{
  unsigned int size = 0;
  for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
    size += Packing<GeomReal>::packable_size(point(i), ctx);

  // Record the size in the first buffer entry
  return size + 1;
}

template <>
template <typename BufferIter>
unsigned int
Packing<Point>::
packed_size(BufferIter iter)
{
  // We recorded the size in the first buffer entry
  return *iter;
}

template <>
template <typename OutputIter, typename Context>
void
Packing<Point>::
pack(const Point & point,
     OutputIter data_out,
     const Context * ctx)
{
  unsigned int size = packable_size(point, ctx);

  // First write out the buffer size
  *data_out++ = cast_int<std::size_t>(size);

  // Now pack the data. Note that TIMPI uses a back_inserter for `pack_range` so we don't (and
  // can't) actually increment the iterator with operator+=. operator++ is a no-op
  for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
    Packing<GeomReal>::pack(point(i), data_out, ctx);
}

template <typename T, typename I>
template <typename BufferIter, typename Context>
Point
Packing<Point>::
unpack(BufferIter in, Context * ctx)
{
  Point point;

  // We don't care about the size
  in++;

  for (unsigned int i = 0; i < LIBMESH_DIM; ++i)
  {
    Packing<GeomReal>::unpack(in, ctx);
    // Make sure we increment the iterator
    in += Packing<GeomReal>::packable_size(point(i), ctx);
  }

  return point;
}
}
}


#endif // LIBMESH_PARALLEL_ALGEBRA_H
