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

#include "libmesh/parallel_algebra.h"

namespace TIMPI
{
namespace
{
template <typename T, typename std::enable_if<T::is_fixed_type, int>::type = 0>
void
point_construct(T & standard_type, const Point * example)
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

    StandardType<libMesh::GeomReal> T_type(&((*ex)(0)));

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

  timpi_call_mpi
    (MPI_Type_create_resized (tmptype, 0, sizeof(Point),
                              &standard_type.operator data_type()));

    timpi_call_mpi
      (MPI_Type_commit (&standard_type.operator data_type()));

    timpi_call_mpi
      (MPI_Type_free (&tmptype));
#endif // #ifdef LIBMESH_HAVE_MPI
}

template <typename T, typename std::enable_if<!T::is_fixed_type, int>::type = 0>
void
point_construct(T &, const Point *)
{
}

template <typename T, typename std::enable_if<T::is_fixed_type, int>::type = 0>
void
point_dup(const T & in, T & out)
{
  timpi_call_mpi (MPI_Type_dup (in, &out.operator data_type()));;
}

template <typename T, typename std::enable_if<!T::is_fixed_type, int>::type = 0>
void
point_dup(const T &, T &)
{
}

template <typename T, typename std::enable_if<T::is_fixed_type, int>::type = 0>
void
point_free(T & standard_type)
{
  standard_type.free();
}

template <typename T, typename std::enable_if<!T::is_fixed_type, int>::type = 0>
void
point_free(T &)
{
}
}

StandardType<Point>::StandardType(const Point * example)
{
  point_construct(*this, example);
}

StandardType<Point>::StandardType(const StandardType<Point> & t)
{
  point_dup(t, *this);
}

void StandardType<Point>::free() {
  point_free(*this);
}

}
