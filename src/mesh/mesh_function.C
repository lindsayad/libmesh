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



// Local Includes
#include "libmesh/mesh_function.h"
#include "libmesh/dense_vector.h"
#include "libmesh/equation_systems.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dof_map.h"
#include "libmesh/point_locator_tree.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fe_compute_data.h"
#include "libmesh/mesh_base.h"
#include "libmesh/point.h"
#include "libmesh/elem.h"
#include "libmesh/int_range.h"
#include "libmesh/fe_map.h"

namespace libMesh
{


//------------------------------------------------------------------
// MeshFunction methods
template <typename Output>
MeshFunction<Output>::MeshFunction (const EquationSystems & eqn_systems,
                            const NumericVector<Number> & vec,
                            const DofMap & dof_map,
                            const std::vector<unsigned int> & vars,
                            const FunctionBase<Output> * master) :
  FunctionBase<Output> (master),
  ParallelObject       (eqn_systems),
  _eqn_systems         (eqn_systems),
  _vector              (vec),
  _dof_map             (dof_map),
  _system_vars         (vars),
  _point_locator       (nullptr),
  _out_of_mesh_mode    (false),
  _out_of_mesh_value   ()
{
}



template <typename Output>
MeshFunction<Output>::MeshFunction (const EquationSystems & eqn_systems,
                            const NumericVector<Number> & vec,
                            const DofMap & dof_map,
                            const unsigned int var,
                            const FunctionBase<Output> * master) :
  FunctionBase<Output> (master),
  ParallelObject       (eqn_systems),
  _eqn_systems         (eqn_systems),
  _vector              (vec),
  _dof_map             (dof_map),
  _system_vars         (1,var),
  _point_locator       (nullptr),
  _out_of_mesh_mode    (false),
  _out_of_mesh_value   ()
{
  //   std::vector<unsigned int> buf (1);
  //   buf[0] = var;
  //   _system_vars (buf);
}







template <typename Output>
MeshFunction<Output>::~MeshFunction ()
{
  delete this->_point_locator;
}




template <typename Output>
void MeshFunction<Output>::init (const Trees::BuildType /*point_locator_build_type*/)
{
  // are indices of the desired variable(s) provided?
  libmesh_assert_greater (this->_system_vars.size(), 0);

  // Don't do twice...
  if (this->_initialized)
    {
      libmesh_assert(this->_point_locator);
      return;
    }

  /*
   * set up the PointLocator: currently we always get this from the
   * MeshBase we're associated with.
   */
  const MeshBase & mesh = this->_eqn_systems.get_mesh();

  this->_point_locator = mesh.sub_point_locator().release();

  // ready for use
  this->_initialized = true;
}


template <typename Output>
void
MeshFunction<Output>::clear ()
{
  // only delete the point locator when we are the master
  if ((this->_point_locator != nullptr) && (this->_master == nullptr))
    {
      delete this->_point_locator;
      this->_point_locator = nullptr;
    }
  this->_initialized = false;
}



template <typename Output>
std::unique_ptr<FunctionBase<Output>> MeshFunction<Output>::clone () const
{
  MeshFunction * mf_clone =
    new MeshFunction(_eqn_systems, _vector, _dof_map, _system_vars, this);

  if (this->initialized())
    mf_clone->init();
  mf_clone->set_point_locator_tolerance(
    this->get_point_locator().get_close_to_point_tol());

  return std::unique_ptr<FunctionBase<Output>>(mf_clone);
}



template <typename Output>
Output MeshFunction<Output>::operator() (const Point & p,
                                 const Real time)
{
  libmesh_assert (this->initialized());

  DenseVector<Output> buf (1);
  this->operator() (p, time, buf);
  return buf(0);
}

template <typename Output>
std::map<const Elem *, Output> MeshFunction<Output>::discontinuous_value (const Point & p,
                                                                  const Real time)
{
  libmesh_assert (this->initialized());

  std::map<const Elem *, DenseVector<Output>> buffer;
  this->discontinuous_value (p, time, buffer);
  std::map<const Elem *, Output> return_value;
  for (const auto & pr : buffer)
    return_value[pr.first] = pr.second(0);
  // NOTE: If no suitable element is found, then the map return_value is empty. This
  // puts burden on the user of this function but I don't really see a better way.
  return return_value;
}

template <typename Output>
typename MeshFunction<Output>::OutputGradient
MeshFunction<Output>::gradient (const Point & p,
                        const Real time)
{
  libmesh_assert (this->initialized());

  std::vector<OutputGradient> buf (1);
  this->gradient(p, time, buf);
  return buf[0];
}

template <typename Output>
std::map<const Elem *, typename MeshFunction<Output>::OutputGradient>
MeshFunction<Output>::discontinuous_gradient (const Point & p,
                                              const Real time)
{
  libmesh_assert (this->initialized());

  std::map<const Elem *, std::vector<OutputGradient>> buffer;
  this->discontinuous_gradient (p, time, buffer);
  std::map<const Elem *, OutputGradient> return_value;
  for (const auto & pr : buffer)
    return_value[pr.first] = pr.second[0];
  // NOTE: If no suitable element is found, then the map return_value is empty. This
  // puts burden on the user of this function but I don't really see a better way.
  return return_value;
}

#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
template <typename Output>
typename MeshFunction<Output>::OutputTensor
MeshFunction<Output>::hessian (const Point & p,
                               const Real time)
{
  libmesh_assert (this->initialized());

  std::vector<OutputTensor> buf (1);
  this->hessian(p, time, buf);
  return buf[0];
}
#endif

template <typename Output>
void MeshFunction<Output>::operator() (const Point & p,
                               const Real time,
                               DenseVector<Output> & output)
{
  this->operator() (p, time, output, nullptr);
}

template <typename Output>
void MeshFunction<Output>::operator() (const Point & p,
                               const Real,
                               DenseVector<Output> & output,
                               const std::set<subdomain_id_type> * subdomain_ids)
{
  libmesh_assert (this->initialized());

  const Elem * element = this->find_element(p,subdomain_ids);

  if (!element)
    {
      // We'd better be in out_of_mesh_mode if we couldn't find an
      // element in the mesh
      libmesh_assert (_out_of_mesh_mode);
      output = _out_of_mesh_value;
    }
  else
    {
      // resize the output vector to the number of output values
      // that the user told us
      output.resize (cast_int<unsigned int>
                     (this->_system_vars.size()));


      {
        const unsigned int dim = element->dim();


        /*
         * Get local coordinates to feed these into compute_data().
         * Note that the fe_type can safely be used from the 0-variable,
         * since the inverse mapping is the same for all FEFamilies
         */
        const Point mapped_point (FEMap::inverse_map (dim, element,
                                                      p));

        // loop over all vars
        for (auto index : index_range(this->_system_vars))
          {
            /*
             * the data for this variable
             */
            const unsigned int var = _system_vars[index];

            if (var == libMesh::invalid_uint)
              {
                libmesh_assert (_out_of_mesh_mode &&
                                index < _out_of_mesh_value.size());
                output(index) = _out_of_mesh_value(index);
                continue;
              }

            const FEType & fe_type = this->_dof_map.variable_type(var);

            /**
             * Build an FEComputeData that contains both input and output data
             * for the specific compute_data method.
             */
            {
              FEComputeData data (this->_eqn_systems, mapped_point);

              FEInterface::compute_data (dim, fe_type, element, data);

              // where the solution values for the var-th variable are stored
              std::vector<dof_id_type> dof_indices;
              this->_dof_map.dof_indices (element, dof_indices, var);

              // interpolate the solution
              {
                Output value = 0.;

                for (auto i : index_range(dof_indices))
                  value += this->_vector(dof_indices[i]) * data.shape[i];

                output(index) = value;
              }

            }

            // next variable
          }
      }
    }
}


template <typename Output>
void MeshFunction<Output>::discontinuous_value (const Point & p,
                                        const Real time,
                                        std::map<const Elem *, DenseVector<Output>> & output)
{
  this->discontinuous_value (p, time, output, nullptr);
}



template <typename Output>
void MeshFunction<Output>::discontinuous_value (const Point & p,
                                        const Real,
                                        std::map<const Elem *, DenseVector<Output>> & output,
                                        const std::set<subdomain_id_type> * subdomain_ids)
{
  libmesh_assert (this->initialized());

  // clear the output map
  output.clear();

  // get the candidate elements
  std::set<const Elem *> candidate_element = this->find_elements(p,subdomain_ids);

  // loop through all candidates, if the set is empty this function will return an
  // empty map
  for (const auto & element : candidate_element)
    {
      const unsigned int dim = element->dim();

      // define a temporary vector to store all values
      DenseVector<Output> temp_output (cast_int<unsigned int>(this->_system_vars.size()));

      /*
       * Get local coordinates to feed these into compute_data().
       * Note that the fe_type can safely be used from the 0-variable,
       * since the inverse mapping is the same for all FEFamilies
       */
      const Point mapped_point (FEMap::inverse_map (dim, element, p));

      // loop over all vars
      for (auto index : index_range(this->_system_vars))
        {
          /*
           * the data for this variable
           */
          const unsigned int var = _system_vars[index];

          if (var == libMesh::invalid_uint)
            {
              libmesh_assert (_out_of_mesh_mode &&
                              index < _out_of_mesh_value.size());
              temp_output(index) = _out_of_mesh_value(index);
              continue;
            }

          const FEType & fe_type = this->_dof_map.variable_type(var);

          /**
           * Build an FEComputeData that contains both input and output data
           * for the specific compute_data method.
           */
          {
            FEComputeData data (this->_eqn_systems, mapped_point);

            FEInterface::compute_data (dim, fe_type, element, data);

            // where the solution values for the var-th variable are stored
            std::vector<dof_id_type> dof_indices;
            this->_dof_map.dof_indices (element, dof_indices, var);

            // interpolate the solution
            {
              Output value = 0.;

              for (auto i : index_range(dof_indices))
                value += this->_vector(dof_indices[i]) * data.shape[i];

              temp_output(index) = value;
            }

          }

          // next variable
        }

      // Insert temp_output into output
      output[element] = temp_output;
    }
}



template <typename Output>
void MeshFunction<Output>::gradient (const Point & p,
                             const Real,
                             std::vector<OutputGradient> & output,
                             const std::set<subdomain_id_type> * subdomain_ids)
{
  libmesh_assert (this->initialized());

  const Elem * element = this->find_element(p,subdomain_ids);

  if (!element)
    {
      output.resize(0);
      return;
    }
  else
    {
      // resize the output vector to the number of output values
      // that the user told us
      output.resize (this->_system_vars.size());


      {
        const unsigned int dim = element->dim();


        /*
         * Get local coordinates to feed these into compute_data().
         * Note that the fe_type can safely be used from the 0-variable,
         * since the inverse mapping is the same for all FEFamilies
         */
        const Point mapped_point (FEMap::inverse_map (dim, element,
                                                      p));

        std::vector<Point> point_list (1, mapped_point);

        // loop over all vars
        for (auto index : index_range(this->_system_vars))
          {
            /*
             * the data for this variable
             */
            const unsigned int var = _system_vars[index];

            if (var == libMesh::invalid_uint)
              {
                libmesh_assert (_out_of_mesh_mode &&
                                index < _out_of_mesh_value.size());
                output[index] = OutputGradient(_out_of_mesh_value(index));
                continue;
              }

            const FEType & fe_type = this->_dof_map.variable_type(var);

            // where the solution values for the var-th variable are stored
            std::vector<dof_id_type> dof_indices;
            this->_dof_map.dof_indices (element, dof_indices, var);

            // interpolate the solution
            OutputGradient grad(0.);
#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
            //The other algorithm works in case of finite elements as well,
            //but this one is faster.
            if (!element->infinite())
              {
#endif
                std::unique_ptr<FEBase> point_fe (FEBase::build(dim, fe_type));
                const auto & dphi = point_fe->get_dphi();
                point_fe->reinit(element, &point_list);

                for (auto i : index_range(dof_indices))
                  grad.add_scaled(dphi[i][0], this->_vector(dof_indices[i]));

#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
              }
            else
              {
                /**
                 * Build an FEComputeData that contains both input and output data
                 * for the specific compute_data method.
                 */
                FEComputeData data (this->_eqn_systems, mapped_point);
                data.enable_derivative();
                FEInterface::compute_data (dim, fe_type, element, data);
                //grad [x] = data.dshape[i](v) * dv/dx  * dof_index [i]
                // sum over all indices
                for (auto i : index_range(dof_indices))
                  {
                    // local coordinates
                    for (std::size_t v=0; v<dim; v++)
                      for (std::size_t xyz=0; xyz<LIBMESH_DIM; xyz++)
                        {
                          // FIXME: this needs better syntax: It is matrix-vector multiplication.
                          grad(xyz) += data.local_transform[v][xyz]
                            * data.dshape[i](v)
                            * this->_vector(dof_indices[i]);
                        }
                  }
              }
#endif
            output[index] = grad;
          }
      }
    }
}


template <typename Output>
void MeshFunction<Output>::discontinuous_gradient (const Point & p,
                                           const Real time,
                                           std::map<const Elem *, std::vector<OutputGradient>> & output)
{
  this->discontinuous_gradient (p, time, output, nullptr);
}



template <typename Output>
void MeshFunction<Output>::discontinuous_gradient (const Point & p,
                                           const Real,
                                           std::map<const Elem *, std::vector<OutputGradient>> & output,
                                           const std::set<subdomain_id_type> * subdomain_ids)
{
  libmesh_assert (this->initialized());

  // clear the output map
  output.clear();

  // get the candidate elements
  std::set<const Elem *> candidate_element = this->find_elements(p,subdomain_ids);

  // loop through all candidates, if the set is empty this function will return an
  // empty map
  for (const auto & element : candidate_element)
    {
      const unsigned int dim = element->dim();

      // define a temporary vector to store all values
      std::vector<OutputGradient> temp_output (cast_int<unsigned int>(this->_system_vars.size()));

      /*
       * Get local coordinates to feed these into compute_data().
       * Note that the fe_type can safely be used from the 0-variable,
       * since the inverse mapping is the same for all FEFamilies
       */
      const Point mapped_point (FEMap::inverse_map (dim, element, p));


      // loop over all vars
      std::vector<Point> point_list (1, mapped_point);
      for (auto index : index_range(this->_system_vars))
        {
          /*
           * the data for this variable
           */
          const unsigned int var = _system_vars[index];

          if (var == libMesh::invalid_uint)
            {
              libmesh_assert (_out_of_mesh_mode &&
                              index < _out_of_mesh_value.size());
              temp_output[index] = OutputGradient(_out_of_mesh_value(index));
              continue;
            }

          const FEType & fe_type = this->_dof_map.variable_type(var);

          // where the solution values for the var-th variable are stored
          std::vector<dof_id_type> dof_indices;
          this->_dof_map.dof_indices (element, dof_indices, var);

          OutputGradient grad(0.);

          // for performance-reasons, we use different algorithms now.
          // TODO: Check that both give the same result for finite elements.
          // Otherwive it is wrong...
#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
          if (!element->infinite())
            {
#endif
              std::unique_ptr<FEBase> point_fe (FEBase::build(dim, fe_type));
              const auto & dphi = point_fe->get_dphi();
              point_fe->reinit(element, & point_list);

              for (auto i : index_range(dof_indices))
                grad.add_scaled(dphi[i][0], this->_vector(dof_indices[i]));
#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
            }
          else
            {
              /**
               * Build an FEComputeData that contains both input and output data
               * for the specific compute_data method.
               */
              //TODO: enable this for a vector of points as well...
              FEComputeData data (this->_eqn_systems, mapped_point);
              data.enable_derivative();
              FEInterface::compute_data (dim, fe_type, element, data);

              //grad [x] = data.dshape[i](v) * dv/dx  * dof_index [i]
              // sum over all indices
              for (auto i : index_range(dof_indices))
                {
                  // local coordinates.
                  for (std::size_t v=0; v<dim; v++)
                    for (std::size_t xyz=0; xyz<LIBMESH_DIM; xyz++)
                      {
                        // FIXME: this needs better syntax: It is matrix-vector multiplication.
                        grad(xyz) += data.local_transform[v][xyz]
                          * data.dshape[i](v)
                          * this->_vector(dof_indices[i]);
                      }
                }
            }
#endif
          temp_output[index] = grad;

          // next variable
        }

      // Insert temp_output into output
      output[element] = temp_output;
    }
}



#ifdef LIBMESH_ENABLE_SECOND_DERIVATIVES
template <typename Output>
void MeshFunction<Output>::hessian (const Point & p,
                            const Real,
                            std::vector<OutputTensor> & output,
                            const std::set<subdomain_id_type> * subdomain_ids)
{
  libmesh_assert (this->initialized());

  const Elem * element = this->find_element(p,subdomain_ids);

  if (!element)
    {
      output.resize(0);
    }
  else
    {
#ifdef LIBMESH_ENABLE_INFINITE_ELEMENTS
      if(element->infinite())
        libmesh_warning("Warning: Requested the Hessian of an Infinite element."
                        << "Second derivatives for Infinite elements"
                        << " are not yet implemented!"
                        << std::endl);
#endif
      // resize the output vector to the number of output values
      // that the user told us
      output.resize (this->_system_vars.size());


      {
        const unsigned int dim = element->dim();


        /*
         * Get local coordinates to feed these into compute_data().
         * Note that the fe_type can safely be used from the 0-variable,
         * since the inverse mapping is the same for all FEFamilies
         */
        const Point mapped_point (FEMap::inverse_map (dim, element,
                                                      p));

        std::vector<Point> point_list (1, mapped_point);

        // loop over all vars
        for (auto index : index_range(this->_system_vars))
          {
            /*
             * the data for this variable
             */
            const unsigned int var = _system_vars[index];

            if (var == libMesh::invalid_uint)
              {
                libmesh_assert (_out_of_mesh_mode &&
                                index < _out_of_mesh_value.size());
                output[index] = OutputTensor(_out_of_mesh_value(index));
                continue;
              }
            const FEType & fe_type = this->_dof_map.variable_type(var);

            std::unique_ptr<FEBase> point_fe (FEBase::build(dim, fe_type));
            const auto & d2phi =
              point_fe->get_d2phi();
            point_fe->reinit(element, &point_list);

            // where the solution values for the var-th variable are stored
            std::vector<dof_id_type> dof_indices;
            this->_dof_map.dof_indices (element, dof_indices, var);

            // interpolate the solution
            OutputTensor hess;

            for (auto i : index_range(dof_indices))
              hess.add_scaled(d2phi[i][0], this->_vector(dof_indices[i]));

            output[index] = hess;
          }
      }
    }
}
#endif

template <typename Output>
const Elem * MeshFunction<Output>::find_element(const Point & p,
                                        const std::set<subdomain_id_type> * subdomain_ids) const
{
  /* Ensure that in the case of a master mesh function, the
     out-of-mesh mode is enabled either for both or for none.  This is
     important because the out-of-mesh mode is also communicated to
     the point locator.  Since this is time consuming, enable it only
     in debug mode.  */
#ifdef DEBUG
  if (this->_master != nullptr)
    {
      const MeshFunction * master =
        cast_ptr<const MeshFunction *>(this->_master);
      if (_out_of_mesh_mode!=master->_out_of_mesh_mode)
        libmesh_error_msg("ERROR: If you use out-of-mesh-mode in connection with master mesh " \
                          << "functions, you must enable out-of-mesh mode for both the master and the slave mesh function.");
    }
#endif

  // locate the point in the other mesh
  const Elem * element = this->_point_locator->operator()(p,subdomain_ids);

  // If we have an element, but it's not a local element, then we
  // either need to have a serialized vector or we need to find a
  // local element sharing the same point.
  if (element &&
      (element->processor_id() != this->processor_id()) &&
      _vector.type() != SERIAL)
    {
      // look for a local element containing the point
      std::set<const Elem *> point_neighbors;
      element->find_point_neighbors(p, point_neighbors);
      element = nullptr;
      for (const auto & elem : point_neighbors)
          if (elem->processor_id() == this->processor_id())
            {
              element = elem;
              break;
            }
    }

  return element;
}

template <typename Output>
std::set<const Elem *> MeshFunction<Output>::find_elements(const Point & p,
                                                   const std::set<subdomain_id_type> * subdomain_ids) const
{
  /* Ensure that in the case of a master mesh function, the
     out-of-mesh mode is enabled either for both or for none.  This is
     important because the out-of-mesh mode is also communicated to
     the point locator.  Since this is time consuming, enable it only
     in debug mode.  */
#ifdef DEBUG
  if (this->_master != nullptr)
    {
      const MeshFunction * master =
        cast_ptr<const MeshFunction *>(this->_master);
      if (_out_of_mesh_mode!=master->_out_of_mesh_mode)
        libmesh_error_msg("ERROR: If you use out-of-mesh-mode in connection with master mesh " \
                          << "functions, you must enable out-of-mesh mode for both the master and the slave mesh function.");
    }
#endif

  // locate the point in the other mesh
  std::set<const Elem *> candidate_elements;
  std::set<const Elem *> final_candidate_elements;
  this->_point_locator->operator()(p,candidate_elements,subdomain_ids);
  for (const auto & element : candidate_elements)
    {
      // If we have an element, but it's not a local element, then we
      // either need to have a serialized vector or we need to find a
      // local element sharing the same point.
      if (element &&
          (element->processor_id() != this->processor_id()) &&
          _vector.type() != SERIAL)
        {
          // look for a local element containing the point
          std::set<const Elem *> point_neighbors;
          element->find_point_neighbors(p, point_neighbors);
          for (const auto & elem : point_neighbors)
            if (elem->processor_id() == this->processor_id())
              {
                final_candidate_elements.insert(elem);
                break;
              }
        }
      else
        final_candidate_elements.insert(element);
    }

  return final_candidate_elements;
}

template <typename Output>
const PointLocatorBase & MeshFunction<Output>::get_point_locator (void) const
{
  libmesh_assert (this->initialized());
  return *_point_locator;
}

template <typename Output>
void MeshFunction<Output>::enable_out_of_mesh_mode(const DenseVector<Output> & value)
{
  libmesh_assert (this->initialized());
  _point_locator->enable_out_of_mesh_mode();
  _out_of_mesh_mode = true;
  _out_of_mesh_value = value;
}

template <typename Output>
void MeshFunction<Output>::enable_out_of_mesh_mode(const Output & value)
{
  DenseVector<Output> v(1);
  v(0) = value;
  this->enable_out_of_mesh_mode(v);
}

template <typename Output>
void MeshFunction<Output>::disable_out_of_mesh_mode(void)
{
  libmesh_assert (this->initialized());
  _point_locator->disable_out_of_mesh_mode();
  _out_of_mesh_mode = false;
}

template <typename Output>
void MeshFunction<Output>::set_point_locator_tolerance(Real tol)
{
  _point_locator->set_close_to_point_tol(tol);
}

template <typename Output>
void MeshFunction<Output>::unset_point_locator_tolerance()
{
  _point_locator->unset_close_to_point_tol();
}

template class MeshFunction<Number>;
template class MeshFunction<GeomNumber>;

} // namespace libMesh
