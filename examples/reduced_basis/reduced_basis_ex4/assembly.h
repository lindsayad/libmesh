#ifndef ASSEMBLY_H
#define ASSEMBLY_H

// libMesh includes
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe.h"
#include "libmesh/quadrature.h"
#include "libmesh/dof_map.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/fe_interface.h"
#include "libmesh/elem.h"
#include "libmesh/utility.h"
#include "libmesh/print_trace.h"

// rbOOmit includes
#include "libmesh/rb_assembly_expansion.h"
#include "libmesh/rb_eim_theta.h"
#include "libmesh/rb_parametrized_function.h"

// C++ includes
#include <cmath>

// Bring in bits from the libMesh namespace.
// Just the bits we're using, since this is a header.
using libMesh::ElemAssembly;
using libMesh::FEMContext;
using libMesh::Number;
using libMesh::Point;
using libMesh::RBAssemblyExpansion;
using libMesh::RBEIMAssembly;
using libMesh::RBEIMConstruction;
using libMesh::RBParametrizedFunction;
using libMesh::RBParameters;
using libMesh::RBTheta;
using libMesh::RBThetaExpansion;
using libMesh::Real;
using libMesh::RealGradient;
using libMesh::Elem;
using libMesh::FEBase;
using libMesh::subdomain_id_type;
using libMesh::dof_id_type;
using libMesh::Utility::pow;

struct ShiftedGaussian : public RBParametrizedFunction
{
  unsigned int get_n_components() const override
  {
    return 1;
  }

  virtual std::vector<Number>
  evaluate(const RBParameters & mu,
           const Point & p,
           dof_id_type /*elem_id*/,
           unsigned int /*qp*/,
           subdomain_id_type /*subdomain_id*/,
           const std::vector<Point> & /*p_perturb*/,
           const std::vector<Real> & /*phi_i_qp*/) override
  {
    // // Old way, there is only 1 entry in the return vector
    // Real center_x = mu.get_value("center_x");
    // Real center_y = mu.get_value("center_y");
    // return std::vector<Number> { std::exp(-2. * (pow<2>(center_x - p(0)) + pow<2>(center_y - p(1)))) };

    // New way, there are get_n_components() * mu.max_n_values() entries in the return vector.
    // Make sure that the same number of values are provided for both relevant parameters.
    auto n_values_x = mu.n_values("center_x");
    auto n_values_y = mu.n_values("center_y");
    libmesh_error_msg_if(n_values_x != n_values_y, "Must specify same number of values for all parameters.");

    // Debugging: print number of values
    // libMesh::out << "Called ShiftedGaussian::evaluate() with " << n_values_x << " value(s) for each parameter." << std::endl;

    std::vector<Number> ret(this->get_n_components() * n_values_x);
    for (std::size_t i=0; i<n_values_x; ++i)
      {
        Real center_x = mu.get_value("center_x", i);
        Real center_y = mu.get_value("center_y", i);
        ret[i] = std::exp(-2. * (pow<2>(center_x - p(0)) + pow<2>(center_y - p(1))));
      }
    return ret;
  }
};

// A simple Theta(mu) function which just returns a constant value
// (hence does not depend on mu). The constant must be specified when
// calling the constructor.
struct ThetaConstant : RBTheta
{
  /**
   * Constructor
   */
  ThetaConstant(Number val) : _val(val) {}

  /**
   * Evaluate theta for a single scalar-valued RBParameters object.
   * In this case, Theta(mu) does not depend on mu explicitly, except
   * to determine the number of "steps" (aka max_n_values()) which mu
   * has, so that the output vector is sized appropriately.
   */
  virtual Number evaluate(const RBParameters & mu) override
  {
    libmesh_error_msg_if(mu.max_n_values() > 1,
                         "You should only call the evaluate_vec() API when using multi-step RBParameters objects.");

    return _val;
  }

  /**
   * Evaluate theta for multiple mu values, each of which may have multiple "steps".
   * This theta still doesn't depend on mu, but the output vector must be sized appropriately.
   */
  virtual std::vector<Number> evaluate_vec(const std::vector<RBParameters> & mus) override
  {
    // Compute the number of values to be returned in the vector. For
    // scalar-valued RBParameters objects, there would be mus.size()
    // values returned in the vector. For step-valued RBParameters
    // objects, there are:
    // sum_i mus[i].max_n_values()
    // total Thetas, i.e. one Theta per step.
    unsigned int count = 0;
    for (const auto & mu : mus)
      count += mu.max_n_values();

    return std::vector<Number>(count, _val);
  }

private:
  Number _val;
};

struct A0 : ElemAssembly
{
  // Assemble the Laplacian operator
  virtual void interior_assembly(FEMContext & c)
  {
    const unsigned int u_var = 0;

    FEBase * elem_fe = nullptr;
    c.get_element_fe(u_var, elem_fe);

    const std::vector<Real> & JxW = elem_fe->get_JxW();

    // The velocity shape function gradients at interior
    // quadrature points.
    const std::vector<std::vector<RealGradient>> & dphi = elem_fe->get_dphi();

    // The number of local degrees of freedom in each variable
    const unsigned int n_u_dofs = c.get_dof_indices(u_var).size();

    // Now we will build the affine operator
    unsigned int n_qpoints = c.get_element_qrule().n_points();

    for (unsigned int qp=0; qp != n_qpoints; qp++)
      for (unsigned int i=0; i != n_u_dofs; i++)
        for (unsigned int j=0; j != n_u_dofs; j++)
          c.get_elem_jacobian()(i,j) += JxW[qp] * dphi[j][qp]*dphi[i][qp];
  }
};


struct EIM_IP_assembly : ElemAssembly
{
  // Use the L2 norm to find the best fit
  virtual void interior_assembly(FEMContext & c)
  {
    const unsigned int u_var = 0;

    FEBase * elem_fe = nullptr;
    c.get_element_fe(u_var, elem_fe);

    const std::vector<Real> & JxW = elem_fe->get_JxW();

    const std::vector<std::vector<Real>> & phi = elem_fe->get_phi();

    const unsigned int n_u_dofs = c.get_dof_indices(u_var).size();

    unsigned int n_qpoints = c.get_element_qrule().n_points();

    for (unsigned int qp=0; qp != n_qpoints; qp++)
      for (unsigned int i=0; i != n_u_dofs; i++)
        for (unsigned int j=0; j != n_u_dofs; j++)
          c.get_elem_jacobian()(i,j) += JxW[qp] * phi[j][qp]*phi[i][qp];
  }
};

struct EIM_F : RBEIMAssembly
{
  EIM_F(RBEIMConstruction & rb_eim_con_in,
        unsigned int basis_function_index_in) :
    RBEIMAssembly(rb_eim_con_in,
                  basis_function_index_in)
  {}

  virtual void interior_assembly(FEMContext & c)
  {
    // PDE variable number
    const unsigned int u_var = 0;

    FEBase * elem_fe = nullptr;
    c.get_element_fe(u_var, elem_fe);

    // EIM variable number
    const unsigned int eim_var = 0;

    const std::vector<Real> & JxW = elem_fe->get_JxW();

    const std::vector<std::vector<Real>> & phi = elem_fe->get_phi();

    // The number of local degrees of freedom in each variable
    const unsigned int n_u_dofs = c.get_dof_indices(u_var).size();

    std::vector<Number> eim_values;
    evaluate_basis_function(c.get_elem().id(),
                            eim_var,
                            eim_values);

    for (unsigned int qp=0; qp != c.get_element_qrule().n_points(); qp++)
      for (unsigned int i=0; i != n_u_dofs; i++)
        c.get_elem_residual()(i) += JxW[qp] * (eim_values[qp]*phi[i][qp]);
  }
};

/**
 * Output assembly object which computes the average value of the
 * solution variable inside a BoundingBox defined by lower corner
 * [min_x_in, min_y_in] and upper corner [max_x_in, max_y_in].
 * OutputAssembly is copied from reduced_basis_ex1 where it is also
 * used.
 */
struct OutputAssembly : ElemAssembly
{
  OutputAssembly(Real min_x_in, Real max_x_in,
                 Real min_y_in, Real max_y_in)
    :
    min_x(min_x_in),
    max_x(max_x_in),
    min_y(min_y_in),
    max_y(max_y_in)
  {}

  // Output: Average value over the region [min_x,max_x]x[min_y,max_y]
  virtual void interior_assembly(FEMContext & c)
  {
    const unsigned int u_var = 0;

    FEBase * fe = nullptr;
    c.get_element_fe(u_var, fe);

    const std::vector<Real> & JxW = fe->get_JxW();
    const std::vector<std::vector<Real>> & phi = fe->get_phi();

    // The number of local degrees of freedom in each variable
    const unsigned int n_u_dofs = c.get_dof_indices(u_var).size();

    // Now we will build the affine operator
    unsigned int n_qpoints = c.get_element_qrule().n_points();

    Real output_area = (max_x-min_x) * (max_y-min_y);

    Point avg = c.get_elem().vertex_average();
    if ((min_x <= avg(0)) && (avg(0) <= max_x) &&
        (min_y <= avg(1)) && (avg(1) <= max_y))
      for (unsigned int qp=0; qp != n_qpoints; qp++)
        for (unsigned int i=0; i != n_u_dofs; i++)
          c.get_elem_residual()(i) += JxW[qp] * phi[i][qp] / output_area;
  }

  // Member variables that define the output region in 2D
  Real min_x, max_x, min_y, max_y;
};

// Define an RBThetaExpansion class for this PDE
struct EimTestRBThetaExpansion : RBThetaExpansion
{
  /**
   * Constructor.
   */
  EimTestRBThetaExpansion() :
    theta_a_0(0.05),
    output_theta(1.0)
  {
    attach_A_theta(&theta_a_0);

    // Note: there are no Thetas associated with the RHS since we use
    // an EIM approximation for the forcing term.

    // Attach an RBTheta object for the output. Here we just use the
    // ThetaConstant class again, but this time with a value of 1.0.
    attach_output_theta(&output_theta);
  }

  // The RBTheta member variables
  ThetaConstant theta_a_0;
  ThetaConstant output_theta;
};

// Define an RBAssemblyExpansion class for this PDE
struct EimTestRBAssemblyExpansion : RBAssemblyExpansion
{
  /**
   * Constructor.
   */
  EimTestRBAssemblyExpansion():
    L0(/*min_x=*/-0.2, /*max_x=*/0.2,
       /*min_y=*/-0.2, /*max_x=*/0.2)
  {
    attach_A_assembly(&A0_assembly);
    attach_output_assembly(&L0);
  }

  // A0 assembly object
  A0 A0_assembly;

  // Assembly object associated with the output functional
  OutputAssembly L0;
};

#endif
