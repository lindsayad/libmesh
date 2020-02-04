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



// Local Includes
#include "libmesh/boundary_info.h"
#include "libmesh/distributed_mesh.h"
#include "libmesh/elem.h"
#include "libmesh/ghosting_functor.h"
#include "libmesh/libmesh_config.h"
#include "libmesh/libmesh_common.h"
#include "libmesh/libmesh_logging.h"
#include "libmesh/mesh_base.h"
#include "libmesh/mesh_communication.h"
#include "libmesh/mesh_inserter_iterator.h"
#include "libmesh/mesh_tools.h"
#include "libmesh/parallel.h"
#include "libmesh/parallel_elem.h"
#include "libmesh/parallel_node.h"
#include "libmesh/parallel_ghost_sync.h"
#include "libmesh/utility.h"
#include "libmesh/remote_elem.h"
#include "libmesh/int_range.h"

// C++ Includes
#include <numeric>
#include <set>
#include <unordered_set>
#include <unordered_map>

#ifndef LIBMESH_MESH_COMMUNICATION_IMPL_H
#define LIBMESH_MESH_COMMUNICATION_IMPL_H

//-----------------------------------------------
// anonymous namespace for implementation details
namespace {

using namespace libMesh;

template <typename RealType = Real>
struct SyncNeighbors
{
  typedef std::vector<dof_id_type> datum;
  typedef MeshBaseTempl<RealType> MeshBase;
  typedef ElemTempl<RealType> Elem;
  typedef RemoteElemTempl<RealType> RemoteElem;

  SyncNeighbors(MeshBase & _mesh) :
    mesh(_mesh) {}

  MeshBase & mesh;

  // Find the neighbor ids for each requested element
  void gather_data (const std::vector<dof_id_type> & ids,
                    std::vector<datum> & neighbors) const
  {
    neighbors.resize(ids.size());

    for (auto i : index_range(ids))
      {
        // Look for this element in the mesh
        // We'd better find every element we're asked for
        const Elem & elem = mesh.elem_ref(ids[i]);

        // Return the element's neighbors
        const unsigned int n_neigh = elem.n_neighbors();
        neighbors[i].resize(n_neigh);
        for (unsigned int n = 0; n != n_neigh; ++n)
          {
            const Elem * neigh = elem.neighbor_ptr(n);
            if (neigh)
              {
                libmesh_assert_not_equal_to(neigh, RemoteElemTempl<RealType>::get_instance());
                neighbors[i][n] = neigh->id();
              }
            else
              neighbors[i][n] = DofObject::invalid_id;
          }
      }
  }

  void act_on_data (const std::vector<dof_id_type> & ids,
                    const std::vector<datum> & neighbors) const
  {
    for (auto i : index_range(ids))
      {
        Elem & elem = mesh.elem_ref(ids[i]);

        const datum & new_neigh = neighbors[i];

        const unsigned int n_neigh = elem.n_neighbors();
        libmesh_assert_equal_to (n_neigh, new_neigh.size());

        for (unsigned int n = 0; n != n_neigh; ++n)
          {
            const dof_id_type new_neigh_id = new_neigh[n];
            const Elem * old_neigh = elem.neighbor_ptr(n);
            if (old_neigh && old_neigh != RemoteElemTempl<RealType>::get_instance())
              {
                libmesh_assert_equal_to(old_neigh->id(), new_neigh_id);
              }
            else if (new_neigh_id == DofObject::invalid_id)
              {
                libmesh_assert (!old_neigh);
              }
            else
              {
                Elem * neigh = mesh.query_elem_ptr(new_neigh_id);
                if (neigh)
                  elem.set_neighbor(n, neigh);
                else
                  elem.set_neighbor(n, const_cast<RemoteElem *>(RemoteElemTempl<RealType>::get_instance()));
              }
          }
      }
  }
};


}



namespace libMesh
{

template <typename RealType>
void query_ghosting_functors(const MeshBaseTempl<RealType> & mesh,
                             processor_id_type pid,
                             typename MeshBaseTempl<RealType>::const_element_iterator elem_it,
                             typename MeshBaseTempl<RealType>::const_element_iterator elem_end,
                             std::set<const ElemTempl<RealType> *, CompareElemIdsByLevel> & connected_elements)
{
  for (auto & gf :
         as_range(mesh.ghosting_functors_begin(),
                  mesh.ghosting_functors_end()))
    {
      typename GhostingFunctorTempl<RealType>::map_type elements_to_ghost;
      libmesh_assert(gf);
      (*gf)(elem_it, elem_end, pid, elements_to_ghost);

      // We can ignore the CouplingMatrix in ->second, but we
      // need to ghost all the elements in ->first.
      for (auto & pr : elements_to_ghost)
        {
          const ElemTempl<RealType> * elem = pr.first;
          libmesh_assert(elem != RemoteElemTempl<RealType>::get_instance());
          connected_elements.insert(elem);
        }
    }

  // The GhostingFunctors won't be telling us about the elements from
  // pid; we need to add those ourselves.
  for (; elem_it != elem_end; ++elem_it)
    connected_elements.insert(*elem_it);
}


template <typename RealType>
void connect_children(const MeshBaseTempl<RealType> & mesh,
                      typename MeshBaseTempl<RealType>::const_element_iterator elem_it,
                      typename MeshBaseTempl<RealType>::const_element_iterator elem_end,
                      std::set<const ElemTempl<RealType> *, CompareElemIdsByLevel> & connected_elements)
{
  // None of these parameters are used when !LIBMESH_ENABLE_AMR.
  libmesh_ignore(mesh, elem_it, elem_end, connected_elements);

#ifdef LIBMESH_ENABLE_AMR
  // Our XdrIO output needs inactive local elements to not have any
  // remote_elem children.  Let's make sure that doesn't happen.
  //
  for (const auto & elem : as_range(elem_it, elem_end))
    {
      if (elem->has_children())
        for (auto & child : elem->child_ref_range())
          if (&child != RemoteElemTempl<RealType>::get_instance())
            connected_elements.insert(&child);
    }
#endif // LIBMESH_ENABLE_AMR
}


template <typename RealType>
void connect_families(std::set<const ElemTempl<RealType> *, CompareElemIdsByLevel> & connected_elements)
{
  // This parameter is not used when !LIBMESH_ENABLE_AMR.
  libmesh_ignore(connected_elements);

#ifdef LIBMESH_ENABLE_AMR

  // Because our set is sorted by ascending level, we can traverse it
  // in reverse order, adding parents as we go, and end up with all
  // ancestors added.  This is safe for std::set where insert doesn't
  // invalidate iterators.
  //
  // This only works because we do *not* cache
  // connected_elements.rend(), whose value can change when we insert
  // elements which are sorted before the original rend.
  //
  // We're also going to get subactive descendents here, when any
  // exist.  We're iterating in the wrong direction to do that
  // non-recursively, so we'll cop out and rely on total_family_tree.
  // Iterating backwards does mean that we won't be querying the newly
  // inserted subactive elements redundantly.

  typename std::set<const ElemTempl<RealType> *, CompareElemIdsByLevel>::reverse_iterator
    elem_rit  = connected_elements.rbegin();

  for (; elem_rit != connected_elements.rend(); ++elem_rit)
    {
      const ElemTempl<RealType> * elem = *elem_rit;
      libmesh_assert(elem);
      const ElemTempl<RealType> * parent = elem->parent();

      // We let ghosting functors worry about only active elements,
      // but the remote processor needs all its semilocal elements'
      // ancestors and active semilocal elements' descendants too.
      if (parent)
        connected_elements.insert (parent);

      if (elem->active() && elem->has_children())
        {
          std::vector<const ElemTempl<RealType> *> subactive_family;
          elem->total_family_tree(subactive_family);
          for (const auto & f : subactive_family)
            {
              libmesh_assert(f != RemoteElemTempl<RealType>::get_instance());
              connected_elements.insert(f);
            }
        }
    }

#  ifdef DEBUG
  // Let's be paranoid and make sure that all our ancestors
  // really did get inserted.  I screwed this up the first time
  // by caching rend, and I can easily imagine screwing it up in
  // the future by changing containers.
  for (const auto & elem : connected_elements)
    {
      libmesh_assert(elem);
      const ElemTempl<RealType> * parent = elem->parent();
      if (parent)
        libmesh_assert(connected_elements.count(parent));
    }
#  endif // DEBUG

#endif // LIBMESH_ENABLE_AMR
}


template <typename RealType>
void reconnect_nodes (const std::set<const ElemTempl<RealType> *, CompareElemIdsByLevel> & connected_elements,
                      std::set<const NodeTempl<RealType> *> & connected_nodes)
{
  // We're done using the nodes list for element decisions; now
  // let's reuse it for nodes of the elements we've decided on.
  connected_nodes.clear();

  for (const auto & elem : connected_elements)
    for (auto & n : elem->node_ref_range())
      connected_nodes.insert(&n);
}




// ------------------------------------------------------------
// MeshCommunication class members
void MeshCommunication::clear ()
{
  //  _neighboring_processors.clear();
}



#ifndef LIBMESH_HAVE_MPI // avoid spurious gcc warnings
// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::redistribute (DistributedMeshTempl<RealType> &, bool) const
{
  // no MPI == one processor, no redistribution
  return;
}

#else
// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::redistribute (DistributedMeshTempl<RealType> & mesh,
                                      bool newly_coarsened_only) const
{
  // This method will be called after a new partitioning has been
  // assigned to the elements.  This partitioning was defined in
  // terms of the active elements, and "trickled down" to the
  // parents and nodes as to be consistent.
  //
  // The point is that the entire concept of local elements is
  // kinda shaky in this method.  Elements which were previously
  // local may now be assigned to other processors, so we need to
  // send those off.  Similarly, we need to accept elements from
  // other processors.

  // This method is also useful in the more limited case of
  // post-coarsening redistribution: if elements are only ghosting
  // neighbors of their active elements, but adaptive coarsening
  // causes an inactive element to become active, then we may need a
  // copy of that inactive element's neighbors.

  // The approach is as follows:
  // (1) send all relevant elements we have stored to their proper homes
  // (2) receive elements from all processors, watching for duplicates
  // (3) deleting all nonlocal elements elements
  // (4) obtaining required ghost elements from neighboring processors
  libmesh_parallel_only(mesh.comm());
  libmesh_assert (!mesh.is_serial());
  libmesh_assert (MeshTools::n_elem(mesh.unpartitioned_elements_begin(),
                                    mesh.unpartitioned_elements_end()) == 0);

  LOG_SCOPE("redistribute()", "MeshCommunication");

  // Get a few unique message tags to use in communications; we'll
  // default to some numbers around pi*1000
  Parallel::MessageTag
    nodestag   = mesh.comm().get_unique_tag(3141),
    elemstag   = mesh.comm().get_unique_tag(3142);

  // Figure out how many nodes and elements we have which are assigned to each
  // processor.  send_n_nodes_and_elem_per_proc contains the number of nodes/elements
  // we will be sending to each processor, recv_n_nodes_and_elem_per_proc contains
  // the number of nodes/elements we will be receiving from each processor.
  // Format:
  //  send_n_nodes_and_elem_per_proc[2*pid+0] = number of nodes to send to pid
  //  send_n_nodes_and_elem_per_proc[2*pid+1] = number of elements to send to pid
  std::vector<dof_id_type> send_n_nodes_and_elem_per_proc(2*mesh.n_processors(), 0);

  std::vector<Parallel::Request>
    node_send_requests, element_send_requests;

  // We're going to sort elements-to-send by pid in one pass, to avoid
  // sending predicated iterators through the whole mesh N_p times
  std::unordered_map<processor_id_type, std::vector<ElemTempl<RealType> *>> send_to_pid;

  const typename MeshBaseTempl<RealType>::const_element_iterator send_elems_begin =
#ifdef LIBMESH_ENABLE_AMR
    newly_coarsened_only ?
      mesh.flagged_elements_begin(ElemTempl<RealType>::JUST_COARSENED) :
#endif
      mesh.active_elements_begin();

  const typename MeshBaseTempl<RealType>::const_element_iterator send_elems_end =
#ifdef LIBMESH_ENABLE_AMR
    newly_coarsened_only ?
      mesh.flagged_elements_end(ElemTempl<RealType>::JUST_COARSENED) :
#endif
      mesh.active_elements_end();

  for (auto & elem : as_range(send_elems_begin, send_elems_end))
    send_to_pid[elem->processor_id()].push_back(elem);

  // If we don't have any just-coarsened elements to send to a
  // pid, then there won't be any nodes or any elements pulled
  // in by ghosting either, and we're done with this pid.
  for (auto pair : send_to_pid)
    {
      const processor_id_type pid = pair.first;
      // don't send to ourselves!!
      if (pid == mesh.processor_id())
        continue;

      // Build up a list of nodes and elements to send to processor pid.
      // We will certainly send all the elements assigned to this processor,
      // but we will also ship off any elements which are required
      // to be ghosted and any nodes which are used by any of the
      // above.

      const auto & p_elements = pair.second;
      libmesh_assert(!p_elements.empty());

      ElemTempl<RealType> * const * elempp = p_elements.data();
      ElemTempl<RealType> * const * elemend = elempp + p_elements.size();

#ifndef LIBMESH_ENABLE_AMR
      // This parameter is not used when !LIBMESH_ENABLE_AMR.
      libmesh_ignore(newly_coarsened_only);
      libmesh_assert(!newly_coarsened_only);
#endif

      auto elem_it =
        const_element_iterator_templ<RealType>
          (elempp, elemend, Predicates::NotNull<ElemTempl<RealType> * const *>());

      const auto elem_end =
        const_element_iterator_templ<RealType>
          (elemend, elemend, Predicates::NotNull<ElemTempl<RealType> * const *>());

      std::set<const ElemTempl<RealType> *, CompareElemIdsByLevel> elements_to_send;

      // See which to-be-ghosted elements we need to send
      query_ghosting_functors (mesh, pid, elem_it, elem_end,
                               elements_to_send);

      // The inactive elements we need to send should have their
      // immediate children present.
      connect_children(mesh, mesh.pid_elements_begin(pid),
                       mesh.pid_elements_end(pid),
                       elements_to_send);

      // The elements we need should have their ancestors and their
      // subactive children present too.
      connect_families(elements_to_send);

      std::set<const NodeTempl<RealType> *> connected_nodes;
      reconnect_nodes(elements_to_send, connected_nodes);

      // the number of nodes we will ship to pid
      send_n_nodes_and_elem_per_proc[2*pid+0] =
        cast_int<dof_id_type>(connected_nodes.size());

      // send any nodes off to the destination processor
      libmesh_assert (!connected_nodes.empty());
      node_send_requests.push_back(Parallel::request());

      mesh.comm().send_packed_range (pid, &mesh,
                                     connected_nodes.begin(),
                                     connected_nodes.end(),
                                     node_send_requests.back(),
                                     nodestag);

      // the number of elements we will send to this processor
      send_n_nodes_and_elem_per_proc[2*pid+1] =
        cast_int<dof_id_type>(elements_to_send.size());

      // send the elements off to the destination processor
      libmesh_assert (!elements_to_send.empty());
      element_send_requests.push_back(Parallel::request());

      mesh.comm().send_packed_range (pid, &mesh,
                                     elements_to_send.begin(),
                                     elements_to_send.end(),
                                     element_send_requests.back(),
                                     elemstag);
    }

  std::vector<dof_id_type> recv_n_nodes_and_elem_per_proc(send_n_nodes_and_elem_per_proc);

  mesh.comm().alltoall (recv_n_nodes_and_elem_per_proc);

  // In general we will only need to communicate with a subset of the other processors.
  // I can't immediately think of a case where we will send elements but not nodes, but
  // these are only bools and we have the information anyway...
  std::vector<bool>
    send_node_pair(mesh.n_processors(),false), send_elem_pair(mesh.n_processors(),false),
    recv_node_pair(mesh.n_processors(),false), recv_elem_pair(mesh.n_processors(),false);

  unsigned int
    n_send_node_pairs=0,      n_send_elem_pairs=0,
    n_recv_node_pairs=0,      n_recv_elem_pairs=0;

  for (auto pid : IntRange<processor_id_type>(0, mesh.n_processors()))
    {
      if (send_n_nodes_and_elem_per_proc[2*pid+0]) // we have nodes to send
        {
          send_node_pair[pid] = true;
          n_send_node_pairs++;
        }

      if (send_n_nodes_and_elem_per_proc[2*pid+1]) // we have elements to send
        {
          send_elem_pair[pid] = true;
          n_send_elem_pairs++;
        }

      if (recv_n_nodes_and_elem_per_proc[2*pid+0]) // we have nodes to receive
        {
          recv_node_pair[pid] = true;
          n_recv_node_pairs++;
        }

      if (recv_n_nodes_and_elem_per_proc[2*pid+1]) // we have elements to receive
        {
          recv_elem_pair[pid] = true;
          n_recv_elem_pairs++;
        }
    }
  libmesh_assert_equal_to (n_send_node_pairs, node_send_requests.size());
  libmesh_assert_equal_to (n_send_elem_pairs, element_send_requests.size());

  // Receive nodes.
  // We now know how many processors will be sending us nodes.
  for (unsigned int node_comm_step=0; node_comm_step<n_recv_node_pairs; node_comm_step++)
    // but we don't necessarily want to impose an ordering, so
    // just process whatever message is available next.
    mesh.comm().receive_packed_range (Parallel::any_source,
                                      &mesh,
                                      mesh_inserter_iterator<NodeTempl<RealType>>(mesh),
                                      (NodeTempl<RealType>**)nullptr,
                                      nodestag);

  // Receive elements.
  // Similarly we know how many processors are sending us elements,
  // but we don't really care in what order we receive them.
  for (unsigned int elem_comm_step=0; elem_comm_step<n_recv_elem_pairs; elem_comm_step++)
    mesh.comm().receive_packed_range (Parallel::any_source,
                                      &mesh,
                                      mesh_inserter_iterator<ElemTempl<RealType>>(mesh),
                                      (ElemTempl<RealType>**)nullptr,
                                      elemstag);

  // Wait for all sends to complete
  Parallel::wait (node_send_requests);
  Parallel::wait (element_send_requests);

  // Check on the redistribution consistency
#ifdef DEBUG
  MeshTools::libmesh_assert_equal_n_systems(mesh);

  MeshTools::libmesh_assert_valid_refinement_tree(mesh);
#endif

  // If we had a point locator, it's invalid now that there are new
  // elements it can't locate.
  mesh.clear_point_locator();

  // We now have all elements and nodes redistributed; our ghosting
  // functors should be ready to redistribute and/or recompute any
  // cached data they use too.
  for (auto & gf : as_range(mesh.ghosting_functors_begin(), mesh.ghosting_functors_end()))
    gf->redistribute();
}
#endif // LIBMESH_HAVE_MPI



#ifndef LIBMESH_HAVE_MPI // avoid spurious gcc warnings
// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::gather_neighboring_elements (DistributedMeshTempl<RealType> &) const
{
  // no MPI == one processor, no need for this method...
  return;
}
#else
// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::gather_neighboring_elements (DistributedMeshTempl<RealType> & mesh) const
{
  // Don't need to do anything if there is
  // only one processor.
  if (mesh.n_processors() == 1)
    return;

  // This function must be run on all processors at once
  libmesh_parallel_only(mesh.comm());

  LOG_SCOPE("gather_neighboring_elements()", "MeshCommunication");

  //------------------------------------------------------------------
  // The purpose of this function is to provide neighbor data structure
  // consistency for a parallel, distributed mesh.  In libMesh we require
  // that each local element have access to a full set of valid face
  // neighbors.  In some cases this requires us to store "ghost elements" -
  // elements that belong to other processors but we store to provide
  // data structure consistency.  Also, it is assumed that any element
  // with a nullptr neighbor resides on a physical domain boundary.  So,
  // even our "ghost elements" must have non-nullptr neighbors.  To handle
  // this the concept of "RemoteElem" is used - a special construct which
  // is used to denote that an element has a face neighbor, but we do
  // not actually store detailed information about that neighbor.  This
  // is required to prevent data structure explosion.
  //
  // So when this method is called we should have only local elements.
  // These local elements will then find neighbors among the local
  // element set.  After this is completed, any element with a nullptr
  // neighbor has either (i) a face on the physical boundary of the mesh,
  // or (ii) a neighboring element which lives on a remote processor.
  // To handle case (ii), we communicate the global node indices connected
  // to all such faces to our neighboring processors.  They then send us
  // all their elements with a nullptr neighbor that are connected to any
  // of the nodes in our list.
  //------------------------------------------------------------------

  // Let's begin with finding consistent neighbor data information
  // for all the elements we currently have.  We'll use a clean
  // slate here - clear any existing information, including RemoteElem's.
  mesh.find_neighbors (/* reset_remote_elements = */ true,
                       /* reset_current_list    = */ true);

  // Get a unique message tag to use in communications
  Parallel::MessageTag
    element_neighbors_tag = mesh.comm().get_unique_tag();

  // Now any element with a nullptr neighbor either
  // (i) lives on the physical domain boundary, or
  // (ii) lives on an inter-processor boundary.
  // We will now gather all the elements from adjacent processors
  // which are of the same state, which should address all the type (ii)
  // elements.

  // A list of all the processors which *may* contain neighboring elements.
  // (for development simplicity, just make this the identity map)
  std::vector<processor_id_type> adjacent_processors;
  for (auto pid : IntRange<processor_id_type>(0, mesh.n_processors()))
    if (pid != mesh.processor_id())
      adjacent_processors.push_back (pid);


  const processor_id_type n_adjacent_processors =
    cast_int<processor_id_type>(adjacent_processors.size());

  //-------------------------------------------------------------------------
  // Let's build a list of all nodes which live on nullptr-neighbor sides.
  // For simplicity, we will use a set to build the list, then transfer
  // it to a vector for communication.
  std::vector<dof_id_type> my_interface_node_list;
  std::vector<const ElemTempl<RealType> *>  my_interface_elements;
  {
    std::set<dof_id_type> my_interface_node_set;

    // Pull objects out of the loop to reduce heap operations
    std::unique_ptr<const ElemTempl<RealType>> side;

    // since parent nodes are a subset of children nodes, this should be sufficient
    for (const auto & elem : mesh.active_local_element_ptr_range())
      {
        libmesh_assert(elem);

        if (elem->on_boundary()) // denotes *any* side has a nullptr neighbor
          {
            my_interface_elements.push_back(elem); // add the element, but only once, even
            // if there are multiple nullptr neighbors
            for (auto s : elem->side_index_range())
              if (elem->neighbor_ptr(s) == nullptr)
                {
                  elem->build_side_ptr(side, s);

                  for (auto n : IntRange<unsigned int>(0, side->n_vertices()))
                    my_interface_node_set.insert (side->node_id(n));
                }
          }
      }

    my_interface_node_list.reserve (my_interface_node_set.size());
    my_interface_node_list.insert  (my_interface_node_list.end(),
                                    my_interface_node_set.begin(),
                                    my_interface_node_set.end());
  }

  // we will now send my_interface_node_list to all of the adjacent processors.
  // note that for the time being we will copy the list to a unique buffer for
  // each processor so that we can use a nonblocking send and not access the
  // buffer again until the send completes.  it is my understanding that the
  // MPI 2.1 standard seeks to remove this restriction as unnecessary, so in
  // the future we should change this to send the same buffer to each of the
  // adjacent processors. - BSK 11/17/2008
  std::vector<std::vector<dof_id_type>>
    my_interface_node_xfer_buffers (n_adjacent_processors, my_interface_node_list);
  std::map<processor_id_type, unsigned char> n_comm_steps;

  std::vector<Parallel::Request> send_requests (3*n_adjacent_processors);
  unsigned int current_request = 0;

  for (unsigned int comm_step=0; comm_step<n_adjacent_processors; comm_step++)
    {
      n_comm_steps[adjacent_processors[comm_step]]=1;
      mesh.comm().send (adjacent_processors[comm_step],
                        my_interface_node_xfer_buffers[comm_step],
                        send_requests[current_request++],
                        element_neighbors_tag);
    }

  //-------------------------------------------------------------------------
  // processor pairings are symmetric - I expect to receive an interface node
  // list from each processor in adjacent_processors as well!
  // now we will catch an incoming node list for each of our adjacent processors.
  //
  // we are done with the adjacent_processors list - note that it is in general
  // a superset of the processors we truly share elements with.  so let's
  // clear the superset list, and we will fill it with the true list.
  adjacent_processors.clear();

  std::vector<dof_id_type> common_interface_node_list;

  // we expect two classes of messages -
  // (1) incoming interface node lists, to which we will reply with our elements
  //     touching nodes in the list, and
  // (2) replies from the requests we sent off previously.
  //  (2.a) - nodes
  //  (2.b) - elements
  // so we expect 3 communications from each adjacent processor.
  // by structuring the communication in this way we hopefully impose no
  // order on the handling of the arriving messages.  in particular, we
  // should be able to handle the case where we receive a request and
  // all replies from processor A before even receiving a request from
  // processor B.

  for (unsigned int comm_step=0; comm_step<3*n_adjacent_processors; comm_step++)
    {
      //------------------------------------------------------------------
      // catch incoming node list
      Parallel::Status
        status(mesh.comm().probe (Parallel::any_source,
                                  element_neighbors_tag));
      const processor_id_type
        source_pid_idx = cast_int<processor_id_type>(status.source()),
        dest_pid_idx   = source_pid_idx;

      //------------------------------------------------------------------
      // first time - incoming request
      if (n_comm_steps[source_pid_idx] == 1)
        {
          n_comm_steps[source_pid_idx]++;

          mesh.comm().receive (source_pid_idx,
                               common_interface_node_list,
                               element_neighbors_tag);

          // const std::size_t
          //   their_interface_node_list_size = common_interface_node_list.size();

          // we now have the interface node list from processor source_pid_idx.
          // now we can find all of our elements which touch any of these nodes
          // and send copies back to this processor.  however, we can make our
          // search more efficient by first excluding all the nodes in
          // their list which are not also contained in
          // my_interface_node_list.  we can do this in place as a set
          // intersection.
          common_interface_node_list.erase
            (std::set_intersection (my_interface_node_list.begin(),
                                    my_interface_node_list.end(),
                                    common_interface_node_list.begin(),
                                    common_interface_node_list.end(),
                                    common_interface_node_list.begin()),
             common_interface_node_list.end());

          // if (false)
          //   libMesh::out << "[" << mesh.processor_id() << "] "
          //                << "my_interface_node_list.size()="       << my_interface_node_list.size()
          //                << ", [" << source_pid_idx << "] "
          //                << "their_interface_node_list.size()="    << their_interface_node_list_size
          //                << ", common_interface_node_list.size()=" << common_interface_node_list.size()
          //                << std::endl;

          // Now we need to see which of our elements touch the nodes in the list.
          // We will certainly send all the active elements which intersect source_pid_idx,
          // but we will also ship off the other elements in the same family tree
          // as the active ones for data structure consistency.
          //
          // FIXME - shipping full family trees is unnecessary and inefficient.
          //
          // We also ship any nodes connected to these elements.  Note
          // some of these nodes and elements may be replicated from
          // other processors, but that is OK.
          std::set<const ElemTempl<RealType> *, CompareElemIdsByLevel> elements_to_send;
          std::set<const NodeTempl<RealType> *> connected_nodes;

          // Check for quick return?
          if (common_interface_node_list.empty())
            {
              // let's try to be smart here - if we have no nodes in common,
              // we cannot share elements.  so post the messages expected
              // from us here and go on about our business.
              // note that even though these are nonblocking sends
              // they should complete essentially instantly, because
              // in all cases the send buffers are empty
              mesh.comm().send_packed_range (dest_pid_idx,
                                             &mesh,
                                             connected_nodes.begin(),
                                             connected_nodes.end(),
                                             send_requests[current_request++],
                                             element_neighbors_tag);

              mesh.comm().send_packed_range (dest_pid_idx,
                                             &mesh,
                                             elements_to_send.begin(),
                                             elements_to_send.end(),
                                             send_requests[current_request++],
                                             element_neighbors_tag);

              continue;
            }
          // otherwise, this really *is* an adjacent processor.
          adjacent_processors.push_back(source_pid_idx);

          std::vector<const ElemTempl<RealType> *> family_tree;

          for (auto & elem : my_interface_elements)
            {
              std::size_t n_shared_nodes = 0;

              for (auto n : IntRange<unsigned int>(0, elem->n_vertices()))
                if (std::binary_search (common_interface_node_list.begin(),
                                        common_interface_node_list.end(),
                                        elem->node_id(n)))
                  {
                    n_shared_nodes++;

                    // TBD - how many nodes do we need to share
                    // before we care?  certainly 2, but 1?  not
                    // sure, so let's play it safe...
                    if (n_shared_nodes > 0) break;
                  }

              if (n_shared_nodes) // share at least one node?
                {
                  elem = elem->top_parent();

                  // avoid a lot of duplicated effort -- if we already have elem
                  // in the set its entire family tree is already in the set.
                  if (!elements_to_send.count(elem))
                    {
#ifdef LIBMESH_ENABLE_AMR
                      elem->family_tree(family_tree);
#else
                      family_tree.clear();
                      family_tree.push_back(elem);
#endif
                      for (const auto & f : family_tree)
                        {
                          elem = f;
                          elements_to_send.insert (elem);

                          for (auto & n : elem->node_ref_range())
                            connected_nodes.insert (&n);
                        }
                    }
                }
            }

          // The elements_to_send and connected_nodes sets now contain all
          // the elements and nodes we need to send to this processor.
          // All that remains is to pack up the objects (along with
          // any boundary conditions) and send the messages off.
          {
            libmesh_assert (connected_nodes.empty() || !elements_to_send.empty());
            libmesh_assert (!connected_nodes.empty() || elements_to_send.empty());

            // send the nodes off to the destination processor
            mesh.comm().send_packed_range (dest_pid_idx,
                                           &mesh,
                                           connected_nodes.begin(),
                                           connected_nodes.end(),
                                           send_requests[current_request++],
                                           element_neighbors_tag);

            // send the elements off to the destination processor
            mesh.comm().send_packed_range (dest_pid_idx,
                                           &mesh,
                                           elements_to_send.begin(),
                                           elements_to_send.end(),
                                           send_requests[current_request++],
                                           element_neighbors_tag);
          }
        }
      //------------------------------------------------------------------
      // second time - reply of nodes
      else if (n_comm_steps[source_pid_idx] == 2)
        {
          n_comm_steps[source_pid_idx]++;

          mesh.comm().receive_packed_range (source_pid_idx,
                                            &mesh,
                                            mesh_inserter_iterator<NodeTempl<RealType>>(mesh),
                                            (NodeTempl<RealType>**)nullptr,
                                            element_neighbors_tag);
        }
      //------------------------------------------------------------------
      // third time - reply of elements
      else if (n_comm_steps[source_pid_idx] == 3)
        {
          n_comm_steps[source_pid_idx]++;

          mesh.comm().receive_packed_range (source_pid_idx,
                                            &mesh,
                                            mesh_inserter_iterator<ElemTempl<RealType>>(mesh),
                                            (ElemTempl<RealType>**)nullptr,
                                            element_neighbors_tag);
        }
      //------------------------------------------------------------------
      // fourth time - shouldn't happen
      else
        {
          libMesh::err << "ERROR:  unexpected number of replies: "
                       << n_comm_steps[source_pid_idx]
                       << std::endl;
        }
    } // done catching & processing replies associated with tag ~ 100,000pi

  // allow any pending requests to complete
  Parallel::wait (send_requests);

  // If we had a point locator, it's invalid now that there are new
  // elements it can't locate.
  mesh.clear_point_locator();

  // We can now find neighbor information for the interfaces between
  // local elements and ghost elements.
  mesh.find_neighbors (/* reset_remote_elements = */ true,
                       /* reset_current_list    = */ false);

  // Ghost elements may not have correct remote_elem neighbor links,
  // and we may not be able to locally infer correct neighbor links to
  // remote elements.  So we synchronize ghost element neighbor links.
  SyncNeighbors<RealType> nsync(mesh);

  Parallel::sync_dofobject_data_by_id
    (mesh.comm(), mesh.elements_begin(), mesh.elements_end(), nsync);
}
#endif // LIBMESH_HAVE_MPI


#ifndef LIBMESH_HAVE_MPI // avoid spurious gcc warnings
// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::send_coarse_ghosts(MeshBaseTempl<RealType> &) const
{
  // no MPI == one processor, no need for this method...
  return;
}
#else
template <typename RealType>
void MeshCommunication::send_coarse_ghosts(MeshBaseTempl<RealType> & mesh) const
{

  // Don't need to do anything if all processors already ghost all non-local
  // elements.
  if (mesh.is_serial())
    return;

  // This algorithm uses the MeshBase::flagged_pid_elements_begin/end iterators
  // which are only available when AMR is enabled.
#ifndef LIBMESH_ENABLE_AMR
  libmesh_error_msg("Calling MeshCommunication::send_coarse_ghosts() requires AMR to be enabled. "
                    "Please configure libmesh with --enable-amr.");
#else
  // When we coarsen elements on a DistributedMesh, we make their
  // parents active.  This may increase the ghosting requirements on
  // the processor which owns the newly-activated parent element.  To
  // ensure ghosting requirements are satisfied, processors which
  // coarsen an element will send all the associated ghosted elements
  // to all processors which own any of the coarsened-away-element's
  // siblings.
  typedef std::unordered_map<processor_id_type, std::vector<ElemTempl<RealType> *>> ghost_map;
  ghost_map coarsening_elements_to_ghost;

  const processor_id_type proc_id = mesh.processor_id();
  // Look for just-coarsened elements
  for (auto elem : as_range(mesh.flagged_pid_elements_begin(ElemTempl<RealType>::COARSEN, proc_id),
                            mesh.flagged_pid_elements_end(ElemTempl<RealType>::COARSEN, proc_id)))
    {
      // If it's flagged for coarsening it had better have a parent
      libmesh_assert(elem->parent());

      // On a distributed mesh:
      // If we don't own this element's parent but we do own it, then
      // there is a chance that we are aware of ghost elements which
      // the parent's owner needs us to send them.
      const processor_id_type their_proc_id = elem->parent()->processor_id();
      if (their_proc_id != proc_id)
        coarsening_elements_to_ghost[their_proc_id].push_back(elem);
    }

  const processor_id_type n_proc = mesh.n_processors();

  // Get a few unique message tags to use in communications; we'll
  // default to some numbers around pi*1000
  Parallel::MessageTag
    nodestag   = mesh.comm().get_unique_tag(3141),
    elemstag   = mesh.comm().get_unique_tag(3142);

  std::vector<Parallel::Request> send_requests;

  // Using unsigned char instead of bool since it'll get converted for
  // MPI use later anyway
  std::vector<unsigned char> send_to_proc(n_proc, 0);

  for (processor_id_type p=0; p != n_proc; ++p)
    {
      if (p == proc_id)
        break;

      // We'll send these asynchronously, but their data will be
      // packed into Parallel:: buffers so it will be okay when the
      // original containers are destructed before the sends complete.
      std::set<const ElemTempl<RealType> *, CompareElemIdsByLevel> elements_to_send;
      std::set<const NodeTempl<RealType> *> nodes_to_send;

      const typename ghost_map::const_iterator it =
        coarsening_elements_to_ghost.find(p);
      if (it != coarsening_elements_to_ghost.end())
        {
          const std::vector<ElemTempl<RealType> *> & elems = it->second;
          libmesh_assert(elems.size());

          // Make some fake element iterators defining this vector of
          // elements
          ElemTempl<RealType> * const * elempp = const_cast<ElemTempl<RealType> * const *>(elems.data());
          ElemTempl<RealType> * const * elemend = elempp+elems.size();

          const auto elem_it =
            const_element_iterator_templ<RealType>(elempp, elemend,
                                                   Predicates::NotNull<ElemTempl<RealType> * const *>());

          const auto elem_end =
            const_element_iterator_templ<RealType>(elemend, elemend,
                                                   Predicates::NotNull<ElemTempl<RealType> * const *>());

          for (auto & gf : as_range(mesh.ghosting_functors_begin(),
                                    mesh.ghosting_functors_end()))
            {
              typename GhostingFunctorTempl<RealType>::map_type elements_to_ghost;
              libmesh_assert(gf);
              (*gf)(elem_it, elem_end, p, elements_to_ghost);

              // We can ignore the CouplingMatrix in ->second, but we
              // need to ghost all the elements in ->first.
              for (auto & pr : elements_to_ghost)
                {
                  const ElemTempl<RealType> * elem = pr.first;
                  libmesh_assert(elem);
                  while (elem)
                    {
                      libmesh_assert(elem != RemoteElemTempl<RealType>::get_instance());
                      elements_to_send.insert(elem);
                      for (auto & n : elem->node_ref_range())
                        nodes_to_send.insert(&n);
                      elem = elem->parent();
                    }
                }
            }

          send_requests.push_back(Parallel::request());

          mesh.comm().send_packed_range (p, &mesh,
                                         nodes_to_send.begin(),
                                         nodes_to_send.end(),
                                         send_requests.back(),
                                         nodestag);

          send_requests.push_back(Parallel::request());

          send_to_proc[p] = 1; // true

          mesh.comm().send_packed_range (p, &mesh,
                                         elements_to_send.begin(),
                                         elements_to_send.end(),
                                         send_requests.back(),
                                         elemstag);
        }
    }

  // Find out how many other processors are sending us elements+nodes.
  std::vector<unsigned char> recv_from_proc(send_to_proc);
  mesh.comm().alltoall(recv_from_proc);

  const processor_id_type n_receives = cast_int<processor_id_type>
    (std::count(recv_from_proc.begin(), recv_from_proc.end(), 1));

  // Receive nodes first since elements will need to attach to them
  for (processor_id_type recv_i = 0; recv_i != n_receives; ++recv_i)
    {
      mesh.comm().receive_packed_range
        (Parallel::any_source,
         &mesh,
         mesh_inserter_iterator<NodeTempl<RealType>>(mesh),
         (NodeTempl<RealType>**)nullptr,
         nodestag);
    }

  for (processor_id_type recv_i = 0; recv_i != n_receives; ++recv_i)
    {
      mesh.comm().receive_packed_range
        (Parallel::any_source,
         &mesh,
         mesh_inserter_iterator<ElemTempl<RealType>>(mesh),
         (ElemTempl<RealType>**)nullptr,
         elemstag);
    }

  // Wait for all sends to complete
  Parallel::wait (send_requests);
#endif // LIBMESH_ENABLE_AMR
}

#endif // LIBMESH_HAVE_MPI

#ifndef LIBMESH_HAVE_MPI // avoid spurious gcc warnings
// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::broadcast (MeshBaseTempl<RealType> &) const
{
  // no MPI == one processor, no need for this method...
  return;
}
#else
// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::broadcast (MeshBaseTempl<RealType> & mesh) const
{
  // Don't need to do anything if there is
  // only one processor.
  if (mesh.n_processors() == 1)
    return;

  // This function must be run on all processors at once
  libmesh_parallel_only(mesh.comm());

  LOG_SCOPE("broadcast()", "MeshCommunication");

  // Explicitly clear the mesh on all but processor 0.
  if (mesh.processor_id() != 0)
    mesh.clear();

  // We may have set extra data only on processor 0 in a read()
  mesh.comm().broadcast(mesh._node_integer_names);
  mesh.comm().broadcast(mesh._elem_integer_names);

  // We may have set mapping data only on processor 0 in a read()
  unsigned char map_type = mesh.default_mapping_type();
  unsigned char map_data = mesh.default_mapping_data();
  mesh.comm().broadcast(map_type);
  mesh.comm().broadcast(map_data);
  mesh.set_default_mapping_type(ElemMappingType(map_type));
  mesh.set_default_mapping_data(map_data);

  // Broadcast nodes
  mesh.comm().broadcast_packed_range(&mesh,
                                     mesh.nodes_begin(),
                                     mesh.nodes_end(),
                                     &mesh,
                                     mesh_inserter_iterator<NodeTempl<RealType>>(mesh));

  // Broadcast elements from coarsest to finest, so that child
  // elements will see their parents already in place.
  //
  // When restarting from a checkpoint, we may have elements which are
  // assigned to a processor but which have not yet been sent to that
  // processor, so we need to use a paranoid n_levels() count and not
  // the usual fast algorithm.
  const unsigned int n_levels = MeshTools::paranoid_n_levels(mesh);

  for (unsigned int l=0; l != n_levels; ++l)
    mesh.comm().broadcast_packed_range(&mesh,
                                       mesh.level_elements_begin(l),
                                       mesh.level_elements_end(l),
                                       &mesh,
                                       mesh_inserter_iterator<ElemTempl<RealType>>(mesh));

  // Make sure mesh_dimension and elem_dimensions are consistent.
  mesh.cache_elem_dims();

  // Broadcast all of the named entity information
  mesh.comm().broadcast(mesh.set_subdomain_name_map());
  mesh.comm().broadcast(mesh.get_boundary_info().set_sideset_name_map());
  mesh.comm().broadcast(mesh.get_boundary_info().set_nodeset_name_map());

  // If we had a point locator, it's invalid now that there are new
  // elements it can't locate.
  mesh.clear_point_locator();

  libmesh_assert (mesh.comm().verify(mesh.n_elem()));
  libmesh_assert (mesh.comm().verify(mesh.n_nodes()));

#ifdef DEBUG
  MeshTools::libmesh_assert_valid_procids<ElemTempl<RealType>>(mesh);
  MeshTools::libmesh_assert_valid_procids<NodeTempl<RealType>>(mesh);
#endif
}
#endif // LIBMESH_HAVE_MPI



#ifndef LIBMESH_HAVE_MPI // avoid spurious gcc warnings
// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::gather (const processor_id_type, DistributedMeshTempl<RealType> &) const
{
  // no MPI == one processor, no need for this method...
  return;
}
#else
// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::gather (const processor_id_type root_id, DistributedMeshTempl<RealType> & mesh) const
{
  // Check for quick return
  if (mesh.n_processors() == 1)
    return;

  // This function must be run on all processors at once
  libmesh_parallel_only(mesh.comm());

  LOG_SCOPE("(all)gather()", "MeshCommunication");

  (root_id == DofObject::invalid_processor_id) ?

    mesh.comm().allgather_packed_range (&mesh,
                                        mesh.nodes_begin(),
                                        mesh.nodes_end(),
                                        mesh_inserter_iterator<NodeTempl<RealType>>(mesh)) :

    mesh.comm().gather_packed_range (root_id,
                                     &mesh,
                                     mesh.nodes_begin(),
                                     mesh.nodes_end(),
                                     mesh_inserter_iterator<NodeTempl<RealType>>(mesh));

  // Gather elements from coarsest to finest, so that child
  // elements will see their parents already in place.
  const unsigned int n_levels = MeshTools::n_levels(mesh);

  for (unsigned int l=0; l != n_levels; ++l)
    (root_id == DofObject::invalid_processor_id) ?

      mesh.comm().allgather_packed_range (&mesh,
                                          mesh.level_elements_begin(l),
                                          mesh.level_elements_end(l),
                                          mesh_inserter_iterator<ElemTempl<RealType>>(mesh)) :

      mesh.comm().gather_packed_range (root_id,
                                       &mesh,
                                       mesh.level_elements_begin(l),
                                       mesh.level_elements_end(l),
                                       mesh_inserter_iterator<ElemTempl<RealType>>(mesh));

  // If we had a point locator, it's invalid now that there are new
  // elements it can't locate.
  mesh.clear_point_locator();

  // If we are doing an allgather(), perform sanity check on the result.
  if (root_id == DofObject::invalid_processor_id)
    {
      libmesh_assert (mesh.comm().verify(mesh.n_elem()));
      libmesh_assert (mesh.comm().verify(mesh.n_nodes()));
    }

  // Inform new elements of their neighbors,
  // while resetting all remote_elem links on
  // the ranks which did the gather.
  mesh.find_neighbors(root_id == DofObject::invalid_processor_id ||
                      root_id == mesh.processor_id());

  // All done, but let's make sure it's done correctly

#ifdef DEBUG
  MeshTools::libmesh_assert_valid_boundary_ids(mesh);
#endif
}
#endif // LIBMESH_HAVE_MPI



// Functor for make_elems_parallel_consistent and
// make_node_ids_parallel_consistent
namespace {

template <typename RealType = Real>
struct SyncIds
{
  typedef MeshBaseTempl<RealType> MeshBase;

  typedef dof_id_type datum;
  typedef void (MeshBase::*renumber_obj)(dof_id_type, dof_id_type);

  SyncIds(MeshBase & _mesh, renumber_obj _renumberer) :
    mesh(_mesh),
    renumber(_renumberer) {}

  MeshBase & mesh;
  renumber_obj renumber;
  // renumber_obj & renumber;

  // Find the id of each requested DofObject -
  // Parallel::sync_* already did the work for us
  void gather_data (const std::vector<dof_id_type> & ids,
                    std::vector<datum> & ids_out) const
  {
    ids_out = ids;
  }

  void act_on_data (const std::vector<dof_id_type> & old_ids,
                    const std::vector<datum> & new_ids) const
  {
    for (auto i : index_range(old_ids))
      if (old_ids[i] != new_ids[i])
        (mesh.*renumber)(old_ids[i], new_ids[i]);
  }
};

template <typename RealType = Real>
struct SyncNodeIds
{
  typedef MeshBaseTempl<RealType> MeshBase;
  typedef dof_id_type datum;

  SyncNodeIds(MeshBase & _mesh) :
    mesh(_mesh) {}

  MeshBase & mesh;

  // We only know a Node id() is definitive if we own the Node or if
  // we're told it's definitive.  We keep track of the latter cases by
  // putting definitively id'd ghost nodes into this set.
  typedef std::unordered_set<const NodeTempl<RealType> *> uset_type;
  uset_type definitive_ids;

  // We should never be told two different definitive ids for the same
  // node, but let's check on that in debug mode.
#ifdef DEBUG
  typedef std::unordered_map<dof_id_type, dof_id_type> umap_type;
  umap_type definitive_renumbering;
#endif

  // Find the id of each requested DofObject -
  // Parallel::sync_* already tried to do the work for us, but we can
  // only say the result is definitive if we own the DofObject or if
  // we were given the definitive result from another processor.
  void gather_data (const std::vector<dof_id_type> & ids,
                    std::vector<datum> & ids_out) const
  {
    ids_out.clear();
    ids_out.resize(ids.size(), DofObject::invalid_id);

    for (auto i : index_range(ids))
      {
        const dof_id_type id = ids[i];
        const NodeTempl<RealType> * node = mesh.query_node_ptr(id);
        if (node && (node->processor_id() == mesh.processor_id() ||
                     definitive_ids.count(node)))
          ids_out[i] = id;
      }
  }

  bool act_on_data (const std::vector<dof_id_type> & old_ids,
                    const std::vector<datum> & new_ids)
  {
    bool data_changed = false;
    for (auto i : index_range(old_ids))
      {
        const dof_id_type new_id = new_ids[i];

        const dof_id_type old_id = old_ids[i];

        NodeTempl<RealType> * node = mesh.query_node_ptr(old_id);

        // If we can't find the node we were asking about, another
        // processor must have already given us the definitive id
        // for it
        if (!node)
          {
            // But let's check anyway in debug mode
#ifdef DEBUG
            libmesh_assert
              (definitive_renumbering.count(old_id));
            libmesh_assert_equal_to
              (new_id, definitive_renumbering[old_id]);
#endif
            continue;
          }

        // If we asked for an id but there's no definitive id ready
        // for us yet, then we can't quit trying to sync yet.
        if (new_id == DofObject::invalid_id)
          {
            // But we might have gotten a definitive id from a
            // different request
            if (!definitive_ids.count(mesh.node_ptr(old_id)))
              data_changed = true;
          }
        else
          {
            if (node->processor_id() != mesh.processor_id())
              definitive_ids.insert(node);
            if (old_id != new_id)
              {
#ifdef DEBUG
                libmesh_assert
                  (!definitive_renumbering.count(old_id));
                definitive_renumbering[old_id] = new_id;
#endif
                mesh.renumber_node(old_id, new_id);
                data_changed = true;
              }
          }
      }
    return data_changed;
  }
};


#ifdef LIBMESH_ENABLE_AMR
template <typename RealType = Real>
struct SyncPLevels
{
  typedef MeshBaseTempl<RealType> MeshBase;
  typedef unsigned char datum;

  SyncPLevels(MeshBase & _mesh) :
    mesh(_mesh) {}

  MeshBase & mesh;

  // Find the p_level of each requested Elem
  void gather_data (const std::vector<dof_id_type> & ids,
                    std::vector<datum> & ids_out) const
  {
    ids_out.reserve(ids.size());

    for (const auto & id : ids)
      {
        ElemTempl<RealType> & elem = mesh.elem_ref(id);
        ids_out.push_back(cast_int<unsigned char>(elem.p_level()));
      }
  }

  void act_on_data (const std::vector<dof_id_type> & old_ids,
                    const std::vector<datum> & new_p_levels) const
  {
    for (auto i : index_range(old_ids))
      {
        ElemTempl<RealType> & elem = mesh.elem_ref(old_ids[i]);
        elem.set_p_level(new_p_levels[i]);
      }
  }
};
#endif // LIBMESH_ENABLE_AMR


#ifdef LIBMESH_ENABLE_UNIQUE_ID
template <typename DofObjSubclass, typename RealType = Real>
struct SyncUniqueIds
{
  typedef MeshBaseTempl<RealType> MeshBase;
  typedef unique_id_type datum;
  typedef DofObjSubclass* (MeshBase::*query_obj)(const dof_id_type);

  SyncUniqueIds(MeshBase &_mesh, query_obj _querier) :
    mesh(_mesh),
    query(_querier) {}

  MeshBase & mesh;
  query_obj query;

  // Find the id of each requested DofObject -
  // Parallel::sync_* already did the work for us
  void gather_data (const std::vector<dof_id_type> & ids,
                    std::vector<datum> & ids_out) const
  {
    ids_out.reserve(ids.size());

    for (const auto & id : ids)
      {
        DofObjSubclass * d = (mesh.*query)(id);
        libmesh_assert(d);
        ids_out.push_back(d->unique_id());
      }
  }

  void act_on_data (const std::vector<dof_id_type> & ids,
                    const std::vector<datum> & unique_ids) const
  {
    for (auto i : index_range(ids))
      {
        DofObjSubclass * d = (mesh.*query)(ids[i]);
        libmesh_assert(d);
        d->set_unique_id() = unique_ids[i];
      }
  }
};
#endif // LIBMESH_ENABLE_UNIQUE_ID
}



// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::make_node_ids_parallel_consistent (MeshBaseTempl<RealType> & mesh)
{
  // This function must be run on all processors at once
  libmesh_parallel_only(mesh.comm());

  // We need to agree on which processor owns every node, but we can't
  // easily assert that here because we don't currently agree on which
  // id every node has, and some of our temporary ids on unrelated
  // nodes will "overlap".
//#ifdef DEBUG
//  MeshTools::libmesh_assert_parallel_consistent_procids<NodeTempl<RealType>> (mesh);
//#endif // DEBUG

  LOG_SCOPE ("make_node_ids_parallel_consistent()", "MeshCommunication");

  SyncNodeIds<RealType> syncids(mesh);
  Parallel::sync_node_data_by_element_id
    (mesh, mesh.elements_begin(), mesh.elements_end(),
     Parallel::SyncEverything(), Parallel::SyncEverything(), syncids);

  // At this point, with both ids and processor ids synced, we can
  // finally check for topological consistency of node processor ids.
#ifdef DEBUG
  MeshTools::libmesh_assert_topology_consistent_procids<NodeTempl<RealType>> (mesh);
#endif
}



template <typename RealType>
void MeshCommunication::make_node_unique_ids_parallel_consistent (MeshBaseTempl<RealType> & mesh)
{
  // Avoid unused variable warnings if unique ids aren't enabled.
  libmesh_ignore(mesh);

  // This function must be run on all processors at once
  libmesh_parallel_only(mesh.comm());

#ifdef LIBMESH_ENABLE_UNIQUE_ID
  LOG_SCOPE ("make_node_unique_ids_parallel_consistent()", "MeshCommunication");

  SyncUniqueIds<NodeTempl<RealType>> syncuniqueids(mesh, &MeshBaseTempl<RealType>::query_node_ptr);
  Parallel::sync_dofobject_data_by_id(mesh.comm(),
                                      mesh.nodes_begin(),
                                      mesh.nodes_end(),
                                      syncuniqueids);

#endif
}




// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::make_elems_parallel_consistent(MeshBaseTempl<RealType> & mesh)
{
  // This function must be run on all processors at once
  libmesh_parallel_only(mesh.comm());

  LOG_SCOPE ("make_elems_parallel_consistent()", "MeshCommunication");

  SyncIds<RealType> syncids(mesh, &MeshBaseTempl<RealType>::renumber_elem);
  Parallel::sync_element_data_by_parent_id
    (mesh, mesh.active_elements_begin(),
     mesh.active_elements_end(), syncids);

#ifdef LIBMESH_ENABLE_UNIQUE_ID
  SyncUniqueIds<ElemTempl<RealType>> syncuniqueids(mesh, &MeshBaseTempl<RealType>::query_elem_ptr);
  Parallel::sync_dofobject_data_by_id
    (mesh.comm(), mesh.active_elements_begin(),
     mesh.active_elements_end(), syncuniqueids);
#endif
}



// ------------------------------------------------------------
#ifdef LIBMESH_ENABLE_AMR
template <typename RealType>
void MeshCommunication::make_p_levels_parallel_consistent(MeshBaseTempl<RealType> & mesh)
{
  // This function must be run on all processors at once
  libmesh_parallel_only(mesh.comm());

  LOG_SCOPE ("make_p_levels_parallel_consistent()", "MeshCommunication");

  SyncPLevels<RealType> syncplevels(mesh);
  Parallel::sync_dofobject_data_by_id
    (mesh.comm(), mesh.elements_begin(), mesh.elements_end(),
     syncplevels);
}
#endif // LIBMESH_ENABLE_AMR



// Functors for make_node_proc_ids_parallel_consistent
namespace {

template <typename RealType = Real>
struct SyncProcIds
{
  typedef MeshBaseTempl<RealType> MeshBase;
  typedef processor_id_type datum;

  SyncProcIds(MeshBase & _mesh) : mesh(_mesh) {}

  MeshBase & mesh;

  // ------------------------------------------------------------
  void gather_data (const std::vector<dof_id_type> & ids,
                    std::vector<datum> & data)
  {
    // Find the processor id of each requested node
    data.resize(ids.size());

    for (auto i : index_range(ids))
      {
        // Look for this point in the mesh
        if (ids[i] != DofObject::invalid_id)
          {
            NodeTempl<RealType> & node = mesh.node_ref(ids[i]);

            // Return the node's correct processor id,
            data[i] = node.processor_id();
          }
        else
          data[i] = DofObject::invalid_processor_id;
      }
  }

  // ------------------------------------------------------------
  bool act_on_data (const std::vector<dof_id_type> & ids,
                    const std::vector<datum> proc_ids)
  {
    bool data_changed = false;

    // Set the ghost node processor ids we've now been informed of
    for (auto i : index_range(ids))
      {
        NodeTempl<RealType> & node = mesh.node_ref(ids[i]);

        // We may not have ids synched when this synchronization is done, so we
        // *can't* use id to load-balance processor id properly; we have to use
        // the old heuristic of choosing the smallest valid processor id.
        //
        // If someone tells us our node processor id is too low, then
        // they're wrong.  If they tell us our node processor id is
        // too high, then we're wrong.
        if (node.processor_id() > proc_ids[i])
          {
            data_changed = true;
            node.processor_id() = proc_ids[i];
          }
      }

    return data_changed;
  }
};


struct ElemNodesMaybeNew
{
  ElemNodesMaybeNew() {}

  template <typename RealType>
  bool operator() (const ElemTempl<RealType> * elem) const
  {
    // If this element was just refined then it may have new nodes we
    // need to work on
#ifdef LIBMESH_ENABLE_AMR
    if (elem->refinement_flag() == ElemTempl<RealType>::JUST_REFINED)
      return true;
#endif

    // If this element has remote_elem neighbors then there may have
    // been refinement of those neighbors that affect its nodes'
    // processor_id()
    for (auto neigh : elem->neighbor_ptr_range())
      if (neigh == RemoteElemTempl<RealType>::get_instance())
        return true;
    return false;
  }
};

template <typename RealType = Real>
struct NodeWasNew
{
  typedef MeshBaseTempl<RealType> MeshBase;
  typedef ElemTempl<RealType> Elem;
  typedef NodeTempl<RealType> Node;

  NodeWasNew(const MeshBaseTempl<RealType> & mesh)
  {
    for (const auto & node : mesh.node_ptr_range())
      if (node->processor_id() == DofObject::invalid_processor_id)
        was_new.insert(node);
  }

  bool operator() (const Elem * elem, unsigned int local_node_num) const
  {
    if (was_new.count(elem->node_ptr(local_node_num)))
      return true;
    return false;
  }

  std::unordered_set<const Node *> was_new;
};

}



// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::make_node_proc_ids_parallel_consistent(MeshBaseTempl<RealType> & mesh)
{
  LOG_SCOPE ("make_node_proc_ids_parallel_consistent()", "MeshCommunication");

  // This function must be run on all processors at once
  libmesh_parallel_only(mesh.comm());

  // When this function is called, each section of a parallelized mesh
  // should be in the following state:
  //
  // All nodes should have the exact same physical location on every
  // processor where they exist.
  //
  // Local nodes should have unique authoritative ids,
  // and processor ids consistent with all processors which own
  // an element touching them.
  //
  // Ghost nodes touching local elements should have processor ids
  // consistent with all processors which own an element touching
  // them.
  SyncProcIds<RealType> sync(mesh);
  Parallel::sync_node_data_by_element_id
    (mesh, mesh.elements_begin(), mesh.elements_end(),
     Parallel::SyncEverything(), Parallel::SyncEverything(), sync);
}



// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::make_new_node_proc_ids_parallel_consistent(MeshBaseTempl<RealType> & mesh)
{
  LOG_SCOPE ("make_new_node_proc_ids_parallel_consistent()", "MeshCommunication");

  // This function must be run on all processors at once
  libmesh_parallel_only(mesh.comm());

  // When this function is called, each section of a parallelized mesh
  // should be in the following state:
  //
  // Local nodes should have unique authoritative ids,
  // and new nodes should be unpartitioned.
  //
  // New ghost nodes touching local elements should be unpartitioned.

  // We may not have consistent processor ids for new nodes (because a
  // node may be old and partitioned on one processor but new and
  // unpartitioned on another) when we start
#ifdef DEBUG
  MeshTools::libmesh_assert_parallel_consistent_procids<NodeTempl<RealType>>(mesh);
  // MeshTools::libmesh_assert_parallel_consistent_new_node_procids(mesh);
#endif

  // We have two kinds of new nodes.  *NEW* nodes are unpartitioned on
  // all processors: we need to use a id-independent (i.e. dumb)
  // heuristic to partition them.  But "new" nodes are newly created
  // on some processors (when ghost elements are refined) yet
  // correspond to existing nodes on other processors: we need to use
  // the existing processor id for them.
  //
  // A node which is "new" on one processor will be associated with at
  // least one ghost element, and we can just query that ghost
  // element's owner to find out the correct processor id.

  auto node_unpartitioned =
    [](const ElemTempl<RealType> * elem, unsigned int local_node_num)
    { return elem->node_ref(local_node_num).processor_id() ==
        DofObject::invalid_processor_id; };

  SyncProcIds<RealType> sync(mesh);

  sync_node_data_by_element_id_once
    (mesh, mesh.not_local_elements_begin(),
     mesh.not_local_elements_end(), Parallel::SyncEverything(),
     node_unpartitioned, sync);

  // Nodes should now be unpartitioned iff they are truly new; those
  // are the *only* nodes we will touch.
#ifdef DEBUG
  MeshTools::libmesh_assert_parallel_consistent_new_node_procids(mesh);
#endif

  NodeWasNew<RealType> node_was_new(mesh);

  // Set the lowest processor id we can on truly new nodes
  for (auto & elem : mesh.element_ptr_range())
    for (auto & node : elem->node_ref_range())
      if (node_was_new.was_new.count(&node))
        {
          processor_id_type & pid = node.processor_id();
          pid = std::min(pid, elem->processor_id());
        }

  // Then finally see if other processors have a lower option
  Parallel::sync_node_data_by_element_id
    (mesh, mesh.elements_begin(), mesh.elements_end(),
     ElemNodesMaybeNew(), node_was_new, sync);

  // We should have consistent processor ids when we're done.
#ifdef DEBUG
  MeshTools::libmesh_assert_parallel_consistent_procids<NodeTempl<RealType>>(mesh);
  MeshTools::libmesh_assert_parallel_consistent_new_node_procids(mesh);
#endif
}



// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::make_nodes_parallel_consistent (MeshBaseTempl<RealType> & mesh)
{
  // This function must be run on all processors at once
  libmesh_parallel_only(mesh.comm());

  // When this function is called, each section of a parallelized mesh
  // should be in the following state:
  //
  // All nodes should have the exact same physical location on every
  // processor where they exist.
  //
  // Local nodes should have unique authoritative ids,
  // and processor ids consistent with all processors which own
  // an element touching them.
  //
  // Ghost nodes touching local elements should have processor ids
  // consistent with all processors which own an element touching
  // them.
  //
  // Ghost nodes should have ids which are either already correct
  // or which are in the "unpartitioned" id space.

  // First, let's sync up processor ids.  Some of these processor ids
  // may be "wrong" from coarsening, but they're right in the sense
  // that they'll tell us who has the authoritative dofobject ids for
  // each node.

  this->make_node_proc_ids_parallel_consistent(mesh);

  // Second, sync up dofobject ids.
  this->make_node_ids_parallel_consistent(mesh);

  // Third, sync up dofobject unique_ids if applicable.
  this->make_node_unique_ids_parallel_consistent(mesh);

  // Finally, correct the processor ids to make DofMap happy
  MeshTools::correct_node_proc_ids(mesh);
}



// ------------------------------------------------------------
template <typename RealType>
void MeshCommunication::make_new_nodes_parallel_consistent (MeshBaseTempl<RealType> & mesh)
{
  // This function must be run on all processors at once
  libmesh_parallel_only(mesh.comm());

  // When this function is called, each section of a parallelized mesh
  // should be in the following state:
  //
  // All nodes should have the exact same physical location on every
  // processor where they exist.
  //
  // Local nodes should have unique authoritative ids,
  // and new nodes should be unpartitioned.
  //
  // New ghost nodes touching local elements should be unpartitioned.
  //
  // New ghost nodes should have ids which are either already correct
  // or which are in the "unpartitioned" id space.
  //
  // Non-new nodes should have correct ids and processor ids already.

  // First, let's sync up new nodes' processor ids.

  this->make_new_node_proc_ids_parallel_consistent(mesh);

  // Second, sync up dofobject ids.
  this->make_node_ids_parallel_consistent(mesh);

  // Third, sync up dofobject unique_ids if applicable.
  this->make_node_unique_ids_parallel_consistent(mesh);

  // Finally, correct the processor ids to make DofMap happy
  MeshTools::correct_node_proc_ids(mesh);
}



// ------------------------------------------------------------
template <typename RealType>
void
MeshCommunication::delete_remote_elements (DistributedMeshTempl<RealType> & mesh,
                                           const std::set<ElemTempl<RealType> *> & extra_ghost_elem_ids) const
{
  // The mesh should know it's about to be parallelized
  libmesh_assert (!mesh.is_serial());

  LOG_SCOPE("delete_remote_elements()", "MeshCommunication");

#ifdef DEBUG
  // We expect maximum ids to be in sync so we can use them to size
  // vectors
  libmesh_assert(mesh.comm().verify(mesh.max_node_id()));
  libmesh_assert(mesh.comm().verify(mesh.max_elem_id()));
  const dof_id_type par_max_node_id = mesh.parallel_max_node_id();
  const dof_id_type par_max_elem_id = mesh.parallel_max_elem_id();
  libmesh_assert_equal_to (par_max_node_id, mesh.max_node_id());
  libmesh_assert_equal_to (par_max_elem_id, mesh.max_elem_id());
#endif

  std::set<const ElemTempl<RealType> *, CompareElemIdsByLevel> elements_to_keep;

  // Don't delete elements that we were explicitly told not to
  for (const auto & elem : extra_ghost_elem_ids)
    {
      std::vector<const ElemTempl<RealType> *> active_family;
#ifdef LIBMESH_ENABLE_AMR
      if (!elem->subactive())
        elem->active_family_tree(active_family);
      else
#endif
        active_family.push_back(elem);

      for (const auto & f : active_family)
        elements_to_keep.insert(f);
    }

  // See which elements we still need to keep ghosted, given that
  // we're keeping local and unpartitioned elements.
  query_ghosting_functors
    (mesh, mesh.processor_id(),
     mesh.active_pid_elements_begin(mesh.processor_id()),
     mesh.active_pid_elements_end(mesh.processor_id()),
     elements_to_keep);
  query_ghosting_functors
    (mesh, DofObject::invalid_processor_id,
     mesh.active_pid_elements_begin(DofObject::invalid_processor_id),
     mesh.active_pid_elements_end(DofObject::invalid_processor_id),
     elements_to_keep);

  // The inactive elements we need to send should have their
  // immediate children present.
  connect_children(mesh, mesh.pid_elements_begin(mesh.processor_id()),
                   mesh.pid_elements_end(mesh.processor_id()),
                   elements_to_keep);
  connect_children(mesh,
                   mesh.pid_elements_begin(DofObject::invalid_processor_id),
                   mesh.pid_elements_end(DofObject::invalid_processor_id),
                   elements_to_keep);

  // The elements we need should have their ancestors and their
  // subactive children present too.
  connect_families(elements_to_keep);

  // Don't delete nodes that our semilocal elements need
  std::set<const NodeTempl<RealType> *> connected_nodes;
  reconnect_nodes(elements_to_keep, connected_nodes);

  // Delete all the elements we have no reason to save,
  // starting with the most refined so that the mesh
  // is valid at all intermediate steps
  unsigned int n_levels = MeshTools::n_levels(mesh);

  for (int l = n_levels - 1; l >= 0; --l)
    for (auto & elem : as_range(mesh.level_elements_begin(l),
                                mesh.level_elements_end(l)))
      {
        libmesh_assert (elem);
        // Make sure we don't leave any invalid pointers
        const bool keep_me = elements_to_keep.count(elem);

        if (!keep_me)
          elem->make_links_to_me_remote();

        // delete_elem doesn't currently invalidate element
        // iterators... that had better not change
        if (!keep_me)
          mesh.delete_elem(elem);
      }

  // Delete all the nodes we have no reason to save
  for (auto & node : mesh.node_ptr_range())
    {
      libmesh_assert(node);
      if (!connected_nodes.count(node))
        mesh.delete_node(node);
    }

  // If we had a point locator, it's invalid now that some of the
  // elements it pointed to have been deleted.
  mesh.clear_point_locator();

  // Much of our boundary info may have been for now-remote parts of
  // the mesh, in which case we don't want to keep local copies.
  mesh.get_boundary_info().regenerate_id_sets();

  // We now have all remote elements and nodes deleted; our ghosting
  // functors should be ready to delete any now-redundant cached data
  // they use too.
  for (auto & gf : as_range(mesh.ghosting_functors_begin(), mesh.ghosting_functors_end()))
    gf->delete_remote_elements();

#ifdef DEBUG
  MeshTools::libmesh_assert_valid_refinement_tree(mesh);
#endif
}

} // namespace libMesh

#define MESH_COMM_INSTANTIATIONS0(RealType)                                                        \
  template void MeshCommunication::broadcast(MeshBaseTempl<RealType> &) const;                     \
  template void MeshCommunication::redistribute(DistributedMeshTempl<RealType> & mesh,             \
                                                bool newly_coarsened_only = false) const;          \
  template void MeshCommunication::gather_neighboring_elements(DistributedMeshTempl<RealType> &)   \
      const;                                                                                       \
  template void MeshCommunication::send_coarse_ghosts(MeshBaseTempl<RealType> &) const;            \
  template void MeshCommunication::gather(const processor_id_type root_id,                         \
                                          DistributedMeshTempl<RealType> &) const;                 \
  template void MeshCommunication::delete_remote_elements(                                         \
      DistributedMeshTempl<RealType> &, const std::set<ElemTempl<RealType> *> &) const;            \
  template void MeshCommunication::make_elems_parallel_consistent(MeshBaseTempl<RealType> &);      \
  template void MeshCommunication::make_node_ids_parallel_consistent(MeshBaseTempl<RealType> &);   \
  template void MeshCommunication::make_node_unique_ids_parallel_consistent(                       \
      MeshBaseTempl<RealType> &);                                                                  \
  template void MeshCommunication::make_node_proc_ids_parallel_consistent(                         \
      MeshBaseTempl<RealType> &);                                                                  \
  template void MeshCommunication::make_new_node_proc_ids_parallel_consistent(                     \
      MeshBaseTempl<RealType> &);                                                                  \
  template void MeshCommunication::make_nodes_parallel_consistent(MeshBaseTempl<RealType> &);      \
  template void MeshCommunication::make_new_nodes_parallel_consistent(MeshBaseTempl<RealType> &);  \
  template void query_ghosting_functors(                                                           \
      const MeshBaseTempl<RealType> & mesh,                                                        \
      processor_id_type pid,                                                                       \
      typename MeshBaseTempl<RealType>::const_element_iterator elem_it,                            \
      typename MeshBaseTempl<RealType>::const_element_iterator elem_end,                           \
      std::set<const ElemTempl<RealType> *, CompareElemIdsByLevel> & connected_elements);          \
  template void connect_children(                                                                  \
      const MeshBaseTempl<RealType> & mesh,                                                        \
      typename MeshBaseTempl<RealType>::const_element_iterator elem_it,                            \
      typename MeshBaseTempl<RealType>::const_element_iterator elem_end,                           \
      std::set<const ElemTempl<RealType> *, CompareElemIdsByLevel> & connected_elements);          \
  template void connect_families(                                                                  \
      std::set<const ElemTempl<RealType> *, CompareElemIdsByLevel> & connected_elements);          \
  template void reconnect_nodes(                                                                   \
      const std::set<const ElemTempl<RealType> *, CompareElemIdsByLevel> & connected_elements,     \
      std::set<const NodeTempl<RealType> *> & connected_nodes)

#ifdef LIBMESH_ENABLE_AMR
#define MESH_COMM_INSTANTIATIONS(RealType)                                                         \
  MESH_COMM_INSTANTIATIONS0(RealType);                                                             \
  template void MeshCommunication::make_p_levels_parallel_consistent(MeshBaseTempl<RealType> &)
#else
#define MESH_COMM_INSTANTIATIONS(RealType) MESH_COMM_INSTANTIATIONS0(RealType)
#endif // LIBMESH_ENABLE_AMR

#endif // LIBMESH_MESH_COMMUNICATION_IMPL_H
