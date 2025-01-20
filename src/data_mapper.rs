//! Maps global data to local data required by processes.
//!
//! Consider ten global dofs with indices 0 to 9 and two processes. The first process
//! may need the dofs 0, 1, 2, 3, 4, 5, 6 and the second process the dofs 3, 4, 5, 6, 7, 8, 9.
//! Hence, some dofs are needed on both processes. The `Global2LocalDataMapper` establishes the corresponding
//! communication and maps distributed vectors of global dofs to the required dofs on each process.

use std::{collections::HashMap, rc::Rc};

use itertools::izip;
use mpi::traits::{Communicator, Equivalence};

use crate::IndexLayout;

/// Maps global data to local data.
pub struct Global2LocalDataMapper<'a, C: Communicator> {
    index_layout: Rc<IndexLayout<'a, C>>,
    ghost_communicator: crate::GhostCommunicator<usize>,
    owned_dofs: Vec<usize>,
    ghost_dofs: Vec<usize>,
    dof_to_position: HashMap<usize, usize>,
}

impl<'a, C: Communicator> Global2LocalDataMapper<'a, C> {
    /// Create a new data mapper.
    ///
    /// The `required_dofs` are the dofs that are required on the local process.
    pub fn new(index_layout: Rc<IndexLayout<'a, C>>, required_dofs: &[usize]) -> Self {
        let comm = index_layout.comm();
        let rank = comm.rank() as usize;

        // First we go through the required dofs, determine which ones are ghosts and
        // create a map from dof index to position in the required dofs.

        let mut owned_dofs = Vec::<usize>::new();
        let mut ghost_dofs = Vec::<usize>::new();
        let mut ghost_owners = Vec::<usize>::new();

        let mut dof_to_position = HashMap::<usize, usize>::new();

        for (pos, &dof) in required_dofs.iter().enumerate() {
            dof_to_position.insert(dof, pos);
            let original_rank = index_layout.rank_from_index(dof).unwrap();
            if original_rank == rank {
                owned_dofs.push(dof);
            } else {
                ghost_dofs.push(dof);
                ghost_owners.push(original_rank);
            }
        }

        // We now initialize the ghost communicator.

        let ghost_communicator = crate::GhostCommunicator::new(&ghost_dofs, &ghost_owners, comm);

        // That's it. Return the struct.

        Self {
            index_layout,
            ghost_communicator,
            owned_dofs,
            ghost_dofs,
            dof_to_position,
        }
    }

    /// Map global data to the local required data
    ///
    /// The input data is a vector of global data. A chunk size can be given in case multiple elements
    /// are associated with each dof.
    pub fn map_data<T: Equivalence + Copy>(&self, data: &[T], chunk_size: usize) -> Vec<T> {
        // First we need to go through the send dofs and set up the data that needs to be sent.

        let rank = self.index_layout.comm().rank() as usize;

        // Prepare the send data

        let send_data = {
            let mut tmp =
                Vec::<T>::with_capacity(self.ghost_communicator.total_send_count() * chunk_size);
            let send_buffer: &mut [T] = unsafe { std::mem::transmute(tmp.spare_capacity_mut()) };
            for (global_send_index, send_buffer_chunk) in izip!(
                self.ghost_communicator.send_indices().iter(),
                send_buffer.chunks_mut(chunk_size)
            ) {
                let local_start_index = self
                    .index_layout
                    .global2local(rank, *global_send_index)
                    .unwrap()
                    * chunk_size;
                let local_end_index = local_start_index + chunk_size;
                send_buffer_chunk.copy_from_slice(&data[local_start_index..local_end_index]);
            }
            unsafe { tmp.set_len(self.ghost_communicator.total_send_count() * chunk_size) };
            tmp
        };

        // Now get the receive data

        let receive_data = {
            let mut tmp =
                Vec::<T>::with_capacity(self.ghost_communicator.total_receive_count() * chunk_size);
            let receive_buffer: &mut [T] = unsafe { std::mem::transmute(tmp.spare_capacity_mut()) };
            self.ghost_communicator.forward_send_values_by_chunks(
                &send_data,
                receive_buffer,
                chunk_size,
            );
            unsafe { tmp.set_len(self.ghost_communicator.total_receive_count() * chunk_size) };
            tmp
        };

        // We have the receive data from the other processes. We now need to setup the output vector
        // and collect the data from what is already on the process and from the ghosts.

        let output_data = {
            let total_number_of_dofs = self.owned_dofs.len() + self.ghost_dofs.len();
            let mut output_data = Vec::<T>::with_capacity(total_number_of_dofs * chunk_size);

            let output_buffer: &mut [T] =
                unsafe { std::mem::transmute(output_data.spare_capacity_mut()) };

            // First we go through the ghost dofs and copy the corresponding data over.

            for (global_receive_index, receive_buffer_chunk) in izip!(
                self.ghost_communicator.receive_indices().iter(),
                receive_data.chunks(chunk_size)
            ) {
                let local_position = self.dof_to_position[global_receive_index];
                let local_start_index = local_position * chunk_size;
                let local_end_index = local_start_index + chunk_size;
                output_buffer[local_start_index..local_end_index]
                    .copy_from_slice(receive_buffer_chunk);
            }

            // Now do the same with the owned dofs.

            for &global_owned_dof in &self.owned_dofs {
                let local_position = self.dof_to_position[&global_owned_dof];
                let local_start_index = local_position * chunk_size;
                let local_end_index = local_start_index + chunk_size;

                let local_data_start_index = self
                    .index_layout
                    .global2local(rank, global_owned_dof)
                    .unwrap()
                    * chunk_size;
                let local_data_end_index = local_data_start_index + chunk_size;
                output_buffer[local_start_index..local_end_index]
                    .copy_from_slice(&data[local_data_start_index..local_data_end_index]);
            }

            unsafe { output_data.set_len(total_number_of_dofs * chunk_size) };
            output_data
        };

        output_data
    }

    /// Return the index layout
    pub fn index_layout(&self) -> Rc<IndexLayout<'a, C>> {
        self.index_layout.clone()
    }

    /// Return the ghost communicator
    pub fn ghost_communicator(&self) -> &crate::GhostCommunicator<usize> {
        &self.ghost_communicator
    }

    /// Return the dof to position map
    pub fn dof_to_position_map(&self) -> &HashMap<usize, usize> {
        &self.dof_to_position
    }
}
