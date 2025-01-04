//? mpirun -n 3

//! This examples demonstrates the use of the ghost communicator.
//!
//! We have three processes. The first process has indices 0, 1, 2, 3, 4.
//! The second process has indices 5, 6, 7, 8, 9. The third process has
//! indices 10, 11, 12, 13, 14.
//! The first process requires indices 5 and 6 from the second process as
//! ghost indices.
//! The second process requires index 4 from the third process as ghost index.
//! The third process requires indices 0, 1, 2 from the first process as ghost.
//! We are setting chunk size to 5 so that each index corresponds to 3 data items.

use bempp_distributed_tools::GhostCommunicator;
use itertools::{izip, Itertools};
use mpi::traits::Communicator;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let chunk_size = 5;

    // This example is designed for two processes.
    assert_eq!(
        world.size(),
        3,
        "This example is designed for three MPI ranks."
    );

    // We setup the ghost communicator.

    let ghost_comm = if rank == 0 {
        GhostCommunicator::new_with_chunk_size(&[5, 6], &[1, 1], chunk_size, &world)
    } else if rank == 1 {
        GhostCommunicator::new_with_chunk_size(&[10], &[2], chunk_size, &world)
    } else {
        GhostCommunicator::new_with_chunk_size(&[5, 0, 1, 2], &[1, 0, 0, 0], chunk_size, &world)
    };

    // Let us now send some data over the ghost communicator. We repeat each index `chunk_size` times.

    let data = if rank == 0 {
        [10, 11, 12]
            .iter()
            .flat_map(|x| std::iter::repeat(x).take(chunk_size))
            .copied()
            .collect_vec()
    } else if rank == 1 {
        [13, 14, 13]
            .iter()
            .flat_map(|x| std::iter::repeat(x).take(chunk_size))
            .copied()
            .collect_vec()
    } else {
        [15].iter()
            .flat_map(|x| std::iter::repeat(x).take(chunk_size))
            .copied()
            .collect_vec()
    };

    let mut received_data = vec![0; ghost_comm.total_receive_count()];

    ghost_comm.forward_send_values(&data, &mut received_data);

    if rank == 0 {
        let expected = [13, 14];
        for (e_slice, &a) in izip!(received_data.chunks(chunk_size), expected.iter()) {
            for &elem in e_slice {
                assert_eq!(elem, a);
            }
        }
    } else if rank == 1 {
        received_data
            .iter()
            .take(chunk_size)
            .for_each(|&x| assert_eq!(x, 15));
    } else {
        let expected = [10, 11, 12, 13];
        for (e_slice, &a) in izip!(received_data.chunks(chunk_size), expected.iter()) {
            for &elem in e_slice {
                assert_eq!(elem, a);
            }
        }
    }

    // We now want to send the received data back to the original owners.

    let mut send_data = vec![0; ghost_comm.total_send_count()];

    ghost_comm.backward_send_values(&received_data, &mut send_data);

    if rank == 0 {
        let expected = [10, 11, 12];
        for (e_slice, &a) in izip!(send_data.chunks(chunk_size), expected.iter()) {
            for &elem in e_slice {
                assert_eq!(elem, a);
            }
        }
    } else if rank == 1 {
        let expected = [13, 14, 13];
        for (e_slice, &a) in izip!(send_data.chunks(chunk_size), expected.iter()) {
            for &elem in e_slice {
                assert_eq!(elem, a);
            }
        }
    } else {
        let expected = [15];
        for (e_slice, &a) in izip!(send_data.chunks(chunk_size), expected.iter()) {
            for &elem in e_slice {
                assert_eq!(elem, a);
            }
        }
    }
}
