//! This examples demonstrates the use of the ghost communicator.
//!
//! We have three processes. The first process has indices 0, 1, 2, 3, 4.
//! The second process has indices 5, 6, 7, 8, 9. The third process has
//! indices 10, 11, 12, 13, 14.
//! The first process requires indices 5 and 6 from the second process as
//! ghost indices.
//! The second process requires index 4 from the third process as ghost index.
//! The third process requires indices 0, 1, 2 from the first process as ghost.

use bempp_ghost::GhostCommunicator;
use mpi::traits::Communicator;

pub fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    // This example is designed for two processes.
    assert_eq!(
        world.size(),
        3,
        "This example is designed for three MPI ranks."
    );

    // We setup the ghost communicator.

    let ghost_comm = if rank == 0 {
        GhostCommunicator::new(&[5, 6], &[1, 1], &world)
    } else if rank == 1 {
        GhostCommunicator::new(&[4], &[2], &world)
    } else {
        GhostCommunicator::new(&[0, 1, 2], &[0, 0, 0], &world)
    };

    // We have now setup the ghost communicator.
    // Let us print the in-ranks and out-ranks for process 0.

    if rank == 0 {
        println!(
            "Process 1: In ranks: {:#?}, Out ranks: {:#?}",
            ghost_comm.in_ranks(),
            ghost_comm.out_ranks()
        );
    }
}
