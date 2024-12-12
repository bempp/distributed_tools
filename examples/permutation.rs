//! Example for the use of global permutations.

use bempp_distributed_tools::permutation::DataPermutation;
use bempp_distributed_tools::{DefaultDistributedIndexLayout, IndexLayout};
use itertools::Itertools;
use mpi::traits::Communicator;
use rand::prelude::*;
use rand::seq::SliceRandom;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);

    let rank = world.rank();

    let n = 13;

    // We setup the index layout.

    // We now create a permutation of the dofs.
    let index_layout = DefaultDistributedIndexLayout::new(n, 1, &world);
    let mut custom_global_layout = (0..n).collect_vec();
    custom_global_layout.shuffle(&mut rng);

    let local_bounds = index_layout.local_range();

    let custom_indices = &custom_global_layout[local_bounds.0..local_bounds.1];

    let permutation = DataPermutation::new(&index_layout, custom_indices, &world);

    // We now want to send some data over.

    let data = (local_bounds.0..local_bounds.1).collect_vec();

    let mut permuted_data = vec![0; custom_indices.len()];

    permutation.forward_permute(&data, &mut permuted_data);

    if rank == 0 {
        println!("Custom indices: {:#?}", custom_indices);
        println!("Permuted data: {:#?}", permuted_data);
    }
}
