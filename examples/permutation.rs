//! Example for the use of global permutations.

use bempp_distributed_tools::permutation::DataPermutation;
use bempp_distributed_tools::{EquiDistributedIndexLayout, IndexLayout};
use itertools::{izip, Itertools};
use rand::prelude::*;
use rand::seq::SliceRandom;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);

    let n = 1537;

    // We setup the index layout.

    // We now create a permutation of the dofs.
    let index_layout = EquiDistributedIndexLayout::new(n, 1, &world);
    let mut custom_global_layout = (0..n).collect_vec();
    custom_global_layout.shuffle(&mut rng);

    let local_bounds = index_layout.local_range();

    let custom_indices = &custom_global_layout[local_bounds.0..local_bounds.1];

    let permutation = DataPermutation::new(&index_layout, custom_indices, &world);

    // We now want to send some data over.

    let data = (local_bounds.0..local_bounds.1).collect_vec();

    let mut permuted_forward_data = vec![0; custom_indices.len()];
    let mut permuted_backward_data = vec![0; index_layout.number_of_local_indices()];

    permutation.forward_permute(&data, &mut permuted_forward_data);

    permutation.backward_permute(&permuted_forward_data, &mut permuted_backward_data);

    for (&actual, expected) in izip!(
        permuted_backward_data.iter(),
        local_bounds.0..local_bounds.1
    ) {
        assert_eq!(actual, expected);
    }
}
