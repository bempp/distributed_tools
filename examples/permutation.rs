//! Example for the use of global permutations.

use std::rc::Rc;

use bempp_distributed_tools::permutation::DataPermutation;
use bempp_distributed_tools::IndexLayout;
use itertools::{izip, Itertools};
use rand::prelude::*;
use rand::seq::SliceRandom;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);

    let n = 1537;

    let chunk_size = 3;

    // We setup the index layout.

    // We now create a permutation of the dofs.
    let index_layout = Rc::new(IndexLayout::from_equidistributed_chunks(n, 1, &world));
    let mut custom_global_layout = (0..n).collect_vec();
    custom_global_layout.shuffle(&mut rng);

    let local_bounds = index_layout.local_range();

    let custom_indices = &custom_global_layout[local_bounds.0..local_bounds.1];

    let permutation = DataPermutation::new(index_layout.clone(), custom_indices);

    // We now want to send some data over.

    let data = (local_bounds.0..local_bounds.1)
        .flat_map(|elem| std::iter::repeat(elem).take(3))
        .collect_vec();

    let mut permuted_forward_data = vec![0; chunk_size * custom_indices.len()];
    let mut permuted_backward_data = vec![0; chunk_size * index_layout.number_of_local_indices()];

    permutation.forward_permute(&data, &mut permuted_forward_data, chunk_size);

    permutation.backward_permute(
        &permuted_forward_data,
        &mut permuted_backward_data,
        chunk_size,
    );

    for (&actual, &expected) in izip!(permuted_backward_data.iter(), data.iter()) {
        assert_eq!(actual, expected);
    }
}
