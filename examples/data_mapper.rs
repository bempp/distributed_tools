//! Example for using the data mapper.

use std::rc::Rc;

use bempp_distributed_tools::Global2LocalDataMapper;
use itertools::Itertools;
use mpi::traits::Communicator;

fn main() {
    let universe = mpi::initialize().unwrap();

    let world = universe.world();

    assert_eq!(world.size(), 2, "This example must be run using two ranks.");

    let n_indices = 5;

    let index_layout = Rc::new(bempp_distributed_tools::IndexLayout::from_local_counts(
        n_indices, &world,
    ));

    let required_indices = if world.rank() == 0 {
        vec![0, 1, 2, 3, 6, 9, 5, 2, 1]
    } else {
        vec![0, 1, 2, 0, 6, 9, 2, 0]
    };

    let data_mapper = Global2LocalDataMapper::new(index_layout.clone(), &required_indices);

    let in_vec = (0..n_indices)
        .map(|index| world.rank() as usize * n_indices + index)
        .collect_vec();

    let out_vec = data_mapper.map_data(&in_vec, 1);

    assert_eq!(out_vec, required_indices);
}
