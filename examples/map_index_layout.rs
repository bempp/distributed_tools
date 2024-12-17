//? mpirun -n 3

//! Map betwen two index layouts

use bempp_distributed_tools::{
    index_layout::{IndexLayout, IndexLayoutFromLocalCounts},
    EquiDistributedIndexLayout,
};
use itertools::{izip, Itertools};
use mpi::traits::Communicator;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    // Create an index layout with 10 indices on each rank.

    let layout1 = EquiDistributedIndexLayout::new(30, 1, &world);

    // Create a second layout with 5 indices on rank 0, 17 on rank 1 and 8 on rank 2.

    let counts = match world.rank() {
        0 => 5,
        1 => 17,
        2 => 8,
        _ => panic!("This example only works with three processes."),
    };

    let layout2 = IndexLayoutFromLocalCounts::new(counts, &world);

    // Now we can map between the two layouts.

    let data = if world.rank() == 0 {
        (0..10).collect_vec()
    } else if world.rank() == 1 {
        (10..20).collect_vec()
    } else {
        (20..30).collect_vec()
    };

    let mapped_data = layout1.remap(&layout2, &data);

    if world.rank() == 0 {
        assert_eq!(mapped_data.len(), 5);
        for (expected, &actual) in izip!(0..5, mapped_data.iter()) {
            assert_eq!(expected, actual);
        }
    } else if world.rank() == 1 {
        assert_eq!(mapped_data.len(), 17);
        for (expected, &actual) in izip!(5..22, mapped_data.iter()) {
            assert_eq!(expected, actual);
        }
    } else if world.rank() == 2 {
        assert_eq!(mapped_data.len(), 8);
        for (expected, &actual) in izip!(22..30, mapped_data.iter()) {
            assert_eq!(expected, actual);
        }
    }

    let remapped_data = layout2.remap(&layout1, &mapped_data);

    assert_eq!(data, remapped_data);
}
