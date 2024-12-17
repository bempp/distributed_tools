//! Default distributed index layout
use crate::index_layout::IndexLayout;
use mpi::traits::{Communicator, CommunicatorCollectives};

/// Specify an index layout from local variable counts
pub struct IndexLayoutFromLocalCounts<'a, C: Communicator> {
    counts: Vec<usize>,
    comm: &'a C,
}

impl<'a, C: Communicator + CommunicatorCollectives> IndexLayoutFromLocalCounts<'a, C> {
    /// Crate new
    pub fn new(local_count: usize, comm: &'a C) -> Self {
        let size = comm.size() as usize;
        let mut counts = vec![0; size + 1];
        comm.all_gather_into(&local_count, &mut counts[1..]);
        for i in 1..=size {
            counts[i] += counts[i - 1];
        }
        Self { counts, comm }
    }
}

impl<C: Communicator> IndexLayout for IndexLayoutFromLocalCounts<'_, C> {
    type Comm = C;

    fn counts(&self) -> &[usize] {
        &self.counts
    }

    fn comm(&self) -> &Self::Comm {
        self.comm
    }
}
