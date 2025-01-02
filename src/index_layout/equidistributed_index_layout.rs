//! Default distributed index layout
use crate::index_layout::IndexLayout;
use mpi::traits::Communicator;

/// Default index layout
pub struct EquiDistributedIndexLayout<'a, C: Communicator> {
    counts: Vec<usize>,
    comm: &'a C,
}

unsafe impl<'a, C: Communicator> Sync for EquiDistributedIndexLayout<'a, C> {}

impl<'a, C: Communicator> EquiDistributedIndexLayout<'a, C> {
    /// Crate new
    pub fn new(nchunks: usize, chunk_size: usize, comm: &'a C) -> Self {
        let nindices = nchunks * chunk_size;
        let comm_size = comm.size() as usize;

        assert!(
            comm_size > 0,
            "Group size is zero. At least one process needs to be in the group."
        );
        let mut counts = vec![0; 1 + comm_size];

        // The following code computes what index is on what rank. No MPI operation necessary.
        // Each process computes it from its own rank and the number of MPI processes in
        // the communicator

        if nchunks <= comm_size {
            // If we have fewer chunks than ranks simply
            // give chunk_size indices to each rank until filled up.
            // Then fill the rest with None.

            for (index, item) in counts.iter_mut().enumerate().take(nchunks) {
                *item = index * chunk_size;
            }

            for item in counts.iter_mut().take(comm_size).skip(nchunks) {
                *item = nindices;
            }

            counts[comm_size] = nindices;
        } else {
            // We want to equally distribute the range
            // among the ranks. Assume that we have 12
            // indices and want to distribute among 5 ranks.
            // Then each rank gets 12 / 5 = 2 indices. However,
            // we have a remainder 12 % 5 = 2. Those two indices
            // are distributed among the first two ranks. So at
            // the end we have the distribution
            // 0 -> (0, 3)
            // 1 -> (3, 6)
            // 2 -> (6, 8)
            // 3 -> (8, 10)
            // 4 -> (10, 12)

            let chunks_per_rank = nchunks / comm_size;
            let remainder = nchunks % comm_size;
            let mut count = 0;
            let mut new_count;

            for index in 0..comm_size {
                if index < remainder {
                    // Add one remainder index to the first
                    // indices.
                    new_count = count + chunks_per_rank * chunk_size + chunk_size;
                } else {
                    // When the remainder is used up just
                    // add chunk size indices to each rank.
                    new_count = count + chunks_per_rank * chunk_size;
                }
                counts[1 + index] = new_count;
                count = new_count;
            }
        }

        Self { counts, comm }
    }
}

impl<C: Communicator> IndexLayout for EquiDistributedIndexLayout<'_, C> {
    type Comm = C;

    fn counts(&self) -> &[usize] {
        &self.counts
    }

    fn comm(&self) -> &Self::Comm {
        self.comm
    }
}
