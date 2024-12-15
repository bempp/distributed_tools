//! Definition of Index Layouts.
//!
//! An [IndexLayout] specified how degrees of freedom are distributed among processes.
//! We always assume that a process has a contiguous set of degrees of freedom.

/// The Index Layout trait. It fully specifies how degrees of freedom are distributed
/// among processes. Each process must hold a contiguous number of degrees of freedom (dofs).
/// However, it is possible that a process holds no dof at all. Local indices are specified by
/// index ranges of the type [first, last). The index `first` is contained on the process. The
/// index `last` is not contained on the process. If `first == last` then there is no index on
/// the local process.
mod equidistributed_index_layout;
mod index_layout_from_local_counts;
pub use equidistributed_index_layout::EquiDistributedIndexLayout;
pub use index_layout_from_local_counts::IndexLayoutFromLocalCounts;
use itertools::Itertools;
use mpi::traits::{Communicator, Equivalence};

use crate::array_tools::redistribute;

// An index layout specifying index ranges on each rank.
//
/// This index layout assumes a contiguous set of indices
/// starting with the first n0 indices on rank 0, the next n1 indices on rank 1, etc.
pub trait IndexLayout {
    /// MPI Communicator;
    type Comm: mpi::topology::Communicator;

    /// The cumulative sum of indices over the ranks.
    ///
    /// The number of indices on rank is is counts[1 + i] - counts[i].
    /// The last entry is the total number of indices.
    fn counts(&self) -> &[usize];

    /// The local index range. If there is no local index
    /// the left and right bound are identical.
    fn local_range(&self) -> (usize, usize) {
        let counts = self.counts();
        (
            counts[self.comm().rank() as usize],
            counts[1 + self.comm().rank() as usize],
        )
    }

    /// The number of global indices.
    fn number_of_global_indices(&self) -> usize {
        *self.counts().last().unwrap()
    }

    /// The number of local indicies, that is the amount of indicies
    /// on my process.
    fn number_of_local_indices(&self) -> usize {
        let counts = self.counts();
        counts[1 + self.comm().rank() as usize] - counts[self.comm().rank() as usize]
    }

    /// Index range on a given process.
    fn index_range(&self, rank: usize) -> Option<(usize, usize)> {
        let counts = self.counts();
        if rank < self.comm().size() as usize {
            Some((counts[rank], counts[1 + rank]))
        } else {
            None
        }
    }

    /// Convert continuous (0, n) indices to actual indices.
    ///
    /// Assume that the local range is (30, 40). Then this method
    /// will map (0,10) -> (30, 40).
    /// It returns ```None``` if ```index``` is out of bounds.
    fn local2global(&self, index: usize) -> Option<usize> {
        let rank = self.comm().rank() as usize;
        if index < self.number_of_local_indices() {
            Some(self.counts()[rank] + index)
        } else {
            None
        }
    }

    /// Convert global index to local index on a given rank.
    /// Returns ```None``` if index does not exist on rank.
    fn global2local(&self, rank: usize, index: usize) -> Option<usize> {
        if let Some(index_range) = self.index_range(rank) {
            if index >= index_range.1 {
                return None;
            }

            Some(index - index_range.0)
        } else {
            None
        }
    }

    /// Get the rank of a given index.
    fn rank_from_index(&self, index: usize) -> Option<usize> {
        for (count_index, &count) in self.counts()[1..].iter().enumerate() {
            if index < count {
                return Some(count_index);
            }
        }
        None
    }

    /// Remap indices from one layout to another.
    fn remap<L: IndexLayout, T: Equivalence>(&self, other: &L, data: &[T]) -> Vec<T> {
        assert_eq!(data.len(), self.number_of_local_indices());
        assert_eq!(
            self.number_of_global_indices(),
            other.number_of_global_indices()
        );

        let my_range = self.local_range();

        let other_bins = (0..other.comm().size() as usize)
            .map(|rank| other.index_range(rank).unwrap().0)
            .collect_vec();

        let sorted_keys = (my_range.0..my_range.1).collect_vec();

        let counts = crate::array_tools::sort_to_bins(&sorted_keys, &other_bins)
            .iter()
            .map(|&key| key as i32)
            .collect_vec();

        redistribute(data, &counts, other.comm())
    }

    /// Return the communicator.
    fn comm(&self) -> &Self::Comm;
}
