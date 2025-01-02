//! Default distributed index layout
use crate::index_layout::IndexLayout;
use mpi::traits::Communicator;

/// Specify an index layout that has dofs only on a single process.
pub struct SingleProcessIndexLayout<'a, C: Communicator> {
    counts: Vec<usize>,
    comm: &'a C,
}

unsafe impl<'a, C: Communicator> Sync for SingleProcessIndexLayout<'a, C> {}

impl<'a, C: Communicator> SingleProcessIndexLayout<'a, C> {
    /// Create new single process index layout that lives on `root` with `ndofs` degrees of freedom.
    pub fn new(root: usize, ndofs: usize, comm: &'a C) -> Self {
        let size = comm.size() as usize;
        assert!(root < size as usize);
        let mut counts = vec![0; comm.size() as usize + 1];
        counts[root] = ndofs;
        for i in 1..=size {
            counts[i] += counts[i - 1];
        }
        Self { counts, comm }
    }
}

impl<C: Communicator> IndexLayout for SingleProcessIndexLayout<'_, C> {
    type Comm = C;

    fn counts(&self) -> &[usize] {
        &self.counts
    }

    fn comm(&self) -> &Self::Comm {
        &self.comm
    }
}
