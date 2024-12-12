//! Array tools
//!
//! This module contains tools for working with distributed arrays.

use itertools::{izip, Itertools};
use mpi::{
    datatype::{Partition, PartitionMut},
    traits::{CommunicatorCollectives, Equivalence},
};

///
/// Distribute a sorted sequence into bins.
///
/// For an array with n elements to be distributed into p bins,
/// the array `bins` has p elements. The bins are defined by half-open intervals
/// of the form [b_j, b_{j+1})). The final bin is the half-open interval [b_{p-1}, \infty).
/// It is assumed that the bins and the elements are both sorted sequences and that
/// every element has an associated bin.
/// The function returns a p element array with the counts of how many elements go to each bin.
/// Since the sequence is sorted this fully defines what element goes into which bin.
pub fn sort_to_bins<T: Ord>(sorted_keys: &[T], bins: &[T]) -> Vec<usize> {
    let nbins = bins.len();

    // Deal with the special case that there is only one bin.
    // This means that all elements are in the one bin.
    if nbins == 1 {
        return vec![sorted_keys.len(); 1];
    }

    let mut bin_counts = vec![0; nbins];

    // This iterates over each possible bin and returns also the associated rank.
    // The last bin position is not iterated over since for an array with p elements
    // there are p-1 tuple windows.
    let mut bin_iter = izip!(
        bin_counts.iter_mut(),
        bins.iter().tuple_windows::<(&T, &T)>(),
    );

    // We take the first element of the bin iterator. There will always be at least one since
    // there are at least two bins (an actual one, and the last half infinite one)
    let mut r: &mut usize;
    let mut bin_start: &T;
    let mut bin_end: &T;
    (r, (bin_start, bin_end)) = bin_iter.next().unwrap();

    let mut count = 0;
    'outer: for key in sorted_keys.iter() {
        if bin_start <= key && key < bin_end {
            *r += 1;
            count += 1;
        } else {
            // Move the bin forward until it fits. There will always be a fitting bin.
            loop {
                if let Some((rn, (bsn, ben))) = bin_iter.next() {
                    if bsn <= key && key < ben {
                        // We have found the next fitting bin for our current element.
                        // Can register it and go back to the outer for loop.
                        *rn += 1;
                        r = rn;
                        bin_start = bsn;
                        bin_end = ben;
                        count += 1;
                        break;
                    }
                } else {
                    // We have no more fitting bin. So break the outer loop.
                    break 'outer;
                }
            }
        }
    }

    // We now have everything but the last bin. Just bunch the remaining elements to
    // the last count.
    *bin_counts.last_mut().unwrap() = sorted_keys.len() - count;

    bin_counts
}

/// Redistribute an array via an all_to_all_varcount operation.
pub fn redistribute<T: Equivalence, C: CommunicatorCollectives>(
    arr: &[T],
    counts: &[i32],
    comm: &C,
) -> Vec<T> {
    assert_eq!(counts.len(), comm.size() as usize);

    // First send the counts around via an alltoall operation.

    let mut recv_counts = vec![0; counts.len()];

    comm.all_to_all_into(counts, &mut recv_counts);

    // We have the recv_counts. Allocate space and setup the partitions.

    let nelems = recv_counts.iter().sum::<i32>() as usize;

    let mut output = Vec::<T>::with_capacity(nelems);
    let out_buf: &mut [T] = unsafe { std::mem::transmute(output.spare_capacity_mut()) };

    let send_partition = Partition::new(arr, counts, displacements(counts));
    let mut recv_partition =
        PartitionMut::new(out_buf, &recv_counts[..], displacements(&recv_counts));

    comm.all_to_all_varcount_into(&send_partition, &mut recv_partition);

    unsafe { output.set_len(nelems) };

    output
}

/// Compute displacements from a vector of counts.
///
/// This is useful for global MPI varcount operations. Let
/// count [ 3, 4, 5]. Then the corresponding displacements are
// [0, 3, 7]. Note that the last element `5` is ignored.
pub fn displacements(counts: &[i32]) -> Vec<i32> {
    counts
        .iter()
        .scan(0, |acc, &x| {
            let tmp = *acc;
            *acc += x;
            Some(tmp)
        })
        .collect()
}
