//! bemmpp-distributed-tools
//!
//! A collection of MPI tools in Rust.
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

mod ghost_communicator;
mod index_layout;

pub use ghost_communicator::GhostCommunicator;
pub use index_layout::{DefaultDistributedIndexLayout, IndexLayout};
