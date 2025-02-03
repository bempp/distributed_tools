//! bemmpp-distributed-tools
//!
//! A collection of MPI tools in Rust.
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

pub mod array_tools;
pub mod data_mapper;
pub mod ghost_communicator;
pub mod index_embedding;
pub mod index_layout;
pub mod permutation;

pub use data_mapper::Global2LocalDataMapper;
pub use ghost_communicator::GhostCommunicator;
pub use index_layout::IndexLayout;
pub use permutation::DataPermutation;
