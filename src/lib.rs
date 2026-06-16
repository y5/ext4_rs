#![feature(error_in_core)]
#![no_std]
#![allow(unused)]

extern crate alloc;

pub mod prelude;
pub mod utils;

pub use prelude::*;
pub use utils::*;

mod ext4_defs;
mod ext4_impls;

// HTree directory hashing + index helpers (publicly testable).
pub use ext4_impls::htree::*;

// Journal (jbd2) engine (publicly testable).
pub use ext4_impls::journal::*;

pub mod fuse_interface;
pub mod simple_interface;

pub use fuse_interface::*;
pub use simple_interface::*;
