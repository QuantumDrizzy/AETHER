//! # AETHER Core
//!
//! Core domain types and logic for the AETHER materials research platform.
//! Provides the universal material model, experiment framework, compatibility
//! engine, knowledge graph, and configuration management.

pub mod material;
pub mod experiment;
pub mod compatibility;
pub mod knowledge;
pub mod config;

pub use material::*;
pub use experiment::*;
pub use compatibility::*;
pub use knowledge::*;
pub use config::*;
