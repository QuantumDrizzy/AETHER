//! Knowledge graph entries — correlations, patterns, and insights
//! discovered during AETHER research sessions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The kind of knowledge captured.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KnowledgeEntryType {
    Correlation,
    Pattern,
    Anomaly,
    Insight,
    Prediction,
}

impl std::fmt::Display for KnowledgeEntryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A single entry in the AETHER knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    /// Unique identifier.
    pub id: Uuid,
    /// Classification of the entry.
    pub entry_type: KnowledgeEntryType,
    /// Short title.
    pub title: String,
    /// Detailed description / explanation.
    pub description: String,
    /// Supporting evidence (free-text references).
    pub evidence: Vec<String>,
    /// Confidence in this entry [0, 1].
    pub confidence: f64,
    /// Searchable tags.
    pub tags: Vec<String>,
    /// Experiment IDs that contributed to this entry.
    pub source_experiments: Vec<Uuid>,
    /// When the entry was first created.
    pub created_at: DateTime<Utc>,
    /// When the entry was last updated.
    pub updated_at: DateTime<Utc>,
}

impl KnowledgeEntry {
    /// Create a new knowledge entry.
    pub fn new(
        entry_type: KnowledgeEntryType,
        title: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            entry_type,
            title: title.into(),
            description: description.into(),
            evidence: Vec::new(),
            confidence: 0.5,
            tags: Vec::new(),
            source_experiments: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a piece of evidence.
    pub fn with_evidence(mut self, ev: impl Into<String>) -> Self {
        self.evidence.push(ev.into());
        self
    }

    /// Set the confidence.
    pub fn with_confidence(mut self, c: f64) -> Self {
        self.confidence = c;
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Link a source experiment.
    pub fn with_experiment(mut self, id: Uuid) -> Self {
        self.source_experiments.push(id);
        self
    }
}
