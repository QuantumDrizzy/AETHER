pub trait SensorDriver: Send + Sync {
    fn read(&mut self) -> Result<SensorReading, anyhow::Error>;
}

#[derive(Debug, Clone)]
pub struct SensorReading {
    pub channel: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub values: Vec<f64>,
    pub metadata: std::collections::HashMap<String, String>,
}

pub struct RingBuffer<T> {
    buffer: Vec<T>,
    capacity: usize,
    head: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            head: 0,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(item);
        } else {
            self.buffer[self.head] = item;
            self.head = (self.head + 1) % self.capacity;
        }
    }
}

pub struct AcquisitionManager {
    // Stub
}

impl AcquisitionManager {
    pub fn new() -> Self {
        Self {}
    }

    pub fn start(&self) {
        // Stub
    }
}
