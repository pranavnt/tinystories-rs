use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};
use derive_new::new;
use serde::{Serialize, Deserialize};

// Derive Deserialize here to satisfy the DeserializeOwned bound
#[derive(new, Clone, Debug, Serialize, Deserialize)]
pub struct TextGenerationItem {
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TinystoriesDbItem {
    pub content: String,
}

pub struct TinystoriesDataset {
    dataset: SqliteDataset<TextGenerationItem>,
}

impl Dataset<TextGenerationItem> for TinystoriesDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        // Directly get the TextGenerationItem as it's now deserializable
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl TinystoriesDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("train")
    }

    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<TextGenerationItem> = HuggingfaceDatasetLoader::new("roneneldan/TinyStories")
            .dataset(split)
            .unwrap();
        Self { dataset }
    }
}
