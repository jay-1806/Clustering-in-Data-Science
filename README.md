# Comprehensive Clustering Algorithms Collection

This repository contains a collection of Jupyter notebooks demonstrating various clustering algorithms and anomaly detection techniques. The projects range from implementing algorithms from scratch to utilizing state-of-the-art libraries and Large Language Model (LLM) embeddings for complex data types like text, images, and audio.

## 📂 Project Structure

The repository is organized into the following notebooks, covering different aspects of unsupervised learning:

### 1. Core Algorithms
- **[K-Means from Scratch](kmeans_from_scratch.ipynb)**
  - Implementation of the K-Means clustering algorithm from the ground up using NumPy.
  - Comparison with `sklearn.cluster.KMeans`.
  - Visualization of cluster formation and convergence.

- **[Hierarchical Clustering](hierarchical_clustering_impl.ipynb)**
  - Demonstration of Agglomerative Hierarchical Clustering.
  - Visualization using Dendrograms to analyze cluster relationships.

- **[Gaussian Mixture Models (GMM)](gaussian_mixture_clustering.ipynb)**
  - Probabilistic clustering using Gaussian Mixture Models.
  - Comparison of GMM with K-Means for non-circular data distributions.

### 2. Advanced Clustering & Anomaly Detection
- **[DBSCAN with PyCaret](dbscan_clustering_pycaret.ipynb)**
  - Density-Based Spatial Clustering of Applications with Noise (DBSCAN) implementation.
  - Utilizes the **PyCaret** low-code machine learning library for efficient modeling.

- **[Anomaly Detection with PyOD](anomaly_detection_pyod.ipynb)**
  - Implementation of anomaly detection techniques using the **PyOD** (Python Outlier Detection) library.
  - Covers univariate and multivariate anomaly detection use cases.

### 3. Specialized Data Clustering
- **[Time Series Clustering](timeseries_analysis_clustering.ipynb)**
  - Clustering of time-series data using pretrained models and specialized metrics.
  - Analysis of temporal patterns and trends.

- **[Document Clustering with LLMs](document_clustering_llm.ipynb)**
  - Clustering text documents using state-of-the-art LLM embeddings (e.g., Sentence Transformers, OpenAI embeddings).
  - Semantic grouping of textual data.

- **[Image Clustering](image_clustering_analysis.ipynb)**
  - Clustering image datasets using advanced embeddings (e.g., ImageBind).
  - Grouping images based on visual similarity and semantic content.

- **[Audio Clustering](audio_clustering_analysis.ipynb)**
  - Clustering audio files using deep learning embeddings.
  - Categorization of audio clips based on features extracted from models like ImageBind or audio-specific transformers.

## 🛠️ Technologies & Libraries

The projects utilize a wide range of Python libraries for data science and machine learning:

- **Core:** `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `SciPy`
- **AutoML & Specialized:** `PyCaret`, `PyOD`
- **Deep Learning & NLP:** `PyTorch`, `Sentence-Transformers`, `Hugging Face Transformers`

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/clustering-algorithms-collection.git
   cd clustering-algorithms-collection
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment. You can install the necessary packages using pip:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn pycaret pyod torch transformers sentence-transformers
   ```
   *Note: Some notebooks (like ImageBind or PyCaret) may have specific dependency requirements. Please refer to the individual notebooks for detailed installation instructions.*

3. **Run the notebooks:**
   Launch Jupyter Lab or Notebook:
   ```bash
   jupyter lab
   ```
   Open any `.ipynb` file to explore the implementation and results.

## 📝 License

This project is open-source and available under the MIT License.