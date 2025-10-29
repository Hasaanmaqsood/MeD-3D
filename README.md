# MeD-3D: Multimodal Fusion Framework for ccRCC Recurrence Prediction


## 📖 Overview

MeD-3D is a comprehensive multimodal deep learning framework for predicting cancer recurrence in clear cell Renal Cell Carcinoma (ccRCC) patients. This implementation integrates electronic health records (EHR), whole slide images (WSI), and CT/MRI scans using both early and late fusion strategies.

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU A100 80GB



## Data Acquisition

### TCGA-KIRC Dataset
- **Source:** [The Cancer Imaging Archive - TCGA-KIRC](https://www.cancerimagingarchive.net/collection/tcga-kirc/)
- **Download Method:** NBIA Data Retriever or TCIA-utils Python package  
- **Modalities:** CT, MRI, Whole Slide Images, Clinical Data  

### CPTAC-CCRCC Dataset
- **Source:** [The Cancer Imaging Archive - CPTAC-CCRCC](https://www.cancerimagingarchive.net/collection/cptac-ccrcc/)
- **Download Method:** NBIA Data Retriever or TCIA-utils Python package  
- **Modalities:** CT, MRI, Whole Slide Images, Clinical Data  


### Installation
```bash
# Clone repository
git clone https://github.com/your-username/MeD-3D.git
cd MeD-3D

# Create conda environment
conda env create -f environment.yml
conda activate med3d

# Install dependencies
pip install -r requirements.txt
```

## Development
## 🔬 WSI Processing with CLAM

### Step 1: CLAM Setup
```bash
# Clone and install CLAM
git clone https://github.com/mahmoodlab/CLAM.git
cd CLAM
pip install -e .
cd ..

```

### 1.2 Tissue Segmentation and Patching

```bash
python CLAM/create_patches_fp.py \
    --source data/raw/wsi/ \
    --save_dir data/processed/clam_patches/ \
    --patch_size 256 \
    --seg --patch --stitch \
    --preset tcga.csv

```

### 1.3 Feature Extraction

```bash
CUDA_VISIBLE_DEVICES=0 python CLAM/extract_features_fp.py \
    --data_h5_dir data/processed/clam_patches/patches/ \
    --data_slide_dir data/raw/wsi/ \
    --csv_path data/processed/clam_patches/process_list_autogen.csv \
    --feat_dir data/features/wsi_features/ \
    --batch_size 512 \
    --slide_ext .svs

```


## Step 2: CT/MRI Data Processing

### 2.1 Radiology Data Download
All CT and MRI data were downloaded programmatically using the TCIA-utils Python package. The complete data acquisition process is implemented in `Dataset_downloading_TCGA.ipynb`.

**Key Steps:**
- Query TCGA-KIRC and CPTAC-CCRCC collections for CT and MR series
- Download complete imaging studies using NBIA API
- Save comprehensive metadata including SeriesInstanceUID and Modality
- Organize data by collection and patient ID


### 2.2 MedicalNet Feature Extraction
3D feature extraction was performed using MedicalNet's pre-trained 3D-ResNet18 model.

**Processing Pipeline:**
- Spatial resampling to uniform dimensions: 448 × 448 × 56
- Modality-specific intensity normalization
- Feature extraction using 3D-ResNet18
- Generation of 512-dimensional feature vectors per scan
- Patient-level feature aggregation

**Output:** `radiology_features.csv` containing 512D feature vectors for each patient


## Step 3: EHR Data Processing

### 3.1 Data Preprocessing and Feature Engineering
Structured clinical data underwent comprehensive preprocessing to ensure data quality and model readiness. The pipeline included:

**Data Cleaning:**
- Missing value imputation using median for numerical features and mode for categorical variables
- Handling of special placeholder values (-1) in tumor staging criteria
- Ordinal encoding of AJCC staging variables to preserve clinical hierarchy
- One-hot encoding of demographic features
- MinMax scaling of all numerical features to [0,1] range

**Class Imbalance Mitigation:**
- Implementation of both SMOTE and ADASYN oversampling techniques
- Comparative evaluation of both methods on model performance
- Stratified train-test split to maintain class distribution

### 3.2 MLP Model Training and Feature Extraction
A Multilayer Perceptron (MLP) classifier was trained for recurrence prediction with the following configuration:

**Model Architecture:**
- Multiple fully-connected layers with batch normalization
- ReLU activation functions and dropout regularization
- Adam optimizer with L2 regularization (weight decay: 1e-4)
- ReduceLROnPlateau scheduler for adaptive learning rate adjustment

**Feature Extraction:**
- 64-dimensional embeddings extracted from the final hidden layer (fc3)
- Patient-level feature vectors preserving original identifiers
- Comprehensive model checkpointing and performance monitoring

**Output:** `ehr_fc3_embeddings.csv` containing 64-dimensional feature vectors for multimodal fusion

## Step 4: Multimodal Fusion

### 4.1 Feature Integration
Combined pre-computed feature vectors from all three modalities:
- **EHR**: 64-dimensional clinical embeddings
- **WSI**: 1024-dimensional histopathology features  
- **CT/MRI**: 512-dimensional radiology embeddings

Created unified patient-level dataset for fusion experiments.

### 4.2 Fusion Strategies
Implemented and compared two fusion approaches:

**Early Fusion**
- Direct feature-level integration
- Methods: Concatenation and Mean Pooling
- Enables cross-modal feature learning

**Late Fusion**
- Decision-level integration  
- Methods: Weighted Sum and Learned Weights
- Leverages modality-specific predictions

Both strategies enhance predictive performance through complementary data integration.

## Project Structure


## 🙌 Acknowledgments

We thank:

- [**Mahmood Lab, Harvard**](https://github.com/mahmoodlab/CLAM) for the CLAM pipeline  
- [**Tencent AI Lab**](https://github.com/Tencent/MedicalNet) for pretrained 3D medical CNNs  
- [**The Cancer Imaging Archive (TCIA)**](https://www.cancerimagingarchive.net/) for data access  
