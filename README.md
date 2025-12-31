# AI Generated Image Detection

<b>Demo website:</b> https://huggingface.co/spaces/mysertkaya/AI-Generated-Image-Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Unsloth](https://img.shields.io/badge/Unsloth-Fast_Finetuning-green)

This repository contains a deep learning project designed to detect whether an image is **Real** or **AI-Generated (Fake)**. It utilizes the **Qwen2.5-VL-7B-Instruct** Vision-Language Model (VLM), fine-tuned using **LoRA** (Low-Rank Adaptation) via the **Unsloth** library for efficiency.

The project explores two different training methodologies to improve robustness against social media compression and hard-to-classify examples.

## 🚀 Features

*   **State-of-the-Art VLM:** Uses Qwen2.5-VL-7B, a powerful multimodal model.
*   **Efficient Fine-Tuning:** Implements 4-bit quantization and LoRA adapters using Unsloth.
*   **Robust Training Pipeline (Method 2):**
    *   **Data Augmentation:** Random resizing, cropping, and color jittering.
    *   **Compression Simulation:** Applies random JPEG compression (Quality 50-95) during training to mimic real-world social media artifacts.
    *   **Focal Loss:** Used to focus the model on hard-to-classify examples (Alpha: 0.75, Gamma: 2.0).
*   **Interactive Inference:** Includes a **Gradio** web interface for easy testing.

## 📂 Repository Structure

*   `first_method_training_code.ipynb`: First Method training pipeline using standard SFT (Supervised Fine-Tuning) with CrossEntropy Loss.
*   `second_method_training_code.ipynb`: Second Method training pipeline incorporating JPEG compression simulation, heavy augmentation, and Focal Loss.
*   `inference_code_to_run.ipynb`: Inference script that loads the trained adapters and launches a Gradio web UI.

## 🛠️ Installation

To run the notebooks, you need to install the required dependencies. It is recommended to use a GPU environment (e.g., Google Colab or a local machine with CUDA).

```bash
pip install unsloth
pip install torch torchvision pillow gradio
pip install transformers --upgrade
pip install --no-deps trl
```
## 📊 Dataset

The model is trained on the **SID_Set (Synthetic Image Detection)** dataset sourced from Hugging Face  
(**saberzl/SID_Set**).

### Preprocessing
- Images with `label = 2` are filtered out.
- Images are filtered to ensure **high quality** (e.g., minimum width of 1024 pixels).
- **Class Balancing:** The dataset is undersampled to ensure an equal number of **Real** and **Fake** examples.

---

## 🧠 Training Methodologies

### Method 1: The High-Quality Specialist
- **Model:** Qwen2.5-VL-7B-Instruct (4-bit)
- **Loss:** Standard Cross-Entropy
- **Scheduler:** Linear
- **Epochs:** 1
- **Focus:** Establishing a good performance on balanced data

---

### Method 2: The Low-Quality Specialist
- **Data Augmentation:**
  - `RandomHorizontalFlip`
  - `ColorJitter`
  - `RandomResizedCrop` (target size: 512×512)
- **Compression Simulation:**
  - Images are compressed and decompressed in memory using **JPEG quality 50–95** to simulate real-world quality degradation.
- **Loss:** Focal Loss (penalizes hard misclassifications more than easy ones)
- **Scheduler:** Cosine Annealing
- **Epochs:** 2
- **Result:** Improved robustness against artifacts commonly found in compressed images  
  (e.g., images shared via Twitter or WhatsApp)

  
## 📈 Results
- **Training Accuracy:** Reached ~99% on the validation set for Method 1.
- **Evaluation:** Tested on external datasets (e.g., Dalle3 generated images) and standard validation splits.
## 🤝 Acknowledgements
Unsloth AI for the optimized fine-tuning library.
Qwen Team for the Qwen2.5-VL model.
saberzl for the SID Dataset.
