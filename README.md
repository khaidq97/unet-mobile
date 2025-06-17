# Nail Segmentation with U-Net Mobile

A nail segmentation project using U-Net architecture optimized for mobile deployment. This project includes model training, export functionality, and a TensorFlow.js testing interface.

## 📋 System Requirements

- Python 3.10
- TensorFlow
- CUDA (optional, for GPU training)

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd unet-mobile
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
├── assets/
│   ├── dataset/          # Training dataset
│   └── model/            # Trained models
├── configs/
│   ├── config.yml        # Main configuration file
│   └── config.py         # Configuration processing script
├── training/
│   ├── train.py          # Main training script
│   ├── export.py         # Model export to different formats
│   ├── inference.py      # Inference script
│   ├── losses.py         # Custom loss functions
│   ├── metrics.py        # Custom metrics
│   └── visualization.py  # Visualization utilities
├── scripts/
│   ├── train.sh          # Training script
│   └── export.sh         # Model export script
├── tools/
│   └── tfjs.html         # TensorFlow.js testing interface
├── models/               # Model architectures
├── data/                 # Data processing utilities
└── logs/                 # Training logs and checkpoints
```

## 🎯 Usage

### Training Model

1. Prepare your dataset in the `assets/dataset/` directory
2. Configure training parameters in `configs/config.yml`
3. Run training:

```bash
# Using the provided script
bash scripts/train.sh

# Or run directly
python training/train.py --config configs/config.yml
```

### Export Model

After training is complete, export the model to TensorFlow.js format:

```bash
# Using the provided script
bash scripts/export.sh

# Or run directly
python training/export.py \
  --model_path logs/unet-nano/best_model.keras \
  --output_dir logs/export/unet-nano \
  --type tfjs
```

### Testing with TensorFlow.js

1. Open `tools/tfjs.html` in your web browser
2. Load the exported model
3. Upload images to test segmentation

### Inference

```bash
python training/inference.py \
  --model_path assets/model/your_model.keras \
  --input_path path/to/your/image.jpg \
  --output_path path/to/output/
```

## 📊 Dataset

The dataset is stored in the `assets/dataset/` directory. Ensure your dataset follows this structure:

```
assets/dataset/
├── images/          # Original images
├── masks/           # Mask annotations
```

## 🔧 Configuration

Edit the `configs/config.yml` file to customize:

- Model architecture
- Hyperparameters
- Dataset paths
- Training settings

## 📈 Monitoring

- Training logs are saved in the `logs/` directory
- Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir logs/
```

## 🎮 Demo

Open `tools/tfjs.html` in your browser to test the model in real-time using webcam or by uploading images.

## 📝 Dependencies

- TensorFlow
- OpenCV
- scikit-image
- scikit-learn
- matplotlib
- tqdm

## 🏗 Model Architecture

The project uses a mobile-optimized U-Net architecture designed for efficient nail segmentation on resource-constrained devices.

## 🚀 Quick Start

1. **Setup Environment:**
```bash
pip install -r requirements.txt
```

2. **Prepare Data:**
   - Place your images in `assets/dataset/images/`
   - Place corresponding masks in `assets/dataset/masks/`

3. **Train Model:**
```bash
bash scripts/train.sh
```

4. **Export for Web:**
```bash
bash scripts/export.sh
```

5. **Test in Browser:**
   - Open `tools/tfjs.html`
   - Load your exported model
   - Start testing!

## 📊 Performance

The model is optimized for:
- Mobile deployment
- Real-time inference
- Low memory footprint
- High accuracy nail segmentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact

- Email: khaidoan.dk97@gmail.com
- Project Link: [https://github.com/khaidq97/unet-mobile](https://github.com/khaidq97/unet-mobile)

## 🙏 Acknowledgments

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- TensorFlow Team
- OpenCV Community
- Mobile ML Community

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{nail-segmentation-unet,
  title={Nail Segmentation with Mobile U-Net},
  author={khaidq},
  year={2025},
  url={https://github.com/khaidq97/unet-mobile}
}
``` 