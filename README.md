# 🎵 KANOYO-RVC 🎵

<div align="center">
  
  ![KANOYO-RVC](./docs/kanoyo-rvc-banner.png)

  [![GitHub](https://img.shields.io/badge/GitHub-Source%20Code-181717?style=for-the-badge&logo=github)](https://github.com/KANOYO-RVC/KANOYO-RVC)
  [![Discord](https://img.shields.io/badge/Discord-Join%20Server-7289DA?style=for-the-badge&logo=discord)](https://discord.gg/your-server)
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-ff9ce3?style=for-the-badge)](https://huggingface.co/spaces/your-space)
  [![Colab](https://img.shields.io/badge/Google%20Colab-Run%20Online-F9AB00?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/your-colab)

</div>

## ✨ Introduction

**KANOYO-RVC** is a powerful and user-friendly voice conversion toolkit that allows you to transform voices with exceptional quality and ease. Built with a focus on simplicity and performance, KANOYO-RVC provides a streamlined experience for both beginners and advanced users.

> 💡 This project is based on [Ilaria-RVC](https://github.com/TheStingerX/Ilaria-RVC-Mainline) with significant code restructuring and performance improvements.

## 🔥 Key Features

<div align="center">
  
| 🚀 Fast Inference | 🎙️ Quality Voice Conversion | 🤖 Built-in TTS |
| :---: | :---: | :---: |
| Optimized for speed | High-fidelity results | Multiple TTS engines |

| 🧠 Easy Training | 🔍 Model Browser | 🔧 Modular Design |
| :---: | :---: | :---: |
| Train your own models | Find and download models | Clean, maintainable code |

</div>

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/KANOYO-RVC.git

# Navigate to the project directory
cd KANOYO-RVC

# Install dependencies
pip install -r requirements.txt
```

For Windows users, we provide convenient batch files:
- `KANOYO-RVC-Launcher.bat` - Start the application
- `KANOYO-RVC-Assistant.bat` - Update and download additional resources

## 🖥️ Usage

```bash
# Start the web interface
python infer-web-new.py
```

The application has several main tabs:
- **Inference**: Convert audio with AI voice models
- **Train**: Create your own voice models
- **Extra**: Additional utilities and features
- **Misc**: Miscellaneous tools and settings

## 🏗️ Project Structure

KANOYO-RVC features a modular code structure for better maintainability:

```
KANOYO-RVC/
├── tabs/
│   ├── __init__.py      # Package initialization
│   ├── common.py        # Shared components and utilities
│   ├── inference.py     # Voice conversion functionality
│   ├── train.py         # Model training functionality
│   ├── extra.py         # Additional features
│   └── misc.py          # Miscellaneous components
├── infer-web-new.py     # Main application entry point
└── ...                  # Other files and directories
```

## 📋 Detailed Features

### Voice Conversion
- 🔊 High-quality voice transformation
- ⚡ Fast processing with GPU acceleration
- 🎛️ Advanced pitch and timbre control

### Model Training
- 🎯 Create custom voice models
- 📊 Training visualization and metrics
- ⚙️ Customizable training parameters

### Utilities
- 🔄 Model downloading and importing
- 🔍 Voice analysis tools
- 🎵 Audio preprocessing

## 🛠️ Technologies

- **PyTorch** - Deep learning framework
- **Gradio** - Web interface
- **Fairseq** - Sequence modeling toolkit
- **librosa** - Audio analysis

## 🙏 Credits

### Original Project
- **Ilaria-RVC** - The foundation this project is built upon

### Special Thanks to Ilaria Team
- **Ilaria**: Founder, Lead Developer
- **Yui**: Training feature
- **GDR-**: Inference feature
- And all other contributors to the original project

### In loving memory of JLabDX 🕊️

## 📄 License

This project is released under the license terms of the original Ilaria-RVC project.

---

<div align="center">
  
  Made with ❤️ by KANOYO
  
  <img src="https://img.shields.io/badge/Powered%20by-AI%20Voice%20Technology-9cf?style=for-the-badge" alt="Powered by AI Voice Technology">
  
</div>
