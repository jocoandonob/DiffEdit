# DiffEdit Image Editor

This is a Gradio web application that implements the DiffEdit pipeline for image editing using Stable Diffusion. The application allows users to edit images by specifying source and target prompts, which will generate a mask and apply the changes accordingly.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://127.0.0.1:7860)

3. Using the application:
   - Upload an image using the "Input Image" upload button
   - Enter the source prompt describing what's currently in the image
   - Enter the target prompt describing what you want to change it to
   - Click "Process Image" to generate the result
   - The output will show three images side by side:
     - Original image
     - Generated mask
     - Edited image

## Example

- Source prompt: "a bowl of fruits"
- Target prompt: "a basket of pears"

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- See requirements.txt for Python package dependencies 