# Examples

This folder contains ready-to-use scripts for nERdy+.

## Before You Start

Make sure you have:
1. Activated the conda environment: `conda activate nerdy`
2. Are in the main nERdy folder (not inside examples/)

## Available Scripts

### 1. Run Inference (`run_inference.py`)

Segment ER structures from your microscopy images.

**Basic usage:**
```bash
python examples/run_inference.py --input path/to/your/image.png
```

**More options:**
```bash
# Save to specific location
python examples/run_inference.py --input image.png --output my_result.png

# Use CPU (if GPU causes problems)
python examples/run_inference.py --input image.png --device cpu

# Use a different model
python examples/run_inference.py --input image.png --model path/to/model.pth
```

**What you'll get:**
- A segmentation mask saved as `{input_name}_segmentation.png`
- White pixels = ER structure, Black pixels = background

### 2. Train Custom Model (`train_custom.py`)

Train nERdy+ on your own annotated data.

**Basic usage:**
```bash
python examples/train_custom.py --data /path/to/your/dataset
```

**More options:**
```bash
# Full customization
python examples/train_custom.py \
    --data /path/to/dataset \
    --output my_model.pth \
    --epochs 200 \
    --batch-size 16 \
    --lr 0.0001
```

**Required dataset structure:**
```
your_dataset/
    class1/
        images/
            image1.png
            image2.png
        masks/
            image1_mask.png    # Same name + "_mask"
            image2_mask.png
```

**Tips:**
- Start with fewer epochs (50) to test if training works
- Reduce batch-size if you get "CUDA out of memory" errors
- Training takes ~30 min on GPU, ~2-3 hours on CPU

## Common Issues

**"No module named 'model'"**
→ Run from the main nERdy folder, not from inside examples/

**"FileNotFoundError"**
→ Check that your file path is correct (use full path if unsure)

**"CUDA out of memory"**
→ Add `--device cpu` or reduce batch size
