# Video Face-Crop/Blur Processing

This script processes **all `.mp4` videos** inside a root input folder (including nested subfolders), detects the largest face in each frame, and creates a processed output video.

It preserves the **same folder structure** in the output directory.

---

## What the script does

For each video frame:

- If a face is detected:
  - selects the **largest face**
  - crops around the face (with fixed-size logic)
  - zooms the crop
  - places it in the center of a black canvas (same size as original frame)

- If no face is detected:
  - applies a **strong blur**
  - darkens the frame
  - writes that frame to output

---



The script saves processed videos into a separate root folder (default: `blur_videos`) while preserving the same subfolder layout.

### Example

Input:
study_videos/agreeableness/high/4T9UUWWRE94.000.mp4

Output:
blur_videos/agreeableness/high/4T9UUWWRE94.000.mp4

This makes it easy to keep the processed dataset organized exactly like the original.

---
**Download/save the Python script**
   - Download the script `blur.py`
   - Put `blur.py` in the **same parent folder** as `study_videos/`

## Requirements

```bash
pip install opencv-python numpy
```
## Requirements
```bash
python blur.py
```
