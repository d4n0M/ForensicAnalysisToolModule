### Forensic Weapon Detection Library

An expert-level library for weapon and violence detection using OWLv2 (Open-Vocabulary Object Detector).
Installation

To install the library in editable mode (recommended for development):bash pip install -e.


## Usage

```python
from inference import create_detector
detector = create_detector(model_name="owlv2", device="cuda")
results = detector.detect_image(your_image_array)
```
