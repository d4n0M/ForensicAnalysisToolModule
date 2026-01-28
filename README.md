### Forensic Weapon Detection Library

An expert-level library for weapon and violence detection using OWLv2 (Open-Vocabulary Object Detector).
Installation

To install the library in editable mode (recommended for development):bash pip install -e.


## Usage

```python
from inference import create_detector
detector = create_detector(model_name="owlv2", device="cuda")
results = detector.detect_image(your_image_array)


### Updated Directory Tree
After adding these files, your project will look like this:

.
├── cpp
│   └── bindings.cpp          <-- NEW: C++ Source & Bindings
├── examples
│   └── example.py
├── inference
│   ├── inference.py
│   ├── __init__.py
│   └── requirements.txt      <-- (Can now be deleted)
├── __init__.py
├── pyproject.toml            <-- NEW: Packaging Metadata
├── setup.py                  <-- NEW: Build Script
└── README.md                 <-- NEW: Documentation

### Installation and Verification
Once these files are in place, you can "install" your library into your Python environment by running `pip install -e.` from the root directory.[2, 12] This command will compile the C++ code in `/cpp` into a shared object file that your Python code can import as `inference.core_cpp`.[4, 13]
