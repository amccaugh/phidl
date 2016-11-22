# device-layout
CAD layout and geometry creation utilities for photonic and superconducting circuits

# Changelog
## 0.5.3
### New features
 - Layers() are now implemented.  See tutorial_example.py "Using Layers" section for a demonstration
 - You can now construct a Device using a set of parameters.  See "Constructing a Device from set of parameters" in tutorial_example.py
 - Usage of the annotate() function has been added to tutorial_example.py
 - quickplot rendering speed has been sped up by a factor of 10x
### Changes
- pg.rectangle() now takes "size" as a parameter rather than "point1" and "point2"
