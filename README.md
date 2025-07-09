# Genetic Programming for Computationally Efficient Land Use Allocation Optimization

## Description

This repository is based on the work by **Moritz J. Hildemann, Alan T. Murray, and Judith A. Verstegen** as part of their study:

> **Genetic Programming for Computationally Efficient Land Use Allocation Optimization**  
> 12th International Conference on Geographic Information Science (GIScience 2023).

In their work, they introduced genetic programming for evolving functions that efficiently solve large-scale land use allocation problems, providing scalable and computationally efficient alternatives to standard optimization methods.

**Original dataset and code:** [Mendeley Data](https://data.mendeley.com/datasets/4tw223jvjv/3)

---

## Credits

- **Original Authors:**
  - Moritz J. Hildemann (Institute for Geoinformatics, University of Münster, Germany)
  - Alan T. Murray (Department of Geography, University of California at Santa Barbara, USA)
  - Judith A. Verstegen (Department of Human Geography and Spatial Planning, Utrecht University, The Netherlands)

- **Original Paper:** [DOI: 10.4230/LIPIcs.GIScience.2023.4](https://doi.org/10.4230/LIPIcs.GIScience.2023.4)

- **Dataset and Supplementary Material:**
  - Hildemann, Moritz Jan (2023). *Genetic programming for computationally efficient land use allocation optimization*. Mendeley Data, V3. [https://doi.org/10.17632/4tw223jvjv.3](https://doi.org/10.17632/4tw223jvjv.3)
  - [Supplementary Illustrations (Figshare)](https://doi.org/10.6084/m9.figshare.21977228.v2)

- **License:** This project is shared under **CC-BY 4.0**, following the license of the original authors.


## Modifications

This repository contains:
- Modifications and extensions on top of the original implementation to run NSGA3 on raster data.
- Changes to the environment.yml file to include specific versions of the imports as other version combinations throw error.
- Changes to the filenames in the input data from German to English as the original name contained special characters and using it on different deskopts might create errors. 
- Added Plotly browser renderer setup:
  ```python
  import plotly.io as pio
  pio.renderers.default = "browser"

All modifications are by [Aditi Singh / @aditisinghxd], 2025.

---
