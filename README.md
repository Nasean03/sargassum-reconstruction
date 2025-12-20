# Puzzle-Based Reconstruction of Cloud-Obscured Sargassum Fields

**Author:** Nasean Belgrave  
**Course:** COMP3490, University of the West Indies  
**Date:** November 2025

## Overview

This project explores a novel approach for reconstructing cloud-obscured satellite imagery to monitor Sargassum influxes in Barbados. Using puzzle-solving-inspired algorithms, multi-date Sentinel-2 imagery is reconstructed to generate continuous, high-resolution offshore Sargassum maps. 

**Key Achievements:**
- Developed a modular puzzle-based reconstruction workflow.
- Improved continuity of Sargassum observations using multi-date Sentinel-2 imagery.
- Demonstrated cloud-obscured tile replacement with Sargassum prioritization using the Floating Algae Index (FAI).

## Methodology

1. **Data Preparation:** Multi-date Sentinel-2 images (RGB + FAI) are preprocessed.
2. **Cloud Detection:** RGB brightness, color neutrality, and blue-channel dominance identify cloudy tiles.
3. **Tile-Based Reconstruction:** Cloudy tiles are replaced with cloud-free donor tiles from other dates, prioritizing Sargassum-rich tiles.
4. **Validation:** Reconstructed images compared visually and quantitatively with cloud-free images and ground-truth observations.

## Results

- High fidelity reconstructions: ~98.9% of clouded tiles successfully replaced.
- FAI heatmaps confirm Sargassum features preserved.
- Visual comparisons show effective removal of clouds and reconstruction of continuous offshore Sargassum patterns.

![Sample Reconstruction](results/sample-reconstructions/reconstructed.png)

## Future Work

- Improve tile blending for seamless transitions.
- Extend methodology to regional/global mapping.
- Incorporate drift prediction using ocean currents.
- Adapt to other environmental monitoring tasks (turbidity, erosion, floods).

## Files

- `requirements.txt` – Python dependencies.
- `docs/report.pdf` – Full research report.
- `results/` – Sample reconstructions and heatmaps.



