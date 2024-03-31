# Assignment 2: Edge and Boundary Detection

## Overview

This project involves the implementation of advanced image processing techniques for edge and boundary detection using the Hough transform and Active Contour Model (Snake). The assignment covers tasks such as detecting edges, lines, circles, and ellipses in given grayscale and color images using the Canny edge detector. Additionally, it includes initializing contours for objects and evolving them using the greedy algorithm, representing the output as chain code, and computing the perimeter and area inside these contours.

## Tasks Implemented

### A) Edge and Shape Detection
1. **Edge Detection**: Utilized the Canny edge detector to identify edges in both grayscale and color images. Detected edges are superimposed on the original images for visualization.

2. **Shape Detection**: Detected lines, circles, and ellipses present in the images, if any, using the Hough transform. The detected shapes are overlaid on the original images.

### B) Active Contour Model (Snake)
1. **Contour Initialization**: Initialized contours for given objects in the images.

2. **Contour Evolution**: Utilized the greedy algorithm to evolve the Active Contour Model (Snake) for each initialized contour. The output is represented as chain code.

3. **Perimeter and Area Calculation**: Computed the perimeter and area inside the evolved contours for further analysis.

## Requirements
- Python 3.7
- OpenCV library
- NumPy library
- Matplotlib library (for visualization)
  
## Collaborators
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/OmarEmad101">
        <img src="https://github.com/OmarEmad101.png" width="100px" alt="@OmarEmad101">
        <br>
        <sub><b>Omar Emad</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Omarnbl">
        <img src="https://github.com/Omarnbl.png" width="100px" alt="@Omarnbl">
        <br>
        <sub><b>Omar Nabil</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/KhaledBadr07">
        <img src="https://github.com/KhaledBadr07.png" width="100px" alt="@KhaledBadr07">
        <br>
        <sub><b>Khaled Badr</b></sub>
      </a>
    </td>
  </tr> 
  <!-- New Row -->
  <tr>
    <td align="center">
      <a href="https://github.com/nourhan-ahmedd">
        <img src="https://github.com/nourhan-ahmedd.png" width="100px" alt="@nourhan-ahmedd">
        <br>
        <sub><b>Nourhan Ahmed </b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/hanaheshamm">
        <img src="https://github.com/hanaheshamm.png" width="100px" alt="@hanaheshamm">
        <br>
        <sub><b>Hana Hesham</b></sub>
      </a>
    </td>
  </tr>
</table>

## Submission
The submission includes a zip file containing the report, codes, results, and any additional files necessary for review by the TAs.
