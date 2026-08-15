# Drishti Kavach Research Paper (IEEE Conference Format)

This directory contains the complete IEEE-formatted LaTeX source code for the research paper:

**Title**: *Drishti Kavach: A Unified Deep Learning and Spatial Clearance Reasoning Framework for Railway Physical Obstacle Detection Under Adverse Weather and Night-Vision Environments*

**Authors**:
1. Gowri Krishnan Nair (Dept. of CSE, MVJ College of Engineering, Bangalore)
2. Ananya Sanjiv (Dept. of CSE, MVJ College of Engineering, Bangalore)
3. Alvin Sonny (Dept. of CSE, MVJ College of Engineering, Bangalore)
4. Ganesha Thejaswi V (Dept. of CSE, MVJ College of Engineering, Bangalore)
5. Prof. Sujitha K L (Guide, Dept. of CSE, MVJ College of Engineering, Bangalore)

---

## How to Compile & View the Paper

### Option 1: Overleaf (Recommended - Zero Setup)
1. Open [Overleaf](https://www.overleaf.com/).
2. Create a **New Project** -> **Upload Project**.
3. Upload `drishti_kavach_paper.tex`.
4. Click **Recompile** to generate the PDF instantly.

### Option 2: Local LaTeX Compilation (pdflatex)
If you have MacTeX / TeXLive installed:
```bash
cd paper
pdflatex drishti_kavach_paper.tex
pdflatex drishti_kavach_paper.tex
```

---

## Paper Sections Included:
- **Abstract & Keywords** (Kavach ATP optical blindspot, RailDrishti multi-task, Active IR 850nm, Defogging, Vector clearance reasoning).
- **Section I**: Introduction (Indian Railways scale, Kavach ATP background, Physical hazard risks).
- **Section II**: Related Work (Track segmentation, Obstacle detection, Adverse weather vision).
- **Section III**: System Architecture & Mathematical Methodology:
  - Active IR (850nm NIR) Sensor Physics equations.
  - Unified Dataset Formulation & 1080p Mask Extraction.
  - Multi-Task Neural Network (*RailDrishti* YOLO11-seg) architecture and loss functions.
  - Koschmieder Atmospheric Scattering Defogger & CLAHE contrast recovery equations.
  - Vector-geometric Spatial Clearance Envelope reasoning equations.
- **Section IV**: Hardware Implementation & Edge Deployment.
- **Section V**: Experimental Evaluation (Quantitative Tables with blank placeholders for results).
- **Section VI**: Conclusion & Future Scope.
- **Acknowledgments & IEEE References**.
