from pathlib import Path

readme = """<div align="center">

# 🎼 Niryo One Music Conductor

### Vision-Based Robotic Imitation of Conducting Gestures

**MSc Electrical and Electronic Engineering Dissertation**  
**London South Bank University**

</div>

---

## Project Overview

This project investigates whether conducting gestures captured from ordinary recorded video can be converted into safe, repeatable robot motion on the Niryo One educational robotic arm.

The system uses MediaPipe Pose and OpenCV to detect the conductor’s shoulder, elbow and wrist positions. The wrist movement is then converted into bounded robot coordinates, stored as timestamped CSV trajectory data and replayed on both the physical Niryo One and a virtual Niryo One in MuJoCo.

The project is a low-cost educational feasibility prototype. It demonstrates how human conducting movements can be captured from video and represented through physical and simulated robot motion. It does not claim full musical understanding, exact human imitation or replacement of a human conductor.

---

## Project Videos

The following videos are hosted on LSBU OneDrive and provide supporting evidence for the main stages and final outcome of the project.

### 1. Project Source Video

[▶ View the Project Source Video on LSBU OneDrive](https://stulsbuac-my.sharepoint.com/:v:/g/personal/s3916437_lsbu_ac_uk/IQDvnds2AKYeT7bqfgx3lpkTATHF0zqcAZDep4W7J-cahNE?e=8WKsTn)

This video contains the conducting movements used as input to the project. The recorded arm movements were analysed to identify the shoulder, elbow and wrist positions before the wrist motion was converted into robot trajectory data.

---

### 2. MuJoCo Simulation Video

[▶ View the MuJoCo Simulation Video on LSBU OneDrive](https://stulsbuac-my.sharepoint.com/:v:/g/personal/s3916437_lsbu_ac_uk/IQAokmSvIuBNR6mm0_ccq2TZAVbWPeuTCs99EWk6OHehvtk?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJTcGFwcCIsInJlZmVycmFsVmlldyI6IlNoYXJlZERpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&e=FEisvr)

This video demonstrates the simulation stage of the project. Stored trajectories generated from the conducting videos are replayed on a virtual Niryo One in MuJoCo. The simulation was used to visually check that the trajectory data could produce controlled conducting-like movement before or alongside physical robot testing.

---

### 3. Compiled Result Video

[▶ View the Compiled Result Video on LSBU OneDrive](https://stulsbuac-my.sharepoint.com/:v:/r/personal/s3916437_lsbu_ac_uk/Documents/result%20compiled%20video.mp4?d=w5cc98422bc8a4155a4d66d6fa0374bf8&csf=1&web=1&e=ZB8n6p)

This video presents the overall project outcome by bringing together the source conducting movement, the extracted motion data and the resulting Niryo One response. It provides a clear visual summary of how ordinary recorded video was converted into constrained, conducting-like robot motion.

---

### 4. Full Physical Niryo One Conducting-Like Video

[▶ View the Full Physical Niryo One Conducting-Like Video on LSBU OneDrive](https://stulsbuac-my.sharepoint.com/:v:/r/personal/s3916437_lsbu_ac_uk/Documents/Full%20Physical%20Niryo%20One%20Conducting-like%20Video.mp4?d=wdef45ce9b3c14c9a98e5610ac5c567ce&csf=1&web=1&e=ajhFG1)

This video shows the physical Niryo One replaying the stored trajectory data as observable, constrained conducting-like movement. It represents the final physical validation of the project and demonstrates that the recorded gestures could be transferred from video to an educational robotic arm while remaining within the programmed movement limits.

---

## System Workflow

```mermaid
flowchart TD
    A["Recorded conducting video"] --> B["OpenCV frame processing"]
    B --> C["MediaPipe Pose landmark extraction"]
    C --> D{"Visibility above 0.15?"}
    D -- "No" --> E["Reject low-confidence sample"]
    D -- "Yes" --> F["Select wrist control point"]
    F --> G["Map movement into bounded robot coordinates"]
    G --> H["Store timestamped CSV trajectory"]
    H --> I{"Playback route"}
    I --> J["Physical Niryo One"]
    I --> K["MuJoCo simulation"]
