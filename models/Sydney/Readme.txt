University of "Sydney" Submission README 

Project Overview:
-----------------
This repository contains the implementation of a Core loss model developed using Python. The repository is designed for the Magnet Challenge 2023 final submission.

Usage:
-------------------
To use the MMINN model, follow thess steps: 

1. Create a folder of a target material inside "./Model/Testing" and name it [Material ?].
2. Download the testing files into "./Model/Testing/Material ?", and files should include 
	2.1. "B_Field.csv": flux density waveform, N by 1024, in T
	2.2. "Frequency.csv": frequency, N by 1, in Hz
	2.3. "Temperature.csv": temperature, N by 1, in C
3. Customize the targeted material in line9 of file "Model_Inference.py" to corresping name [Material ?].
4. Run "Model_Inference" and results are stored in "./Model/Testing/Result/Volumetri_Loss_Material ?"

Additional note:
-------------------
A UI is designed to make data interpretation a visually engaging experience and help juging panel to evaluate the MMINN model's performance. A screenshot is shown in "UI_mainPage" and the developer will present the UI in the code review meeting.
