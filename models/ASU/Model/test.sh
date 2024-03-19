#!/usr/bin/bash
# Use python installed or install python. Assume it could be called by python command
# 
# remove after installation
pip install -r requirement.txt 

# cd Model #change directory to Model sub directory(references to folders is made from there)

python Model_inference.py "Material A"
python Model_inference.py "Material B"
python Model_inference.py "Material C"
python Model_inference.py "Material D"
python Model_inference.py "Material E"