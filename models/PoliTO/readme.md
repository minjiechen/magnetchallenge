## PoliTO Proposal

### Basic Description
The file hierarchy is the same specified in the template
The main file **"Model_inference.py"** is able to produce an output file for whatever material is specified in the global variable **"MATERIAL"**
We also included one file for each material that is able to run on his own, using the same file hierarchy as "Model_inference.py"

### Requirements
All the requirements needed to run all the files are specified in **"requirements.txt"**

### Output
The programs create a file named "Volumetric_Loss_Material X.csv" directly inside the folder "Testing/Material X" (where X is the material letter) 
which is supposed to contain also the input files (such as "B_Field.csv").
In the "stdout" there will be some warnings of the libraries **"tensorflow"** and **"pandas"**, but the program should run anyway.
When the execution is over the text "Model inference is finished" will be printed on the "stdout".