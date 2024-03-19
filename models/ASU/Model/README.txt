# Model
 
The given model_inference.py was modified to to give prediction for a core loss.
We added no extra requirements besides numpy which was already used in the model inference.py
The model was run with python 11 and 12 on windows. We expect to run most setups, but can't guarantee it
 

As requested the results are written to the the results folder which is in same folder as Model folder that the scripts resides in. We assume base folder to be parent folder of Model folder.  

The instructions to run the code are listed in test.sh, runing "model_inferencel.py" directly with mofification of Material variable works or run code as shown in the test.sh file. models are in v0.1 subfolder in Model folder to explicitly show multiple that trained versions of same model can be hosted or made available. 

Models are shared as json file to improve the evaluation process by making the model easily testable. We reduced the  need to install multiple packages.

# Run

click test.sh or open git bash and enter: "bash test.sh"
All 5 material
 $ bash test.sh

One material at time
 $ python model_inference.py "Material A"

Edit model_inference.py script by changing the variable Material
 $ python model_inference.py

# Results

+ will populate the results folder with predictions as in model_inference.py template code


# Sample interaction(approx 6 min on our computer.)

ASUMag\Model> bash test.sh
Testing material Material A
7651
input size (7651, 29)
Model size:  1576
Model inference is finished!
Testing material Material B
3172
input size (3172, 29)
Model size:  1576
Model inference is finished!
Testing material Material C
5357
input size (5357, 29)
Model size:  1576
Model inference is finished!
Testing material Material D
7299
input size (7299, 29)
Model size:  1576
Model inference is finished!
Testing material Material E
3738
input size (3738, 29)
Model size:  1576
Model inference is finished!