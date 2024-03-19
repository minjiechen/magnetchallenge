The function to compute the magnetic core loss, utilizing the neuronal network, is located in `StoiberReynvannEquation.py` and is called respectively.
This function is applicable for all materials using the following meterial ids. 
If this script is executed it reads the data for all materials and creates a csv containing the loss for each material.
| Material | Id |
|----------|----|
| A        | 10 |
| B        | 11 |
| C        | 12 |
| D        | 13 |
| E        | 14 |

To simplify the utilization of this script the scripts `MaterialAEquation.py`, `MaterialBEquation.py`, ... exist.
These scipts contain a function to compute the magnetic core loss for a single frequency, temperature and magnetic flux curve.
These functions are called `MaterialAEquation`, `MaterialBEquation`, ... .
Reading and writing csv files is done on script exectuion and is located in the functions `MaterialA2CSV`, `MaterialB2CSV`, ... .