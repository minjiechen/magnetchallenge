# MagNet Challenge 2023
Team: Fuzhou University

## Outline
- **Model**: `.ckpt` models from Pytorch and `.json` files contained normalization infomation.
- **Result**: `.csv` resulst for 5 materials.
- **MagNet_Data.py**: a python module to create Pytorch dataset, which will be imported in `Model_Inference.py`.
- **MagNet_Model.py**: a python module to create Pytorch model, which will be imported in `Model_Inference.py`.
- **Model_Inference.py**: a python script for model inference.
- **3F4_self-measure.zip**: Self-measured data for 3F4 that do not used in this MagNet Challenge. It is submitted for possible research purposes.
- **requirements.txt**: Some Python pakages required to be installed. 

## Usage
1. Install required pyhton packages.
```
cd Fuzhou University
pip install requirements.txt
```
2. Modify the `DATA_ROOT=''` in `Model_Inference.py`, which contains the data from [2023 MagNet Challenge Testing Data.zip].
3. Modify the `Material = 'A'` for each material from [A, B, C, D, E].
4. Run `Model_Inference.py`
```
python Model_Inference.py
```

The inference results `Volumetric_Loss_{Material}.csv` of each material will be saved to `./Fuzhou University/Result`. If you want to test this program, please back up the results in `./Fuzhou University/Result` beforehand or change the save address in:
```python
# Save Results
# None: Please backup files in './Result' or change another directory to save.
os.makedirs('./Result', exist_ok=True)
data_P = model.results
with open(rf'./Result/Volumetric_Loss_{Material}.csv', "w") as f:
    np.savetxt(f, data_P)
    f.close()
```

We use `batch_size=128` for both training and testing, if the test computer does not have enough GPU memory, please modify the `batch_size` in:
```python
# Create Pytorch Lightning Dataset
# Note: If the test PC does not have enough GPU memory, set a smaller [batch_size].
#       However, changing the [batch_size] will make a small difference in performance.
print('------------/ Start load dataset... /------------')
dm = MagNetDataModule(data_B, data_F, data_T, batch_size=128,
                        norm_info_path=rf'./Model/norm_info_{Material}.json')
dm.prepare_data()
dm.setup('inference')
print('------------/ Successfully load dataset! /------------')
```

If the test computer does not have GPUs, change `accelerator="gpu"` to `"cpu"` and disable mixed precision:
```python
trainer = pl.Trainer(
    accelerator="cpu",
    benchmark=False,
    deterministic=True,
    # precision='16-mixed', # commenting out this line if use 'cpu'
    logger=False
)
```

If any issues are encountered in running this code, please don’t hesitate to contact us [xinyu3307@163.com].