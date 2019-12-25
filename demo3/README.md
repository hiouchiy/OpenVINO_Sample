# Quantize the CheXNet by OpenVINO

## Prepare
- Install OpenVINO (Tested 2019R3.1 on this notebook)
- Install Intel-PyTorch following [here](https://software.intel.com/en-us/articles/getting-started-with-intel-optimization-of-pytorch)
- Clone [CheXNet Repo](https://github.com/taneishi/CheXNet) first. Then clone this repository.
- Put all files (besides README.md) in this repository into the root folder of the cloned CheXNet repository as above.
- Download a dataset from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC) and unzip it and put them into "ChestX-ray14/images" in CheXNet repository.
- Run the notebook.