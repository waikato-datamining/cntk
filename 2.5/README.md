# CNTK 2.5

Collection of modified CNTK 2.5 scripts.

## Installation

* create a virtual environment using Python 3.5

  ```
  virtualenv -p /usr/bin/python3.5 venv
  ``` 
  
* install CNTK for either ([source](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Linux-Python?tabs=cntkpy25)

  * CPU only (not all scripts are supported on CPU)
  
    ```
    ./venv/bin/pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.5-cp35-cp35m-linux_x86_64.whl
    ```
  
  * GPU
  
    ```
    ./venv/bin/pip install https://cntk.ai/PythonWheel/GPU/cntk_gpu-2.5-cp35-cp35m-linux_x86_64.whl
    ```
  
* install additional requirements

  ```
  ./venv/bin/pip install -r requirements.txt
  ```