# CNTK 2.6

Collection of modified CNTK 2.6 scripts.

## Installation

### Windows

**GPU instructions taken from here:**

https://docs.microsoft.com/en-us/cognitive-toolkit/setup-gpu-specific-packages

Steps:

* install CUDA 9.0 and patches

  https://developer.nvidia.com/cuda-90-download-archive
  
* check for NVSMI DLL

  ```
  dir "C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"
  ```

* check CUDA 9.0 installation

   ```
  dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin\cudart64_90.dll"
  ```
  
* install cuDNN to `C:\cudnn-9.0-v7.0`

  http://developer.download.nvidia.com/compute/redist/cudnn/v7.0.4/cudnn-9.0-windows10-x64-v7.zip
  
* check cuDNN installation

  ``` 
  dir "C:\local\cudnn-9.0-v7.0\cuda\bin\cudnn64_7.dll"
  ```

* add the following paths to your PATH environment variable (if not already present)

  * `C:\Program Files\NVIDIA Corporation\NVSMI`
  * `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin`
  * `C:\local\cudnn-9.0-v7.0\cuda\bin`

**MKL instructions taken from here:**

https://docs.microsoft.com/en-us/cognitive-toolkit/setup-mkl-on-windows

Steps:

* download MKLML 2018.0.1

  https://github.com/01org/mkl-dnn/releases/download/v0.12/mklml_win_2018.0.1.20171227.zip

* unzip to `C:\local\mklml-2018.0.1` and add the `lib` directory to the PATH environment variable

**OpenCV instructions taken from here:**

https://docs.microsoft.com/en-us/cognitive-toolkit/setup-opencv-on-windows

Steps:

* download OpenCV 3.1

  http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.1.0/opencv-3.1.0.exe/download

* install to `C:\local\opencv3.10`

* add `C:\local\opencv3.10\opencv\build\x64\vc14\bin` to PATH


**CNTK Instructions taken from here:**

https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-python?tabs=cntkpy25

Steps:

* install anaconda:

  https://repo.continuum.io/archive/Anaconda3-4.1.1-Windows-x86_64.exe

* install graphviz (stable, windows, msi)

  https://graphviz.gitlab.io/_pages/Download/Download_windows.html

* add graphviz binary directory (eg `C:\Program Files (x86)\Graphviz2.38\bin`) to PATH environment variable
* open command prompt
* create virtual environment

  ```
  conda create --name cntk26-py35 python=3.5 numpy scipy h5py jupyter
  ```

* activate environment

  ```
  activate cntk26-py35
  ```
  
* install cntk 

  * Python 3.5, CPU only

    ```
    pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.6-cp35-cp35m-win_amd64.whl
    ```

  * Python 3.5, GPU

    ```
    pip install https://cntk.ai/PythonWheel/GPU/cntk_gpu-2.6-cp35-cp35m-win_amd64.whl
    ```

  * clone this repository
  
    ```
    git clone https://github.com/waikato-datamining/cntk.git
    ```
    
  * change into 2.6 directory
  
  * install additional requirements
  
    ```
    pip install -r requirements.txt
    ```
    
  * (optional) install samples (change into the root of your home directory)

    ```
    python -m cntk.sample_installer
    ```

  * change into Faster R-CNN directory

    ```
    cd CNTK-Samples-2-6\Examples\Image\Detection\FasterRCNN
    ```
  
  * install data/models

    ```
    python install_data_and_model.py
    ```
