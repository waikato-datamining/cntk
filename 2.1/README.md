# CNTK 2.1

Collection of modified CNTK 2.1 scripts.

## Installation

### Windows

**Instructions taken from here:**

https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-python?tabs=cntkpy21

Steps:

* install anaconda
  
  https://repo.continuum.io/archive/Anaconda3-4.1.1-Windows-x86_64.exe

* install graphviz (stable, windows, msi)

  https://graphviz.gitlab.io/_pages/Download/Download_windows.html
  
* add graphviz binary directory (eg `C:\Program Files (x86)\Graphviz2.38\bin`) 
  to PATH environment variable (see eg https://www.techjunkie.com/environment-variables-windows-10/)

* open command prompt
* create virtual environment

  ```
  conda create --name cntk21-py35 python=3.5 numpy scipy h5py jupyter
  ```

* activate environment

  ```
  activate cntk21-py35
  ```
  
* install cntk

  * Python 3.5, cpu only
  
    ```
    pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.1-cp35-cp35m-win_amd64.whl
    ```

  * Python 3.5, GPU

    ```
    pip install https://cntk.ai/PythonWheel/GPU/cntk-2.1-cp35-cp35m-win_amd64.whl
    ```

  * clone this repository
  
    ```
    git clone https://github.com/waikato-datamining/cntk.git
    ```
    
  * change into 2.1 directory
  
  * install additional requirements
  
    ```
    pip install -r requirements.txt
    ```
    
  * (optional) install CNTK samples

    ```
    python -m cntk.sample_installer
    ```
   
  * (optional) install Faster R-CNN data and models
       
      * change into Faster R-CNN directory
    
        ```
        cd CNTK-Samples-2-1\Examples\Image\Detection\FasterRCNN
        ```
    
      * install data/models
    
        ```
        python install_data_and_model.py
        ```
