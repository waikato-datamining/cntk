# Transfer Learning 2.1

Based on transfer learning example files from CNTK-Samples-2-1:

```
Examples/Image/TransferLearning/
```

Combined the following two files:

* TransferLearning_Extended.py
* TransferLearning.py

Into `Transfer.py` and pushed parameters into a config file
to avoid having to modify the code for different models or what
GPU to run the process on.

See `TransferExample.yaml` for an example config file.

Use `--help` to display all the available options of the `Transfer.py`
class.
