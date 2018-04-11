# Tools 2.1

## model_info

Simply prints the outputs of the supplied model(s) to stdout.

The following command:

```bash
python model_info.py /some/where/PretrainedModels/ResNet_18.model 
```

Will output something like this:

```
Output('ce', [], [])
Output('errs', [], [])
Output('top5Errs', [], [])
Output('z', [#, ], [1000])
Output('ce', [], [])
Output('z', [#, ], [1000])
Output('z.PlusArgs[0]', [#, ], [1000])
```

## onnx_to_cntk

Converts an ONNX model into a native CNTK one.

```bash
python onnx_to_cntk.py onnx.model cntk.model 
```
