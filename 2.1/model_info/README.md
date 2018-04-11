# Model info 2.1

Simply prints the outputs of the supplied model to stdout.

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
