# Recognizer

To run, first initalize a virtualenv:

```
  $ python -m virtualenv venv
```

Then activate said virtualenv:

```
  $ .\venv\Scripts\Activate.ps1
```

(or equivalent for your shell)

Then, install the requirements:

```
  $ pip install -r requirements.txt
```

And finally, run with:

```
  $ Measure-Command {python .\recognize.py .\data\test_data\sensorLog_20200331T105141e5xup4ssiuI_mix.txt | Out-Default }
```