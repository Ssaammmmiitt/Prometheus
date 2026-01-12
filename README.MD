Run `pip install -r requirements.txt`
Run `python server.py`

Backend will be hosted at `localhost:8000`

use the endpoint `/predict`

## **EXPECTS**
```
{
    "output_date": 20180305,
    "patch_row": 80,
    "patch_col": 32,
}
```

## **RETURNS**

```
{
  "result": {
    "probability": 0.519,
    "prediction": 1,
    "coords": [
      64,
      48
    ]
  }
}
```
