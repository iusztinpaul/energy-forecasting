# energy-forecasting

## Feature Engineering Pipeline

### Run

```shell
docker run -it -p 6789:6789 -v $(pwd):/home/src mageai/mageai \
  /app/run_app.sh mage start energy-forecasting-feature-pipeline
```

If you are on Windows the command is slightly different. You can check it out [here](https://docs.mage.ai/getting-started/setup).
