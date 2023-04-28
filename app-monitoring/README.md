# Monitoring - Web APP

## Install for Development

Run:
```shell
cd app-monitoring
poetry shell
poetry install
```

**NOTE:** Be sure that the API is already running.

To start the app, run the following:
```shell
streamlit run monitoring/main.py --server.port 8502
```

Access http://127.0.0.1:8502/ to see the app.
