# Monitoring - Web APP

Check out [Lesson 6](https://towardsdatascience.com/fastapi-and-streamlit-the-python-duo-you-must-know-about-72825def1243) on Medium to better understand how we built the Streamlit monitoring dashboard.

## Install for Development

Create virtual environment:
```shell
cd app-monitoring
poetry shell
poetry install
```

**NOTE:** Be sure that the API is already running.


## Usage for Development

To start the app, run the following:
```shell
streamlit run monitoring/main.py --server.port 8502
```

Access http://127.0.0.1:8502/ to see the app.
