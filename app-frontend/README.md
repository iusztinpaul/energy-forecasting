# Frontend - Web APP

Check out [Lesson 6](https://towardsdatascience.com/fastapi-and-streamlit-the-python-duo-you-must-know-about-72825def1243) on Medium to better understand how we built the Streamlit predictions dashboard.

## Install for Development

Create virtual environment:
```shell
cd app-frontend
poetry shell
poetry install
```

**NOTE:** Be sure that the API is already running.


## Usage for Development

To start the app, run the following:
```shell
streamlit run frontend/main.py --server.port 8501
```

Access http://127.0.0.1:8501/ to see the app.
