# SimpleManus


## Set up the virtual environment

1. Ensure poetry is installed, if not, install it with:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
2. Configure Poetry to Use a Local .venv Directory (Recommended)
To have the virtual environment created inside your project folder (as .venv), run:
```bash
poetry config virtualenvs.in-project true
```

3. Install the dependencies
```bash
poetry install
```


