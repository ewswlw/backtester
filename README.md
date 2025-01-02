# Backtester

A Python-based backtesting framework for financial trading strategies, with support for various data sources including Bloomberg and Yahoo Finance.

## Features

- Multiple data source support (Bloomberg, Yahoo Finance)
- Vectorized backtesting using vectorbt
- Jupyter notebook integration
- Customizable strategy implementation
- Performance analytics and visualization

## Project Structure

- `src/`: Core backtesting engine and utilities
- `notebooks/`: Jupyter notebooks for strategy development and analysis
- `config/`: Configuration files for different strategies and data sources
- `data/`: Processed data files
- `raw_data/`: Raw data files
- `utils/`: Utility functions and helper scripts
- `example_scripts/`: Example strategy implementations
- `Outputs/`: Backtesting results and reports
- `plots/`: Generated charts and visualizations

## Setup

This project uses Poetry for dependency management. To get started:

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

4. Install the Jupyter kernel for the poetry environment:
```bash
poetry run python -m ipykernel install --user --name tajana-MtNpgr4y-py3.11 --display-name "Poetry (backtester)"
```

## Usage

1. Start by exploring the example notebooks in the `notebooks/` directory
2. Configure your data sources in the `config/` directory
3. Implement your trading strategy using the provided framework
4. Run backtests and analyze results using the built-in analytics tools

## Dependencies

Key packages include:
- vectorbt: For vectorized backtesting
- pandas: For data manipulation
- numpy: For numerical computations
- matplotlib/seaborn: For visualization
- yfinance: For Yahoo Finance data
- xbbg: For Bloomberg data (requires Bloomberg terminal)

## License

[Add your license here]
