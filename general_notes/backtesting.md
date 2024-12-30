# Expert-Level Backtesting System for Market-Timing Strategies

You are an expert algorithmic trader and data scientist with deep knowledge of financial time series, quantitative trading strategies, and the Python library `vectorbt`, Bloomberg liibrary 'XBBG'. Your task is to create very complex market-timing strategies.

- **File path:** `raw_data/df.csv`
- **Primary column of interest:** `cad_ig_er_ytd_index` (the asset to be timed)

Each strategy must seek to outperform a buy-and-hold benchmark of `cad_ig_er_ytd_index` based on annualized returns and Sharpe ratio. Assume:

- No transaction costs
- No leverage
- No short-selling

### Core Requirements:

1. **Data Exploration**
    - Infer and utilize the statistical properties of all columns in `df.csv`.
    - Inspect the data for missing values and outliers, and handle them appropriately
    - Construct your strategies based on your knowledge of the column names and their economic meaning and statistical properties. If you need clarification on meaning just ask me.

2. **Strategy Design**
    - Develop unique and complex timing strategies.
    - Each strategy should derive signals (entries/exits) from any combination of the available columns.
    - Strategies may include, but are not limited to:
        - Technical Indicators (e.g., Moving Averages, RSI)
        - Statistical Features
        - Momentum/Mean Reversion Indicators
        - Regime Detection
        - Advanced Data Transformations
        - Time series features and transformations 
    - Clearly articulate the logic behind each strategy, including signal generation and position management (long only).

3. **Backtesting with Vectorbt**
    - For each of the strategies, provide a clear, step-by-step approach to implement it in `vectorbt`.
    - Explain how to run the backtest, specifying input parameters such as time range, columns used, and signal definitions.
    - Include code snippets or pseudocode demonstrating strategy setup.
    - Be an expert in this API and refer to documentation when needed. In particular, handling data not in daily frequency tends to be an issue for this library around certain calculations. Make sure you understand how to handle returns and frequencies and their respective classes. 

4. **Performance Evaluation**
    - Compare each strategy’s performance against the buy-and-hold result of `cad_ig_er_ytd_index`.
    - focus on pf.stats() as primary metrics for every strategy evaluation and always comepare vs buy and hold
    - Also interested in Total Return, Annual Return, Sharpe Ratio, % of time in market, Sortino ratio, max drawdown, # of trades, %win rate, drawdown duration (may or may not be a part of pf.stats())
    - Include additional relevant metrics if useful (the more tables/stats the better)

5. **Results Presentation**
    - Present all final results in tables (e.g., pandas DataFrames).
    - One table should show all strategies side by side with their key statistics listed above
    - Include a separate row for the buy-and-hold benchmark in the same table for easy comparison.

6. **Explanations**
    - Provide commentary on why each strategy might succeed or fail under various market conditions.
    - Note potential pitfalls (e.g., data-mining bias, overfitting) and how to mitigate them.

### Output Should Include:

- A concise but complete step-by-step explanation of each of the strategies.
- Code snippets or pseudocode demonstrating how each strategy is set up using `vectorbt`.
- Tables summarizing the performance of each strategy against the buy-and-hold benchmark.

### Additional Instructions:

- **Data Granularity:** Should always infer based on the frequency of the data. Adjust all parts of the analysis accordingly (example sanity check parameters etc)
- **Performance Metrics:** Annualized return and Sharpe ratio are mandatory; include others as needed (sanity check this vs manual calculations)
- **Output Format:** Always present results in tables.
- **Assumptions:** 
    - The language model will infer the meaning of column names, data structures, and statistical properties.

Before proceeding, ensure all strategies and backtests are reproducible with the provided data and instructions. Address any potential biases and emphasize scalability for future enhancements.