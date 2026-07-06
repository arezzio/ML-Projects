"""
Predictive Drift Navigator
A forward looking portfolio rebalancing tool.

Core idea:
Instead of telling the user what to buy today, this tool forecasts prices
30 days into the future using Prophet, runs a portfolio optimization for
each of those future days, and compares each future optimal portfolio to
the portfolio we would build today. The size of that difference is the
"Drift Score." When drift crosses a threshold, the tool signals that the
user should rebalance now rather than wait.

How the pieces connect, at a glance:
    get_price_history           -> pulls raw data that everything else needs
    forecast_prices              -> called once per ticker inside build_forecast_table
    build_forecast_table         -> combines all ticker forecasts into one table
    expected_returns_and_covariance -> turns any price table (real or simulated)
                                       into the inputs optimize_portfolio needs
    optimize_portfolio           -> called once for "today" and once per future day
    calculate_drift_score        -> compares today's weights to each future result
    run_drift_navigator          -> the conductor that calls everything in order

Required libraries:
    pip install yfinance prophet pandas numpy scipy
"""

# numpy handles the numerical arrays and math (means, sums, square roots) used
# throughout the optimization and drift calculations.
import numpy as np

# pandas handles all the tabular data: price histories, forecast tables, and
# the final results table. Almost every function passes data around as a
# pandas DataFrame or Series.
import pandas as pd

# yfinance is the library that actually reaches out to Yahoo Finance and
# downloads historical stock prices.
import yfinance as yf

# Prophet is Facebook's forecasting library. We give it a history of prices
# for one asset and it predicts future values.
from prophet import Prophet

# minimize is a general purpose optimizer from scipy. We use it to find the
# portfolio weights that maximize the Sharpe ratio (see optimize_portfolio).
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
# Keep all the knobs you might want to change in one place. This makes it
# easy to experiment without hunting through the whole file. Every other
# section reads from these constants rather than hardcoding values inline.
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
LOOKBACK_DAYS = 365 * 2
FORECAST_HORIZON = 30

# The annual risk free rate used inside the Sharpe ratio formula. This
# represents the return of a "safe" asset like a T-bill, which the Sharpe
# ratio measures excess return against.
RISK_FREE_RATE = 0.04

# If the drift score (see calculate_drift_score) exceeds this number, the
# tool will flag that day as a recommended rebalance point. 0.10 means the
# optimal portfolio would need to shift by a combined 10 percentage points
# across all holdings before we trigger an alert.
DRIFT_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# 2. DATA COLLECTION
# ---------------------------------------------------------------------------

def get_price_history(tickers, lookback_days):
    """
    Downloads daily adjusted close prices for each ticker.

    Returns a DataFrame where each column is a ticker and each row is a date.
    This DataFrame is the starting point for both today's optimization and
    every Prophet forecast, so if this function returns bad data, every
    downstream step is affected.
    """
    # pd.Timestamp.today() grabs the current date and time. This becomes the
    # end of our historical window.
    end_date = pd.Timestamp.today()

    # pd.Timedelta lets us subtract a number of days from a timestamp.
    # start_date is therefore lookback_days before today.
    start_date = end_date - pd.Timedelta(days=lookback_days)

    # yf.download does the actual network call to Yahoo Finance. Passing a
    # list of tickers returns a DataFrame with a multi level column index
    # (one set of columns per ticker). auto_adjust=True adjusts prices for
    # splits and dividends so returns are comparable across time.
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    # raw_data has several price types (Open, High, Low, Close, Volume). We
    # only need the closing price for each day, so we select that slice.
    prices = raw_data["Close"]

    # If any ticker is missing a price on a given day (holiday, IPO date,
    # data gap), dropna removes that entire row so every asset lines up on
    # the exact same set of dates. This matters because the covariance
    # calculation later requires equal length, aligned return series.
    prices = prices.dropna()

    # Return the cleaned price table. Every other function in this file
    # either uses this directly or uses something derived from it.
    return prices


# ---------------------------------------------------------------------------
# 3. FORECASTING WITH PROPHET
# ---------------------------------------------------------------------------

def forecast_prices(price_series, horizon):
    """
    Fits a Prophet model to a single asset's price history and forecasts
    forward by `horizon` days.

    price_series: a pandas Series of prices indexed by date, for ONE ticker.
    Returns a DataFrame with columns ds (date) and yhat (predicted price)
    for just the future forecasted days.

    This function is called once per ticker inside build_forecast_table, so
    if you have 4 tickers, this function runs 4 separate times, each with
    its own independent Prophet model.
    """
    # Prophet has a strict input format: a DataFrame with exactly two
    # columns, "ds" for the date and "y" for the value being forecasted.
    # Here we rebuild the single ticker's price series into that shape.
    df = pd.DataFrame({
        "ds": price_series.index,   # the dates, taken from the Series index
        "y": price_series.values    # the actual prices on those dates
    })

    # Create the Prophet model object. We turn off daily seasonality since
    # stock data does not have meaningful within day patterns at this
    # granularity, but we keep weekly and yearly seasonality on since stocks
    # can show patterns tied to trading days and calendar years.
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)

    # fit() is where Prophet actually learns from the historical data. This
    # is the training step, equivalent to model.fit() in most ML libraries.
    model.fit(df)

    # make_future_dataframe extends the date range by `horizon` more days
    # beyond the last date in the training data. It returns a DataFrame with
    # just a "ds" column covering both the historical and future dates.
    future = model.make_future_dataframe(periods=horizon)

    # predict() generates forecasts for every date in `future`, including
    # the historical dates we already had. The result includes many columns
    # (trend, seasonality components, etc), but we only need yhat, which is
    # the final predicted value.
    forecast = model.predict(future)

    # We only want the rows that are actually NEW, meaning dates after the
    # last real historical date. This filters out the re-predicted history.
    future_only = forecast[forecast["ds"] > df["ds"].max()][["ds", "yhat"]]

    # Reset the row index so it starts cleanly at 0 for these future rows
    # only. This makes it easier to line up rows across different tickers
    # later in build_forecast_table.
    future_only = future_only.reset_index(drop=True)

    return future_only


def build_forecast_table(prices, horizon):
    """
    Runs forecast_prices for every ticker and combines the results into one
    DataFrame. Rows are future dates, columns are tickers, values are
    predicted prices.

    This function is the bridge between the per ticker Prophet forecasts and
    the portfolio level calculations in run_drift_navigator, which expect a
    single table shaped like the original price history.
    """
    # An empty dictionary that will hold one forecasted price array per
    # ticker. Dictionaries let us build up columns before assembling the
    # final DataFrame in one step, which is more efficient than appending
    # columns one at a time.
    forecast_columns = {}

    # Loop over every column name in the price table, meaning every ticker.
    for ticker in prices.columns:
        # Call forecast_prices on just this ticker's price Series. This is
        # the only place forecast_prices gets called, once per ticker.
        forecast_df = forecast_prices(prices[ticker], horizon)

        # Store just the predicted values (yhat), not the dates, since all
        # tickers share the same future dates and we only need to store
        # those once.
        forecast_columns[ticker] = forecast_df["yhat"].values

    # Since every ticker was forecasted over the identical horizon, we can
    # grab the future dates from any one of them. Here we just reuse the
    # first ticker in the list as a convenient source of the date column.
    dates = forecast_prices(prices[prices.columns[0]], horizon)["ds"].values

    # Assemble the dictionary of ticker forecasts into a single DataFrame,
    # using the shared future dates as the row index. The resulting shape
    # matches the original price history: dates as rows, tickers as columns.
    forecast_table = pd.DataFrame(forecast_columns, index=dates)

    return forecast_table


# ---------------------------------------------------------------------------
# 4. PORTFOLIO OPTIMIZATION
# ---------------------------------------------------------------------------

def expected_returns_and_covariance(price_table):
    """
    Converts a price table (dates x tickers) into:
        - expected daily returns (mean of percent changes)
        - a covariance matrix of those returns

    This function is called twice inside run_drift_navigator: once on the
    real historical prices (for today's portfolio) and once per simulated
    future day (for each future portfolio). It is the shared translator
    between raw prices and the inputs optimize_portfolio actually needs.
    """
    # pct_change() computes the day over day percentage change for every
    # column (ticker). The first row becomes NaN since there is no prior
    # day to compare against, so dropna() removes that row.
    returns = price_table.pct_change().dropna()

    # mean_returns is the average daily return for each ticker, a single
    # number per column. This becomes the "expected return" assumption fed
    # into the Sharpe ratio calculation.
    mean_returns = returns.mean()

    # cov_matrix captures how the tickers move together (or against) each
    # other. It is a square matrix, tickers by tickers, and is essential for
    # estimating portfolio level risk, not just individual asset risk.
    cov_matrix = returns.cov()

    return mean_returns, cov_matrix


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    The objective function we minimize. Since scipy's minimize() only
    minimizes, and we want to MAXIMIZE the Sharpe ratio, we minimize its
    negative instead. This function is never called directly by us, it is
    passed into scipy's minimize() inside optimize_portfolio, which calls it
    repeatedly while searching for the best weights.
    """
    # np.dot(weights, mean_returns) computes the weighted average of the
    # expected daily returns, which is the portfolio's expected daily
    # return. Multiplying by 252 (roughly the number of trading days per
    # year) annualizes it.
    portfolio_return = np.dot(weights, mean_returns) * 252

    # This line computes portfolio variance using the standard formula
    # w^T * Cov * w, then takes the square root to get volatility
    # (standard deviation), and annualizes it the same way as the return.
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights)) * 252
    )

    # The Sharpe ratio is excess return (return above the risk free rate)
    # divided by volatility. A higher Sharpe ratio means better return per
    # unit of risk taken.
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # We return the negative because scipy's minimize() only knows how to
    # minimize, and minimizing the negative of something is the same as
    # maximizing the original value.
    return -sharpe_ratio


def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
    """
    Finds the portfolio weights that maximize the Sharpe ratio, subject to:
        - weights sum to 1 (fully invested)
        - no shorting (each weight is between 0 and 1)

    Returns a numpy array of weights, one per asset, in the same order as
    mean_returns. This function is called exactly once for "today" and once
    for every future simulated day inside run_drift_navigator, meaning it
    runs FORECAST_HORIZON + 1 times in total during a full run.
    """
    # The number of assets determines the size of our weights array. This
    # is inferred from mean_returns rather than hardcoded, so the function
    # automatically adapts if you change the TICKERS list.
    num_assets = len(mean_returns)

    # Every optimizer needs a starting point to search from. We start with
    # an equal weight portfolio (for example, 25 percent each with 4
    # assets) since it is a neutral, unbiased starting guess.
    initial_guess = np.repeat(1 / num_assets, num_assets)

    # This constraint tells scipy that the sum of all weights must equal 1,
    # meaning we invest exactly 100 percent of the portfolio, nothing held
    # back in cash and nothing borrowed.
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})

    # Bounds restrict each individual weight to be between 0 and 1. This
    # rules out short selling (negative weights) and leverage (weights
    # above 1) for this first version of the tool.
    bounds = tuple((0, 1) for _ in range(num_assets))

    # This is the actual optimization call. SLSQP (Sequential Least Squares
    # Programming) is a solver capable of handling the equality constraint
    # and bounds we defined above. It repeatedly calls
    # negative_sharpe_ratio with different candidate weight arrays until it
    # finds the combination that minimizes that function, which is
    # equivalent to maximizing the true Sharpe ratio.
    result = minimize(
        negative_sharpe_ratio,
        initial_guess,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    # result.x holds the best weights the optimizer found. This is the
    # single array that gets returned and used everywhere else in the
    # pipeline as "the optimal portfolio."
    return result.x


# ---------------------------------------------------------------------------
# 5. DRIFT SCORE CALCULATION
# ---------------------------------------------------------------------------

def calculate_drift_score(today_weights, future_weights):
    """
    Measures how far a future optimal portfolio has drifted from today's
    optimal portfolio. Uses a simple L1 distance (sum of absolute
    differences in allocation).

    A drift score of 0 means no change is needed. Larger scores mean the
    optimal portfolio is expected to move further away from what we hold
    today. This function is called once per future day inside
    run_drift_navigator's main loop.
    """
    # np.abs takes the absolute value of the element wise difference between
    # the two weight arrays, so a ticker going from 25 percent to 15 percent
    # contributes 0.10 regardless of direction. np.sum then adds up these
    # differences across every ticker into one single number: the drift
    # score for that day. float() converts the result from a numpy type
    # into a plain Python float for cleaner printing and storage.
    return float(np.sum(np.abs(today_weights - future_weights)))


def run_drift_navigator():
    """
    Ties every stage together:
        1. Pull historical prices
        2. Compute today's optimal portfolio from historical returns
        3. Forecast the next 30 days of prices with Prophet
        4. For each future day, treat the forecast up to that day as a new
           "history" and recompute the optimal portfolio
        5. Compare each future portfolio to today's portfolio
        6. Flag any day where the drift crosses the threshold

    This is the only function called from the bottom of the file. Every
    other function above exists to support one of the steps here.
    """
    print("Step 1: downloading price history...")

    # Calls get_price_history using the tickers and lookback window defined
    # in the CONFIGURATION section. This is the single source of real data
    # for the entire script.
    prices = get_price_history(TICKERS, LOOKBACK_DAYS)

    print("Step 2: computing today's optimal portfolio...")

    # Runs the real historical prices through expected_returns_and_covariance
    # to get the two inputs optimize_portfolio needs.
    today_mean_returns, today_cov_matrix = expected_returns_and_covariance(prices)

    # Finds the best weights based purely on historical data, with no
    # forecasting involved. This is our baseline, "what we would build if we
    # rebalanced today" portfolio.
    today_weights = optimize_portfolio(today_mean_returns, today_cov_matrix, RISK_FREE_RATE)

    print("Today's optimal weights:")
    # zip pairs up each ticker name with its corresponding weight so we can
    # print them together, ticker by ticker.
    for ticker, weight in zip(TICKERS, today_weights):
        # :.2% formats a decimal like 0.25 as "25.00%".
        print(f"  {ticker}: {weight:.2%}")

    print("\nStep 3: forecasting the next 30 days with Prophet...")

    # Calls build_forecast_table, which internally calls forecast_prices
    # once per ticker, and returns one combined table of future prices.
    forecast_table = build_forecast_table(prices, FORECAST_HORIZON)

    print("Step 4: recomputing optimal weights across the forecast horizon...")

    # An empty list that will collect one dictionary of results per future
    # day, which we turn into a DataFrame at the end.
    results = []

    # Loop over every row (future day) in the forecast table. range(len(...))
    # gives us numeric positions 0, 1, 2, and so on up to FORECAST_HORIZON - 1.
    for day_index in range(len(forecast_table)):
        # pd.concat stacks the real historical prices on top of the
        # forecasted prices, but only up through the current day_index. This
        # simulates "what history would look like if we fast forwarded to
        # this future date," which lets us reuse the exact same
        # expected_returns_and_covariance and optimize_portfolio functions
        # that we used for today's real portfolio.
        simulated_history = pd.concat([prices, forecast_table.iloc[:day_index + 1]])

        # Same two step process as before: turn prices into expected
        # returns and covariance, then optimize.
        future_mean_returns, future_cov_matrix = expected_returns_and_covariance(simulated_history)
        future_weights = optimize_portfolio(future_mean_returns, future_cov_matrix, RISK_FREE_RATE)

        # Compares this future day's optimal weights against today's
        # optimal weights, producing a single drift score for this day.
        drift_score = calculate_drift_score(today_weights, future_weights)

        # Store this day's date, drift score, and weights together as one
        # record. Storing them together (rather than in separate lists)
        # keeps each day's data bundled and easy to turn into rows later.
        results.append({
            "date": forecast_table.index[day_index],
            "drift_score": drift_score,
            "weights": future_weights
        })

    print("\nStep 5: checking for rebalance triggers...")

    # Converts our list of per day dictionaries into a proper DataFrame,
    # where each dictionary becomes one row and each key becomes a column.
    drift_df = pd.DataFrame(results)

    # Filters drift_df down to only the rows where drift_score is above our
    # configured DRIFT_THRESHOLD. This uses pandas boolean indexing: the
    # condition inside the brackets creates a True or False for every row,
    # and only True rows are kept.
    trigger_days = drift_df[drift_df["drift_score"] > DRIFT_THRESHOLD]

    # If there is at least one day that crossed the threshold...
    if len(trigger_days) > 0:
        # ...grab the very first one, since that is the soonest date the
        # tool recommends rebalancing by.
        first_trigger = trigger_days.iloc[0]
        print(f"ALERT: Rebalance recommended by {first_trigger['date']} "
              f"(drift score {first_trigger['drift_score']:.3f})")
    else:
        # No day crossed the threshold, meaning the current portfolio is
        # expected to stay close enough to optimal for the full horizon.
        print("No rebalance needed in the next 30 days based on current forecasts.")

    # Return the full table so it can be inspected, plotted, or tested
    # outside of this function too.
    return drift_df


# This block only runs when the file is executed directly (for example,
# "python predictive_drift_navigator.py"), not when it is imported as a
# module into another script. This is standard Python practice for keeping
# reusable functions separate from the code that actually runs them.
if __name__ == "__main__":
    # Runs the entire pipeline start to finish and stores the resulting
    # DataFrame of drift scores.
    drift_results = run_drift_navigator()

    print("\nFull drift score table:")
    # Prints just the date and drift_score columns for a clean, readable
    # summary of how drift evolves over the 30 day horizon.
    print(drift_results[["date", "drift_score"]])
