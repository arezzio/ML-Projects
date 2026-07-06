# The Complete Beginner's Cheat Sheet
### Prophet Forecasting · Portfolio Optimization · The Predictive Drift Navigator

**How to use this:** you've (basically) never read Python before, so every concept below follows the same pattern — plain English, then an analogy, then the *real* code from this project so you see it "in the wild," not a toy example. Skim it once top to bottom so you know what's in here, then keep it open and jump to whatever section you need while you're actually coding. You are not supposed to memorize this.

## Table of Contents
- [Part 1: Python From Zero](#part-1-python-from-zero)
- [Part 2: Pandas From Zero](#part-2-pandas-from-zero)
- [Part 3: NumPy From Zero](#part-3-numpy-from-zero)
- [Part 4: Finance Concepts From Zero](#part-4-finance-concepts-from-zero)
- [Part 5: Prophet & Forecasting From Zero](#part-5-prophet--forecasting-from-zero)
- [Part 6: Optimization From Zero](#part-6-optimization-from-zero)
- [Part 7: The Original Repo — Fast Lookup](#part-7-the-original-repo--fast-lookup)
- [Part 8: The Predictive Drift Navigator — New Concepts](#part-8-the-predictive-drift-navigator--new-concepts)
- [Part 9: Debugging Toolkit](#part-9-debugging-toolkit)
- [Part 10: Glossary (A–Z)](#part-10-glossary-az)
- [Part 11: Full Syntax Quick-Reference](#part-11-full-syntax-quick-reference)

---

## PART 1: PYTHON FROM ZERO

### 1.1 What is a program, really?
A list of instructions the computer runs top to bottom, one at a time — with a few tricks (loops, functions, `if` statements) that make it skip around instead of always going straight down. Every concept below is just a different way of organizing that list.

### 1.2 Variables — labeled boxes
```python
ticker = "AAPL"
price = 202.45
```
**Think of it like:** a sticky note (`ticker`) stuck on a box, with something (`"AAPL"`) inside. Whenever the code says `ticker` later, Python opens that exact box.

### 1.3 Data types — what KIND of thing is in the box
| Type | Holds | Example |
|---|---|---|
| `str` | Text | `"AAPL"` |
| `int` | Whole number | `12` |
| `float` | Decimal number | `202.45` |
| `bool` | True or False | `True` |
| `None` | Deliberately *nothing* | `None` |

`None` isn't zero and isn't an empty string — it's Python's way of saying "this box is intentionally empty." You'll see it constantly as a return value when something fails (like a bad stock ticker).

### 1.4 Lists — a row of numbered lockers
```python
tickers = ["AMD", "MSFT", "AAPL"]
tickers[0]    # "AMD"  -> first locker
tickers[-1]   # "AAPL" -> last locker (negative = counting from the end)
```
**Think of it like:** school lockers in a row, numbered starting at 0, not 1. `[-1]` is a shortcut for "the last one" so you never have to count how many there are.

### 1.5 Dictionaries — a labeled filing cabinet
```python
holiday_map = {"New Year's Day": "new_years", "Christmas": "christmas"}
holiday_map["Christmas"]        # "christmas"
holiday_map.get("Easter")       # None (doesn't crash!)
holiday_map["Easter"]           # crashes: KeyError
```
**Think of it like:** a filing cabinet where every drawer has a label. You don't care that "Christmas" happens to be the 5th drawer — you just pull the drawer labeled `"Christmas"`. Use `.get()` when a key might not exist and you don't want a crash; use `[...]` when you're sure it's there.

### 1.6 Sets — a bag that refuses duplicates
```python
dates_a = {1, 2, 3}
dates_b = {2, 3, 4}
dates_a & dates_b     # {2, 3} -> only what's in BOTH
```
Like a list, but auto-removes duplicates and is built for fast "is this in here?" checks and comparing groups. The project uses this to find dates that *every single stock* traded on — no gaps, no mismatches.

### 1.7 Tuples — a list that can't be changed
```python
bounds = (0.05, 1)
```
Same idea as a list but uses `()`, and once made, it's locked — you can't accidentally overwrite it later. Used for things that should never mutate, like a min/max pair.

### 1.8 If / else — decision forks
```python
if price > 100:
    print("expensive")
else:
    print("cheap")
```
Reads like English. The indentation (spaces before `print`) is NOT optional — it's literally how Python knows what's "inside" the `if`.

### 1.9 For loops — do this to every item
```python
for ticker in tickers:
    print(ticker)
```
**Think of it like:** walking down the row of lockers, opening each one in order, doing the same thing to whatever's inside.

### 1.10 Functions — recipe cards
```python
def add_tax(price, tax_rate):
    return price * (1 + tax_rate)

add_tax(100, 0.08)   # 108.0
```
A function is a reusable recipe: you hand it ingredients (`price`, `tax_rate` — called *parameters*), it does steps, and hands back a result with `return`. You "call" it later by writing its name with `()`.

### 1.11 `return` vs `print` — the #1 beginner mix-up
- `print(x)` — shows `x` on screen. Does **not** give the value back to your program.
- `return x` — hands `x` back to whatever called the function, so it can be stored and used later.

```python
def get_price():
    print(202.45)          # you SEE 202.45 on screen...

result = get_price()       # ...but result is actually None! print() gave nothing back.
```
Nearly every function in this project uses `return`, because the result has to flow into the *next* step of the pipeline — nothing would work if functions only `print()`ed their answers.

### 1.12 `import` — borrowing tools from other files
```python
import pandas as pd
from .settings import START_DATE, END_DATE
```
- `import pandas as pd` — loads the whole `pandas` library and nicknames it `pd`.
- `from .settings import START_DATE` — reaches into another file (`settings.py`) and grabs just the two names you asked for. The leading `.` means "look in this same folder."

### 1.13 Classes and objects — a cookie cutter and its cookies
```python
class ProphetModel:
    def __init__(self):
        self.model = None

    def fit(self, price_series):
        self.model = Prophet()
        ...
```
A `class` is a blueprint. Every time you write `ProphetModel()`, Python stamps out a brand-new object following that blueprint — a "cookie" from the "cutter." `__init__` runs automatically the instant a new object is created; it sets up that object's starting state.

### 1.14 `self` — "this particular cookie"
Every method inside a class takes `self` as its first parameter — it means "the specific object this method was called on."
```python
model_a = ProphetModel()
model_b = ProphetModel()
# model_a.model and model_b.model are two totally separate boxes,
# even though they came from the same blueprint
```

### 1.15 Type hints — sticky notes, not rules
```python
def extract_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
```
`tickers: list[str]` is a note saying "this should be a list of strings." Python does **not** check this while running — it's pure documentation, for you and your editor. `-> dict[str, pd.DataFrame]` is the same idea for what the function hands back.

### 1.16 f-strings — fill-in-the-blank sentences
```python
ticker, price = "AAPL", 202.45
print(f"{ticker}: ${price:.2f}")   # "AAPL: $202.45"
```
The `f` before the quote means "this string has blanks." Anything inside `{}` gets calculated and dropped into the text. `:.2f` means "round to 2 decimal places."

### 1.17 `try` / `except` — having a backup plan
```python
try:
    df = stock.history(...)     # might fail: bad ticker, no internet, etc.
except Exception as e:
    print(f"failed: {e}")
    df = None
```
**Think of it like:** "try crossing the bridge; if it collapses, don't fall in the river — use the ladder instead." Without this, ONE bad stock ticker would crash your entire program, even if the other 11 worked fine.

### 1.18 Comprehensions — loops written in shorthand
These two do the *exact same thing*:
```python
# the long way
squared = []
for x in [1, 2, 3]:
    squared.append(x * x)

# the comprehension way — identical result
squared = [x * x for x in [1, 2, 3]]
```
Once you can read a normal `for` loop, read a comprehension by mentally "unrolling" it back into that longer form. Dictionaries work the same way:
```python
{ticker: price for ticker, price in zip(tickers, prices)}
```

### 1.19 `lambda` — a tiny function with no name
```python
square = lambda x: x * x
# identical to:
def square(x):
    return x * x
```
Used when a function is small enough that a full `def` + name feels like overkill — usually when handing it *directly* to another function as an argument (like telling SciPy's optimizer "here's your constraint rule").

### 1.20 The walrus operator `:=`
```python
if mapped := holiday_map.get(name):
    return mapped
```
Assigns AND checks, in one line. Identical to:
```python
mapped = holiday_map.get(name)
if mapped:
    return mapped
```

### 1.21 `*args` and `**kwargs` — unpacking a backpack
```python
prophet_params = {"yearly_seasonality": True, "weekly_seasonality": True}
Prophet(**prophet_params)
# identical to:
Prophet(yearly_seasonality=True, weekly_seasonality=True)
```
`**` in front of a dict dumps its contents out as individually named arguments — like unzipping a backpack and handing each item over separately. `*` does the same for lists/tuples, but as unnamed (positional) arguments — e.g. `set.intersection(*date_sets)`.

### 1.22 Closures — a function that remembers its surroundings
```python
def make_optimizer(risk_aversion):
    def objective(weights):
        return risk_aversion * some_math(weights)   # uses risk_aversion from OUTSIDE
    return objective
```
The inner function keeps access to variables from the outer function even after the outer one's finished running. This is exactly how `optimiser.py`'s `objective()` function uses `mu`, `cov`, and `risk_aversion` without ever receiving them as its own arguments.

### 1.23 Method chaining — `.this().then_this()`
```python
ProphetModel().fit(prices).predict_next(prices)
```
Only possible because `fit()` ends with `return self` — handing back the same object so you can immediately call another method on it, like a conveyor belt.

### 1.24 Indexing and slicing
```python
prices[0]      # first item
prices[-1]     # last item
prices[1:3]    # items at index 1 and 2 (the end, 3, is EXCLUDED)
prices[:5]     # first 5 items
prices[1:]     # everything from index 1 to the end
```

---

## PART 2: PANDAS FROM ZERO
*(pandas = the library for working with tables of data — this is what `pd` means everywhere)*

### 2.1 What is a DataFrame?
Picture an Excel spreadsheet living inside Python — rows and columns, both with labels.
```python
import pandas as pd
df = pd.DataFrame({"Price": [100, 102, 99], "Returns": [0.0, 0.02, -0.03]})
```

### 2.2 What is a Series?
A single column pulled out of a DataFrame — still has row labels, just one column of values.
```python
df["Price"]   # this is a Series, not a DataFrame
```

### 2.3 The Index — row labels
Every DataFrame has an "index," the row labels on the left. In this project, that's almost always a date.
```python
df.index          # all the dates
df.index[-1]      # the most recent date
```

### 2.4 Peeking at data
```python
df.head()   # first 5 rows
df.tail()   # last 5 rows
df.shape    # (num_rows, num_columns)
```

### 2.5 Selecting a column
```python
df["Price"]                # one column -> a Series
df[["Price", "Returns"]]   # multiple columns -> a DataFrame (note the DOUBLE brackets)
```

### 2.6 Selecting rows: `.loc`, `.iloc`, and plain `[]` — slow down here
This trips up almost every beginner, so read carefully:

- **`.iloc[0]`** — "integer location." Get a row by POSITION, like a list. `.iloc[0]` is always the first row physically in the table, no matter what its label says.
- **`.loc[some_date]`** — "label location." Get a row by its actual LABEL (a date, here).
- **`df.index[-1]`** — gets the LABEL of the last row (a date) — not the row's contents.

```python
df.iloc[-1]              # last row, by position
df.loc[some_date]        # the row labeled with that specific date
df["Price"].iloc[-1]     # last value in the Price column — you'll see this everywhere
```

### 2.7 Adding a new column
```python
df["Returns"] = df["Price"].pct_change()
```
Just assign to a column name that doesn't exist yet — pandas creates it automatically.

### 2.8 `.pct_change()` — daily returns, with real numbers
```
Price:    100,   102,    99
Returns:  NaN,  0.02,  -0.0294
```
Formula: `(today - yesterday) / yesterday`. Day 1 has nothing before it, so it becomes `NaN` ("Not a Number" — pandas' word for missing data). That's why the very next line is almost always...

### 2.9 `.dropna()`
```python
df = df.dropna()
```
Deletes any row containing a `NaN`. Used right after `.pct_change()` to remove that empty first row.

### 2.10 Filtering rows with conditions
```python
df[df["Price"] > 100]              # only rows where price is over 100
df.loc[df.index >= some_date]      # only rows from some_date onward
```
`df["Price"] > 100` actually produces a Series of `True`/`False` values, one per row — a "boolean mask." Putting that mask inside `df[...]` keeps only the `True` rows. This exact pattern is how `collect_recent_prices()` grabs the trailing 30 days of data.

### 2.11 Combining tables: `pd.concat`
```python
df_combined = pd.concat([df_old, df_new])
```
Stacks DataFrames on top of each other. Used to glue a freshly predicted row onto the end of real historical data.

### 2.12 The confusing date family
| Type | What it is | Comes from |
|---|---|---|
| `datetime.date` | A plain calendar date, e.g. `2026-07-03` | Python's built-in `datetime` module |
| `pd.Timestamp` | pandas' richer date/time type (can hold hours, timezone, etc.) | pandas |
| `datetime.timedelta` | A *duration* you add/subtract from dates, e.g. "1 day" | `datetime` module |

```python
from datetime import timedelta
last_date = df.index[-1]                    # might be a date OR a Timestamp
next_date = last_date + timedelta(days=1)   # move forward one calendar day
```
A very common real-world bug: comparing a `datetime.date` to a `pd.Timestamp` directly can misbehave, because technically they're different types even though both "mean a date." If you hit a weird date comparison bug, this mismatch is the first thing to check.

### 2.13 Common pandas gotchas
- **`SettingWithCopyWarning`** — pandas isn't sure if you meant to edit the real DataFrame or a temporary copy of it. Fix: call `.copy()` explicitly when you intend to branch off an independent copy — you'll see `df_copy = df.copy()` constantly in this codebase for exactly this reason.
- **Chained assignment** — `df[df.Price > 100]["Returns"] = 0` looks reasonable but often silently fails to actually change anything. Use `.loc[]` instead: `df.loc[df.Price > 100, "Returns"] = 0`.

---

## PART 3: NUMPY FROM ZERO
*(numpy = the library for fast math on arrays of numbers — this is what `np` means everywhere)*

### 3.1 Arrays vs. lists
A NumPy array looks like a list but supports fast math on *every element at once*.
```python
import numpy as np
prices = np.array([100, 102, 99])
prices * 2        # array([200, 204, 198]) -> every element multiplied, no loop needed
```
A plain Python list can't do this — `[100, 102, 99] * 2` would just repeat the list twice (`[100, 102, 99, 100, 102, 99]`), not multiply each number.

### 3.2 The dot product — explained with a shopping cart
```python
weights = np.array([0.5, 0.3, 0.2])       # 50%, 30%, 20% of your money
returns = np.array([0.01, 0.02, -0.01])   # each stock's return
np.dot(weights, returns)                   # 0.5*0.01 + 0.3*0.02 + 0.2*(-0.01) = 0.009
```
**Think of it like:** buying different quantities of items at different prices — "quantity × price, added up for the whole cart." Here it's "weight × return, added up for every stock" — which is precisely your portfolio's overall return.

### 3.3 Matrices in five sentences
A matrix is just a grid of numbers — a table of tables. The covariance matrix (`cov`) in this project is a grid where entry `(i, j)` tells you how much stock `i` and stock `j` move together; the diagonal `(i, i)` is just each stock's own volatility. `weights.T @ cov @ weights` is the standard formula for collapsing that whole grid into one number: total portfolio risk. You don't need to multiply matrices by hand — NumPy does it — you just need to recognize this pattern on sight.

---

## PART 4: FINANCE CONCEPTS FROM ZERO

### 4.1 Ticker
A short code identifying a stock on an exchange — `AAPL` = Apple, `TSLA` = Tesla. Just a name tag.

### 4.2 Price vs. Return
- **Price** — what one share costs right now, in dollars.
- **Return** — the *percentage* change in price over some period. $100 → $102 is a +2% return.

Models and optimizers work with returns, not raw prices, because returns are comparable across stocks that cost wildly different amounts — a $10 stock and a $1,000 stock can both have a 2% day.

### 4.3 Portfolio, weight, allocation
A **portfolio** is your whole collection of holdings. A **weight** (or "allocation") is the fraction of your total money sitting in each stock — weights always add up to 100% (1.0). $5,000 AAPL / $3,000 MSFT / $2,000 TSLA out of $10,000 total = weights of 50% / 30% / 20%.

### 4.4 Diversification
Spreading money across multiple assets so one bad stock doesn't wreck your whole portfolio. This is *why* the optimizer has a `MINIMUM_ALLOCATION` — it forces at least a little money into every stock instead of dumping 100% into whatever looks best on paper today (fragile, and paper answers change).

### 4.5 Volatility / Variance / Standard deviation
All three describe "how much does the price wiggle around." **Variance** is the mathematical measure; **standard deviation** is its square root (same idea, friendlier units); **volatility** is the casual finance word for the same thing. Higher = bigger, less predictable swings, both up and down.

### 4.6 Covariance & correlation — do two stocks move together?
**Covariance** measures whether two stocks tend to rise/fall at the same time (positive), move opposite (negative), or have no relationship (near zero). **Correlation** is the same idea rescaled to always sit between -1 and +1, making it easier to compare across pairs. This matters because two risky stocks moving in *opposite* directions can make a calmer portfolio together than either one alone — that's the actual mathematical basis of diversification, not just a saying.

### 4.7 Expected return
Your best *guess* at what a stock will return going forward. In this project, that guess comes from Prophet's forecast. It's a prediction, not a promise — markets don't actually know the future, and neither does the model.

### 4.8 Risk aversion (λ / lambda)
One number representing "how much do I hate risk, relative to how much I want return." Higher risk aversion → the optimizer plays it safer, trading away some expected return for a smoother ride.

### 4.9 Rebalancing
Buying/selling to bring your *actual* holdings back in line with your *target* weights. If AAPL rockets up, it might now be 60% of your portfolio instead of the 50% you wanted — rebalancing means trimming AAPL and topping up the others to get back to 50/30/20.

### 4.10 Portfolio drift
The gap between what you actually hold right now and what the "ideal" portfolio looks like. Drift grows over time as prices move — and that gap, projected into the *future* instead of only measured today, is exactly what your Drift Score will track.

### 4.11 Trading days, holidays, market calendar
The market isn't open every day — no weekends, no holidays (Thanksgiving, Christmas, etc.). A "trading day" (or "session") is a day it's actually open. This matters a lot for forecasting: naively assuming "tomorrow" is always a trading day makes your forecast dates drift out of sync with reality. That's exactly why `calendar_utils.py` asks the *real* NYSE calendar for the next valid sessions instead of blindly adding `timedelta(days=1)` in a loop.

---

## PART 5: PROPHET & FORECASTING FROM ZERO

### 5.1 What's a time series?
Data points in order over time — Monday's price, Tuesday's price, Wednesday's price. Order matters here, unlike a normal spreadsheet you could shuffle without losing information.

### 5.2 Forecasting vs. "just predicting one number"
Forecasting means predicting a whole *sequence* of future values, usually with some sense of confidence at each point. Not "AAPL will be exactly $210.14 tomorrow" — more like "AAPL will probably land around $208–$212, and that range gets wider the further out we look."

### 5.3 Trend, seasonality, holidays — what Prophet actually models
Prophet breaks a time series into pieces and adds them back together:
- **Trend** — the big, slow drift up or down over months/years.
- **Seasonality** — repeating patterns (day-of-week, time-of-year). `weekly_seasonality=True` lets Prophet learn things like "this stock tends to dip on Mondays."
- **Holiday effects** — one-off bumps around specific dates — the whole reason this project builds a holiday calendar and feeds it in.

Prophet sums these pieces to build its forecast — which is also why it hands back a smooth, sensible-looking future path instead of a jagged guess.

### 5.4 `ds` and `y` — Prophet's two required columns
Prophet insists on this exact naming, no exceptions:
- `ds` — "datestamp," the date column
- `y` — the value you're trying to predict (price, here)

```python
df = pd.DataFrame({"ds": price_series.index, "y": price_series.values})
```

### 5.5 `yhat`, `yhat_lower`, `yhat_upper`
- `yhat` — "y-hat," the predicted value (the hat symbol traditionally means "estimate of" in statistics).
- `yhat_lower` / `yhat_upper` — the uncertainty band, Prophet's honest admission of "it's probably somewhere in this range."

### 5.6 Why the uncertainty band gets WIDER further into the future
The model is more confident about tomorrow than about 29 days from now — small uncertainties compound the further out you project, the same way a weather forecast for tomorrow beats one for three weeks out. You saw this directly in our test run: day 1's gap between `yhat_lower`/`yhat_upper` was noticeably tighter than day 30's.

### 5.7 Fit vs. Predict — training vs. asking questions
- **`.fit(df)`** — show the model your historical data so it can learn the pattern. This is the slow, expensive step.
- **`.predict(future)`** — given an already-fitted model, ask "what do you think happens on these specific future dates?" Cheap, once fitted.

### 5.8 Why fit ONCE and predict 30 days, instead of predicting 1 day, 30 times
Every `.fit()` call redoes the expensive learning step from scratch. The original repo's `predict_next()` calls `.fit()` every time because it only ever needs one day ahead. Looping that 30 times for a month would re-train the model 30 times over on the *exact same historical data* — wasteful, with zero new information gained. Fitting once and handing Prophet a 30-day-wide `future` DataFrame gets the entire path in one `.predict()` call — this is exactly what `model_horizon.py` does, and it's the main efficiency idea behind your whole project.

---

## PART 6: OPTIMIZATION FROM ZERO

### 6.1 The "objective function" — a scorecard
A function that takes a candidate answer (here: a set of portfolio weights) and returns one number scoring how good it is. The optimizer's entire job is trying lots of weight combinations and finding the one with the best score.

### 6.2 Minimize vs. maximize — the sign-flip trick
SciPy's `minimize()` only knows how to make a number as SMALL as possible. To *maximize* something (portfolio return), multiply it by -1 and minimize that instead — the smallest possible negative number corresponds to the largest possible positive one.
```python
return -(port_return - 0.5 * risk_aversion * port_var)
#      ^ this minus sign is the entire trick
```

### 6.3 Constraints — unbreakable rules
A rule the final answer absolutely must satisfy — e.g., "all weights must sum to exactly 1 (100%)." The optimizer will never return an answer breaking a constraint, even if it would otherwise score higher.

### 6.4 Bounds — a fence around each variable
Simpler than a constraint: it limits the allowed range for one individual variable. `(0.05, 1)` means "this stock's weight must be between 5% and 100%" — can't be 0% (forces diversification) or negative (no short-selling in this model).

### 6.5 What SLSQP is doing (intuition only)
SLSQP = "Sequential Least Squares Programming." Skip the math, keep this mental model: start from an initial guess (equal weights, here), repeatedly nudge the weights toward a better score while constantly checking constraints/bounds haven't broken, and stop once no nearby nudge improves things further. That stopping point is called **convergence**.

### 6.6 The Markowitz formula, in plain English, piece by piece
```
maximize:  w·μ − (λ/2)·wᵀΣw
```
| Symbol | Meaning |
|---|---|
| `w` | the weights you're solving for (the unknowns) |
| `μ` (mu) | expected return of each asset (from Prophet, here) |
| `w·μ` | dot product — your portfolio's overall expected return |
| `Σ` (cov) | the covariance matrix — how assets move together |
| `wᵀΣw` | your portfolio's overall risk (variance) |
| `λ` (lambda) | risk aversion — how heavily you penalize that risk |

**In one sentence:** find the weights that get you the most expected return, minus a risk penalty dialed up or down by how cautious you want to be.

### 6.7 Reading the solver's output
```python
result = minimize(...)
result.success   # True/False - did it actually converge?
result.x         # the winning weights, as a plain array
result.message   # human-readable explanation if it failed
```
Always check `result.success` before trusting `result.x` — this is exactly why `optimiser.py` raises a `ValueError` if `result.success` is `False`, instead of silently handing back garbage weights.

---

## PART 7: THE ORIGINAL REPO — FAST LOOKUP

### 7.1 Pipeline map
```
extract_data()              → raw prices per ticker, from Yahoo Finance
preprocess_data()           → align all tickers to shared trading dates
model.predict_for_tickers() → Prophet, 1 day ahead, per ticker
collect_recent_prices()     → grab trailing 30 days (for charts/storage)
append_predictions()        → tack the forecast row onto the history
optimize_portfolio_mean_variance() → Markowitz weights, using the forecast as μ
save_results_to_supabase()  → persist to the database
```

### 7.2 File-by-file, one line each
| File | Job |
|---|---|
| `settings.py` | All the constants in one place |
| `extractor.py` | Downloads raw prices |
| `processor.py` | Aligns dates, appends predictions, trims windows |
| `model.py` | Prophet wrapper, ONE day ahead, refits from scratch every call |
| `optimiser.py` | Markowitz math → weights |
| `database.py` | Saves results to Supabase |
| `main.py` | Runs the whole pipeline in order — **read this file first when lost** |
| `streamlit_app.py` | Dashboard |

### 7.3 The shape of data flowing through everything
```python
dict[str, pd.DataFrame]
# {"AAPL": <DataFrame with Price/Returns columns, date index>, "MSFT": <...>, ...}
```
Nearly every function takes this shape in and hands this shape out. Lost reading a function? First ask: *"am I looking at ONE DataFrame, or a DICTIONARY of DataFrames (one per ticker)?"* — that question resolves most confusion instantly.

---

## PART 8: THE PREDICTIVE DRIFT NAVIGATOR — NEW CONCEPTS

### 8.1 Drift Score — a compass, not a speedometer
Your current portfolio is where you're **standing**. The optimizer's answer for some future day is where you **should** be standing on that day. The Drift Score is the distance between those two points — how far off course you'd end up if you did nothing and just let the future arrive. Big drift = heading somewhere very different from where you are today. Small drift = you're basically already where you should be, no action needed.

### 8.2 L1 distance (Manhattan distance) — walking city blocks
```python
drift = sum(abs(w_now[t] - w_future[t]) for t in tickers)
```
**Think of it like:** in Manhattan you can't cut diagonally through buildings — you walk blocks north/south and east/west and add up the total blocks walked. L1 distance does the same thing with portfolio weights: for each stock, measure how far off that *one* weight is, then add up every gap. This has a genuinely useful real-world meaning too — the L1 distance is roughly *"what percentage of your total portfolio you'd need to trade"* to get from where you are to where you should be. That's exactly the number a real investor cares about, because trading costs money (fees, taxes, bid-ask spread).

### 8.3 Why 30 separate optimizations?
Because "the optimal portfolio" isn't one fixed thing — it depends on expected returns, and expected returns change every day as the forecasted price path unfolds. Day 5's optimal weights and day 25's optimal weights can look quite different if, say, the forecast shows one stock climbing steadily while another slumps. Running the optimizer once per future day (using that day's forecasted price as `μ`) gives you a whole **movie** of "what's optimal," not just a single snapshot.

### 8.4 The trigger threshold — a tripwire
One number (e.g. `DRIFT_THRESHOLD = 0.10`) marking "this much drift is worth acting on." Below it: normal wobble, ignore. At or above it: fire the "Trigger Rebalance Now" alert.

**Think of it like:** a smoke detector's sensitivity dial. Too low, and it screams at every puff of steam — annoying, and you'll start ignoring it. Too high, and it stays silent during an actual fire. Tuning this number is a genuine design decision, not just a technical detail — it's the difference between a tool people trust and one they mute.

### 8.5 The new pipeline map
```
extract_data()               → (reused, unchanged)
preprocess_data()            → (reused, unchanged)
forecast_portfolio_horizon() → NEW: Prophet, 30 days ahead, fit ONCE per ticker
   ↓ for t = day 1 .. day 30:
       build that day's forecasted "μ" (expected returns) from the price path
       optimize_portfolio_mean_variance()  → that day's optimal weights (REUSED, unchanged!)
       drift_t = L1_distance(current weights, that day's optimal weights)
   ↓
find first day where drift_t > DRIFT_THRESHOLD  → that's your alert
dashboard: plot drift-over-time, current vs. recommended weights
```

### 8.6 Status: what's built, what's next
| Piece | Status | File |
|---|---|---|
| Multi-day Prophet forecast | ✅ Done | `model_horizon.py` |
| Trading-day calendar helper | ✅ Done | `calendar_utils.py` |
| Settings (horizon, threshold) | ✅ Done | `settings.py` |
| Reused: extractor, processor, optimiser | ✅ Copied over unchanged | `extractor.py`, `processor.py`, `optimiser.py` |
| Daily-optimize loop (30 weight vectors) | 🔲 Next | — |
| Drift score calculation | 🔲 Next | — |
| Trigger/alert logic | 🔲 Next | — |
| Dashboard | 🔲 Later | — |

---

## PART 9: DEBUGGING TOOLKIT — ERRORS YOU WILL DEFINITELY SEE

**How to read a traceback:** read it from the **BOTTOM up**. The last line usually names the actual error; the lines above it show the chain of function calls that led there. Beginners often start reading top-down and panic at the wall of text — the top is just "here's everywhere the error passed through," the bottom is "here's what actually broke."

| Error | What it actually means | Usual fix |
|---|---|---|
| `KeyError: 'AAPL'` | You tried `dict["AAPL"]` but that key doesn't exist | Use `.get("AAPL")` to check safely, or `print(dict.keys())` to see what's actually there |
| `IndexError: list index out of range` | You asked for `list[10]` but the list is shorter than that | Check `len(list)` before indexing, especially after filtering data |
| `AttributeError: 'NoneType' object has no attribute 'X'` | Something that was supposed to be a real object is actually `None` | Trace back to where it was set — a failed extraction or an un-fitted model are common causes here |
| `TypeError: can't compare datetime.date to Timestamp` | You mixed the two date types from Part 2.12 | Convert both sides with `pd.to_datetime(...)` before comparing |
| `ModuleNotFoundError: No module named 'prophet'` | The package isn't installed in this environment | `pip install prophet` (or whatever's missing) — remember `--break-system-packages` if your setup needs it |
| `ValueError: Optimisation failed: ...` | SciPy's `minimize()` couldn't find weights satisfying all constraints | Usually your bounds/constraints conflict — e.g. `MINIMUM_ALLOCATION * num_assets > 1` is mathematically impossible (you can't force everything above 5% if you have 25 assets) |
| `SettingWithCopyWarning` | pandas isn't sure if you meant to edit a copy or the original | Add `.copy()` where you intend to branch off a new DataFrame |
| Prophet/Stan convergence warnings | The underlying optimizer struggled while fitting | Usually harmless for stock data, but if predictions look insane, check for missing or duplicate dates in your input first |
| `ImportError: attempted relative import with no known parent package` | You ran a file directly (`python model_horizon.py`) that uses `from .settings import ...` | Relative imports (the leading `.`) only work when the file is run as part of a package — run it via `python -m src.model_horizon` from one level up instead |

---

## PART 10: GLOSSARY (A–Z)

**Allocation** — see *Weight*.
**Backtest** — testing a strategy against past data to see how it would have performed.
**Bounds** — the allowed min/max range for one variable in an optimization.
**Class** — a blueprint for creating objects.
**Closure** — an inner function that remembers variables from its outer function.
**Constraint** — a rule an optimizer's answer must satisfy exactly.
**Correlation** — how closely two things move together, scaled between -1 and +1.
**Covariance** — how closely two things move together, in raw (unscaled) units.
**`ds`** — Prophet's required date column name.
**DataFrame** — a table of data (rows + columns) in pandas.
**Dict / Dictionary** — a collection of key → value pairs.
**Diversification** — spreading investments across assets to reduce overall risk.
**Drift Score** — the distance between your current portfolio and some future day's optimal portfolio.
**Expected return** — a forecasted or estimated future return; a guess, not a fact.
**f-string** — a Python string with `{}` fill-in-the-blanks.
**Fit** — training a model on historical data.
**Forecast** — a prediction of future values, usually with a confidence range attached.
**Horizon** — how far into the future you're forecasting (e.g. 30 days).
**Index (pandas)** — the row labels of a DataFrame/Series.
**KeyError** — Python error thrown when a dictionary key doesn't exist.
**L1 distance** — sum of absolute differences; "Manhattan distance."
**Lambda (Python)** — an anonymous, one-line function.
**Lambda / λ (finance)** — the risk-aversion coefficient in the Markowitz formula. (Yes, same word, two totally different meanings — context tells you which.)
**List** — an ordered collection of items.
**Markowitz / Mean-Variance optimization** — the classic "maximize return, penalize risk" portfolio math.
**Method chaining** — calling `.method()` repeatedly on the result of the previous call.
**`mu` (μ)** — the expected-returns vector.
**`None`** — Python's explicit "no value here" marker.
**Objective function** — the scorecard an optimizer tries to maximize or minimize.
**Portfolio** — your collection of holdings.
**Rebalancing** — trading to bring actual holdings back to target weights.
**Return (finance)** — the percentage change in price over a period.
**`self`** — refers to the current object inside a class method.
**Seasonality** — a repeating pattern over time (weekly, yearly).
**Series (pandas)** — a single column of data with an index.
**Set** — an unordered collection with no duplicates.
**SLSQP** — the optimization algorithm SciPy uses here ("Sequential Least Squares Programming").
**Ticker** — a stock's short code (e.g. `AAPL`).
**Time series** — data points ordered by time.
**Trading day** — a day the stock market is actually open.
**Trend** — the slow, long-term direction of a time series.
**Type hint** — a note about what type a variable/argument is expected to be.
**Variance** — a mathematical measure of how spread out values are.
**Volatility** — the casual finance term for the same idea as variance/standard deviation.
**Walrus operator (`:=`)** — assign and use a value in the same expression.
**Weight** — the fraction of a portfolio held in one asset.
**`y`** — Prophet's required value-to-predict column.
**`yhat`** — Prophet's predicted value.
**`yhat_lower` / `yhat_upper`** — the bounds of Prophet's confidence interval.

---

## PART 11: FULL SYNTAX QUICK-REFERENCE
*(once the concepts above have clicked, this table is for fast lookups while you're actually typing code)*

| Syntax | Meaning | Example |
|---|---|---|
| `x: str` | Type hint — documents expected type, not enforced | `def f(ticker: str):` |
| `-> dict[str, float]` | Return type hint | `def f() -> dict[str, float]:` |
| `X \| None` | "X or nothing" (Python 3.10+ union syntax) | `def f() -> pd.DataFrame \| None:` |
| `_name` | Convention: "private," internal use only | `_process_ticker_dataframe()` |
| `[expr for x in iterable]` | List comprehension | `[df.index for df in dfs]` |
| `{k: v for k, v in ...}` | Dict comprehension | `{t: w for t, w in zip(tickers, weights)}` |
| `*args` | Unpack a list/tuple into separate positional args | `set.intersection(*date_sets)` |
| `**kwargs` | Unpack a dict into keyword args | `Prophet(**prophet_params)` |
| `x := expr` | Walrus operator — assign and use in one line | `if mapped := d.get(name):` |
| `lambda x: expr` | Anonymous inline function | `lambda w: np.sum(w) - 1` |
| `try / except Exception as e` | Catch errors so one failure doesn't crash everything | see `extractor.py` |
| `zip(a, b, strict=True)` | Pair up two lists elementwise; error if lengths differ | `zip(tickers, weights)` |
| `class X: def __init__(self):` | Blueprint for objects; `__init__` = constructor | `class ProphetModel:` |
| `self` | "This particular instance" — always first arg in a method | `def fit(self, ...):` |
| `return self` | Enables method chaining: `Model().fit(x).predict(y)` | |
| `df["Price"]` | Select one column → Series | |
| `df[["Price","Returns"]]` | Select multiple columns → DataFrame | |
| `df.loc[label]` | Select row(s) by label | `df.loc["2026-01-01"]` |
| `df.iloc[pos]` | Select row(s) by position | `df.iloc[-1]` |
| `df[df.Price > 100]` | Filter rows with a boolean mask | |
| `df.copy()` | Explicit independent copy — avoids `SettingWithCopyWarning` | |
| `pd.concat([a, b])` | Stack DataFrames together | |
| `np.dot(a, b)` | Dot product | `np.dot(weights, mu)` |
| `a.T` | Transpose (flip rows/columns) of an array/matrix | |
| `result.success` / `result.x` | SciPy optimizer's convergence flag / winning answer | |

## Data shape used everywhere
```python
dict[str, pd.DataFrame]
# {"AAPL": <df with Price/Returns columns, date index>, "MSFT": <df>, ...}
```

---

**Last thing:** none of this is meant to be read once and memorized. Treat it like a dictionary — when you hit something in the code that makes you go "wait, what's that," come back here (Ctrl+F is your friend), find it, and get back to building. Understanding clicks a lot faster from real code you're stuck on than from reading straight through a reference doc.
