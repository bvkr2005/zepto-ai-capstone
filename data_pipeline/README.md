# Module 1 — Data Pipeline

This module implements an end-to-end data-engineering pipeline using Books to Scrape.

The pipeline performs:

* Web scraping using `requests` and `BeautifulSoup`
* Data cleaning using pandas
* GBP to INR conversion
* Normalized SQLite database creation
* SQL querying
* pandas `read_sql()` and `merge()` comparison

## Data Source

Books are scraped from:

`https://books.toscrape.com/`

The implementation scrapes books from at least three categories:

* Fiction
* Fantasy
* Romance

The final dataset contains at least 60 books.

## Project Files

```text
data_pipeline/
├── scrape_books.py
├── database.py
├── sql_queries.py
├── pandas_analysis.py
├── scraped_books.csv
├── cleaned_books.csv
├── zepto_books.db
├── sql_query_outputs.txt
└── README.md
```

## Installation

From the project root, create and activate a Python virtual environment.

Install dependencies using:

```bash
pip install -r requirements.txt
```

The main required Python packages are:

* requests
* beautifulsoup4
* pandas

SQLite support is provided through Python's built-in `sqlite3` module.

## Run Steps

### 1. Scrape, clean, and convert the data

```bash
python data_pipeline/scrape_books.py
```

This script:

1. Scrapes all books from at least three categories.
2. Saves the raw results to `scraped_books.csv`.
3. Converts the scraped price to numeric `price_gbp`.
4. Converts text star ratings to integer `rating` values from 1 to 5.
5. Converts availability text to boolean `in_stock`.
6. Creates `price_inr`.
7. Saves the final cleaned dataset to `cleaned_books.csv`.

## Cleaning Decisions

### Price

The raw book price may contain currency or encoding characters such as:

`£51.77`

The pipeline extracts the numeric portion and converts it to a floating-point value.

Example:

`£51.77 → 51.77`

If a numeric price fails to parse, `pd.to_numeric(..., errors="coerce")` converts it to a missing value. The missing numeric price is then filled using the median valid price.

### Rating

Text ratings are mapped as follows:

* One → 1
* Two → 2
* Three → 3
* Four → 4
* Five → 5

If a rating cannot be parsed, the numeric rating is filled using the median valid rating and converted to an integer.

### Availability

Availability text is converted into a boolean:

* In stock → `True`
* Out of stock → `False`

If availability contains an unexpected value that cannot be interpreted safely, that row is dropped instead of guessing its stock status.

## Currency Conversion

The project uses the required fixed baseline conversion rate:

**1 GBP = 105.50 INR**

The INR value is calculated using:

`price_inr = price_gbp × 105.50`

This is a project-defined fixed conversion rate. No external currency-conversion API is required.

## SQLite Database

Run:

```bash
python data_pipeline/database.py
```

This creates:

`zepto_books.db`

The normalized database contains two tables.

### categories

* `category_id` — Primary Key
* `category_name` — Unique category name

### books

* `book_id` — Primary Key
* `title`
* `price_gbp`
* `price_inr`
* `rating`
* `in_stock`
* `category_id` — Foreign Key referencing `categories.category_id`

This creates a one-to-many relationship where one category can contain many books.

## SQL Queries

Run:

```bash
python data_pipeline/sql_queries.py
```

The SQL queries demonstrate:

* SELECT
* WHERE
* ORDER BY
* LIMIT
* DISTINCT
* BETWEEN
* IN
* JOIN

The query strings and their outputs are saved to:

`sql_query_outputs.txt`

## pandas SQL and Merge Comparison

Run:

```bash
python data_pipeline/pandas_analysis.py
```

This script:

1. Loads at least two SQL query results into pandas using `pd.read_sql()`.
2. Executes a SQL JOIN using `pd.read_sql()`.
3. Loads the `books` and `categories` tables separately.
4. Reproduces the same relationship using `pd.merge()`.
5. Compares the SQL JOIN and pandas merge outputs to verify equivalence.

## Run Order

From the repository root:

```bash
python data_pipeline/scrape_books.py
python data_pipeline/database.py
python data_pipeline/sql_queries.py
python data_pipeline/pandas_analysis.py
```

Running the scripts in this order recreates the Module 1 data pipeline from scratch.
