"""Tests for Module 1 (data pipeline).

Run from the repository root:  python -m pytest tests -v
(plain `python -m unittest discover tests` also works).
"""
import sqlite3
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "data_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from scrape_books import parse_availability  # noqa: E402

CLEAN_CSV = PIPELINE_DIR / "cleaned_books.csv"
DB_FILE = PIPELINE_DIR / "zepto_books.db"
GBP_TO_INR = 105.50


class ParseAvailabilityTests(unittest.TestCase):
    def test_in_stock(self):
        self.assertIs(parse_availability("In stock"), True)
        self.assertIs(parse_availability("  In stock (19 available) "), True)

    def test_out_of_stock(self):
        self.assertIs(parse_availability("Out of stock"), False)

    def test_unexpected_or_missing_returns_none(self):
        self.assertIsNone(parse_availability("Coming soon"))
        self.assertIsNone(parse_availability(None))
        self.assertIsNone(parse_availability(float("nan")))


class CleanedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(CLEAN_CSV)

    def test_meets_size_requirements(self):
        self.assertGreaterEqual(len(self.df), 60)
        self.assertGreaterEqual(self.df["category"].nunique(), 3)

    def test_no_missing_values(self):
        self.assertEqual(int(self.df.isna().sum().sum()), 0)

    def test_ratings_are_integers_between_1_and_5(self):
        self.assertTrue(self.df["rating"].between(1, 5).all())
        self.assertTrue((self.df["rating"] == self.df["rating"].round()).all())

    def test_prices_are_positive_numbers(self):
        self.assertTrue((self.df["price_gbp"] > 0).all())

    def test_inr_conversion_uses_fixed_rate(self):
        expected = self.df["price_gbp"] * GBP_TO_INR
        diff = (self.df["price_inr"] - expected).abs()
        self.assertLess(diff.max(), 0.01)

    def test_in_stock_is_boolean(self):
        self.assertTrue(self.df["in_stock"].isin([True, False]).all())


class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect(DB_FILE)
        cls.conn.execute("PRAGMA foreign_keys = ON;")
        cls.csv = pd.read_csv(CLEAN_CSV)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_expected_tables_exist(self):
        names = {r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"categories", "books"} <= names)

    def test_row_counts_match_csv(self):
        n_books = self.conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
        n_cats = self.conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
        self.assertEqual(n_books, len(self.csv))
        self.assertEqual(n_cats, self.csv["category"].nunique())

    def test_foreign_keys_are_valid(self):
        self.assertEqual(self.conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_sql_join_matches_pandas_merge(self):
        sql_df = pd.read_sql(
            "SELECT b.title, b.price_gbp, c.category_name "
            "FROM books b JOIN categories c ON b.category_id = c.category_id",
            self.conn,
        )
        books = pd.read_sql("SELECT * FROM books", self.conn)
        cats = pd.read_sql("SELECT * FROM categories", self.conn)
        merged = books.merge(cats, on="category_id")[
            ["title", "price_gbp", "category_name"]]
        cols = ["title", "price_gbp", "category_name"]
        a = sql_df.sort_values(cols).reset_index(drop=True)
        b = merged.sort_values(cols).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    unittest.main()
