import sqlite3
from pathlib import Path

import pandas as pd


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

CLEAN_DATA_FILE = BASE_DIR / "cleaned_books.csv"

DATABASE_FILE = BASE_DIR / "zepto_books.db"


# =========================================================
# CREATE DATABASE
# =========================================================

def create_database():

    print("\n========================================")
    print("Q4 - SQLITE DATABASE")
    print("========================================")

    # -----------------------------------------------------
    # READ CLEANED CSV
    # -----------------------------------------------------

    print("\nReading cleaned data...")

    df = pd.read_csv(
        CLEAN_DATA_FILE
    )

    print(
        f"Rows loaded from cleaned_books.csv: "
        f"{len(df)}"
    )

    print(
        f"Categories found: "
        f"{df['category'].nunique()}"
    )

    # -----------------------------------------------------
    # CONNECT TO SQLITE
    # -----------------------------------------------------

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    # Enable foreign-key enforcement in SQLite
    cursor.execute(
        "PRAGMA foreign_keys = ON;"
    )

    # -----------------------------------------------------
    # DROP OLD TABLES
    # -----------------------------------------------------

    # Drop books first because it depends on categories
    cursor.execute(
        "DROP TABLE IF EXISTS books;"
    )

    cursor.execute(
        "DROP TABLE IF EXISTS categories;"
    )

    # -----------------------------------------------------
    # CREATE CATEGORIES TABLE
    # -----------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        );
        """
    )

    print(
        "\nCreated table: categories"
    )

    # -----------------------------------------------------
    # CREATE BOOKS TABLE
    # -----------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,

            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        );
        """
    )

    print(
        "Created table: books"
    )

    # -----------------------------------------------------
    # INSERT UNIQUE CATEGORIES
    # -----------------------------------------------------

    print(
        "\nInserting categories..."
    )

    categories = sorted(
        df["category"]
        .dropna()
        .unique()
    )

    for category_name in categories:

        cursor.execute(
            """
            INSERT INTO categories (
                category_name
            )
            VALUES (?);
            """,
            (category_name,)
        )

    connection.commit()

    # -----------------------------------------------------
    # DISPLAY CATEGORIES
    # -----------------------------------------------------

    cursor.execute(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        ORDER BY category_id;
        """
    )

    category_rows = cursor.fetchall()

    print("\nCategories inserted:")

    for row in category_rows:
        print(row)

    # -----------------------------------------------------
    # CREATE CATEGORY LOOKUP DICTIONARY
    # -----------------------------------------------------

    category_lookup = {
        category_name: category_id
        for category_id, category_name
        in category_rows
    }

    print(
        "\nCategory ID mapping:"
    )

    print(
        category_lookup
    )

    # -----------------------------------------------------
    # INSERT BOOKS
    # -----------------------------------------------------

    print(
        "\nInserting books..."
    )

    for _, row in df.iterrows():

        category_id = category_lookup[
            row["category"]
        ]

        # Convert bool-like CSV value safely
        in_stock_value = str(
            row["in_stock"]
        ).strip().lower()

        if in_stock_value == "true":
            in_stock = 1
        else:
            in_stock = 0

        cursor.execute(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                in_stock,
                category_id
            )
        )

    connection.commit()

    # -----------------------------------------------------
    # VERIFY RECORD COUNTS
    # -----------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM categories;
        """
    )

    category_count = (
        cursor.fetchone()[0]
    )

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM books;
        """
    )

    book_count = (
        cursor.fetchone()[0]
    )

    print(
        "\n========================================"
    )

    print(
        "DATABASE VALIDATION"
    )

    print(
        "========================================"
    )

    print(
        f"Categories in database: "
        f"{category_count}"
    )

    print(
        f"Books in database: "
        f"{book_count}"
    )

    # -----------------------------------------------------
    # VALIDATION CHECKS
    # -----------------------------------------------------

    if category_count < 3:

        connection.close()

        raise ValueError(
            "Q4 failed: fewer than "
            "3 categories were inserted."
        )

    if book_count < 60:

        connection.close()

        raise ValueError(
            "Q4 failed: fewer than "
            "60 books were inserted."
        )

    # -----------------------------------------------------
    # TEST THE FOREIGN KEY RELATIONSHIP WITH A JOIN
    # -----------------------------------------------------

    cursor.execute(
        """
        SELECT
            b.book_id,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            c.category_name
        FROM books AS b
        INNER JOIN categories AS c
            ON b.category_id = c.category_id
        LIMIT 10;
        """
    )

    sample_rows = cursor.fetchall()

    print(
        "\nSample books using JOIN:"
    )

    for row in sample_rows:
        print(row)

    # -----------------------------------------------------
    # DISPLAY TABLE SCHEMA
    # -----------------------------------------------------

    print(
        "\nCategories table structure:"
    )

    cursor.execute(
        "PRAGMA table_info(categories);"
    )

    for row in cursor.fetchall():
        print(row)

    print(
        "\nBooks table structure:"
    )

    cursor.execute(
        "PRAGMA table_info(books);"
    )

    for row in cursor.fetchall():
        print(row)

    # -----------------------------------------------------
    # CLOSE CONNECTION
    # -----------------------------------------------------

    connection.close()

    print(
        "\n========================================"
    )

    print(
        "Q4 COMPLETED SUCCESSFULLY"
    )

    print(
        "========================================"
    )

    print(
        "categories table created: PASS"
    )

    print(
        "books table created: PASS"
    )

    print(
        "Primary keys created: PASS"
    )

    print(
        "Foreign key relationship created: PASS"
    )

    print(
        "Normalized database created: PASS"
    )

    print(
        f"\nSQLite database saved to:\n"
        f"{DATABASE_FILE}"
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    create_database()