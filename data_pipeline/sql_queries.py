import sqlite3
from pathlib import Path

import pandas as pd


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

DATABASE_FILE = BASE_DIR / "zepto_books.db"

OUTPUT_FILE = BASE_DIR / "sql_query_outputs.txt"


# =========================================================
# HELPER FUNCTION
# =========================================================

def save_query_output(
    file_handle,
    query_number,
    description,
    query,
    dataframe
):
    """
    Print and save a SQL query and its result.
    """

    section = f"""
========================================
QUERY {query_number} - {description}
========================================

SQL:
{query}

OUTPUT:
{dataframe.to_string(index=False)}

"""

    print(section)

    file_handle.write(section)


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    print("\n========================================")
    print("Q5 - SQL QUERIES")
    print("========================================")

    # Connect to SQLite database
    connection = sqlite3.connect(
        DATABASE_FILE
    )

    # Open output file
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output_file:

        # =================================================
        # QUERY 1
        # SELECT + WHERE
        # =================================================

        query1 = """
        SELECT
            book_id,
            title,
            price_gbp,
            rating
        FROM books
        WHERE rating = 5;
        """

        result1 = pd.read_sql(
            query1,
            connection
        )

        save_query_output(
            output_file,
            1,
            "BOOKS WITH RATING 5",
            query1,
            result1
        )

        # =================================================
        # QUERY 2
        # ORDER BY + LIMIT
        # =================================================

        query2 = """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
        """

        result2 = pd.read_sql(
            query2,
            connection
        )

        save_query_output(
            output_file,
            2,
            "10 MOST EXPENSIVE BOOKS",
            query2,
            result2
        )

        # =================================================
        # QUERY 3
        # DISTINCT
        # =================================================

        query3 = """
        SELECT DISTINCT
            category_name
        FROM categories
        ORDER BY category_name;
        """

        result3 = pd.read_sql(
            query3,
            connection
        )

        save_query_output(
            output_file,
            3,
            "DISTINCT CATEGORIES",
            query3,
            result3
        )

        # =================================================
        # QUERY 4
        # BETWEEN
        # =================================================

        query4 = """
        SELECT
            book_id,
            title,
            price_gbp,
            rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 30
        ORDER BY price_gbp ASC;
        """

        result4 = pd.read_sql(
            query4,
            connection
        )

        save_query_output(
            output_file,
            4,
            "BOOKS PRICED BETWEEN GBP 20 AND GBP 30",
            query4,
            result4
        )

        # =================================================
        # QUERY 5
        # JOIN + ORDER BY + LIMIT
        # =================================================

        query5 = """
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
        ORDER BY b.rating DESC,
                 b.price_gbp DESC
        LIMIT 20;
        """

        result5 = pd.read_sql(
            query5,
            connection
        )

        save_query_output(
            output_file,
            5,
            "BOOKS WITH CATEGORY JOIN",
            query5,
            result5
        )

        # =================================================
        # QUERY 6
        # IN + JOIN
        # =================================================

        query6 = """
        SELECT
            b.title,
            b.price_gbp,
            b.rating,
            c.category_name
        FROM books AS b
        INNER JOIN categories AS c
            ON b.category_id = c.category_id
        WHERE c.category_name
            IN ('Fiction', 'Fantasy')
        ORDER BY
            c.category_name,
            b.title;
        """

        result6 = pd.read_sql(
            query6,
            connection
        )

        save_query_output(
            output_file,
            6,
            "FICTION AND FANTASY BOOKS",
            query6,
            result6
        )

    # Close connection
    connection.close()

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n========================================")
    print("Q5 COMPLETED SUCCESSFULLY")
    print("========================================")

    print("SELECT: PASS")
    print("WHERE: PASS")
    print("ORDER BY: PASS")
    print("LIMIT: PASS")
    print("DISTINCT: PASS")
    print("BETWEEN: PASS")
    print("IN: PASS")
    print("JOIN: PASS")

    print(
        f"\nQuery outputs saved to:\n"
        f"{OUTPUT_FILE}"
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()