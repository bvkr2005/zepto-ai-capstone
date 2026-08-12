import sqlite3
from pathlib import Path

import pandas as pd


# =========================================================
# FILE PATH
# =========================================================

BASE_DIR = Path(__file__).parent
DATABASE_FILE = BASE_DIR / "zepto_books.db"


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    print("\n========================================")
    print("Q6 - PANDAS SQL + MERGE COMPARISON")
    print("========================================")

    # Connect to SQLite database
    connection = sqlite3.connect(
        DATABASE_FILE
    )

    # =====================================================
    # PART 1 - READ SQL QUERY RESULT INTO PANDAS
    # =====================================================

    print("\n========================================")
    print("PART 1 - pd.read_sql QUERY 1")
    print("========================================")

    query1 = """
    SELECT
        book_id,
        title,
        price_gbp,
        rating
    FROM books
    WHERE rating = 5
    ORDER BY price_gbp DESC
    LIMIT 10;
    """

    df_query1 = pd.read_sql(
        query1,
        connection
    )

    print(
        df_query1.to_string(
            index=False
        )
    )

    # =====================================================
    # PART 2 - SECOND SQL QUERY INTO PANDAS
    # =====================================================

    print("\n========================================")
    print("PART 2 - pd.read_sql QUERY 2")
    print("========================================")

    query2 = """
    SELECT
        book_id,
        title,
        price_gbp,
        price_inr,
        rating
    FROM books
    WHERE price_gbp BETWEEN 20 AND 30
    ORDER BY price_gbp ASC
    LIMIT 10;
    """

    df_query2 = pd.read_sql(
        query2,
        connection
    )

    print(
        df_query2.to_string(
            index=False
        )
    )

    # =====================================================
    # PART 3 - SQL JOIN USING pd.read_sql
    # =====================================================

    print("\n========================================")
    print("PART 3 - SQL JOIN USING pd.read_sql")
    print("========================================")

    join_query = """
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
    ORDER BY
        b.book_id;
    """

    sql_join_df = pd.read_sql(
        join_query,
        connection
    )

    print(
        sql_join_df.head(10).to_string(
            index=False
        )
    )

    # =====================================================
    # PART 4 - READ TABLES SEPARATELY
    # =====================================================

    print("\n========================================")
    print("PART 4 - READ TABLES INTO PANDAS")
    print("========================================")

    books_df = pd.read_sql(
        """
        SELECT *
        FROM books;
        """,
        connection
    )

    categories_df = pd.read_sql(
        """
        SELECT *
        FROM categories;
        """,
        connection
    )

    print(
        f"\nBooks rows: "
        f"{len(books_df)}"
    )

    print(
        f"Categories rows: "
        f"{len(categories_df)}"
    )

    # =====================================================
    # PART 5 - PANDAS MERGE
    # =====================================================

    print("\n========================================")
    print("PART 5 - JOIN USING pd.merge")
    print("========================================")

    pandas_join_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    # Select same columns as SQL JOIN
    pandas_join_df = pandas_join_df[
        [
            "book_id",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name"
        ]
    ]

    # Sort exactly the same way as SQL
    pandas_join_df = (
        pandas_join_df
        .sort_values(
            by="book_id"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        pandas_join_df.head(10).to_string(
            index=False
        )
    )

    # =====================================================
    # PART 6 - PREPARE SQL RESULT FOR COMPARISON
    # =====================================================

    sql_join_df = (
        sql_join_df
        .reset_index(
            drop=True
        )
    )

    # =====================================================
    # PART 7 - COMPARE RESULTS
    # =====================================================

    print("\n========================================")
    print("PART 6 - COMPARE SQL JOIN VS pd.merge")
    print("========================================")

    same_shape = (
        sql_join_df.shape
        ==
        pandas_join_df.shape
    )

    print(
        f"\nSame number of rows and columns: "
        f"{same_shape}"
    )

    same_columns = (
        list(sql_join_df.columns)
        ==
        list(pandas_join_df.columns)
    )

    print(
        f"Same column names/order: "
        f"{same_columns}"
    )

    # Compare actual values
    equivalent = sql_join_df.equals(
        pandas_join_df
    )

    print(
        f"Exact DataFrame equality: "
        f"{equivalent}"
    )

    # =====================================================
    # OPTIONAL DIAGNOSTIC
    # =====================================================

    if not equivalent:

        print(
            "\nExact equality returned False."
        )

        print(
            "Checking values while ignoring "
            "possible dtype differences..."
        )

        try:

            pd.testing.assert_frame_equal(
                sql_join_df,
                pandas_join_df,
                check_dtype=False
            )

            equivalent = True

            print(
                "Values match; only dtype "
                "differences were present."
            )

        except AssertionError as exc:

            print(
                "\nThe results do not match:"
            )

            print(exc)

    # =====================================================
    # SIDE-BY-SIDE PREVIEW
    # =====================================================

    print("\n========================================")
    print("SIDE-BY-SIDE COMPARISON")
    print("========================================")

    sql_preview = (
        sql_join_df
        .head(10)
        .add_prefix("SQL_")
    )

    pandas_preview = (
        pandas_join_df
        .head(10)
        .add_prefix("PANDAS_")
    )

    side_by_side = pd.concat(
        [
            sql_preview,
            pandas_preview
        ],
        axis=1
    )

    print(
        side_by_side.to_string(
            index=False
        )
    )

    # =====================================================
    # FINAL VALIDATION
    # =====================================================

    if not same_shape:
        raise ValueError(
            "Q6 failed: SQL JOIN and pandas merge "
            "have different shapes."
        )

    if not same_columns:
        raise ValueError(
            "Q6 failed: SQL JOIN and pandas merge "
            "have different columns."
        )

    if not equivalent:
        raise ValueError(
            "Q6 failed: SQL JOIN and pandas merge "
            "results do not match."
        )

    # Close database connection
    connection.close()

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n========================================")
    print("Q6 COMPLETED SUCCESSFULLY")
    print("========================================")

    print(
        "pd.read_sql result 1 created: PASS"
    )

    print(
        "pd.read_sql result 2 created: PASS"
    )

    print(
        "SQL JOIN loaded with pd.read_sql: PASS"
    )

    print(
        "JOIN reproduced with pd.merge: PASS"
    )

    print(
        "SQL JOIN and pd.merge outputs match: PASS"
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()