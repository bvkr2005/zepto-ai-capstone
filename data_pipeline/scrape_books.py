import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


# =========================================================
# CONFIGURATION
# =========================================================

BASE_URL = "https://books.toscrape.com/catalogue/category/books/"

# At least 3 categories are required
CATEGORIES = {
    "Fiction": "fiction_10",
    "Fantasy": "fantasy_19",
    "Romance": "romance_8",
}

# Required fixed project conversion rate
GBP_TO_INR_RATE = 105.50

# Output files
RAW_OUTPUT_FILE = Path(__file__).parent / "scraped_books.csv"
CLEAN_OUTPUT_FILE = Path(__file__).parent / "cleaned_books.csv"


# =========================================================
# FUNCTION 1: SCRAPE ONE CATEGORY
# =========================================================

def scrape_category(category_name, category_path):
    """
    Scrape all books from one Books to Scrape category.

    Captures:
    - title
    - price
    - star_rating
    - availability
    - category
    """

    books_data = []
    page_number = 1

    while True:

        # First page
        if page_number == 1:
            url = f"{BASE_URL}{category_path}/index.html"

        # Page 2, 3, 4...
        else:
            url = f"{BASE_URL}{category_path}/page-{page_number}.html"

        print(f"\nScraping {category_name} - Page {page_number}")
        print(f"URL: {url}")

        # -------------------------------------------------
        # REQUEST PAGE
        # -------------------------------------------------

        try:
            response = requests.get(
                url,
                timeout=10
            )

            response.raise_for_status()

            # Helps prevent price encoding issues
            response.encoding = "utf-8"

        except requests.RequestException as exc:
            print(
                f"Request failed for {url}: {exc}"
            )
            break

        # -------------------------------------------------
        # PARSE PAGE
        # -------------------------------------------------

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        books = soup.find_all(
            "article",
            class_="product_pod"
        )

        if not books:
            print(
                f"No books found on {url}"
            )
            break

        # -------------------------------------------------
        # EXTRACT EACH BOOK
        # -------------------------------------------------

        for book in books:

            # TITLE
            title = book.h3.a.get(
                "title",
                ""
            ).strip()

            # PRICE
            price_element = book.find(
                "p",
                class_="price_color"
            )

            if price_element:
                price = price_element.get_text(
                    strip=True
                )
            else:
                price = None

            # STAR RATING
            rating_element = book.find(
                "p",
                class_="star-rating"
            )

            star_rating = None

            if rating_element:

                rating_classes = rating_element.get(
                    "class",
                    []
                )

                for rating_text in [
                    "One",
                    "Two",
                    "Three",
                    "Four",
                    "Five"
                ]:

                    if rating_text in rating_classes:
                        star_rating = rating_text
                        break

            # AVAILABILITY
            availability_element = book.find(
                "p",
                class_="instock availability"
            )

            if availability_element:
                availability = availability_element.get_text(
                    " ",
                    strip=True
                )
            else:
                availability = None

            # CREATE RECORD
            book_record = {
                "title": title,
                "price": price,
                "star_rating": star_rating,
                "availability": availability,
                "category": category_name
            }

            books_data.append(book_record)

        # -------------------------------------------------
        # PAGINATION
        # -------------------------------------------------

        next_button = soup.find(
            "li",
            class_="next"
        )

        if next_button is None:
            break

        page_number += 1

        # Polite delay between requests
        time.sleep(0.5)

    return books_data


# =========================================================
# FUNCTION 2: PARSE AVAILABILITY
# =========================================================

def parse_availability(value):
    """
    Convert availability text into boolean.

    In stock     -> True
    Out of stock -> False
    Unexpected   -> None
    """

    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if "in stock" in text:
        return True

    if "out of stock" in text:
        return False

    return None


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    print("\n========================================")
    print("ZEPTO DATA PIPELINE")
    print("Q1 - SCRAPING")
    print("Q2 - DATA CLEANING")
    print("Q3 - CURRENCY CONVERSION")
    print("========================================")

    all_books = []

    # =====================================================
    # Q1 - SCRAPING
    # =====================================================

    for category_name, category_path in CATEGORIES.items():

        category_books = scrape_category(
            category_name,
            category_path
        )

        print(
            f"\n{category_name}: "
            f"{len(category_books)} books scraped"
        )

        all_books.extend(category_books)

    # -----------------------------------------------------
    # CREATE RAW DATAFRAME
    # -----------------------------------------------------

    raw_df = pd.DataFrame(all_books)

    # -----------------------------------------------------
    # Q1 SUMMARY
    # -----------------------------------------------------

    print("\n========================================")
    print("Q1 - SCRAPING SUMMARY")
    print("========================================")

    print(
        f"\nTotal books scraped: "
        f"{len(raw_df)}"
    )

    print(
        f"Number of categories: "
        f"{raw_df['category'].nunique()}"
    )

    print("\nBooks by category:")

    print(
        raw_df["category"].value_counts()
    )

    print("\nFirst 10 raw records:")

    print(
        raw_df.head(10).to_string(
            index=False
        )
    )

    print("\nMissing values in raw data:")

    print(
        raw_df.isnull().sum()
    )

    # -----------------------------------------------------
    # Q1 VALIDATION
    # -----------------------------------------------------

    if len(raw_df) < 60:
        raise ValueError(
            "Q1 failed: fewer than 60 books were scraped."
        )

    if raw_df["category"].nunique() < 3:
        raise ValueError(
            "Q1 failed: fewer than 3 categories were scraped."
        )

    print("\nQ1 requirements passed!")
    print("At least 60 books: PASS")
    print("At least 3 categories: PASS")

    # -----------------------------------------------------
    # SAVE RAW DATA
    # -----------------------------------------------------

    raw_df.to_csv(
        RAW_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nRaw data saved to:\n"
        f"{RAW_OUTPUT_FILE}"
    )

    # =====================================================
    # Q2 - DATA CLEANING
    # =====================================================

    print("\n========================================")
    print("Q2 - DATA CLEANING")
    print("========================================")

    # Work on a copy
    df = raw_df.copy()

    # =====================================================
    # Q2 STEP 1 - CLEAN PRICE
    # =====================================================

    print("\nCleaning price...")

    # Extract numeric portion of price
    #
    # £51.77 -> 51.77
    # Â£51.77 -> 51.77
    # GBP 51.77 -> 51.77
    df["price_gbp"] = (
        df["price"]
        .astype(str)
        .str.extract(
            r"(\d+(?:\.\d+)?)",
            expand=False
        )
    )

    # Convert to float
    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    invalid_price_count = (
        df["price_gbp"]
        .isna()
        .sum()
    )

    print(
        f"Invalid price values found: "
        f"{invalid_price_count}"
    )

    # Median imputation for invalid numeric prices
    price_median = (
        df["price_gbp"]
        .median()
    )

    if pd.isna(price_median):
        raise ValueError(
            "Q2 failed: no valid price values "
            "were available."
        )

    print(
        f"Median price used for imputation: "
        f"{price_median:.2f}"
    )

    df["price_gbp"] = (
        df["price_gbp"]
        .fillna(price_median)
    )

    print(
        "Missing price_gbp values after imputation:",
        df["price_gbp"].isna().sum()
    )

    # =====================================================
    # Q2 STEP 2 - CLEAN RATING
    # =====================================================

    print("\nCleaning star ratings...")

    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    df["rating"] = (
        df["star_rating"]
        .map(rating_map)
    )

    invalid_rating_count = (
        df["rating"]
        .isna()
        .sum()
    )

    print(
        f"Invalid rating values found: "
        f"{invalid_rating_count}"
    )

    # Median imputation
    rating_median = (
        df["rating"]
        .median()
    )

    if pd.isna(rating_median):
        raise ValueError(
            "Q2 failed: no valid rating values "
            "were available."
        )

    print(
        f"Median rating used for imputation: "
        f"{rating_median}"
    )

    df["rating"] = (
        df["rating"]
        .fillna(rating_median)
    )

    # Ensure integer
    df["rating"] = (
        df["rating"]
        .round()
        .astype(int)
    )

    # =====================================================
    # Q2 STEP 3 - CLEAN AVAILABILITY
    # =====================================================

    print("\nCleaning availability...")

    df["in_stock"] = (
        df["availability"]
        .apply(parse_availability)
    )

    invalid_availability_count = (
        df["in_stock"]
        .isna()
        .sum()
    )

    print(
        f"Invalid availability values found: "
        f"{invalid_availability_count}"
    )

    # For unexpected availability text,
    # drop the row instead of guessing.
    before_drop = len(df)

    df = (
        df
        .dropna(
            subset=["in_stock"]
        )
        .copy()
    )

    after_drop = len(df)

    rows_dropped = (
        before_drop - after_drop
    )

    print(
        f"Rows dropped due to invalid availability: "
        f"{rows_dropped}"
    )

    # Ensure boolean type
    df["in_stock"] = (
        df["in_stock"]
        .astype(bool)
    )

    # =====================================================
    # Q2 VALIDATION
    # =====================================================

    print("\n========================================")
    print("Q2 - VALIDATION")
    print("========================================")

    if df["price_gbp"].isna().any():
        raise ValueError(
            "Q2 failed: missing price_gbp values."
        )

    if df["rating"].isna().any():
        raise ValueError(
            "Q2 failed: missing rating values."
        )

    if not df["rating"].between(
        1,
        5
    ).all():
        raise ValueError(
            "Q2 failed: rating values must "
            "be between 1 and 5."
        )

    if df["in_stock"].isna().any():
        raise ValueError(
            "Q2 failed: missing in_stock values."
        )

    if len(df) < 60:
        raise ValueError(
            "Q2 failed: fewer than 60 books "
            "remain after cleaning."
        )

    if df["category"].nunique() < 3:
        raise ValueError(
            "Q2 failed: fewer than 3 categories "
            "remain after cleaning."
        )

    print("\nCleaned data types:")

    print(
        df[
            [
                "price_gbp",
                "rating",
                "in_stock"
            ]
        ].dtypes
    )

    print("\nMissing values after cleaning:")

    print(
        df[
            [
                "price_gbp",
                "rating",
                "in_stock"
            ]
        ]
        .isnull()
        .sum()
    )

    print("\nCleaned data preview:")

    print(
        df[
            [
                "title",
                "price_gbp",
                "rating",
                "in_stock",
                "category"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print("\nQ2 requirements passed!")
    print("price_gbp is numeric: PASS")
    print("rating is integer 1-5: PASS")
    print("in_stock is boolean: PASS")

    # =====================================================
    # Q3 - CURRENCY CONVERSION
    # =====================================================

    print("\n========================================")
    print("Q3 - CURRENCY CONVERSION")
    print("========================================")

    print(
        f"\nUsing fixed project-defined rate: "
        f"1 GBP = {GBP_TO_INR_RATE:.2f} INR"
    )

    # -----------------------------------------------------
    # CREATE price_inr COLUMN
    # -----------------------------------------------------

    df["price_inr"] = (
        df["price_gbp"]
        * GBP_TO_INR_RATE
    ).round(2)

    # -----------------------------------------------------
    # Q3 VALIDATION
    # -----------------------------------------------------

    if df["price_inr"].isna().any():
        raise ValueError(
            "Q3 failed: missing price_inr values."
        )

    if (df["price_inr"] < 0).any():
        raise ValueError(
            "Q3 failed: negative price_inr values found."
        )

    # Calculate expected values again for verification
    expected_price_inr = (
        df["price_gbp"]
        * GBP_TO_INR_RATE
    ).round(2)

    if not df["price_inr"].equals(
        expected_price_inr
    ):
        raise ValueError(
            "Q3 failed: incorrect currency conversion."
        )

    # -----------------------------------------------------
    # PRINT Q3 OUTPUT
    # -----------------------------------------------------

    print("\nCurrency conversion preview:")

    print(
        df[
            [
                "title",
                "price_gbp",
                "price_inr",
                "category"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print("\nQ3 requirements passed!")
    print(
        "Fixed conversion rate: "
        "1 GBP = 105.50 INR"
    )
    print(
        "price_inr column created: PASS"
    )

    # =====================================================
    # SAVE FINAL CLEANED DATA
    # =====================================================

    # IMPORTANT:
    # Save AFTER Q3 so price_inr is included.
    df.to_csv(
        CLEAN_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nFinal cleaned data saved to:\n"
        f"{CLEAN_OUTPUT_FILE}"
    )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print("\n========================================")
    print("FINAL DATA COLUMNS")
    print("========================================")

    print(
        df.columns.tolist()
    )

    print("\n========================================")
    print("FINAL DATA PREVIEW")
    print("========================================")

    print(
        df[
            [
                "title",
                "price_gbp",
                "rating",
                "in_stock",
                "price_inr",
                "category"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print("\n========================================")
    print("Q1 + Q2 + Q3 COMPLETED SUCCESSFULLY")
    print("========================================")

    print(
        f"Final cleaned rows: "
        f"{len(df)}"
    )

    print(
        f"Final categories: "
        f"{df['category'].nunique()}"
    )

    print("Q1 - >= 60 books: PASS")
    print("Q1 - >= 3 categories: PASS")
    print("Q2 - price_gbp float: PASS")
    print("Q2 - rating integer 1-5: PASS")
    print("Q2 - in_stock boolean: PASS")
    print("Q3 - price_inr created: PASS")
    print(
        "Q3 - 1 GBP = 105.50 INR: PASS"
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()