import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/catalogue/category/books/"

CATEGORIES = {
    "Fiction": "fiction_10",
    "Fantasy": "fantasy_19",
    "Romance": "romance_8",
}

OUTPUT_FILE = Path(__file__).parent / "scraped_books.csv"


def scrape_category(category_name, category_path):
    """
    Scrape all books from one Books to Scrape category.

    Returns:
        list[dict]: Scraped book records.
    """

    books_data = []
    page_number = 1

    while True:

        if page_number == 1:
            url = f"{BASE_URL}{category_path}/index.html"
        else:
            url = f"{BASE_URL}{category_path}/page-{page_number}.html"

        print(f"Scraping: {category_name} - page {page_number}")
        print(f"URL: {url}")

        try:
            response = requests.get(url, timeout=10)

            # Handle HTTP errors explicitly
            response.raise_for_status()

        except requests.RequestException as exc:
            print(f"Request failed for {url}: {exc}")
            break

        soup = BeautifulSoup(response.text, "html.parser")

        books = soup.find_all("article", class_="product_pod")

        if not books:
            print(f"No books found on {url}")
            break

        for book in books:

            title = book.h3.a.get("title", "").strip()

            price_element = book.find("p", class_="price_color")
            price = (
                price_element.get_text(strip=True)
                if price_element
                else None
            )

            rating_element = book.find("p", class_="star-rating")

            star_rating = None

            if rating_element:
                rating_classes = rating_element.get("class", [])

                for rating in ["One", "Two", "Three", "Four", "Five"]:
                    if rating in rating_classes:
                        star_rating = rating
                        break

            availability_element = book.find(
                "p",
                class_="instock availability"
            )

            availability = (
                availability_element.get_text(
                    " ",
                    strip=True
                )
                if availability_element
                else None
            )

            book_record = {
                "title": title,
                "price": price,
                "star_rating": star_rating,
                "availability": availability,
                "category": category_name,
            }

            books_data.append(book_record)

        next_button = soup.find("li", class_="next")

        if next_button is None:
            break

        page_number += 1

        # Small polite delay between requests
        time.sleep(0.5)

    return books_data


def main():

    all_books = []

    for category_name, category_path in CATEGORIES.items():

        category_books = scrape_category(
            category_name,
            category_path
        )

        print(
            f"{category_name}: "
            f"{len(category_books)} books scraped"
        )

        all_books.extend(category_books)

    # Convert scraped books into a DataFrame
    df = pd.DataFrame(all_books)

    # Display scraping results
    print("\n--------------------------------")
    print("SCRAPING SUMMARY")
    print("--------------------------------")

    print(f"Total books scraped: {len(df)}")

    print(
        f"Number of categories: "
        f"{df['category'].nunique()}"
    )

    print("\nBooks by category:")
    print(df["category"].value_counts())

    # STEP 13 - Check at least 60 books
    if len(df) < 60:
        raise ValueError(
            "Requirement failed: fewer than 60 books scraped."
        )

    # STEP 13 - Check at least 3 categories
    if df["category"].nunique() < 3:
        raise ValueError(
            "Requirement failed: fewer than 3 categories scraped."
        )

    # Save the data
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nQ1 requirements passed!")
    print("At least 60 books: PASS")
    print("At least 3 categories: PASS")
    print(f"CSV saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()