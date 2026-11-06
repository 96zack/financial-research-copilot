import re

from bs4 import BeautifulSoup


def extract_filing_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove content that isn't part of the filing narrative
    for tag in soup(["script", "style", "ix:header", "ix:hidden", "ix:resources"]):
        tag.decompose()

    # Insert line breaks around block-level elements
    block_tags = [
        "p",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "tr",
        "li",
        "br",
    ]

    for tag in soup.find_all(block_tags):
        tag.insert_before("\n")
        tag.insert_after("\n")

    # Extract text with spaces rather than newlines
    raw_text = soup.get_text(separator=" ")

    # Normalize whitespace within each line
    lines = []

    for line in raw_text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)

        if line:
            lines.append(line)

    return "\n".join(lines)


def remove_page_footers(text: str) -> str:
    pattern = r"^.+? \| \d{4} Form 10-K \| \d+$"

    cleaned_lines = []

    for line in text.splitlines():
        # Check whether the line matches the footer pattern
        if re.fullmatch(pattern, line.strip()):
            continue

        # Keep the line if it is not a footer
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
