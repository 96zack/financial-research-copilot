from financial_research_copilot.processing.sentence_chunker import (
    split_long_paragraph,
    split_long_sentence,
)

print(split_long_sentence("Apple relies on suppliers", 15))
# ["Apple relies on", "suppliers"]

print(split_long_sentence("ABCDEFGHIJ", 4))
# ["ABCD", "EFGH", "IJ"]

print(split_long_sentence("Apple ABCDEFGHIJ", 5))
# ["Apple", "ABCDE", "FGHIJ"]

print(split_long_sentence("", 10))
# []

text = (
    "Apple relies on suppliers. "
    "Supply shortages increase costs. "
    "Margins may decline."
)

print(split_long_paragraph(text, chunk_size=55))