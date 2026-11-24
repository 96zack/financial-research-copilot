import re


def citation_coverage(answer: str) -> float:
    """Fraction of nonempty sentences containing a citation."""

    # TODO 1: Split the answer into sentences.
    # Hint: re.split(r"(?<=[.!?])\s+", answer.strip())
    sen_list = re.split(
        r"(?<=[.!?])\s+(?!\[Source \d+\])",
        answer.strip(),
    )

    # TODO 2: Remove empty sentences.
    non_empty_sen_list = [s for s in sen_list if s]

    # TODO 3: Return 1.0 for an empty answer.
    if not non_empty_sen_list:
        return 1.0

    # TODO 4: Count sentences containing [Source N].
    sen_with_src = []
    for non_empty_sen in non_empty_sen_list:
        if re.search(r"\[Source \d+\]", non_empty_sen):
            sen_with_src.append(non_empty_sen)

    # TODO 5: Return the count divided by total sentences.
    return len(sen_with_src) / len(non_empty_sen_list)


def citation_precision(
    answer: str,
    available_source_numbers: set[int],
) -> float:
    """Fraction of citation occurrences referencing available sources."""

    # TODO 1: Extract all citation numbers using re.findall().
    cit_list = re.findall(r"\[Source (\d+)\]", answer)

    # TODO 2: Return 1.0 if no citations are present.
    if not cit_list:
        return 1.0

    # TODO 3: Count citations that exist in available_source_numbers.
    valid_count = sum(
        int(citation) in available_source_numbers for citation in cit_list
    )

    # TODO 4: Divide valid citation count by total citation count.
    return valid_count / len(cit_list)
