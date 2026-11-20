import re

def clean_text(text:str) -> str:
    """clean the extracted PDF text.
    
    Operations performed:
    - Remove null characters.
    - Normalize line endings.
    - Remove unnecessary spaces.
    - Preserve paragraphs separation.
    """

    if not text:
        return ""

    #Remove null characters
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # clean individual lines
    lines = []

    for line in text.split("\n"):
        cleaned_line = re.sub(
            r"[ \t]+",
            " ",
            line
            ).strip()

        lines.append(cleaned_line)
    
    text = "\n" . join(lines)

    # Limit excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()