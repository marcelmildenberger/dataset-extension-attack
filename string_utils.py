def extract_two_grams(input_string, remove_spaces=False):
    """
    Generate 2-grams from a string, optionally removing spaces and certain characters.
    Args:
        input_string (str): The input string to process.
        remove_spaces (bool): Whether to remove spaces before generating 2-grams.
    Returns:
        list: List of 2-gram strings.
    """
    chars_to_remove = '"./'
    translation_table = str.maketrans('', '', chars_to_remove)
    cleaned = input_string.translate(translation_table).strip().lower()
    if remove_spaces:
        cleaned = cleaned.replace(' ', '')
    # Generate 2-grams, excluding those containing spaces
    return [cleaned[i:i+2] for i in range(len(cleaned) - 1) if ' ' not in cleaned[i:i+2]]


def format_birthday(date_str):
    """
    Format a date string as MM/DD/YYYY from a string like MMDDYYYY.
    Args:
        date_str (str): Date string in MMDDYYYY format.
    Returns:
        str: Date string in MM/DD/YYYY format.
    """
    return f"{date_str[:2]}/{date_str[2:4]}/{date_str[4:]}"


def process_file(filepath):
    """
    Read a file and return a list of (name, set of 2-grams, count) tuples.
    Args:
        filepath (str): Path to the file.
    Returns:
        list: List of tuples (name, set of 2-grams, count).
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            name, _, count = line.strip().split(',')
            grams = extract_two_grams(name)
            records.append((name.lower(), set(grams), int(count)))
    return records


def lowercase_df(df):
    """
    Lowercase all string columns in a DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with all string columns lowercased.
    """
    return df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)