import string


def extract_bi_grams(input_string, remove_spaces=False):
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


def lowercase_df(df):
    """
    Lowercase all string columns in a DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with all string columns lowercased.
    """
    return df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)

def get_all_bi_grams():
    """
    Generate all possible 2-grams from lowercase letters and digits.
    Returns:
        list: List of all possible 2-gram strings.
    """
    alphabet = string.ascii_lowercase
    digits = string.digits
    letter_letter_grams = [a + b for a in alphabet for b in alphabet]
    digit_digit_grams = [d1 + d2 for d1 in digits for d2 in digits]
    letter_digit_grams = [l + d for l in alphabet for d in digits]
    return letter_letter_grams + letter_digit_grams + digit_digit_grams