import os
import re
import pandas as pd
from matplotlib import pyplot as plt

SEARCH_INFLECTIONS = [
    'חיפשתי', 'חיפשת', 'חיפשתם', 'חיפשתן', 'חיפש', 'חיפשה', 'חיפשו',
    'מחפש', 'מחפשת', 'מחפשים', 'מחפשות',
    'נחפש', 'תחפש', 'תחפשי', 'תחפשו', 'נחפש', 'אחפש'
]



def extract_num_rooms(text):
    text = str(text)
    hebrew_num_words = {
        "שניים": 2, "שתיים": 2,
        "שלושה": 3, "שלוש": 3,
        "ארבעה": 4, "ארבע": 4,
        "חמישה": 5, "חמש": 5,
        "שישה": 6, "שש": 6
    }

    # Handle studio and "room and a half" cases
    if re.search(r'חדר\s+וחצי', text):
        return 1.5
    elif re.search(r'דיר[ה|ת]?\s*חדר\b', text):
        return 1.0

    # Combine digit and word matching
    digit_part = r'(\d(?:\.\d)?)'
    room_suffix = r'(?:חדרים|חד[\'״׳]?)'  # Supports חדרים, חד, חד', חד״, חד׳

    # word_part = r'(שניים|שתיים|שלושה|שלוש|ארבעה|ארבע|חמישה|חמש|שישה|שש)'
    # room_suffix = r'(?:חדרים|חד[\'״׳]?)'
    #
    # # Try just this expression:
    # pattern = re.compile(word_part + r'\s*' + room_suffix)
    #
    # match = pattern.search(text)
    # if match:
    #     print("MATCH:", match.group())

    word_part = r'(' + '|'.join(hebrew_num_words.keys()) + ')'

    # Match cases like "דירת 3 חדרים", "דירה שלוש וחצי חדרים", etc.
    pattern = re.compile(
        rf'{digit_part}?(?:\s*וחצי)?\s*{room_suffix}|' +
        rf'{word_part}(?:\s*וחצי)?\s*{room_suffix}|' +
        rf'{digit_part}?\s*{room_suffix}\s*וחצי|' +
        rf'{word_part}\s*{room_suffix}\s*וחצי'
    )

    # pattern = re.compile(
    #     rf'(?:דיר[ה|ת]?\s*){digit_part}(?:\s*וחצי)?\s*{room_suffix}|' +
    #     rf'(?:דיר[ה|ת]?\s*){word_part}(?:\s*וחצי)?\s*{room_suffix}|' +
    #     rf'(?:דיר[ה|ת]?\s*){digit_part}\s*{room_suffix}\s*וחצי|' +
    #     rf'(?:דיר[ה|ת]?\s*){word_part}\s*{room_suffix}\s*וחצי',
    #     re.IGNORECASE
    # )

    match = pattern.search(text)
    if not match:
        return None

    # Find the first non-None match among the groups
    for group in match.groups():
        if group:
            group = group.strip()
            if group in hebrew_num_words:
                base = hebrew_num_words[group]
            else:
                try:
                    base = float(group)
                except ValueError:
                    continue

            if 'וחצי' in text:
                return base + 0.5
            else:
                return base
    return None

def extract_porch(text):
    text = str(text)

    # Normalize whitespace to avoid false negatives due to \n, \r etc.
    text = re.sub(r'\s+', ' ', text)

    # If מרפסת appears preceded by negation
    negation_pattern = re.compile(r'(אין|בלי|ללא|לא כוללת)\s+מרפסת(?:ות)?\b')
    if negation_pattern.search(text):
        return False

    # If מרפסת appears anywhere
    porch_pattern = re.compile(r'\bמרפסת(?:ות)?\b')
    return bool(porch_pattern.search(text))

def extract_sublet(text):
    SUBLET_INFLECTIONS = ['סבלט', 'סאבלט','לסבלט' ,'מסבלט' ,'מסאבלט'
                          'מסבלט', 'מסבלטת','מסאבלטת', 'לסאבלט']

    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    pattern = re.compile(r'\b(?:' + '|'.join(SUBLET_INFLECTIONS) + r')\b')
    return bool(pattern.search(text))

def extract_shabbat_kosher(text):
    SHOMER_INFLECTIONS = ['לשמור', 'שומרות', 'שומרים', 'שומרת', 'שומר']
    SHABBAT_KOSHER_INFLECTIONS = [ 'כשר', 'כשרות', 'שבת']
    NEGATION_PHRASES = ['לא', 'בלי', 'ללא', 'אין','אינו']

    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    base_pattern = rf'(?:{"|".join(SHOMER_INFLECTIONS)})[ \-]?(?:{"|".join(SHABBAT_KOSHER_INFLECTIONS)})'

    # 1. Check if a **negated** version appears
    negation_pattern = re.compile(
        rf'(?:{"|".join(NEGATION_PHRASES)})\s+{base_pattern}',
        re.UNICODE
    )
    if negation_pattern.search(text):
        return False

    # 2. If not negated, check for a **positive** match
    positive_pattern = re.compile(rf'\b{base_pattern}\b', re.UNICODE)
    return bool(positive_pattern.search(text))

def extract_washing_machine(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text)

    negation_pattern = re.compile(r'(אין|בלי|ללא|לא כוללת)\s+מכונת כביסה?\b')
    if negation_pattern.search(text):
        return False

    pattern = re.compile(r'\bמכונת כביסה\b')
    return bool(pattern.search(text))

def extract_dryer(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

    # Negation check
    negation_pattern = re.compile(r'(אין|בלי|ללא|לא כוללת)\s+מייבש?\b')
    if negation_pattern.search(text):
        return False

    # Positive match
    pattern = re.compile(r'\bמייבש\b')
    return bool(pattern.search(text))

def extract_rent_price(text):
    text = str(text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Rent-related keywords
    rent_keywords = r'(שכ(?:ד|\"ד)?|שכר דירה|לחודש|מחיר)'

    # Currency symbols/phrases (can appear before or after number)
    currency = r'(?:₪|ש"ח|שח|שקל(?:ים)?)'

    # Regex to find numbers possibly followed or preceded by currency
    number_pattern = r'(?<!\d)(\d{1,3}(?:,\d{3})?|\d{4})(?!\d)'

    # Full pattern to detect price mentions around rent keywords
    pattern = re.compile(
        rf'({rent_keywords})?[^₪שח\d]{{0,15}}{number_pattern}\s*({currency})?|'
        rf'({currency})\s*{number_pattern}', re.UNICODE
    )

    candidates = []

    for match in pattern.finditer(text):
        # Extract matched number
        nums = match.groups()
        numbers = [n for n in nums if n and re.match(number_pattern, n)]
        for raw in numbers:
            number = int(raw.replace(',', ''))
            if 1200 <= number <= 20000:
                candidates.append(number)

    # Secondary Pass
    dot_pattern = re.compile(r'(?<!\d)(\d{1,2}\.\d{3})(?!\d)')
    for dot_match in dot_pattern.finditer(text):
        raw = dot_match.group(1)
        num = int(raw.replace(".", ""))
        if 1200 <= num <= 20000:
            candidates.append(num)

    return min(candidates) if candidates else None

def extract_no_brokerage(text):
    BROKERAGE_INFLECTIONS = ['תיווך', 'מתווך', 'מתווכים']
    NEGATION_PHRASES = ['ללא', 'בלי', 'לא', 'אין']

    text = str(text)
    text = re.sub(r'\s+', ' ', text)

    # Construct regex for negated brokerage mentions
    negation_pattern = re.compile(
        rf'(?:{"|".join(NEGATION_PHRASES)})\s+(?:{"|".join(BROKERAGE_INFLECTIONS)})',
        re.UNICODE
    )

    if negation_pattern.search(text):
        return True  # explicitly no brokerage

    return False  # no indication, or possibly with brokerage

# # # notes:כניסה\זמינה מידית\מיידית -> post date? / כניסה -> dont confuse with entrnace to the building, יציאה ב -???
# # # keywords: מתפנה, מפנים,תחילת,בתחילת,מוקדם,אמצע,לקראת,סוף,החודש,במהלך,חתימה,חתימת חוזה,חידוש חוזה,החל מה,החל מ,בסוף,
# # # notes: keep a bank of months in hebrew to compare to. look for date format in strings-must be connect to some keywords
# # # what do we do with sublet and then contract?. כניסה גמישה -> ignore
def extract_contract_month(text, post_date):
    # Hebrew month names → numeric
    MONTH_MAP = {
        'ינואר': 1, 'פברואר': 2, 'מרץ': 3, 'אפריל': 4,
        'מאי': 5, 'יוני': 6, 'יולי': 7, 'אוגוסט': 8,
        'ספטמבר': 9, 'אוקטובר': 10, 'נובמבר': 11, 'דצמבר': 12
    }

    CONTRACT_KEYWORDS = [
              'כניסה', 'מתפנה', 'מפנים', 'תחילת', 'בתחילת',
              'לקראת', 'בסוף', 'החל מ', 'החל מה',
               'חידוש חוזה',
               'אפשרות לחידוש חוזה',
                'וחידוש'
        ]

    text = re.sub(r'\s+', ' ', str(text))

    # 0) Immediate entry/contract start → use post_date.month
    immediate_re = re.compile(
                rf'\b(?:הכניסה|כניסה|חתימה|זמינה|לכניסה).{{0,15}}מ?יידית\b',
                re.UNICODE)
    if immediate_re.search(text):
        # assume post_date is a datetime.date or pd.Timestamp
        return post_date.month

    # 1) Numeric dates: e.g. "כניסה ב05/09/2025" or "מתפנה ב05.09.24"
    num_pat = (
        r'(?P<kw>\b(?:' + '|'.join(CONTRACT_KEYWORDS) + r')\b)\s*'
        r'ב[ \t]*[–—-]?[ \t]*' 
        r'(?P<day>\d{1,2})'
        r'[\/\.,](?P<month>\d{1,2})'
        r'(?:[\/\.,]\d{2,4})?[-–—]?[ \t]*'
        r'\b'
    )
    m_num = re.search(num_pat, text)
    if m_num:
        month = int(m_num.group('month'))
        if 1 <= month <= 12:
            return month

    # 2) Month names
    month_names_re = '|'.join(MONTH_MAP.keys())
    for kw in CONTRACT_KEYWORDS:
        name_pat = (
            rf'\b{kw}\s*'              # keyword
            r'(?:חודש\s+)?'            # optional 'חודש'
            r'ב?'                      # optional 'ב'
            rf'(?P<mn>\d{{1,2}}|{month_names_re})\b'
        )
        m_name = re.search(name_pat, text)
        if m_name:
            mn = m_name.group('mn')
            if mn.isdigit():
                mn_i = int(mn)
                if 1 <= mn_i <= 12:
                    return mn_i
            else:
                return MONTH_MAP[mn]

    return None

def extract_post_time_date(df):
    # Convert to datetime
    df["datetime"] = df["date"].str.extract(r'^(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})')
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y_%m_%d__%H_%M_%S")

    # Assign clean date and time
    df["date"] = df["datetime"].dt.date  # -> type: datetime.date
    df["time_of_post"] = df["datetime"].dt.time  # -> type: datetime.time

    # Drop the temp column
    df.drop(columns="datetime", inplace=True)
    return df


def extract_features(df):
    df["num_rooms"] = df["post_text"].apply(extract_num_rooms)
    df["porch"] = df["post_text"].apply(extract_porch)
    df["washing_machine"] = df["post_text"].apply(extract_washing_machine)
    df["dryer"] = df["post_text"].apply(extract_dryer)
    df["rent_price"] = df["post_text"].apply(extract_rent_price)
    df["is_sublet"] = df["post_text"].apply(extract_sublet)
    df["is_shabbat_kosher"] = df["post_text"].apply(extract_shabbat_kosher)
    df["no_brokerage"] = df["post_text"].apply(extract_no_brokerage)
    df= extract_post_time_date(df)
    df["contract_month"] = df.apply(
        lambda row: extract_contract_month(row["post_text"], row["date"]),
        axis=1
    )
    return df

def plot_prices(df):
    rent_values = df["rent_price"].dropna().astype(int)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rent_values, bins=30, edgecolor='black')
    plt.title("Distribution of Rent Prices")
    plt.xlabel("Rent (NIS)")
    plt.ylabel("Number of Posts")
    plt.grid(True)
    plt.tight_layout()

    # Save to file
    output_path = "../../output/feature_extraction/RentPriceHist.png"
    plt.savefig(output_path)
    plt.close()

def filter_problematic_posts(df):
    filtered = df[df["contract_month"].isna() & ~df["search_post"]]
    return filtered

def main():
    df= pd.read_csv("../../output/FacebookApartmentPostsNoDups.csv")
    df = extract_features(df)
    plot_prices(df)
    # for easier debugging
    problematic_df = filter_problematic_posts(df)
    # Ensure the output folder exists
    output_dir = "../../output/feature_extraction"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv("../../output/feature_extraction/PostsWithFeatures.csv", encoding='utf-8-sig')

if __name__ == '__main__':
    main()

