import os
import json
import re

import pandas as pd
import pytz




def contains_hebrew(text):
    """
    Check if text contains any Hebrew characters.
    Hebrew Unicode range: \u0590-\u05FF (includes Hebrew letters, vowels, and punctuation)
    """
    if pd.isna(text) or not isinstance(text, str):
        return False

    # Check for Hebrew letters (main range)
    hebrew_pattern = r'[\u05D0-\u05EA]'
    return bool(re.search(hebrew_pattern, text))


def main():
    output_folder = "../../output/crawler_folder_output"
    print(output_folder)

    df = load_posts_to_df(output_folder)

    df['createdAt'] = pd.to_datetime(df['createdAt'], unit='s', utc=True)
    df['createdAt'] = df['createdAt'].dt.tz_convert(pytz.timezone('Asia/Jerusalem'))

    df = df.drop(columns=[
        'folderNames', "collectionId", "_attachments", "attachmentsDetails",
        "attachedStoryAttachments", "comments", "feedbackId", "storyId"
    ])

    # Extract post text and photo count
    df[['post_text', 'photo_count']] = df['content'].apply(extract_post_parts)
    df['photo_count'] = df['photo_count'].astype('Int64')  # for nullable int

    df = extract_searching(df)[extract_searching(df)["search_post"] == False]

    # Filter posts that contain Hebrew characters
    df['contains_hebrew'] = df['post_text'].apply(contains_hebrew)
    df_hebrew = df[df['contains_hebrew'] == True].copy()

    # Save full cleaned version
    #save_path = '../../output/FacebookApartmentPosts.csv'
    dedup_path = '../../output/FacebookApartmentPostsNoDups.csv'
    #hebrew_path = '../../output/FacebookApartmentPostsHebrew.csv'
    hebrew_dedup_path = '../../output/FacebookApartmentPostsHebrewNoDups.csv'

    #df.to_csv(save_path, index=False, encoding='utf-8-sig')

    # Save deduplicated version
    df_no_dups = df.drop_duplicates(subset='post_text', keep='first')
    df_no_dups.to_csv(dedup_path, index=False, encoding='utf-8-sig')
    print(f"total {df_no_dups.shape[0]} unique posts with apartments")

    # Save Hebrew posts (can be mixed language, just needs some Hebrew)
    #df_hebrew.to_csv(hebrew_path, index=False, encoding='utf-8-sig')
    #print(f"total {df_hebrew.shape[0]} posts containing Hebrew characters")

    # Save deduplicated Hebrew posts
    df_hebrew_no_dups = df_hebrew.drop_duplicates(subset='post_text', keep='first')
    df_hebrew_no_dups.to_csv(hebrew_dedup_path, index=False, encoding='utf-8-sig')
    print(f"total {df_hebrew_no_dups.shape[0]} unique posts containing Hebrew characters")

    # Optional: Remove the helper column if you don't want it in the final files
    # df = df.drop(columns=['contains_hebrew'])
    # df_hebrew = df_hebrew.drop(columns=['contains_hebrew'])


def extract_post_parts(text):
    if not isinstance(text, str):
        return pd.Series(["", 0])

    # Match "with a Facebook Post:", "with a video:", "with 3 photos:", etc.
    match = re.search(
        r'(?s)(.*?)\s*with\s+(?:(\d+)\s+\w+|a\s+(Facebook Post|\w+)):', text
    )
    if match:
        post_text = match.group(1).strip()
        count_str = match.group(2)
        word = match.group(3)
        if count_str:
            photo_count = int(count_str)
        elif word == "Facebook Post":
            photo_count = 0
        else:
            photo_count = 1
    else:
        post_text = text.strip()
        photo_count = 0

    return pd.Series([post_text, photo_count])
def load_posts_to_df(base_dir):
    posts = []

    # Walk through the directory tree
    for group_name in os.listdir(base_dir):
        group_path = os.path.join(base_dir, group_name)
        if not os.path.isdir(group_path):
            continue

        for date_folder in os.listdir(group_path):
            date_path = os.path.join(group_path, date_folder)
            if not os.path.isdir(date_path):
                continue

            post_file_path = os.path.join(date_path, "post.json")
            if os.path.isfile(post_file_path):
                try:
                    with open(post_file_path, 'r', encoding='utf-8-sig') as f:
                        post_data = json.load(f)
                        post_data["group_name"] = group_name
                        post_data["date"] = date_folder
                        posts.append(post_data)
                except Exception as e:
                    print(f"Error reading {post_file_path}: {e}")

    return pd.DataFrame(posts)


SEARCH_INFLECTIONS = [
    'חיפשתי', 'חיפשת', 'חיפשתם', 'חיפשתן', 'חיפש', 'חיפשה', 'חיפשו',
    'מחפש', 'מחפשת', 'מחפשים', 'מחפשות',
    'נחפש', 'תחפש', 'תחפשי', 'תחפשו', 'נחפש', 'אחפש'
]

# Compile a regex pattern to match any inflection

def extract_searching(df):
    search_words = '|'.join(SEARCH_INFLECTIONS)
    target_terms = r'סבלט|להיכנס|להכנס|סאבלט|דירה|דירת|לשכור|להשכרה|להיכנס|חדר|להצטרף'

    # Match: search word followed immediately by target term (separated by space or punctuation)
    pattern = re.compile(
        rf'(?:{search_words})\s*(?:{target_terms})',
        re.UNICODE
    )

    df["search_post"] = df["post_text"].astype(str).apply(lambda x: bool(pattern.search(x)))
    return df


if __name__ == "__main__":
    main()