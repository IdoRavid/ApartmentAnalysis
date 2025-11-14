import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import matplotlib.dates as mdates
import re
from bidi.algorithm import get_display
import arabic_reshaper

def main():
    # Call the functions
    df = pd.read_csv("../output/FacebookApartmentPostsNoDups.csv")


    plot_reactions_comments_hist(df)
    plot_post_dates_line_by_group(df)
    plot_wordcloud_from_content(df, "..\\utils\\Alef-Regular.ttf", ["4", "5", "6", "7", "8", "9", "1", "a", "ד", "facebook","לא" ,"עם", "או" ,"של","את", "א", "ש", "עד", "יש", "ללא","2", "3" ,"ח"])



# Apply to DataFrame



def fix_rtl(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

def plot_post_dates_line_by_group(df, cutoff_date='2024-07-01'):
    # Parse 'date' column with custom format
    df['datetime'] = pd.to_datetime(df['date'], format="%Y_%m_%d__%H_%M_%S", errors='coerce')
    df = df.dropna(subset=['datetime'])

    # Convert cutoff_date string to Timestamp
    cutoff_date = pd.Timestamp(cutoff_date)

    # Floor to day
    df['only_date'] = df['datetime'].dt.floor('D')

    # Filter out dates before cutoff
    df = df[df['only_date'] >= cutoff_date]

    # Check that 'group_name' exists
    if 'group_name' not in df.columns:
        raise ValueError("The dataframe must contain a 'group_name' column.")

    # Get the unique group names
    group_names = df['group_name'].unique()
    if len(group_names) != 2:
        raise ValueError(f"Expected exactly 2 groups, found {len(group_names)}: {group_names}")

    group1, group2 = group_names
    label1 = fix_rtl(group1)
    label2 = fix_rtl(group2)

    # Count posts per day for each group
    group1_counts = df[df['group_name'] == group1].groupby('only_date').size()
    group2_counts = df[df['group_name'] == group2].groupby('only_date').size()

    # Create a combined date range
    all_dates = pd.date_range(start=df['only_date'].min(), end=df['only_date'].max())
    group1_counts = group1_counts.reindex(all_dates, fill_value=0)
    group2_counts = group2_counts.reindex(all_dates, fill_value=0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(group1_counts.index, group1_counts.values, label=label1, color='blue')
    ax.plot(group2_counts.index, -group2_counts.values, label=label2, color='red')  # negative for below-axis

    # Format the y-axis to show absolute values
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([abs(int(y)) for y in yticks])

    # Labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Posts')
    ax.set_title('Daily Post Counts by Group (Split Above/Below Axis)')
    ax.legend()

    # X-axis formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig("../output/milestone/post_timestamp_line_graph.png")
    plt.show()







# 1. Histogram of reactionsCount and commentCount
def plot_reactions_comments_hist(df):
    plt.figure(figsize=(10, 6))
    plt.hist([df['reactionsCount'], df['commentCount']],
             label=['Reactions', 'Comments'],
             color=['skyblue', 'salmon'], bins=100)
    plt.legend()
    plt.title('Histogram of Reactions and Comments Count')
    plt.xlabel('Count')
    plt.ylabel('Log Frequency')
    plt.yscale('log')  # Use log scale for y-axis
    plt.xlim(0, 100)    # Limit x-axis
    plt.grid(True)
    plt.savefig("../output/milestone/reactionsHist.png")
    plt.show()

# 3. Word cloud of 30 most frequent words in content
def plot_wordcloud_from_content(df, font_path, remove_words=None, amount=40):
    if remove_words is None:
        remove_words = set()
    else:
        remove_words = set(remove_words)  # convert list to set for faster lookup

    text = ' '.join(df['post_text'].dropna().astype(str).tolist())
    words = re.findall(r'\b\w+\b', text.lower())
    freq = Counter(words).most_common()  # get all frequencies

    # Filter out unwanted words
    freq_filtered = [(word, count) for word, count in freq if word not in remove_words]

    # Take top `amount` after filtering
    freq_filtered = freq_filtered[:amount]

    # Reshape and reorder for RTL if needed (example for Hebrew)
    reshaped_freq = {}
    for word, count in freq_filtered:
        reshaped_word = arabic_reshaper.reshape(word)
        bidi_word = get_display(reshaped_word)
        reshaped_freq[bidi_word] = count

    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate_from_frequencies(reshaped_freq)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Top {amount} Frequent Words in Content')
    plt.savefig("../output/milestone/frequentWordsCloud.png")
    plt.show()
if __name__ == "__main__":
    main()


