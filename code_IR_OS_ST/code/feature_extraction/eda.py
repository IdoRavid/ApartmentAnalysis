import pandas as pd
from matplotlib import pyplot as plt
from ydata_profiling import ProfileReport
import seaborn as sns

def main():
    df = pd.read_csv("../../output/geolocating/posts_improved_geolocation.csv", encoding='utf-8-sig')
    # Combine date and time_of_post into one string
    df['datetime_str'] = df['date'].astype(str) + ' ' + df['time_of_post'].astype(str)

    # Parse combined datetime string
    df['datetime'] = pd.to_datetime(df['datetime_str'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

    # Check parsing success
    print(df['datetime'].head())
    print(df['datetime'].isna().sum(), "rows failed to parse")

    # Then extract features
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month

    X = df[['commentCount', 'reactionsCount', 'shareCount', 'group_name', 'photo_count', 'num_rooms', 'porch',
            'washing_machine', 'dryer', 'rent_price', 'is_sublet', 'is_shabbat_kosher', 'no_brokerage',
            'contract_month', 'location_type', 'longitude', 'latitude', 'hour',
            'dayofweek', 'month']]
    # profile = ProfileReport(df, title="Pandas Profiling Report")
    # profile.to_file("../../output/feature_extraction/report.html")  # saves a detailed interactive report

    categorical_cols = [
        'group_name',
        'porch',
        'washing_machine',
        'dryer',
        'is_sublet',
        'is_shabbat_kosher',
        'no_brokerage',
        'location_type'
    ]

    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    corr = X.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", square=True, cbar_kws={"shrink": .8}, annot=False)

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("../../output/feature_extraction/correlation_heatmap.png",dpi=300)
    plt.show()
    # Save to PNG



if __name__ == '__main__':
    main()