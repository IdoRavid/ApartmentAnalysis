import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("../../output/FacebookApartmentPostsNoDups.csv", encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'], format="%Y_%m_%d__%H_%M_%S", errors='coerce')

    # Extract datetime features
    df['created_hour'] = df['date'].dt.hour
    df['created_dayofweek'] = df['date'].dt.dayofweek
    df['created_month'] = df['date'].dt.month
    df = df.dropna()
    target_columns = ['reactionsCount']
    y = df[target_columns]
    X = df[['shareCount', 'photo_count', 'created_hour', 'created_dayofweek', 'created_month']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')

    print("Mean Squared Error per target column:")
    for col, err in zip(target_columns, mse):
        print(f"{col}: {err:.4f}")

if __name__ == '__main__':
    main()