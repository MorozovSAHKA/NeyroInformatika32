import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def initialize_parameters():
    np.random.seed(42)
    feature_names = ['Feature1', 'Feature2', 'Feature3']
    return feature_names, None, None, None


def generate_data(n_samples=200):
    X = np.random.randn(n_samples, 3)
    y = 2.5 * X[:, 0] + 1.5 * X[:, 1] - 0.8 * X[:, 2] + np.random.randn(n_samples) * 0.5

    feature_names, _, _, _ = initialize_parameters()
    df = pd.DataFrame(X, columns=feature_names)
    df['Target'] = y

    print("\nПервые 5 строк данных:")
    print(df.head())

    return X, y, feature_names, df


def train_model():
    X, y, feature_names, df = generate_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nКоэффициенты: {model.coef_}")
    print(f"Свободный член: {model.intercept_:.4f}")
    print(f"R² score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")

    return model, scaler, X_test, y_test, y_pred, feature_names


def get_user_input(feature_names):
    print("\nВвод данных для прогноза")

    while True:
        try:
            print(f"\nВведите значения для {len(feature_names)} признаков:")
            values = []
            for feature in feature_names:
                value = float(input(f"{feature}: "))
                values.append(value)

            return np.array([values])

        except ValueError:
            print("Ошибка! Введите корректные числовые значения.")
            continue


def predict_from_input(model, scaler, user_data, feature_names):
    if model is None:
        print("Сначала обучите модель!")
        return None

    try:
        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)

        print("\nРезультаты прогноза")
        print(f"\nВведенные данные:")
        for feature, value in zip(feature_names, user_data[0]):
            print(f"{feature}: {value:.4f}")

        print(f"\nПрогнозируемое значение: {prediction[0]:.4f}")

        confidence_interval = 1.96 * 0.5
        lower_bound = prediction[0] - confidence_interval
        upper_bound = prediction[0] + confidence_interval

        print(f"95% доверительный интервал: [{lower_bound:.4f}, {upper_bound:.4f}]")

        return prediction[0]

    except Exception as e:
        print(f"Ошибка при прогнозировании: {e}")
        return None


def show_model_stats(model, X_test, y_test, y_pred, feature_names):
    print("\nСтатистика модели")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Коэффициенты: {model.coef_}")
    print(f"Свободный член: {model.intercept_:.4f}")
    print(f"R² score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")


def visualize_results(model, X_test, y_test, y_pred, feature_names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Фактические значения')
    axes[0].set_ylabel('Предсказанные значения')
    axes[0].set_title('Фактические vs предсказанные значения')

    axes[1].bar(feature_names, model.coef_)
    axes[1].set_xlabel('Признаки')
    axes[1].set_ylabel('Коэффициенты')
    axes[1].set_title('Важность признаков')
    axes[1].axhline(y=0, color='grey', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.show()


def test_with_examples(model, scaler, feature_names):
    examples = [
        [1.0, 0.5, -0.3],
        [-0.5, 1.2, 0.8],
        [0.0, 0.0, 0.0],
        [2.0, -1.0, 0.5]
    ]

    print("\nТестирование на примерных данных")

    for i, example in enumerate(examples, 1):
        print(f"\nПример {i}: {example}")
        user_data = np.array([example])
        predict_from_input(model, scaler, user_data, feature_names)


def interactive_loop():
    print("Интерактивная линейная регрессия")

    feature_names, model, scaler, df = initialize_parameters()
    model, scaler, X_test, y_test, y_pred, feature_names = train_model()

    while True:
        print("\nМеню:")
        print("1. Сделать прогноз")
        print("2. Показать статистику модели")
        print("3. Визуализировать результаты")
        print("4. Протестировать на примерных данных")
        print("5. Переобучить модель")
        print("6. Выйти")

        choice = input("\nВыберите опцию (1-6): ").strip()

        if choice == '1':
            user_data = get_user_input(feature_names)
            predict_from_input(model, scaler, user_data, feature_names)

            cont = input("\nСделать еще один прогноз? (y/n): ").strip().lower()
            if cont != 'y':
                continue

        elif choice == '2':
            show_model_stats(model, X_test, y_test, y_pred, feature_names)

        elif choice == '3':
            visualize_results(model, X_test, y_test, y_pred, feature_names)

        elif choice == '4':
            test_with_examples(model, scaler, feature_names)

        elif choice == '5':
            print("\nПереобучение модели...")
            model, scaler, X_test, y_test, y_pred, feature_names = train_model()

        elif choice == '6':
            print("Выход из программы...")
            break

        else:
            print("Неверный выбор. Пожалуйста, выберите от 1 до 6.")


def main():
    interactive_loop()


if __name__ == "__main__":
    main()
