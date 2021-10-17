from packages.preprocessing import preprocessing_csv
from packages.predict import predict, save_to_csv


def main():
    file = str(input("Which file do you want to predict?"))
    if not ".csv" in file:
        file += ".csv"

    file_pred = preprocessing_csv(file)
    predictions = predict(file_pred)
    save_to_csv(file_pred, predictions)


if __name__ == "__main__":
    main()
