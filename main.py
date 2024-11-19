import logging

from snow_classifier.run import run_model, train_model

if __name__ == "__main__":
    logging.getLogger("snow_classifier").setLevel(logging.INFO)

    train_model()
    print(run_model())
