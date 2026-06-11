import pandas as pd

from drgnn.model import DRGNN


def predict_drug_disease(model: DRGNN, df_test: pd.DataFrame):
    return model.predict(df_test)


def load_and_predict(model_path: str, df_test: pd.DataFrame,
                     device: str = 'cuda:0'):
    model = DRGNN(data=None, device=device)
    model.load_pretrained(model_path)
    return model.predict(df_test)
