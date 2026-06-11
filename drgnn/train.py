from drgnn.model import DRGNN
from drgnn.data import DataControl


def train_pipeline(data_folder: str, save_path: str = './drgnn_model',
                   split: str = 'complex_disease', seed: int = 42,
                   pretrain_epochs: int = 2, finetune_epochs: int = 500,
                   device: str = 'cuda:0'):
    data = DataControl(data_folder=data_folder, split=split, seed=seed)
    data.load_preprocessed(data_folder)

    model = DRGNN(data=data, device=device)
    model.model_initialize()

    model.pretrain(n_epoch=pretrain_epochs)
    model.finetune(n_epoch=finetune_epochs)

    model.save_model(save_path)

    return model
