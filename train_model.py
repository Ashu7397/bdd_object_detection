from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from model import ObjectDetectionModel, get_pretrained_model
from data import BDDObjectDetectionDataset, custom_collate_fn

from config import DATA_ROOT
from loguru import logger as LOGGER


def train_function():
    train_dataset = BDDObjectDetectionDataset(DATA_ROOT, split='train')
    val_dataset = BDDObjectDetectionDataset(DATA_ROOT, split='val')

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=4, num_workers=0, collate_fn=custom_collate_fn, persistent_workers=False)

    num_classes_bdd100k = 11  # 10 object classes + background
    pretrained_model = get_pretrained_model(num_classes_bdd100k)
    lightning_module = ObjectDetectionModel(pretrained_model, learning_rate=1e-3, num_classes=num_classes_bdd100k)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="bdd100k-{epoch:02d}-{val/loss:.2f}",
        save_top_k=1,
        monitor="val/loss",
        mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min"
    )

    logger = TensorBoardLogger("lightning_logs", name="bdd100k_object_detection")

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
    )

    # --- 6. Training ---
    LOGGER.info('Starting training')
    trainer.fit(lightning_module, train_dataloader, val_dataloader)

    LOGGER.info('Training completed')


if __name__ == "__main__":
    train_function()