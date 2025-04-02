from dataloader import get_loaders
import pytorch_lightning as pl
import config
from model import HybridSegmentor
import torch
from callback import MyPrintingCallBack, checkpoint_callback, early_stopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    # 1. Logger cho TensorBoard
    logger = TensorBoardLogger("tb_logs", name="v7_BCEDICE_0_2_final")

    # 2. Load data
    train_loader, val_loader, test_loader = get_loaders(
        config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR, config.VAL_MASK_DIR,
        config.TEST_IMG_DIR, config.TEST_MASK_DIR,
        config.BATCH_SIZE, config.NUM_WORKERS, config.PIN_MEMORY
    )

    # 3. Khởi tạo mô hình
    model = HybridSegmentor(learning_rate=config.LEARNING_RATE).to(config.DEVICE)

    # 4. Trainer
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=config.NUM_EPOCHS,
        min_epochs=1,
        precision="16-mixed",
        callbacks=[checkpoint_callback, early_stopping, MyPrintingCallBack()]
    )

    # 5. Train
    trainer.fit(model, train_loader, val_loader)

    # 6. Validate
    trainer.validate(model, val_loader)

    # 7. Test
    trainer.test(model, test_loader, ckpt_path="best")
