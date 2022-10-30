

CONFIG = dict(
    model_name="microsoft/deberta-v3-large",
    num_classes=6,
    lr=2e-5,

    batch_size=8,
    num_workers=8,
    max_length=512,
    weight_decay=0.01,

    accelerator='gpu',
    max_epochs=5,
    accumulate_grad_batches=4,
    precision=16,
    gradient_clip_val=1000,
    train_size=0.8,
)

# we can also try nn.SmoothL1Loss
