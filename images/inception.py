import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

def calculate_inception_score(
    sample_dataloader,
    test_dataloader,
    device="cpu",
    num_images=50000,
    splits=10,
):
    inception_model = InceptionScore(device=device)
    inception_model.eval()

    preds = np.zeros((num_images, 1000))

    for i, batch in enumerate(sample_dataloader, 0):
        batch = batch.to(device=device)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        predictions = inception_model(batchv)
        start = i * test_dataloader.batch_size
        n_predictions = len(preds[start : start + batch_size_i])
        preds[start : start + batch_size_i] = predictions[:n_predictions]

    split_scores = []

    for k in range(splits):
        part = preds[
            k * (num_images // splits) : (k + 1) * (num_images // splits), :
        ]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return {"Inception Score": np.mean(split_scores)} 
