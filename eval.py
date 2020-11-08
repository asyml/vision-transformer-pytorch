import torch
from data_loaders import *
from model import VisionTransformer
from checkpoint import load, convert


def main():
    keys, values = load('/Users/leon/Downloads/imagenet21k+imagenet2012_ViT-B_16.npz')
    state_dict = convert(keys, values)

    torch_model = VisionTransformer(
                 image_size=(384, 384),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=1000,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1)
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    data_loader = ImageNetDownSampleDataLoader(image_size=384, batch_size=4, num_workers=0, split='val')

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            pred_logits = torch_model(data)
            pred = pred_logits.argmax(-1)

            _, pred = pred.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            print(correct)


if __name__ == '__main__':
    main()