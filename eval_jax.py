from data_loaders import *

import jax
import jax.numpy as jnp
import numpy as onp
import flax
from vit_jax import models
from vit_jax import checkpoint
from vit_jax import log
logger = log.setup_logger('./logs')

import torch
from data_loaders import *
from model import VisionTransformer
from checkpoint import load, convert


def main():
    image_size = 384

    # jax model
    jax_model = models.KNOWN_MODELS['ViT-B_16'].partial(num_classes=1000, representation_size=None)
    _, params = jax_model.init_by_shape(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension of the batch for initialization.
        [((4, image_size, image_size, 3), 'float32')])
    params = checkpoint.load_pretrained(
        pretrained_path='/Users/leon/Downloads/imagenet21k+imagenet2012_ViT-B_16.npz',
        init_params=params,
        model_config=models.CONFIGS['ViT-B_16'],
        logger=logger)
    params_repl = flax.jax_utils.replicate(params)
    # Then map the call to our model's forward pass onto all available devices.
    vit_apply_repl = jax.pmap(jax_model.call)

    # torch_model
    keys, values = load('/Users/leon/Downloads/imagenet21k+imagenet2012_ViT-B_16.npz')
    state_dict = convert(keys, values)

    torch_model = VisionTransformer(
                 image_size=(image_size, image_size),
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


    data_loader = ImageNetDownSampleDataLoader(image_size=image_size, batch_size=1, num_workers=0, split='val')

    for batch_idx, (data, target) in enumerate(data_loader):

        # jax prediction
        target_numpy = target.cpu().numpy()
        data_numpy = data.cpu().numpy().transpose(0, 3, 1, 2).reshape(1, -1, image_size, image_size, 3)
        jax_predicted = vit_apply_repl(params_repl, data_numpy)._value[0]
        # pred = jax_predicted.argmax(axis=-1)
        # is_same = pred == target_numpy
        # print(is_same)

        # torch prediction
        with torch.no_grad():
            torch_predicted = torch_model(data)
        torch_predicted = torch_predicted.cpu().numpy()

        # check difference
        jax_predicted = jax_predicted.transpose(0, 3, 1, 2)
        assert onp.allclose(jax_predicted, torch_predicted, rtol=1e-5, atol=1e-8)

        print(torch_predicted)

if __name__ == '__main__':
    main()