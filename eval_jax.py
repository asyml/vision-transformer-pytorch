from data_loaders import *

import jax
import flax
from vit_jax import models
from vit_jax import checkpoint
from vit_jax import log
logger = log.setup_logger('./logs')


def main():
    jax_model = models.KNOWN_MODELS['ViT-B_16'].partial(num_classes=1000, representation_size=None)
    _, params = jax_model.init_by_shape(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension of the batch for initialization.
        [((4, 384, 384, 3), 'float32')])
    params = checkpoint.load_pretrained(
        pretrained_path='/Users/leon/Downloads/imagenet21k+imagenet2012_ViT-B_16.npz',
        init_params=params,
        model_config=models.CONFIGS['ViT-B_16'],
        logger=logger)
    params_repl = flax.jax_utils.replicate(params)
    # Then map the call to our model's forward pass onto all available devices.
    vit_apply_repl = jax.pmap(jax_model.call)

    data_loader = ImageNetDownSampleDataLoader(image_size=384, batch_size=4, num_workers=0, split='val')

    for batch_idx, (data, target) in enumerate(data_loader):
        target_numpy = target.cpu().numpy().reshape(1, 4, 1)
        data_numpy = data.cpu().numpy().transpose(0, 3, 1, 2).reshape(1, 4, 384, 384, 3)
        jax_predicted = vit_apply_repl(params_repl, data_numpy)
        pred = jax_predicted.argmax(axis=-1)
        is_same = pred == target_numpy
        print(is_same)


if __name__ == '__main__':
    main()