from .vanilla_vae import VanillaVAE
from .beta_vae_h import BetaVAE_H
from .beta_vae_b import BetaVAE_B
from .factor_vae import FactorVAE
from .vae_gan import VAEGAN
from .dc_gan import DCGAN
from .latent_nn import LatentNN
from .end_to_end import EndToEnd

# Model Dictionary
vae_model = {
    'vanilla_vae': VanillaVAE,
    'beta_vae_h': BetaVAE_H,
    'beta_vae_b': BetaVAE_B,
    'factor_vae': FactorVAE,
}

vaegan_model = {
    'vae_gan': VAEGAN
}

gan_model = {
    'dc_gan': DCGAN
}