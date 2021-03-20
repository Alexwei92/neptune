from .vanilla_vae import VanillaVAE
from .beta_vae import BetaVAE
from .vae_gan import VAEGAN
from .dc_gan import DCGAN
# from .latent_models import *


# Model Dictionary
vae_model = {
    'vanilla_vae': VanillaVAE,
    'beta_vae': BetaVAE,
}

vaegan_model = {
    'vae_gan': VAEGAN
}

gan_model = {
    'dc_gan': DCGAN,
}