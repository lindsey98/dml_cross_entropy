from sacred import Ingredient

from .architectures import __all__, __dict__

model_ingredient = Ingredient('vae')


@model_ingredient.config
def config():
    in_channels = 3
    latent_dim = 512
    
@model_ingredient.named_config
def vanillavae():
    arch = 'VanillaVAE'
   

@model_ingredient.named_config
def ae():
    arch = 'AE'
    
@model_ingredient.capture
def get_model(arch, in_channels, latent_dim):
    keys = list(map(lambda x: x.lower(), __all__))
    index = keys.index(arch.lower())
    arch = __all__[index]
    return __dict__[arch](in_channels=in_channels, latent_dim=latent_dim)

