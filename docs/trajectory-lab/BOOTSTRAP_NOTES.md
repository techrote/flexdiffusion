# Bootstrap notes

The callback tap is deliberately placed before Easy Diffusion's optional live-preview decode. This keeps trajectory observation independent from the live-preview interval and avoids making every requested checkpoint depend on browser preview settings.

The current manifest recorder records tensor metadata only. Latent persistence will detach/copy selected tensors to CPU and hand them to a bounded writer; no GPU tensor history will be retained.
