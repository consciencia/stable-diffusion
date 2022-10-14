<h1 align="center">Optimized Stable Diffusion</h1>

This repo is a modified version of the basujindal fork of Stable
Diffusion with the goal to reduce VRAM usage even more.

With this, you can generate 1088x1088 images with only 4GB GPUs.

To reduce the VRAM usage, following additional optimizations were used:
* Better tensor memory management. Inspiration was from [here](https://github.com/Doggettx/stable-diffusion).
* Flash attention is used instead of normal attention. Inspiration was
  from [here](https://www.photoroom.com/tech/stable-diffusion-100-percent-faster-with-memory-efficient-attention/).
* First stage image encoding model and last stage image decoding model
  were moved to CPU because both are very fast and very memory hungry
  so it makes no sense to use GPU for them.

Additionally, support for negative prompts was added.

<h1 align="center">Installation</h1>

First, install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

**If you already have `ldm` conda environment because you already used Stable
Diffusion, remove it because this fork uses different package versions than
other forks in order to be compatible with `xformers`.**
```shell
conda env remove -n ldm
```

Then clone this repository somewhere and open terminal in its directory
and type:
``` shell
conda env create -f environment_<platform>.yaml
```

Where `<platform>` stands for `linux` or `windows`.

Before calling stable diffusion in a terminal session, don't forget to
activate conda environment with:
``` shell
conda activate ldm
```

Then download snapshot of SD model with:
``` shell
curl https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media > sd-v1-4.ckpt
```

And you are done.

**Warning: Never ever try to install explicitly `xformers`, it will fail
because it is dependant on specific version of GCC and pytorch. Let conda
handle this.**

<h1 align="center">Usage</h1>

## txt2img

``` shell
python -B scripts/txt2img.py --prompt "dog" --nprompt "dry" --precision full --ckpt sd-v1-4.ckpt --H 512 --W 512 --n_samples 10 --ddim_steps 50 --scale 7.5
```
* `--prompt` - Textual image description.
* `--nprompt` - Negative textual image description. Things which you
                don't want are placed here.
* `--H` - Image height in pixels. Must be multiple of 64.
* `--W` - Image width in pixels. Must be multiple of 64.
* `--n_samples` - Number of images to generate at once. When
                  generating 1088x1088 images, only one sample is
                  supported on 4GB GPUs.
* `--ddim_steps` - Number of sampler steps. Usually 50 is good enough.
* `--scale` - Guidance scale. Higher number results in more literal
              interpretation of your prompts. Default is 7.5 and its
              not recommended to go above 20. This parameter is also
              known as CFG.

## img2img

``` shell
python -B scripts/img2img.py --prompt "dog" --nprompt "dry" --init-img path/to/init/image.jpg --strength 0.75 --precision full --ckpt sd-v1-4.ckpt --H 512 --W 512 --n_samples 1 --ddim_steps 50 --scale 7.5
```
* `--prompt` - Textual image description.
* `--nprompt` - Negative textual image description. Things which you
                don't want are placed here.
* `--init-img` - Path to initialization image.
* `--strength` - Amount of noise to be added into initialization
                 image. Value of 0.75 stands for 75% of initialization
                 image to be noise. Keep this value low when you want
                 to prevent SD from doing too much pervasive changes to
                 initialization image.
* `--H` - Image height in pixels. Must be multiple of 64.
* `--W` - Image width in pixels. Must be multiple of 64.
* `--n_samples` - Number of images to generate at once. When generating
                  1088x1088 images, only one sample is supported on 4GB
                  GPUs.
* `--ddim_steps` - Number of sampler steps. Usually 50 is good enough.
* `--scale` - Guidance scale. Higher number results in more literal
              interpretation of your prompts. Default is 7.5 and its
              not recommended to go above 20. This parameter is also
              known as CFG.

### Inpainting

Use `img2img.py` as with standard img2img but with additional
parameters.

Inpainting require higher sampler step count. Using `--ddim_steps 100`
should be enough.

Option `--strength` works as usual.

* `--mask` - Path to image mask. Must have same resolution as generated
             image. Surface filled with white pixels will be regenerated
             by Stable Diffusion while surface filled with black pixels
             will be untouched.
* `--invert-mask` - Inverts mask.

Please note that model does not know which part of image should be
changed, inpainting is implemented through discarding changes affecting
masked parts of initialization image. This imply that you should use
inpainting together with very similar prompt to one which created
original image in order to get desired effect.

Inpainting is generally good for refining and correcting previously
generated images. Using it for anything else would probably result in
fail.

<h1 align="center">Weight blocks</h1>

You can use weight blocks for standard prompts and for negative prompts.

Example:

``` text
picture of:1 small:0.5 cat sitting on:1 big:2 dog
```

Will be interpreted as 5 different prompts, each with its own weight:
``` text
picture of -> weight 1
small -> weight 0.5
cat sitting on -> weight 1
big -> weight 2
dog -> weight 1
```

These 5 prompts will be separately processed and results averaged with
the respect to specified weights.

To sum it up, always make sure that weight blocks make sense from
semantical point of view because prompt encoder interprets them in
isolation. Just for clarity, example above is wrong because weight
blocks does not make desired sense when interpreted in isolation.

<h1 align="center">Troubleshooting</h1>

## Green colored output images
If you have a Nvidia GTX series GPU, the output images may be
entirely green in color. This is because GTX series does not support
half precision calculations. To overcome this issue, use the `--precision full`
argument. The downside is that it will lead to higher GPU VRAM usage.

## Distorted images in higher resolution
Stable diffusion was trained on 512x512 images so it does not know how
to fill the space in larger images so it just combines content of
multiple smaller images into one which is obviously wrong because
resulting composition does not make sense.

You can mitigate this by first generating 512x512 image. Then resizing
it (not upscaling) to desired resolution and feeding it to img2img
together with original prompt and `--strength 0.5`. Stable diffusion
will then just polish things up without trying to come up with some
new composition.

## Faces and bodies are deformed
Stable diffusion is known to not work well with faces and generally
human bodies. You can mitigate this by using negative prompts.

Try this:
``` shell
--nprompt "extra limbs, deformed body, blurred, long neck"
```
