<h1 align="center">Optimized Stable Diffusion</h1>

This repo is a modified version of the basujindal fork of Stable
Diffusion with the goal to reduce VRAM usage even more.

With this, you can generate 1088x1088 images with only 4GB GPUs.

To reduce the VRAM usage, following additional optimizations were used:
* Better tensor memory management. Inspirations was from [here](https://github.com/Doggettx/stable-diffusion).
* Flash attention is used instead of normal attention. Inspiration was
  from [here](https://www.photoroom.com/tech/stable-diffusion-100-percent-faster-with-memory-efficient-attention/).
* First stage image encoding model and last stage image decoding model
  were moved to CPU because both are very fast and very memory hungry
  so it makes no sense to use GPU for them.

Additionally, support for negative prompts was added.

<h1 align="center">Installation</h1>

Clone this repo somewhere and open terminal in its directory and type:
``` shell
conda env create -f environment.yaml
```

Before calling stable diffusion in a terminal session, don't forget to
activate conda environment with:
``` shell
conda activate ldm
```

Then download snapshot of SD model with:
``` shell
curl https://www.googleapis.com/storage/v1/b/aai-blog-files/o/sd-v1-4.ckpt?alt=media > sd-v1-4.ckpt
```

And you are done. For linux, that is.

Support for windows is untested because I don't have windows
box. There will be likely issues during building xformers. On linux,
you need gcc<10.0.0 but on windows, I have no idea so its up to you to
solve it somehow by editing `environment.yaml`...

Just for a side note, you can uninstall conda environment using:

``` shell
conda deactivate
conda env remove -n ldm
rm -rf src
```

<h1 align="center">Usage</h1>

## txt2img

``` shell
python -B optimizedSD/optimized_txt2img.py --prompt "dog" --nprompt "dry" --precision full --ckpt sd-v1-4.ckpt --H 512 --W 512 --sampler euler_a --n_samples 10
```
* `--prompt` - Textual image description.
* `--nprompt` - Negative textual image description. Things which you
                don't want are placed here.
* `--H` - Image height in pixels. Must be multiple of 64.
* `--W` - Image width in pixels. Must be multiple of 64.
* `--n_samples` - Number of images to generate at once. When
                  generating 1088x1088 images, only one sample is
                  supported on 4GB GPUs.

## img2img

``` shell
python -B $THISDIR/optimizedSD/optimized_img2img.py --prompt "dog" --nprompt "dry" --init-img path/to/init/image.jpg --strength 0.75 --precision full --ckpt $THISDIR/sd-v1-4.ckpt --H 512 --W 512 --n_samples 1
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
* `--n_samples` - Number of images to generate at once. Because
                  img2img is more VRAM intensive than txt2img, keep
                  sample count low on 4GB GPUs.

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
If you have a Nvidia GTX series GPU, the output images maybe
entirely green in color. This is because GTX series do not support
half precision calculation, which is the default mode of calculation
in this repository. To overcome the issue, use the `--precision full`
argument. The downside is that it will lead to higher GPU VRAM usage.

## Distorted images in higher resolution
Stable diffusion was trained on 512x512 images so it does not know how
to fill the space in larger images so it just combines content of
multiple smaller images into one which is obviously wrong because
resulting composition does not make sense.

You can mitigate this by first generating 512x512 image. Then resizing
it (not upscaling) to desired resolution and feeding it to img2img
together with original prompt and `--strength 0.75`. Stable diffusion
will then just polish things up without trying to come up with some
new composition.

## Faces and bodies are deformed
Stable diffusion is known to not work well with faces and generally
human bodies. You can mitigate this by using negative prompts.

Try this:
``` shell
--nprompt "extra limbs, deformed body, blurred, long neck"
```
