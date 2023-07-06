# Stable Diffusion Web UI Cloud Inference

## Benefits:
1. **No expensive GPUs required**
2. **No need to change your workflow**, compatible with the usage and scripts of sd-webui, such as X/Y/Z Plot, Prompt from file, etc.
3. **Support for 1000+ Checkpoint models**

## Quick Start

1. Install the sd-webui-cloud-inference extension.
2. Go to the `Extensions` tab, click the `Install from URL` button, enter `https://github.com/omniinfer/sd-webui-cloud-inference`, and click the `Install` button.
3. Obtain an API Key from [omniinfer.io](https://omniinfer.readme.io/reference/try-api#find-your-key).
    1. In sd-webui, go to the `Cloud Inference` tab.
    2. Paste the API Key into the `API Key` input box.
    3. Click the `Test Connection` button.
       * ✅ If `Connection Success` appears, the connection is successful.
       * ❌ If `Connection Failed` appears, the connection failed. Please check if the API Key is correct.
4. Go to the `txt2img` tab, check ✅ the `Enable` checkbox under `Cloud Inference`.
   * ✅ The `Generate` button in the top right corner will change to `Generate (cloud)`, indicating that cloud inference is enabled.
   * The list of available models for cloud inference will be displayed in the dropdown menu.
5. Click the `Generate (cloud)` button and wait for the results.

## Compatibility and Limitations

| Feature                    | Compatibility | Limitations                                                                   |
| -------------------------- | ------------- | ----------------------------------------------------------------------------- |
| txt2img                    | ✅✅✅           | 🚫 Hires.fix, Tiling, restore face                                             |
| txt2img_controlnet         | ✅✅✅           | 🚫 Hires.fix, Tiling, restore face, Ending Control Step, Starting Control Step |
| img2img                    | ✅✅✅           |                                                                               |
| img2img_controlnet         | ✅✅✅           | 🚫 Hires.fix, Tiling, restore face, Ending Control Step, Starting Control Step |
| scripts - X/Y/Z plot       | ✅✅✅✅✅         | 🚫 Tiling, restore face                                                        |
| scripts - Prompt matrix    | ✅✅✅✅✅         |                                                                               |
| scripts - Prompt from file | ✅✅✅✅✅         |                                                                               |
