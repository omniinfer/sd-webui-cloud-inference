# Stable Diffusion Web UI Cloud Inference


[![](https://dcbadge.vercel.app/api/server/nzqq8UScpx)](https://discord.gg/nzqq8UScpx)


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/B8s2L_o3DrU/0.jpg)](https://www.youtube.com/watch?v=B8s2L_o3DrU)



## What capabilities does this extension offer?

This extension enables faster image generation without the need for expensive GPUs and seamlessly integrates with the AUTOMAIC1111 UI.

## Benefits:
1. **No expensive GPUs required**, can even use the CPU.
2. **No need to change your workflow**, compatible with the usage and scripts of sd-webui, such as X/Y/Z Plot, Prompt from file, etc.
3. **Support for 10000+ Checkpoint models**, don't need download



## How it works

![how it works](./docs/how-it-works.png)

## Guide

## 1. Install sd-webui-cloud-inference

<img width="1512" alt="image" src="https://github.com/omniinfer/sd-webui-cloud-inference/assets/16937838/2bc8c7b2-4ea6-447a-a82f-89ffb45eb45e">
<img width="1512" alt="image" src="https://github.com/omniinfer/sd-webui-cloud-inference/assets/16937838/f0c43b34-6feb-4d84-ba4e-1afa97abf31e">

## 2. Get your [omniinfer.io](https://www.omniinfer.io/user/login?utm_source=github_wiki) Key

Open [omniinfer.io](https://www.omniinfer.io/user/login?utm_source=github_wiki) in browser 

We can choice "Google Login" or "Github Login"
<img width="1512" alt="image" src="https://github.com/omniinfer/sd-webui-cloud-inference/assets/16937838/5dd02f42-f824-402e-99d7-8fd0ecc8776f">

<img width="1512" alt="image" src="https://github.com/omniinfer/sd-webui-cloud-inference/assets/16937838/05db40e4-ba65-4a9d-baed-bcbb17c6f605">

<img width="1512" alt="image" src="https://github.com/omniinfer/sd-webui-cloud-inference/assets/16937838/74973595-c022-4fd3-aabd-44199b47a1f7">

<img width="1234" alt="image" src="https://github.com/omniinfer/sd-webui-cloud-inference/assets/16937838/4fe56d2d-54cd-41d0-8c5c-93647302e961">


## 3. Enable Cloud Inference feature

Let us back to `Cloud Inference` tab of stable-diffusion-webui 

<img width="1508" alt="image" src="https://github.com/omniinfer/sd-webui-cloud-inference/assets/16937838/3b91e85c-cafc-4eb2-86b4-b350c0f073cd">

## 4. Test Txt2Img

Let us back to `Txt2Img` tab of stable-diffusion-webui


<img width="1512" alt="image" src="https://github.com/omniinfer/sd-webui-cloud-inference/assets/16937838/a947bcc5-93e2-436b-beb8-b72f79523c29">

From now on, you can give it a try and enjoy your creative journey.

Furthermore, you are welcome to freely discuss your user experience, share suggestions, and provide feedback on our Discord channel.
<br>​[![](https://dcbadge.vercel.app/api/server/kJCEK9zf)](https://discord.gg/kJCEK9zf)


## 5. Advanced - Lora

<img width="1714" alt="image" src="https://github.com/omniinfer/sd-webui-cloud-inference/assets/16937838/ddfc0578-fac6-48a6-a765-a12fb06ecfe4">

## 6. Advanced - Lora with X/Y/Z

<img width="1714" alt="image" src="https://github.com/omniinfer/sd-webui-cloud-inference/assets/16937838/c0c72f3e-b915-4579-b0f9-b3a6d1c3ba73">



## Compatibility and Limitations

| Feature                    | Compatibility | Limitations                                                                   |
| -------------------------- | ------------- | ----------------------------------------------------------------------------- |
| txt2img                    | ✅✅✅           | 🚫 Hires.fix, Tiling, restore face                                             |
| txt2img_controlnet         | ✅✅✅           | 🚫 Hires.fix, Tiling, restore face, Ending Control Step, Starting Control Step |
| img2img                    | ✅✅✅           | 🚫 Tiling, restore face                                                        |
| img2img_controlnet         | ✅✅✅           | 🚫 Hires.fix, Tiling, restore face, Ending Control Step, Starting Control Step |
| scripts - X/Y/Z plot       | ✅✅✅✅✅         | 🚫 Checkpoint name                                                             |
| scripts - Prompt matrix    | ✅✅✅✅✅         |                                                                               |
| scripts - Prompt from file | ✅✅✅✅✅         |                                                                               |
