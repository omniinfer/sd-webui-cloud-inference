# Stable Diffusion Web UI Cloud Inference

[![](https://dcbadge.vercel.app/api/server/kJCEK9zf)](https://discord.gg/kJCEK9zf)

## 这个插件能做什么？
该插件能够让你通过云端更快的生成图片，而不需要购买昂贵的 GPU 资源，并且能够和  AUTOMAIC1111 UI 无缝兼容

## 收益:
1. **不需要昂贵的 GPU**
2. **不需要改变工作流**， 基本兼容 sd-webui 的使用和脚本，例如 X/Y/Z Plot、Prompt from file 等
3. **支持 10000+ Checkpoint 模型**

## 相关文档

* [快速开始 - Stable Diffusion WebUI Cloud Inference Tutorial](https://github.com/omniinfer/sd-webui-cloud-inference/wiki/Stable-Diffusion-WebUI-Cloud-Inference-Tutorial)


# 这个插件是如何工作的？

![how it works](./docs/how-it-works.png)


##  兼容性和限制

| 功能                       | 兼容性 | 限制                                                                          |
| -------------------------- | ------ | ----------------------------------------------------------------------------- |
| txt2img                    | ✅✅✅    | 🚫 Hires.fix, Tiling, restore face                                             |
| txt2img_controlnet         | ✅✅✅    | 🚫 Hires.fix, Tiling, restore face, Ending Control Step, Starting Control Step |
| img2img                    | ✅✅✅    | 🚫 Tiling, restore face                                                        |
| img2img_controlnet         | ✅✅✅    | 🚫 Hires.fix, Tiling, restore face, Ending Control Step, Starting Control Step |
| scripts - X/Y/Z plot       | ✅✅✅✅✅  | 🚫 Checkpoint name                                                             |
| scripts - Prompt matrix    | ✅✅✅✅✅  |                                                                               |
| scripts - Prompt from file | ✅✅✅✅✅  |                                                                               |
