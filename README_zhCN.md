# Stable Diffusion Web UI Cloud Inference

##  收益:
1. **不需要昂贵的 GPU**
2. **不需要改变工作流**， 基本兼容 sd-webui 的使用和脚本，例如 X/Y/Z Plot、Prompt from file 等
3. **支持 1000+ Checkpoint 模型**

##  快速开始

1. Install sd-webui-cloud-inference extension
1. 进入 `Extensions` 标签页，点击 `Install from URL` 按钮，输入 `https://github.com/omniinfer/sd-webui-cloud-inference` 点击 `Install` 按钮
2. 从 [omniinfer.io](https://omniinfer.readme.io/reference/try-api#find-your-key) 获取 API Key
    1. 在 sd-webui 中进入 `Cloud Inference` 标签页
    2. 粘贴 API Key 到 `API Key` 输入框
    3. 点击 `Test Connection` 按钮
       * ✅ 如果出现 `Connection Success` 则说明连接成功
       * ❌ 如果出现 `Connection Failed` 则说明连接失败，请检查 API Key 是否正确
3. 进入 `txt2img` 标签页， 勾选 ✅  `Cloud Inference` 中的 `Enable` 复选框
   * ✅ 这时候右上角的 `Geneate` 会变为 `Genearte (cloud)`，说明已经启用了云推理
   * 云端可用的模型列表会显示在下拉框中
4. 点击 `Genearte (cloud)` 按钮，等待结果


##  兼容性和限制

| 功能                       | 兼容性 | 限制                                                                          |
| -------------------------- | ------ | ----------------------------------------------------------------------------- |
| txt2img                    | ✅✅✅    | 🚫 Hires.fix、Tiling、restore face                                             |
| txt2img_controlnet         | ✅✅✅    | 🚫 Hires.fix、Tiling、restore face、Ending Control Step、Starting Control Step |
| img2img                    | ✅✅✅    |                                                                               |
| img2img_controlnet         | ✅✅✅    | 🚫 Hires.fix、Tiling、restore face、Ending Control Step、Starting Control Step |
| scripts - X/Y/Z plot       | ✅✅✅✅✅  | 🚫 Tiling、restore face                                                        |
| scripts - Prompt matrix    | ✅✅✅✅✅  |                                                                               |
| scripts - Prompt from file | ✅✅✅✅✅  |                                                                               |
