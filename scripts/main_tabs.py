import modules.scripts as scripts
import gradio as gr

from modules import script_callbacks
from extension import api


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Tab(label="Omniinfer"):
            with gr.Blocks():
                gr.Markdown("""
            # Omniinfer
            Omniinfer is a cloud inference service that allows you to run txt2img and img2img in the cloud.
            """)
                with gr.Row():
                    key_textbox = gr.Textbox(
                        value=api.get_instance().__dict__.get('_token')
                        if api.get_instance() is not None else "",
                        label="Omniinfer Key",
                        type="password",
                        placeholder="Enter omniinfer key here",
                        elem_id="settings_remote_inference_omniinfer_key",
                    )

                    test_button = gr.Button(
                        "Test Connection",
                        label="Test Connection",
                        variant="primary",
                        elem_id="settings_remote_inference_omniinfer_test",
                    )

                test_message_textbox = gr.Textbox(label="Test Message Results",
                                                  interactive=False)

                gr.HTML(value="""
                                Register for a free key at <u><a href="https://github.com/omniinfer/sd-webui-cloud-inference/wiki/Stable-Diffusion-WebUI-Cloud-Inference-Tutorial#2-obtain-key-from-omniinferio" target="_blank">Stable Diffusion WebUI Cloud Inference Tutorial</a></u>.
                                """)

                def test_callback(key):
                    try:
                        ok_msg = api.OmniinferAPI.test_connection(key)
                        api.OmniinferAPI.update_key_to_config(key)
                        api.refresh_instance()
                        return ok_msg
                    except Exception as exp:
                        return str(exp)

                test_button.click(fn=test_callback,
                                  inputs=[key_textbox],
                                  outputs=[test_message_textbox])
        with gr.Tab(label="Additional Providers"):
            gr.Markdown("""
            # Do you require support for additional api providers?
            discuss in [Github](https://github.com/omniinfer/sd-webui-cloud-inference/discussions/new?category=general)
            """)

        return [(ui_component, "Cloud Inference", "extension_template_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)