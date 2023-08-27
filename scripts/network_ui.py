from pathlib import Path

from modules import shared, ui_extra_networks_user_metadata, errors, extra_networks, script_callbacks, ui_extra_networks
from modules.images import read_info_from_image, save_image_with_geninfo
import gradio as gr
import json
import html
from fastapi.exceptions import HTTPException

from modules.generation_parameters_copypaste import image_from_url_text
from modules.ui_components import ToolButton


class ExtraNetworksPage:
    def __init__(self, title):
        self.title = title
        self.name = title.lower()
        self.id_page = self.name.replace(" ", "_")
        self.card_page = shared.html("extra-networks-card.html")
        self.allow_negative_prompt = False
        self.metadata = {}
        self.items = {}

    def refresh(self):
        pass

    def create_html(self, tabname):
        pass

    def create_item(self, name, index=None, enable_filter=True):
      return {
        "name": "test",
        "filename": "test",
        "preview": "https://next-app-static.s3.amazonaws.com/images-prod/xG1nkqKTMzGDvpLrqFT7WA/7e6f18a0-e02a-4934-70e8-359c4c302f00/width=450/53255.jpeg",
        "description": "1234",
        "search_term": "1234",
        "metadata": {},
      }

    def list_items(self):
        return [{
            "name": "test",
            "filename": "test",
            "preview": "https://next-app-static.s3.amazonaws.com/images-prod/xG1nkqKTMzGDvpLrqFT7WA/7e6f18a0-e02a-4934-70e8-359c4c302f00/width=450/53255.jpeg",
            "description": "1234",
            "search_term": "1234",
            # "onclick": '"' + html.escape(f"""return selectCheckpoint({quote_js(name)})""") + '"',
            # "local_preview": f"{path}.{shared.opts.samples_format}",
            "metadata": {},
            # "sort_keys": {'default': index, **self.get_sort_keys(checkpoint.filename)},
        }]

    def allowed_directories_for_previews(self):
        return []

    def create_html_for_item(self, item, tabname):
        import ipdb; ipdb.set_trace()
        return shared.html("abc")

    def get_sort_keys(self, path):
        """
        List of default keys used for sorting in the UI.
        """
        pth = Path(path)
        stat = pth.stat()
        return {
            "date_created": int(stat.st_ctime or 0),
            "date_modified": int(stat.st_mtime or 0),
            "name": pth.name.lower(),
        }

    def create_user_metadata_editor(self, ui, tabname):
        return ui_extra_networks_user_metadata.UserMetadataEditor(ui, tabname, self)


def register_page(*args, **kwargs):
    ui_extra_networks.register_page(ExtraNetworksPage("Cloud Models"))


script_callbacks.on_before_ui(register_page)
