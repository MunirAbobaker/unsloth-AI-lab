def get_template(name):
    templates = {
        "default": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                      {SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>
                      {INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                      {OUTPUT}<|eot_id|>"""
    }
    return templates.get(name, templates["default"])
