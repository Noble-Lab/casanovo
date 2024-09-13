def process_description(app, ctx, lines):
    if ctx.command.name == "main":
        del lines[:3]


def setup(app):
    app.connect("sphinx-click-process-description", process_description)
