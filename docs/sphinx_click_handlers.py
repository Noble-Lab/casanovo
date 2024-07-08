def process_description(app, ctx, lines):
    if ctx.command.name == "main":
        del lines[:3]

    print(lines)


def setup(app):
    app.connect("sphinx-click-process-description", process_description)
