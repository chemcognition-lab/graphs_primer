def set_visualization_style():
    try:
        from pyfonts import load_font

        # Add custom fonts
        font = load_font(
            font_url="https://github.com/googlefonts/dm-fonts/raw/refs/heads/main/Sans/fonts/ttf/DMSans24pt-Regular.ttf"
        )
        font_manager.fontManager.addfont(font_manager.findfont(font))
        mpl.rcParams["font.sans-serif"] = [font.get_name()]
        print("Load preferred font.")
    except ValueError:
    mpl.rcParams["font.sans-serif"] = ['Fira Code']
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.pad_inches"] = 0.1
    mpl.rcParams["savefig.transparent"] = True
    # mpl.rcParams['axes.linewidth'] = 2.5
    mpl.rcParams["legend.markerscale"] = 1.0
    mpl.rcParams["legend.fontsize"] = "small"
    # seaborn color palette
    sns.set_palette("colorblind")
