configs = {
    (
        f"{sample_size}_{grid_size}x{grid_size}{rescale_suffix}{conf_thresh_suffix}{conf_thresh}"
    ): dict(
        n_imgs=grid_size * grid_size,
        sample_size=sample_size,
        do_rescale=do_rescale,
        conf_thresh=conf_thresh,
    )
    for sample_size in [500, 250, 56, 50, 20, 15, 10]
    for grid_size in [2, 3]
    for rescale_suffix, do_rescale in [("_rescale", True), ("", False)]
    for conf_thresh_suffix, conf_thresh in [("_noconfthresh", 0), ("", 0.5), ("", 0.0)]
}
