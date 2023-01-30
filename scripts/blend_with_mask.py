import argparse

from ai_prototypes.utils.PIL_image import load_image, blend, smoothe_blend


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, default='examples/mask_off/ray_by_finn.jpg')
    parser.add_argument("-m", "--mask", type=str, default='examples/mask_off/ray_by_finn.png')
    parser.add_argument("-t", "--target", type=str, default='examples/mask_off/stargan-v2/result/aligned/stargan-v2__ray_reference__ray_by_finn_sacgan.jpg')
    parser.add_argument("-o", "--output", type=str, default='examples/mask_off/stargan-v2/result/stargan-v2__ray_reference__ray_by_finn_sacgan__blended.jpg')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scale = 0.99
    y_offset = -5

    src_image = load_image(args.source)
    mask_image = load_image(args.mask, src_image.size, "L")
    trg_image = load_image(args.target, src_image.size)

    if scale != 1.0:
        size = (int(trg_image.width * scale), int(trg_image.height * scale))
        scaled_image = trg_image.resize(size)
        box = [
            int((trg_image.width - size[0])/2),
            int((trg_image.height - size[1])/2) + y_offset,
            int((trg_image.width - size[0])/2) + size[0],
            int((trg_image.height - size[1])/2) + size[1] + y_offset]
        trg_image.paste(scaled_image, box)

    assert src_image.size == mask_image.size == trg_image.size

    result = smoothe_blend(src_image, mask_image, trg_image)
    result.save(args.output)
