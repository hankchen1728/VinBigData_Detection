import cv2


def draw_bbox(image, box, label, color, thickness=10):
    alpha = 0.1
    alpha_box = 0.4
    font_scale = 1.5
    font_thick = 2
    overlay_bbox = image.copy()
    overlay_text = image.copy()
    output = image.copy()

    text_width, text_height = cv2.getTextSize(
        text=label.upper(),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        thickness=font_thick
    )[0]
    cv2.rectangle(overlay_bbox, (box[0], box[1]), (box[2], box[3]), color, -1)
    cv2.addWeighted(overlay_bbox, alpha, output, 1 - alpha, 0, output)
    cv2.rectangle(
        overlay_text,
        (box[0], box[1] - 10 - text_height),
        (box[0] + text_width + 5, box[1]),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay_text, alpha_box, output, 1 - alpha_box, 0, output)
    cv2.rectangle(
        output,
        (box[0], box[1]),
        (box[2], box[3]),
        color,
        thickness
    )
    cv2.putText(
        img=output,
        text=label.upper(),
        org=(box[0], box[1] - 5),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=font_thick,
        lineType=cv2.LINE_AA
    )
    return output
