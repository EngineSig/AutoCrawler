import easyocr
import cv2
import numpy as np
import os
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def run_ocr(keyword=None, do_recognize=False):
    print("Sorting out images without text")
    img_path = os.getcwd() + "\\"

    # Can use only a pair of language with english at a time. So far.
    languages = ['ko', 'en']
    reader = easyocr.Reader(lang_list=languages)

    # If the folder is mixed with non-image files
    img_types = (
        ".bmp", ".dib", ".jpg", ".jpeg", ".jp2", ".png", ".webp", ".pbm", ".pgm", ".pgm", ".ppm", ".pxm", ".pnm", ".sr",
        ".ras", ".tiff", ".tif", ".exr", ".hdr", ".pic")

    if keyword is None:
        file_name = "test.json"
    else:
        file_name = img_path + keyword + ".json"
        img_path = img_path + "download" + "\\" + keyword + "\\"

    data = {}
    # iterate through image
    for entry in os.listdir(img_path):
        # If the folder is mixed with non-image files
        if entry.endswith(img_types):

            #print(os.path.isfile(img_path + entry))
            np_img_array = np.fromfile(img_path + entry, np.uint8)
            image = cv2.imdecode(np_img_array, cv2.IMREAD_COLOR)
            #image = cv2.imread(img_path + entry, encoding = 'utf-8')
            horizontal_list, free_list = reader.detect(image)

            if len(horizontal_list) == 0 and len(free_list) == 0:
                os.remove(img_path + entry)
            #if reconizer enabled, save tags into json file
            elif do_recognize:
                recognize_results = reader.recognize(image, horizontal_list, free_list)
                add_dict = {"words": []}
                # iterate through words in image
                for result in recognize_results:
                    tags = {"points": result[0],
                            "transcription": result[1],
                            "confidence": float(result[2])}

                    add_dict["words"].append(tags)
                data[entry] = add_dict

    if do_recognize:
        with open(file_name, "wt", encoding="utf8") as json_file:
            json_file.write(json.dumps(data, sort_keys=True, ensure_ascii=False, indent=4, cls=NpEncoder))


if __name__ == '__main__':
    run_ocr(keyword='테스트용검색아무거나', do_recognize=False)
