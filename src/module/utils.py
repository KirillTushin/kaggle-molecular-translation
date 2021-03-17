from Levenshtein import distance

def convert_image_id_2_path(image_id: str, path: str) -> str:
    return f'{path}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.png'


def clean_tokens(seq, after = [0, 2], skip = [0, 1, 2]):
    i = 0
    res = []
    for t in seq:
        if t not in skip:
            res.append(t)
        if t in after:
            break
    return res

def lev_score(predict, target):
    return [distance(p, r) for p, r in zip(predict, target)]