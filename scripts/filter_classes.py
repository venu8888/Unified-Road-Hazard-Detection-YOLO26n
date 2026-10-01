"""
Class Filtering Module for Option C (9-Class Unified Taxonomy)
In-Scope Training Classes:
  - N-RDD2024: D00, D10, D20, D30, D40
  - Supplementary: Good, Broken, Lose, Uncovered
Excluded from Training:
  - N-RDD2024: D70 (Generic manhole cover - retained for qualitative comparison only)
  - N-RDD2024: D50, D60, D80, D90 (Road markings / other non-hazard defects)
  - Supplementary: circle (Verified absent from dataset)
"""

IN_SCOPE_NRDD_TRAIN = {'D00', 'D10', 'D20', 'D30', 'D40'}
EXCLUDED_NRDD_TRAIN = {'D50', 'D60', 'D70', 'D80', 'D90'}

IN_SCOPE_SUPP_TRAIN = {'Good', 'Broken', 'Lose', 'Uncovered'}
EXCLUDED_SUPP_TRAIN = {'circle'}

def filter_nrdd_objects(objects_list):
    """
    Given a list of (cname, xmin, ymin, xmax, ymax),
    returns in-scope objects for Option C training.
    """
    kept = []
    excluded = []
    for obj in objects_list:
        cname = obj[0]
        if cname in IN_SCOPE_NRDD_TRAIN:
            kept.append(obj)
        else:
            excluded.append(obj)
    return kept, excluded

def filter_supp_boxes(boxes_list, class_map):
    """
    Given a list of (cid, xc, yc, bw, bh),
    returns in-scope boxes for Option C training.
    """
    kept = []
    excluded = []
    for box in boxes_list:
        cid = box[0]
        cname = class_map.get(cid, '')
        if cname in IN_SCOPE_SUPP_TRAIN:
            kept.append(box)
        else:
            excluded.append(box)
    return kept, excluded
