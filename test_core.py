import numpy as np
import pandas as pd

from rsna_ich_xai.pipeline import apply_window, parse_label_row, read_rsna_labels


def test_parse_label_row():
    image_id, label = parse_label_row("ID_abc123_intraventricular")
    assert image_id == "ID_abc123"
    assert label == "intraventricular"


def test_apply_window_range():
    image = np.array([-100.0, 40.0, 100.0], dtype=np.float32)
    out = apply_window(image, center=40.0, width=80.0)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_long_to_wide_labels(tmp_path):
    rows = []
    values = {"epidural": 0, "intraparenchymal": 1, "intraventricular": 1,
              "subarachnoid": 0, "subdural": 0, "any": 1}
    for name, value in values.items():
        rows.append({"ID": f"Image001_{name}", "Label": value})
    path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    wide = read_rsna_labels(str(path))
    assert len(wide) == 1
    assert int(wide.loc[0, "intraparenchymal"]) == 1
    assert int(wide.loc[0, "intraventricular"]) == 1
    assert int(wide.loc[0, "any"]) == 1
