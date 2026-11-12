"""
Verifier, check 11: can the demonstrated wrist be predicted from where the tip is and how the forearm is twisted?
Fitted on five sheets, applied to the sixth (start-mark episodes). Measured from frames.csv; my own fits: least squares
on reach, twist and their squares and product, a table of medians, and a random forest (if scikit-learn is there) that
also gets the tip's x and y and the travelled share of the episode.
    D:/AI_Project/.venv/Scripts/python.exe v11_fitted_wrist.py
"""
import numpy
import pandas

frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "row_in_episode", "sheet", "sheet_orientation", "tip_x_m", "tip_y_m", "joint_4", "joint_5", "step_fraction"])
frames = frames[frames["sheet_orientation"].str.contains("START MARK", na = False)].sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
frames["wrist"] = numpy.degrees(frames["joint_5"])
frames["twist"] = numpy.degrees(frames["joint_4"])
frames["reach"] = 100 * numpy.hypot(frames["tip_x_m"], frames["tip_y_m"])
print("start-mark rows:", len(frames))


def design(part):
    reach, twist = part["reach"].to_numpy(), part["twist"].to_numpy()
    return numpy.column_stack([numpy.ones(len(part)), reach, reach ** 2, numpy.abs(twist), twist, twist ** 2, reach * numpy.abs(twist)])


rows = []
for sheet in sorted(frames["sheet"].unique()):
    train, test = frames[frames["sheet"] != sheet], frames[frames["sheet"] == sheet]
    truth = test["wrist"].to_numpy()
    row = {"sheet": sheet, "rows": len(test), "constant_13.5": numpy.abs(truth - 13.5).mean(), "train_median": numpy.abs(truth - train["wrist"].median()).mean()}
    weights = numpy.linalg.lstsq(design(train), train["wrist"].to_numpy(), rcond = None)[0]
    row["least_squares"] = numpy.abs(design(test) @ weights - truth).mean()
    bins_reach, bins_twist = [0, 20, 26, 32, 38, 60], [0, 30, 60, 90, 130]
    key_train = pandas.cut(train["reach"], bins_reach).astype(str) + pandas.cut(train["twist"].abs(), bins_twist, include_lowest = True).astype(str)
    medians = train.groupby(key_train)["wrist"].median()
    key_test = pandas.cut(test["reach"], bins_reach).astype(str) + pandas.cut(test["twist"].abs(), bins_twist, include_lowest = True).astype(str)
    row["table_of_medians"] = numpy.abs(key_test.map(medians).fillna(13.5).to_numpy() - truth).mean()
    try:
        from sklearn.ensemble import RandomForestRegressor
        features = ["reach", "twist", "tip_x_m", "tip_y_m", "step_fraction"]
        forest = RandomForestRegressor(n_estimators = 60, min_samples_leaf = 30, n_jobs = 2, random_state = 0).fit(train[features], train["wrist"])
        row["random_forest"] = numpy.abs(forest.predict(test[features]) - truth).mean()
    except ImportError:
        row["random_forest"] = numpy.nan
    rows.append(row)
table = pandas.DataFrame(rows)
print(table.round(2).to_string(index = False))
weights = table["rows"] / table["rows"].sum()
print("\nmean distance to the demonstrated wrist over all start-mark rows (degrees):")
for column in ("constant_13.5", "train_median", "least_squares", "table_of_medians", "random_forest"):
    print("  %-18s %.2f" % (column, float((table[column] * weights).sum())))
