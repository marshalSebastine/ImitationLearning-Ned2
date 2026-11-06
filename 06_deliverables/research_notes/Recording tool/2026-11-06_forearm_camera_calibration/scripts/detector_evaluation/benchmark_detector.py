"""
Time the detector on every picture of episodes 037 to 045, with the pictures already
in memory (no disk reading): each picture is detected several times and the median
and the fastest of its runs are kept, so a moment of load from other programs on the
laptop does not count against one picture.

Usage: python benchmark_detector.py [repeats]
"""

import os
import sys
import json
import time
import statistics

import cv2
import numpy

from frame_catalogue import catalogue_pictures
from puck_detector import PuckDetector, detect_puck


OUTPUT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def time_calls(function, pictures: list, repeats: int) -> dict:
    """
    Time ``function(picture)`` on every picture.

    Parameters
    ----------
    function : callable
        Called with one BGR picture.
    pictures : list of numpy.ndarray
        Pictures in memory.
    repeats : int
        Calls per picture.

    Returns
    -------
    dict
        Mean, median, 95th and 99th percentile and maximum, in milliseconds, of the
        per-picture medians, and the maximum of the per-picture fastest runs.
    """

    for picture in pictures[:20]:
        function(picture)
    medians = []
    fastest = []
    for picture in pictures:
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            function(picture)
            durations.append(1000.0 * (time.perf_counter() - start))
        medians.append(statistics.median(durations))
        fastest.append(min(durations))
    medians = numpy.array(medians)
    return {
        "pictures": len(pictures),
        "repeats_per_picture": repeats,
        "mean_ms": round(float(medians.mean()), 3),
        "median_ms": round(float(numpy.median(medians)), 3),
        "percentile_95_ms": round(float(numpy.percentile(medians, 95)), 3),
        "percentile_99_ms": round(float(numpy.percentile(medians, 99)), 3),
        "max_of_medians_ms": round(float(medians.max()), 3),
        "max_of_fastest_ms": round(float(max(fastest)), 3),
        "pictures_over_5_ms_by_median": int((medians > 5.0).sum())
    }


def main():

    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    pictures = [cv2.imread(entry["path"]) for entry in catalogue_pictures()]
    detector = PuckDetector()
    results = {}
    for thread_count in (1, cv2.getNumberOfCPUs()):
        cv2.setNumThreads(thread_count)
        results[f"kept_detector_opencv_threads_{thread_count}"] = time_calls(detector.detect, pictures, repeats)
        results[f"detect_puck_function_opencv_threads_{thread_count}"] = time_calls(detect_puck, pictures, repeats)
    results["opencv_version"] = cv2.__version__
    results["numpy_version"] = numpy.__version__
    for name, values in results.items():
        print(name, values)
    with open(os.path.join(OUTPUT_FOLDER, "benchmark.json"), "w") as output_file:
        json.dump(results, output_file, indent = 1)


if __name__ == "__main__":
    main()
