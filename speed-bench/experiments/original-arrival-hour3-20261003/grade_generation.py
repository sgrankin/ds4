"""Reuse the item-2 factual grader with the item-3 output directory."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ITEM2 = HERE.parent / 'resident-functional-hour2-20261003'
spec = importlib.util.spec_from_file_location('item2_generation_grade', ITEM2 / 'grade_generation.py')
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)
grader.HERE = ITEM2
grader.OUT = HERE / 'generation'
if __name__ == '__main__':
    grader.main()
