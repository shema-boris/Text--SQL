import json
from pathlib import Path
from typing import Any, Dict, List
AGG_MAP={
    0:"",
    1:"MAX",
    2:"MIN",
    3:"COUNT",
    4:"SUM",
    5:"AVG",
}
# This may need adjustment once we see the exact WikiSQL format you download
OP_MAP = {
    0: "=",
    1: ">",
    2: "<",
    3: ">=",    # placeholder; we will verify in Colab
    4: "<=",
    5: "!=",
}

def load_jsonl(path:Path)->List[Dict[str, Any]]:
    rows:List[Dict[str,Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def sql_to_string(entry: Dict[str, Any]) -> str:
    """
    For this Kaggle variant, the SQL is already given as a plain string
    in the 'answer' field.
    """
    sql_str = entry["answer"]
    # Optional: normalize whitespace / case if you like
    return sql_str
def process_split(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)  # list of dicts

    with output_path.open("w", encoding="utf-8") as out_f:
        for entry in rows:
            question = entry["question"]
            sql_str = sql_to_string(entry)
            obj = {"question": question, "sql": sql_str}
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
def main() -> None:
    root = Path("/content/data/external/wikisql")  # adjust to actual folder in Colab
    out_root = Path("data/raw")
    train_in = root / "wikisql_train.json"
    dev_in = root / "wikisql_validation.json"
    test_in = root / "wikisql_test.json"
    train_out = out_root / "wikisql_train.jsonl"
    dev_out = out_root / "wikisql_dev.jsonl"
    test_out = out_root / "wikisql_test.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)
    process_split(train_in, train_out)
    process_split(dev_in, dev_out)
    process_split(test_in, test_out)
if __name__ == "__main__":
    main()
    

