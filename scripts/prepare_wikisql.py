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

def laod_jsonl(path:Path)->List[Dict[str, Any]]:
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
    Convert WikiSQL structured SQL + table schema into a SQL string.
    This function depends on the exact structure of the WikiSQL dataset you download.
    We'll implement it once we inspect a real example in Colab.
    """
    # Example structure (to be confirmed in Colab):
    # sql = entry["sql"]
    # table = entry["table"]
    #
    # sel_idx = sql["sel"]            # selected column index
    # agg_id = sql["agg"]             # aggregation id
    # conds = sql["conds"]            # list of [col_idx, op_id, value]
    #
    # header = table["header"]        # list of column names
    #
    # Then build: SELECT ... FROM ... WHERE ...
    raise NotImplementedError("Implement sql_to_string after inspecting WikiSQL format in Colab.")
def process_split(input_path: Path, output_path: Path) -> None:
    rows = load_jsonl(input_path)
    with output_path.open("w", encoding="utf-8") as out_f:
        for entry in rows:
            question = entry["question"]
            sql_str = sql_to_string(entry)
            obj = {"question": question, "sql": sql_str}
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
def main() -> None:
    # Adjust these paths in Colab once you know where Kaggle / WikiSQL files are
    root = Path("data/external/wikisql")
    out_root = Path("data/raw")
    # Example filenames; will confirm in Colab:
    train_in = root / "train.jsonl"
    dev_in = root / "dev.jsonl"
    test_in = root / "test.jsonl"
    train_out = out_root / "wikisql_train.jsonl"
    dev_out = out_root / "wikisql_dev.jsonl"
    test_out = out_root / "wikisql_test.jsonl"
    process_split(train_in, train_out)
    process_split(dev_in, dev_out)
    process_split(test_in, test_out)
if __name__ == "__main__":
    main()

