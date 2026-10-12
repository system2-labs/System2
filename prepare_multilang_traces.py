"""prepare_multilang_traces.py - Ingest and prepare multi-language code execution traces.

Languages and sources:
1. Python: cruxeval, mbpp, humaneval, mental_traces, diwank
2. Systems & Compiled: Rust, C++, C# (CS), Go, Swift, Scala, Java (from xhwl/cruxeval-x)
3. Scripting & Web: TypeScript, Bash (Shell) (from xhwl/cruxeval-x)
4. Relational Database / Query: SQL (from b-mc2/sql-create-context)

Standardizes all examples into System2 feedforward 4-choice deliberation schema:
- question: str
- choices: list[str] (4 candidate outputs: 1 ground truth, 3 plausible distractors)
- text: str (formatted prompt with choices A-D)
- label: int (0..3)
- source: str (dataset identifier with language prefix)

Saves to ./data/code_traces as a DatasetDict on disk.
"""

import ast
import random
import re
from datasets import Dataset, DatasetDict, load_dataset


COMMON_EXCEPTIONS = [
    "IndexError: list index out of range",
    "ValueError: invalid literal",
    "TypeError: unsupported operand type",
    "KeyError: 'result'",
    "RecursionError: maximum recursion depth exceeded",
    "None",
    "[]",
    "{}",
    "0",
    "-1",
    "True",
    "False"
]


def synthesize_distractors(target, pool=None):
    """Generate 3 plausible distractors for a given target output string."""
    target_str = str(target).strip()
    distractors = set()

    # Try literal eval to determine python type if applicable
    try:
        val = ast.literal_eval(target_str)
    except Exception:
        val = None

    if isinstance(val, bool):
        opp = "False" if val is True else "True"
        distractors.add(opp)
        distractors.add("None")
        distractors.add("1" if val is False else "0")

    elif isinstance(val, int) and not isinstance(val, bool):
        distractors.add(str(val + 1))
        distractors.add(str(val - 1))
        distractors.add(str(val + 2) if val >= 0 else str(val - 2))
        distractors.add(str(0 if val != 0 else 1))
        distractors.add(str(-val if val != 0 else -1))
        distractors.add(str(val * 2 if val != 0 else 2))

    elif isinstance(val, float):
        distractors.add(str(round(val + 1.0, 4)))
        distractors.add(str(round(val - 1.0, 4)))
        distractors.add(str(round(val * 2.0, 4)))
        distractors.add(str(round(val / 2.0, 4)))

    elif isinstance(val, list):
        if len(val) > 0:
            distractors.add(str(list(reversed(val))))
            distractors.add(str(val[:-1]))
            distractors.add(str(val[1:]))
            if isinstance(val[0], int):
                new_v = [x + 1 if isinstance(x, int) else x for x in val]
                distractors.add(str(new_v))
        distractors.add("[]")
        distractors.add("IndexError: list index out of range")

    elif isinstance(val, tuple):
        if len(val) > 0:
            distractors.add(str(tuple(reversed(val))))
            distractors.add(str(val[:-1]))
            distractors.add(str(val[1:]))
        distractors.add("()")
        distractors.add(str(list(val)))

    elif isinstance(val, set):
        if len(val) > 0:
            distractors.add("set()")
        distractors.add(str(list(val)))

    elif isinstance(val, dict):
        if len(val) > 0:
            distractors.add("{}")
            distractors.add(str(list(val.keys())))
            distractors.add(str(list(val.values())))
        distractors.add("{}")

    elif isinstance(val, str):
        if len(val) > 1:
            distractors.add(repr(val[:-1]))
            distractors.add(repr(val[1:]))
            distractors.add(repr(val[::-1]))
            distractors.add(repr(val.upper() if val != val.upper() else val.lower()))
        distractors.add("''")
        distractors.add("None")

    # Clean distractors to avoid matching target and reject runaway outputs
    distractors = {d for d in distractors if d != target_str and 0 < len(d.strip()) <= 1000}

    # Pool fallback
    if pool and len(distractors) < 3:
        shuffled_pool = random.sample(pool, min(len(pool), 50))
        for p in shuffled_pool:
            p_str = str(p).strip()
            if p_str != target_str and p_str not in distractors and 0 < len(p_str) <= 1000:
                distractors.add(p_str)
                if len(distractors) >= 6:
                    break

    # Exception fallback
    for exc in COMMON_EXCEPTIONS:
        if len(distractors) >= 3:
            break
        if exc != target_str:
            distractors.add(exc)

    if len(distractors) < 3:
        return None, None

    chosen = list(distractors)[:3]
    choices = [target_str] + chosen
    random.shuffle(choices)
    label = choices.index(target_str)
    return choices, label


def create_sample(question, target_answer, source, pool=None):
    """Formats question and target into System2 record dict."""
    target_str = str(target_answer).strip()
    if len(target_str) == 0 or len(target_str) > 1000 or len(question.strip()) > 4000:
        return None

    choices, label = synthesize_distractors(target_str, pool=pool)
    if choices is None or any(len(c) > 1000 for c in choices):
        return None

    formatted_text = (
        f"Question: {question.strip()}\n"
        f"(A) {choices[0]}\n"
        f"(B) {choices[1]}\n"
        f"(C) {choices[2]}\n"
        f"(D) {choices[3]}"
    )
    return {
        "question": question.strip(),
        "choices": choices,
        "text": formatted_text,
        "label": label,
        "source": source
    }


def parse_cruxeval_x_lang(lang):
    """Parse one language split from xhwl/cruxeval-x."""
    print(f"Loading xhwl/cruxeval-x split '{lang}'...")
    ds = load_dataset("xhwl/cruxeval-x", split=lang)
    
    all_targets = []
    pairs = []
    for ex in ds:
        code = ex["code"]
        out_r = ex["output_reasoning"]
        parts = out_r.split("????")
        if len(parts) == 2:
            prefix, suffix = parts[0], parts[1]
            p_idx = code.find(prefix)
            if p_idx != -1:
                start = p_idx + len(prefix)
                s_idx = code.find(suffix, start)
                if s_idx != -1:
                    target = code[start:s_idx].strip()
                    if target:
                        q = f"Given the following {lang} program:\n{code}\nWhat is the expected execution output?"
                        pairs.append((q, target))
                        all_targets.append(target)

    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.8)
    train_raw = pairs[:split_idx]
    test_raw = pairs[split_idx:]

    train_samples, test_samples = [], []
    src = f"cruxeval_x_{lang.lower()}"
    for q, t in train_raw:
        s = create_sample(q, t, src, pool=all_targets)
        if s:
            train_samples.append(s)
    for q, t in test_raw:
        s = create_sample(q, t, src, pool=all_targets)
        if s:
            test_samples.append(s)

    print(f"CruxEval-X ({lang}) parsed: {len(train_samples)} train, {len(test_samples)} test.")
    return train_samples, test_samples


def parse_sql_traces(max_samples=5000):
    """Parse relational database SQL query reasoning from b-mc2/sql-create-context."""
    print("Loading b-mc2/sql-create-context...")
    ds = load_dataset("b-mc2/sql-create-context", split="train")
    
    all_answers = ds[:max_samples * 2]["answer"]
    
    indices = list(range(len(ds)))
    random.seed(42)
    random.shuffle(indices)
    chosen = indices[:max_samples]
    
    pairs = []
    for idx in chosen:
        ex = ds[idx]
        q_text = ex["question"]
        ctx = ex["context"]
        ans = ex["answer"]
        q = f"Given the database schema:\n{ctx}\nWhich SQL query correctly executes the request: '{q_text}'?"
        pairs.append((q, ans))

    split_idx = int(len(pairs) * 0.8)
    train_raw = pairs[:split_idx]
    test_raw = pairs[split_idx:]

    train_samples, test_samples = [], []
    for q, t in train_raw:
        s = create_sample(q, t, "sql_create_context", pool=all_answers)
        if s:
            train_samples.append(s)
    for q, t in test_raw:
        s = create_sample(q, t, "sql_create_context", pool=all_answers)
        if s:
            test_samples.append(s)

    print(f"SQL Traces parsed: {len(train_samples)} train, {len(test_samples)} test.")
    return train_samples, test_samples


def parse_cruxeval():
    print("Loading cruxeval-org/cruxeval...")
    ds = load_dataset("cruxeval-org/cruxeval", split="test")
    all_targets = [str(ex["output"]).strip() for ex in ds]
    train_idx = set(range(0, 800, 2))
    train_samples, test_samples = [], []
    for i, ex in enumerate(ds):
        q = f"Given the following Python function:\n{ex['code']}\nWhat is the output of `{ex['code'].split('def ')[1].split('(')[0]}({ex['input']})`?"
        target = ex["output"]
        sample = create_sample(q, target, "cruxeval_py", pool=all_targets)
        if sample:
            if i in train_idx:
                train_samples.append(sample)
            else:
                test_samples.append(sample)
    print(f"CRUXEval (Py) parsed: {len(train_samples)} train, {len(test_samples)} test.")
    return train_samples, test_samples


def parse_mbpp():
    print("Loading google-research-datasets/mbpp...")
    ds = load_dataset("google-research-datasets/mbpp")
    all_targets = []
    for split_name in ds.keys():
        for ex in ds[split_name]:
            for t in ex.get("test_list", []):
                m = re.match(r"assert\s+(.*?)\s*==\s*(.*?)$", t.strip())
                if m:
                    all_targets.append(m.group(2).strip())

    train_samples, test_samples = [], []
    for split_name in ["train", "validation"]:
        for ex in ds[split_name]:
            code = ex["code"]
            for t in ex.get("test_list", []):
                m = re.match(r"assert\s+(.*?)\s*==\s*(.*?)$", t.strip())
                if m:
                    call = m.group(1).strip()
                    expected = m.group(2).strip()
                    q = f"Given the following Python function:\n{code}\nWhat is the return value of `{call}`?"
                    sample = create_sample(q, expected, "mbpp", pool=all_targets)
                    if sample:
                        train_samples.append(sample)

    for ex in ds["test"]:
        code = ex["code"]
        for t in ex.get("test_list", []):
            m = re.match(r"assert\s+(.*?)\s*==\s*(.*?)$", t.strip())
            if m:
                call = m.group(1).strip()
                expected = m.group(2).strip()
                q = f"Given the following Python function:\n{code}\nWhat is the return value of `{call}`?"
                sample = create_sample(q, expected, "mbpp", pool=all_targets)
                if sample:
                    test_samples.append(sample)

    print(f"MBPP parsed: {len(train_samples)} train, {len(test_samples)} test.")
    return train_samples, test_samples


def parse_humaneval():
    print("Loading openai/openai_humaneval...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    all_targets = []
    for ex in ds:
        entry = ex["entry_point"]
        for line in ex["test"].split("\n"):
            line = line.strip()
            if line.startswith("assert "):
                m = re.search(r"assert\s+(?:candidate|" + re.escape(entry) + r")\s*\((.*?)\)\s*==\s*(.*?)$", line)
                if m:
                    all_targets.append(m.group(2).strip().split("#")[0].strip())

    task_ids = [ex["task_id"] for ex in ds]
    random.seed(42)
    random.shuffle(task_ids)
    train_tasks = set(task_ids[:110])

    train_samples, test_samples = [], []
    for ex in ds:
        code = ex["prompt"] + ex["canonical_solution"]
        entry = ex["entry_point"]
        for line in ex["test"].split("\n"):
            line = line.strip()
            if line.startswith("assert "):
                m = re.search(r"assert\s+(?:candidate|" + re.escape(entry) + r")\s*\((.*?)\)\s*==\s*(.*?)$", line)
                if m:
                    args = m.group(1).strip()
                    expected = m.group(2).strip().split("#")[0].strip()
                    q = f"Given the following Python function:\n{code}\nWhat is the return value of `{entry}({args})`?"
                    sample = create_sample(q, expected, "humaneval", pool=all_targets)
                    if sample:
                        if ex["task_id"] in train_tasks:
                            train_samples.append(sample)
                        else:
                            test_samples.append(sample)

    print(f"HumanEval parsed: {len(train_samples)} train, {len(test_samples)} test.")
    return train_samples, test_samples


def parse_mental_traces():
    print("Loading ILoveBuns/python-mental-execution-traces...")
    ds = load_dataset("ILoveBuns/python-mental-execution-traces", split="train")
    all_targets = []
    extracted = []
    for ex in ds:
        p = ex["prompt"]
        c = ex["completion"]
        code_match = re.search(r"```python\n(.*?)\n```", p, re.DOTALL)
        code = code_match.group(1).strip() if code_match else p.strip()
        out_match = re.search(r"Exact output:\s*```(?:text)?\n(.*?)\n```", c, re.DOTALL)
        out = out_match.group(1).strip() if out_match else ""
        if code and out:
            extracted.append((code, out))
            all_targets.append(out)

    random.seed(42)
    random.shuffle(extracted)
    train_raw = extracted[:10500]
    test_raw = extracted[10500:]

    train_samples, test_samples = [], []
    for code, out in train_raw:
        q = f"What is the standard output of the following Python program?\n{code}"
        s = create_sample(q, out, "mental_traces", pool=all_targets)
        if s:
            train_samples.append(s)

    for code, out in test_raw:
        q = f"What is the standard output of the following Python program?\n{code}"
        s = create_sample(q, out, "mental_traces", pool=all_targets)
        if s:
            test_samples.append(s)

    print(f"Mental Traces parsed: {len(train_samples)} train, {len(test_samples)} test.")
    return train_samples, test_samples


def parse_diwank():
    print("Loading diwank/python-code-execution-output...")
    ds_train = load_dataset("diwank/python-code-execution-output", split="train")
    ds_test = load_dataset("diwank/python-code-execution-output", split="test")
    all_targets = [str(ex["output"]).strip() for ex in ds_train]

    train_samples, test_samples = [], []
    for ex in ds_train:
        q = f"What is the execution output of the following Python code?\n{ex['code']}"
        s = create_sample(q, str(ex["output"]).strip(), "diwank", pool=all_targets)
        if s:
            train_samples.append(s)

    for ex in ds_test:
        q = f"What is the execution output of the following Python code?\n{ex['code']}"
        s = create_sample(q, str(ex["output"]).strip(), "diwank", pool=all_targets)
        if s:
            test_samples.append(s)

    print(f"Diwank parsed: {len(train_samples)} train, {len(test_samples)} test.")
    return train_samples, test_samples


def main():
    print("================================================================")
    print("PREPARING MULTI-LANGUAGE SYSTEM2 CODE DELIBERATION (TRACK B)")
    print("================================================================")
    
    random.seed(42)

    # 1. Python foundation
    c_train, c_test = parse_cruxeval()
    m_train, m_test = parse_mbpp()
    h_train, h_test = parse_humaneval()
    mt_train, mt_test = parse_mental_traces()
    d_train, d_test = parse_diwank()

    all_train = c_train + m_train + h_train + mt_train + d_train
    all_test = c_test + m_test + h_test + mt_test + d_test

    # 2. Multi-language CruxEval-X splits
    languages = ["Rust", "CS", "Cpp", "Shell", "TypeScript", "Java", "Go", "Swift", "Scala"]
    for lang in languages:
        l_train, l_test = parse_cruxeval_x_lang(lang)
        all_train.extend(l_train)
        all_test.extend(l_test)

    # 3. SQL Relational Queries
    sql_train, sql_test = parse_sql_traces(max_samples=5000)
    all_train.extend(sql_train)
    all_test.extend(sql_test)

    random.shuffle(all_train)
    random.shuffle(all_test)

    train_ds = Dataset.from_list(all_train)
    test_ds = Dataset.from_list(all_test)

    dataset_dict = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })

    output_dir = "./data/code_traces"
    print("\n----------------------------------------------------------------")
    print(f"Total Prepared System2 Multi-Language Dataset:")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Test samples:  {len(test_ds)}")
    print(f"Saving to {output_dir}...")
    dataset_dict.save_to_disk(output_dir)
    print(f"Successfully saved {output_dir}!")
    print("================================================================")


if __name__ == "__main__":
    main()
