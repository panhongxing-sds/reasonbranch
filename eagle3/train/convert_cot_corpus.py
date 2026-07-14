#!/usr/bin/env python3
"""Convert ShareGPT-style {from,value} corpus to SpecForge {role,content} format."""
import argparse, json

ROLE = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = 0
    with open(args.inp) as f, open(args.out, "w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            convs = r.get("conversations", [])
            out_convs = []
            for c in convs:
                role = ROLE.get(c.get("from") or c.get("role"))
                content = c.get("value") if "value" in c else c.get("content")
                if role is None or content is None:
                    continue
                out_convs.append({"role": role, "content": content})
            if len(out_convs) >= 2:
                g.write(json.dumps({"id": r.get("id", f"s{n}"), "conversations": out_convs}, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} -> {args.out}")


if __name__ == "__main__":
    main()
