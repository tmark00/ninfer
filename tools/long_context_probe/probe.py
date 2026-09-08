# -*- coding: utf-8 -*-
"""Long-context probe: does the model still read, and still reason, at a given depth?

Each arm starts its own server, sends one request, and is scored programmatically. One request an
arm is deliberate: this engine keeps a single rewrite checkpoint per lane, so a second question
against the same haystack diverges before it and pays a full prefill. Asking every question at
once costs one prefill and still scores per item.

  python probe.py --exe PATH --model PATH [--mode needle|reasoning] [--arm CTX:SCALE:BYTES ...]

Default arms compare an unscaled baseline against YaRN on a byte-identical haystack, then push
past the native ceiling. The first pair is the one that isolates scaling from depth.
"""
import argparse, io, json, os, re, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import haystack

DEFAULT_ARMS = ["262144:1.0:640000", "393216:1.5:640000", "393216:1.5:1340000"]


def start(exe, model, context, scale, log_path):
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [exe, model, "--max-context", context, "--kv-capacity", context, "--max-concurrency", "1",
         "--kv-dtype", "rk2v4-e8", "--prefill-chunk", "1024", "--rope-scale", scale,
         "--spec", "mtp", "--draft-tokens", "6", "--lm-head-draft",
         "--default-max-tokens", "16384"],
        stdout=log, stderr=subprocess.STDOUT)
    for _ in range(1800):
        time.sleep(0.5)
        if "listening on" in open(log_path, encoding="utf-8", errors="replace").read():
            time.sleep(1.0)
            return process, log
    process.kill()
    raise RuntimeError("server did not come up for context=%s scale=%s" % (context, scale))


def ask(port, text):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": text}],
            "max_tokens": 6000, "temperature": 0.0}
    request = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    message = json.loads(urllib.request.urlopen(request, timeout=7200).read())["choices"][0]["message"]
    return message.get("content") or "", len(message.get("reasoning_content") or "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("needle", "reasoning"), default="reasoning")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--arm", action="append", default=None,
                        help="CONTEXT:ROPE_SCALE:HAYSTACK_BYTES, repeatable")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    if args.mode == "needle":
        question, expected = haystack.NEEDLE_QUESTION, haystack.needle_expected()
    else:
        question, expected = haystack.REASONING_QUESTION, haystack.reasoning_expected()

    print("== long-context probe: %s ==  %s" % (args.mode, time.strftime("%Y-%m-%d %H:%M")))
    print("== expected: %s ==\n" % expected)

    for arm in (args.arm or DEFAULT_ARMS):
        context, scale, size = arm.split(":")
        corpus = haystack.build(int(size))
        log_path = os.path.join(args.out, "probe-%s-%s-%s.log" % (context, scale, size))
        process, log = start(args.exe, args.model, context, scale, log_path)
        try:
            answer, reasoning_chars = ask(args.port, corpus + question)
        finally:
            time.sleep(0.5)
            process.kill()
            log.close()
            time.sleep(1.0)

        pattern = r"(P[1-6])\s*=\s*(\S+)" if args.mode == "reasoning" else r"([A-Z]+)\s*=\s*(\d+)"
        got = {k.upper(): v.strip() for k, v in re.findall(pattern, answer)}
        hits = sum(1 for k in expected if got.get(k) == expected[k])
        detail = "  ".join("%s:%s" % (k, "OK" if got.get(k) == expected[k] else (got.get(k) or "-"))
                           for k in sorted(expected))
        text = open(log_path, encoding="utf-8", errors="replace").read()
        done = re.search(r"\[req 1\] done.*?prompt=(\d+) gen=(\d+).*?prefill=([0-9.]+)tok/s.*?"
                         r"speculative=mtp ([0-9.]+)tok/round \(([0-9.]+)%\)", text)
        print("ctx=%-7s scale=%-4s  %d/%d   %s" % (context, scale, hits, len(expected), detail))
        if done:
            print("    prompt=%s tokens  gen=%s  reasoning=%d chars  prefill=%s tok/s  MTP %s (%s%%)"
                  % (done.group(1), done.group(2), reasoning_chars, done.group(3),
                     done.group(4), done.group(5)))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
