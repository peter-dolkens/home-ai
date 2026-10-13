#!/usr/bin/env python3
"""Summarise a run logged by anthropic-metrics-proxy.py: one row per request, then session totals.

  python3 scripts/anthropic-metrics-report.py outputs/claude-metrics/run1 [--since EPOCH]
"""
import json, sys, time


def main():
    run = sys.argv[1]
    since = float(sys.argv[sys.argv.index("--since") + 1]) if "--since" in sys.argv else 0
    recs = [json.loads(l) for l in open(f"{run}/requests.jsonl") if l.strip()]
    recs = [r for r in recs if r["t_start"] >= since and r["method"] != "HEAD"]  # HEAD = Claude Code's ping
    msgs = [r for r in recs if r["path"].startswith("/v1/messages") and "count_tokens" not in r["path"]]
    if not msgs:
        print("no /v1/messages requests logged")
        return
    t0 = min(r["t_start"] for r in recs)
    print(f"{'#':>3} {'start':>7} {'ttft':>7} {'visible':>7} {'total':>7} {'in(new)':>7} {'c_read':>7} {'c_write':>7} {'out':>6} {'tok/s':>6} {'msgs':>4} {'stop':<10} blocks")
    for r in msgs:
        print(f"{r['id']:>3} {r['t_start'] - t0:>6.1f}s {r.get('ttft_s') or 0:>6.2f}s {r.get('first_visible_s') or 0:>6.2f}s {r.get('total_s') or 0:>6.1f}s "
              f"{r.get('input_tokens') or 0:>7} {r.get('cache_read_input_tokens') or 0:>7} {r.get('cache_creation_input_tokens') or 0:>7} "
              f"{r.get('output_tokens') or 0:>6} {r.get('decode_tok_s') or '-':>6} "
              f"{r.get('n_messages') or 0:>4} {str(r.get('stop_reason')):<10} "
              f"{','.join(b or '?' for b in r.get('blocks') or [])}{'  ERR ' + str(r.get('error') or r.get('status')) if r.get('status', 200) >= 400 or r.get('error') else ''}")
    end = max(r["t_start"] + (r.get("total_s") or 0) for r in recs)
    tin = sum(r.get("input_tokens") or 0 for r in msgs)
    tout = sum(r.get("output_tokens") or 0 for r in msgs)
    busy = sum(r.get("total_s") or 0 for r in msgs)
    ttfts = [r["ttft_s"] for r in msgs if r.get("ttft_s") is not None]
    print()
    print(f"started            {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}")
    print(f"wall clock         {end - t0:.1f}s (first request to last response)")
    print(f"model busy         {busy:.1f}s across {len(msgs)} /v1/messages requests "
          f"({len(recs) - len(msgs)} other requests)")
    print(f"first token        {msgs[0].get('ttft_s')}s on the first request; "
          f"mean TTFT {sum(ttfts) / len(ttfts):.2f}s, max {max(ttfts):.2f}s" if ttfts else "")
    cr = sum(r.get("cache_read_input_tokens") or 0 for r in msgs)
    cw = sum(r.get("cache_creation_input_tokens") or 0 for r in msgs)
    print(f"tokens             input {tin:,} uncached + cache read {cr:,} + cache write {cw:,}, output {tout:,}"
          " (Ollama reports only uncached input)")
    rates = [r["decode_tok_s"] for r in msgs if r.get("decode_tok_s")]
    if rates:
        print(f"decode             {sum(rates) / len(rates):.1f} tok/s mean over {len(rates)} streamed-text replies")
    print(f"errors             {sum(1 for r in recs if r.get('status', 200) >= 400 or r.get('error'))}")


if __name__ == "__main__":
    main()
