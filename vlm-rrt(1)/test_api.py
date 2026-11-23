"""Throwaway check: which NVIDIA vision models respond for your key? Delete after use."""
import math
import sys
import time

import requests

from vlm_rrt import Environment2D, Node, PromptManager, SnapshotRenderer, VLMRRTConfig

cfg = VLMRRTConfig()
BASE = "https://integrate.api.nvidia.com/v1"
HEADERS = {"Authorization": f"Bearer {cfg.api_key}", "Accept": "application/json"}
print(f"key loaded: {bool(cfg.api_key)}\n")

# Usage: python test_api.py <model-id>  -> runs the planner's kind of image request on that model
if len(sys.argv) > 1:
    model = sys.argv[1]
    if "sanity" in sys.argv[2:]:
        # Does the model follow the goal? Empty map, goal placed in each of 8 directions.
        extra = {"chat_template_kwargs": {"enable_thinking": False}} if "--nothink" in sys.argv else {}
        names = list(PromptManager.DIRECTIONS)
        for mode in ("few_shot", "zero_shot"):
            hits, answers = 0, []
            for name in names:
                ang = PromptManager.DIRECTIONS[name]
                goal = (250 + 180 * math.cos(ang), 250 + 180 * math.sin(ang))
                env = Environment2D(500, 500, (250, 250), goal, [])
                img = SnapshotRenderer.render_base64(env, [Node(250, 250)], Node(250, 250))
                content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                           {"type": "text", "text": PromptManager.build_prompt(mode)}]
                payload = {"model": model, "messages": [{"role": "user", "content": content}],
                           "max_tokens": 100, "temperature": 0.2, **extra}
                time.sleep(1.8)  # stay under the rate limit
                try:
                    r = requests.post(f"{BASE}/chat/completions", headers=HEADERS, json=payload, timeout=60)
                    got = PromptManager.parse_direction(r.json()["choices"][0]["message"].get("content") or "") if r.ok else f"HTTP {r.status_code}"
                except Exception as e:
                    got = type(e).__name__
                hits += got == name
                answers.append((name, got))
            print(f"\n[{mode}] correct {hits}/8")
            for name, got in answers:
                print(f"   goal is {name:10s} -> model said {got}")
        sys.exit()
    env = Environment2D(500, 500, (50, 50), (450, 450), [(200, 200, 80, 80), (120, 300, 70, 90)])
    nodes = [Node(50, 50), Node(65, 60, 0), Node(80, 72, 1)]
    img = SnapshotRenderer.render_base64(env, nodes, nodes[-1])
    print(f"model: {model} | image payload: {len(img) / 1024:.0f} KB")
    variants = [
        ("few_shot, max_tokens=100 (current)", "few_shot", 100, {}),
        ("few_shot, max_tokens=1500", "few_shot", 1500, {}),
        ("few_shot, max_tokens=100, thinking off", "few_shot", 100,
         {"chat_template_kwargs": {"enable_thinking": False}}),
    ]
    for label, mode, mt, extra in variants:
        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                   {"type": "text", "text": PromptManager.build_prompt(mode)}]
        payload = {"model": model, "messages": [{"role": "user", "content": content}],
                   "max_tokens": mt, "temperature": 0.2, **extra}
        t = time.time()
        print(f"\n[{label}]")
        try:
            r = requests.post(f"{BASE}/chat/completions", headers=HEADERS, json=payload, timeout=90)
            dt = time.time() - t
            if not r.ok:
                print(f"   HTTP {r.status_code} in {dt:.1f}s: {r.text[:300]}")
                continue
            j = r.json()
            ch = j["choices"][0]
            msg = ch["message"]
            text = msg.get("content") or ""
            print(f"   {dt:.1f}s | finish_reason={ch.get('finish_reason')} | tokens={j.get('usage', {}).get('completion_tokens')}")
            print(f"   content: {text[:300]!r}")
            print(f"   reasoning (start): {(msg.get('reasoning_content') or '')[:200]!r}")
            print(f"   parsed direction: {PromptManager.parse_direction(text)}")
        except Exception as e:
            print(f"   FAILED after {time.time() - t:.1f}s: {type(e).__name__}")
    sys.exit()

# 1) Is the key accepted at all? (cheap request, no model involved)
t = time.time()
try:
    r = requests.get(f"{BASE}/models", headers=HEADERS, timeout=20)
    print(f"[list models] HTTP {r.status_code} in {time.time() - t:.1f}s")
    ids = [m["id"] for m in r.json().get("data", [])] if r.ok else []
except Exception as e:
    print(f"[list models] FAILED after {time.time() - t:.1f}s: {type(e).__name__}")
    ids = []
if not ids:
    print("\nCould not list models. Check: key in .env, internet/VPN/firewall, or try again later.")

# 2) Which vision models does your account see?
wanted = ["meta/llama-3.2-90b-vision-instruct", "meta/llama-3.2-11b-vision-instruct"]
vision = [i for i in ids if any(k in i.lower() for k in ("vision", "-vl", "llava", "neva", "vila"))]
print(f"\nvision-like models listed: {len(vision)}")
for m in vision[:15]:
    print("   ", m)
for w in wanted:
    print(f"{w} listed: {w in ids}")

# 3) Text-only ping to each candidate, 30 s limit each
candidates = [w for w in wanted if w in ids] + [v for v in vision if v not in wanted][:4]
print("\nping (text only, 30 s limit):")
for model in candidates or wanted:
    payload = {"model": model, "messages": [{"role": "user", "content": "Reply with one word: ready"}],
               "max_tokens": 10, "temperature": 0.2}
    t = time.time()
    try:
        r = requests.post(f"{BASE}/chat/completions", headers=HEADERS, json=payload, timeout=30)
        dt = time.time() - t
        if r.ok:
            print(f"  OK     {dt:5.1f}s  {model}")
        else:
            print(f"  HTTP {r.status_code} {dt:5.1f}s  {model}  {r.text[:120]!r}")
    except Exception as e:
        print(f"  FAIL   {time.time() - t:5.1f}s  {model}  {type(e).__name__}: {str(e)[:200]}")