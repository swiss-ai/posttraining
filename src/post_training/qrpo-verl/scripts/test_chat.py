#!/usr/bin/env python3

import argparse
import os
import sys

import httpx
from openai import OpenAI


DEFAULT_BASE_URL = "http://172.28.33.144:30000/v1"
DEFAULT_MODEL = "Qwen/Qwen3.6-27B-smatrenok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL, e.g. http://<router-ip>:30000/v1",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help="Served model name",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key. Usually EMPTY for this vLLM setup.",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4*4096,
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output",
    )
    args = parser.parse_args()

    # Avoid cluster proxy issues when talking to internal IPs.
    http_client = httpx.Client(
        timeout=httpx.Timeout(300.0),
        trust_env=False,
    )

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        http_client=http_client,
    )

    messages = [
        {"role": "system", "content": args.system},
    ]

    print(f"Base URL: {args.base_url}")
    print(f"Model:    {args.model}")
    print("")
    print("Commands: /exit, /quit, /reset")
    print("")

    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not user_text:
            continue

        if user_text in {"/exit", "/quit"}:
            break

        if user_text == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("conversation reset")
            continue

        messages.append({"role": "user", "content": user_text})

        try:
            if args.no_stream:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )

                assistant_text = response.choices[0].message.content or ""
                print(f"assistant> {assistant_text}")
                messages.append({"role": "assistant", "content": assistant_text})

            else:
                print("assistant> ", end="", flush=True)

                stream = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    stream=True,
                )

                chunks = []

                for event in stream:
                    delta = event.choices[0].delta.content
                    if delta:
                        print(delta, end="", flush=True)
                        chunks.append(delta)

                print()
                assistant_text = "".join(chunks)
                messages.append({"role": "assistant", "content": assistant_text})

        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
            messages.pop()


if __name__ == "__main__":
    main()
