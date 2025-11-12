#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steam Reviews Crawler
- Requires: Python 3.9+
- Install: pip install aiohttp aiofiles tqdm
"""
import asyncio
import aiohttp
import aiofiles
import argparse
import json
import random
import time
from typing import List, Optional
from tqdm import tqdm

STEAM_APP_LIST_URL = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
APP_REVIEWS_URL_TEMPLATE = "https://store.steampowered.com/appreviews/{app_id}?json=1"

# --- helper utilities ---
def chunked(iterable, n):
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(n):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        if chunk:
            yield chunk

# --- core crawler class ---
class SteamReviewsCrawler:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        concurrency: int = 10,
        reviews_per_app: int = 1000,
        per_request: int = 100,          # Steam allows up to 100 per request
        max_retries: int = 5,
        backoff_base: float = 1.0,
        language: str = "all"            # "all" or "english" etc.
    ):
        self.session = session
        self.semaphore = asyncio.Semaphore(concurrency)
        self.reviews_per_app = reviews_per_app
        self.per_request = per_request
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.language = language

    async def fetch_json(self, url, params=None, retry=0):
        # Generic fetcher with retries/backoff
        try:
            async with self.semaphore:
                async with self.session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    # handle 429/5xx with retry
                    if resp.status in (429, 500, 502, 503, 504) and retry < self.max_retries:
                        wait = self.backoff_base * (2 ** retry) + random.random()
                        await asyncio.sleep(wait)
                        return await self.fetch_json(url, params=params, retry=retry+1)
                    else:
                        text = await resp.text()
                        raise aiohttp.ClientResponseError(
                            status=resp.status,
                            request_info=resp.request_info,
                            history=resp.history,
                            message=f"Non-200 status {resp.status} body: {text[:200]}"
                        )
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            if retry < self.max_retries:
                wait = self.backoff_base * (2 ** retry) + random.random()
                await asyncio.sleep(wait)
                return await self.fetch_json(url, params=params, retry=retry+1)
            raise

    async def fetch_app_reviews(self, app_id: int):
        """
        Fetch reviews for a single app_id, using pagination via cursor.
        Returns a list of review dicts (may be large).
        """
        collected = []
        cursor = "*"  # initial cursor required by Steam
        fetched_count = 0

        # We'll loop until we have required reviews or no more reviews
        while fetched_count < self.reviews_per_app:
            to_fetch = min(self.per_request, self.reviews_per_app - fetched_count)
            params = {
                "json": 1,
                "num_per_page": to_fetch,   # alternative param name used by older docs
                "num_per_page": to_fetch,   # ensure present
                "num_per_page": to_fetch,   # repeated intentionally - Steam is forgiving
                "cursor": cursor,
                "language": self.language if self.language != "all" else None,
                "purchase_type": "all",     # all reviews
                "filter": "all"             # all reviews, not only recent
            }
            # Clean None params
            params = {k: v for k, v in params.items() if v is not None}
            url = APP_REVIEWS_URL_TEMPLATE.format(app_id=app_id)
            try:
                resp = await self.fetch_json(url, params=params)
            except Exception as e:
                # If one page fails, break or continue based on policy — here we stop for this app.
                print(f"[WARN] app {app_id} fetch error: {e}")
                break

            # Steam returns something like { 'success': 1, 'query_summary': {...}, 'reviews': [...] }
            reviews = resp.get("reviews") or []
            if not reviews:
                break

            for r in reviews:
                # attach app_id for context
                r["_app_id"] = app_id
                collected.append(r)

            fetched_count += len(reviews)

            # Update cursor; Steam returns a 'cursor' field (sometimes in 'cursor' or inside 'query_summary').
            new_cursor = resp.get("cursor") or resp.get("query_summary", {}).get("cursor")
            if not new_cursor or new_cursor == cursor:
                # no progress - stop
                break
            cursor = new_cursor

            # safety sleep small random to avoid hammering
            await asyncio.sleep(0.1 + random.random() * 0.2)

        return collected

# --- helpers to get app list ---
async def get_app_list(session: aiohttp.ClientSession) -> List[dict]:
    async with session.get(STEAM_APP_LIST_URL, timeout=60) as resp:
        resp.raise_for_status()
        data = await resp.json()
        apps = data.get("applist", {}).get("apps", [])
        return apps

# --- main runner ---
async def run(
    mode: str,
    app_list_file: Optional[str],
    n_apps: int,
    reviews_per_app: int,
    concurrency: int,
    output_file: str,
    randomize: bool = True
):
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": "steam-reviews-crawler/1.0 (respectful-crawler)"}) as session:
        # decide app ids
        if mode == "list":
            if not app_list_file:
                raise ValueError("mode=list requires --app-list-file")
            # read app ids from the file
            with open(app_list_file, "r", encoding="utf-8") as f:
                app_ids = [int(line.strip()) for line in f if line.strip()]
            if n_apps:
                app_ids = app_ids[:n_apps]
        else:
            # mode == "top" -> fetch full app list and pick top N randomly or first N
            apps = await get_app_list(session)
            app_ids = [int(a["appid"]) for a in apps]
            if randomize:
                random.shuffle(app_ids)
            app_ids = app_ids[:n_apps]

        print(f"[INFO] will fetch reviews for {len(app_ids)} apps (reviews_per_app={reviews_per_app})")

        crawler = SteamReviewsCrawler(session=session, concurrency=concurrency, reviews_per_app=reviews_per_app)

        # create tasks in batches to keep memory reasonable
        # We'll process apps sequentially but fetch pages concurrently inside crawler.
        # Optionally do groups of apps concurrently by using asyncio.gather on limited batches.
        out_f = await aiofiles.open(output_file, mode="w", encoding="utf-8")
        try:
            # Process with a progress bar
            for aid in tqdm(app_ids, desc="Apps"):
                try:
                    reviews = await crawler.fetch_app_reviews(aid)
                except Exception as e:
                    print(f"[ERROR] failed to fetch for app {aid}: {e}")
                    reviews = []

                # write reviews as JSON lines
                for r in reviews:
                    await out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

                # polite rate-limiting between apps
                await asyncio.sleep(0.2 + random.random() * 0.3)
        finally:
            await out_f.close()

# --- CLI ---
def parse_args():
    p = argparse.ArgumentParser(description="Steam Reviews Crawler")
    p.add_argument("--mode", choices=["top", "list"], default="top", help="top: sample from all Steam apps; list: use app ids from file")
    p.add_argument("--app-list-file", help="when mode=list: a txt file with appids (one per line)")
    p.add_argument("--n-apps", type=int, default=200, help="number of apps to fetch (100~1000 recommended)")
    p.add_argument("--reviews-per-app", type=int, default=1000, help="max reviews to collect per app")
    p.add_argument("--concurrency", type=int, default=10, help="concurrent HTTP requests")
    p.add_argument("--output", default="steam_reviews.jsonl", help="output JSONL file")
    return p.parse_args()

def main():
    args = parse_args()
    if args.n_apps < 1 or args.n_apps > 5000:
        print("[WARN] n_apps out of recommended range; ensure you're not overloading Steam.")
    # run asyncio
    asyncio.run(run(
        mode=args.mode,
        app_list_file=args.app_list_file,
        n_apps=args.n_apps,
        reviews_per_app=args.reviews_per_app,
        concurrency=args.concurrency,
        output_file=args.output
    ))
    print(f"[DONE] saved reviews to {args.output}")

if __name__ == "__main__":
    main()
