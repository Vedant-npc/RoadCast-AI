from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from traffic_collector import TrafficCollector
from traffic_store import TrafficStore

DEFAULT_DATABASE = Path(__file__).resolve().parent / "data" / "traffic_history.sqlite3"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TomTom traffic observations.")
    parser.add_argument("--once", action="store_true", help="Collect one polling cycle and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Query providers without database writes.")
    parser.add_argument("--status", action="store_true", help="Print database status and exit.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE, help="SQLite database path.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    store = TrafficStore(args.db)
    if args.status:
        print(json.dumps({"locations": len(TrafficCollector(store).locations), **store.status()}, indent=2))
        return

    collector = TrafficCollector(store, dry_run=args.dry_run)
    if args.dry_run or args.once:
        print(json.dumps(collector.collect_once(), indent=2))
        return
    collector.run_forever()


if __name__ == "__main__":
    main()
