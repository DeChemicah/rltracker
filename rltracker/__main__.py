import argparse

from .app import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rocket League scoreboard stats tracker")
    ap.add_argument("--port", type=int, default=5177)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    main(port=a.port, open_browser=not a.no_browser)
