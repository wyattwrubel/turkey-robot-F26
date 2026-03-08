"""
main.py — Server entrypoint

Usage:
    python main.py           start the server
    python main.py --test    run test suite only (no server)
"""

import sys
import uvicorn
   
def main():
    if "--test" in sys.argv:
        from tests import run_tests
        ok = run_tests()
        sys.exit(0 if ok else 1)

    print("\n  CDS Control Server")
    print("  http://localhost:8000\n")

    uvicorn.run(
        "app:app",
        host      = "0.0.0.0",
        port      = 8000,
        log_level = "warning",   # uvicorn noise suppressed — our logger handles output
        reload    = False,
    )


if __name__ == "__main__":     
    main()
