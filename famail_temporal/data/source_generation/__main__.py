"""Allow `python -m famail_temporal.data.source_generation`."""
import sys
from famail_temporal.data.source_generation.cli import main

if __name__ == "__main__":
    sys.exit(main())
