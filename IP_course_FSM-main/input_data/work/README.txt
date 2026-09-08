This is the disposable scratch directory for prepare_inputs.py.

A rebuild may create:
  npvm/   extracted OMX demand matrices
  skims/  direct-mode and public-transport skim matrices
  gtfs/   processed GTFS tables, diagnostics, and stop lookups

These files are packaged into input_data/prepared/ and removed after a
successful build. Use --keep-work to retain them for teaching or diagnostics.
Main.py does not read this directory.
